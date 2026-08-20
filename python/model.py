"""Entity-encoder + causal temporal transformer with a KV cache.

Two execution paths sharing weights:
  forward_seq: full-sequence causal attention (training; parallel over steps)
  step:        incremental decoding with per-stream KV caches (rollout; O(1)
               new work per decision instead of O(T) recompute)

Equivalence of the two paths is asserted by `python3 python/model.py`.

Window semantics: positions are window-relative (0..CTX-1). Streams whose
episodes exceed CTX slide (cache rolls left); cached K/V keep their original
position embeddings, a slight train/rollout mismatch for >CTX episodes (<1%%
of gen1 battles) - accepted, documented.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_ACTIONS = 10
N_MON, MON_INTS, MON_FLOATS = 12, 6, 8
CTX = 128


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward_seq(self, x, pad):
        """x (B,T,d); pad (B,T) True where padded."""
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        shape = (B, T, self.h, self.dh)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        keep = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        keep = keep[None, None] & ~pad[:, None, None, :]
        keep = keep | torch.eye(T, dtype=torch.bool, device=x.device)[None, None]  # NaN guard
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
        x = x + self.proj(o.transpose(1, 2).reshape(B, T, d))
        return x + self.ff(self.ln2(x))

    def forward_step(self, x, kc, vc, idx, pos):
        """x (k,d) new-step activations; kc/vc (N,CTX,H,dh) caches; pos (k,)."""
        K, d = x.shape[0], x.shape[1]
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        kc[idx, pos] = k.view(K, self.h, self.dh).to(kc.dtype)
        vc[idx, pos] = v.view(K, self.h, self.dh).to(vc.dtype)
        t = int(pos.max()) + 1
        Kc = kc[idx, :t].transpose(1, 2).to(q.dtype)  # (k,H,t,dh)
        Vc = vc[idx, :t].transpose(1, 2).to(q.dtype)
        keep = (torch.arange(t, device=x.device)[None] <= pos[:, None])[:, None, None, :]
        o = F.scaled_dot_product_attention(
            q.view(K, 1, self.h, self.dh).transpose(1, 2), Kc, Vc, attn_mask=keep)
        x = x + self.proj(o.transpose(1, 2).reshape(K, d))
        return x + self.ff(self.ln2(x))


class ActorCache:
    """Per-stream KV caches + lengths for incremental rollout."""

    def __init__(self, n, layers, heads, dh, device, dtype=torch.float32):
        self.k = [torch.zeros(n, CTX, heads, dh, device=device, dtype=dtype)
                  for _ in range(layers)]
        self.v = [torch.zeros(n, CTX, heads, dh, device=device, dtype=dtype)
                  for _ in range(layers)]
        self.len = torch.zeros(n, dtype=torch.long, device=device)

    def make_room(self, idx):
        """Slide full windows left by one; returns write positions for idx."""
        full = idx[self.len[idx] >= CTX]
        if len(full):
            for kc, vc in zip(self.k, self.v):
                kc[full] = torch.roll(kc[full], -1, dims=1)
                vc[full] = torch.roll(vc[full], -1, dims=1)
            self.len[full] = CTX - 1
        return self.len[idx]

    def reset(self, idx):
        self.len[idx] = 0


class Model(nn.Module):
    def __init__(self, d=384, e_layers=3, t_layers=6, heads=6):
        super().__init__()
        self.d, self.heads, self.t_layers = d, heads, t_layers
        self.species = nn.Embedding(152, 48)
        self.move = nn.Embedding(166, 48)
        self.status = nn.Embedding(256, 16)
        self.mon_in = nn.Linear(48 + 48 + 16 + MON_FLOATS, d)
        self.global_in = nn.Linear(160 - N_MON * MON_FLOATS, d)
        self.tok_pos = nn.Parameter(torch.randn(1, N_MON + 1, d) * 0.02)
        mk = lambda: nn.TransformerEncoderLayer(  # noqa: E731
            d, heads, dim_feedforward=4 * d, batch_first=True,
            norm_first=True, activation="gelu", dropout=0.0)
        self.entity = nn.TransformerEncoder(
            mk(), e_layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(t_layers))
        self.ln_f = nn.LayerNorm(d)
        self.seq_pos = nn.Parameter(torch.randn(1, CTX, d) * 0.02)
        self.pi = nn.Linear(d, N_ACTIONS)
        self.v = nn.Linear(d, 1)
        self.belief_sp = nn.Linear(d, 6 * 152)
        self.belief_mv = nn.Linear(d, 24 * 166)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)

    def new_cache(self, n_streams, device, dtype=torch.float32):
        return ActorCache(n_streams, self.t_layers, self.heads,
                          self.d // self.heads, device, dtype)

    def embed_step(self, ints, floats):
        """(B,80) ints, (B,160) floats -> (B,d)."""
        mi = ints[:, :72].view(-1, N_MON, MON_INTS).long()
        mf = floats[:, :96].view(-1, N_MON, MON_FLOATS)
        mon = self.mon_in(torch.cat([
            self.species(mi[..., 0].clamp(0, 151)),
            self.move(mi[..., 2:6].clamp(0, 165)).mean(dim=2),
            self.status(mi[..., 1].clamp(0, 255)),
            mf,
        ], dim=-1))
        glob = self.global_in(floats[:, 96:]).unsqueeze(1)
        x = torch.cat([glob, mon], dim=1) + self.tok_pos
        return self.entity(x)[:, 0]

    def forward_seq(self, emb, lens):
        """emb (B,T,d) left-aligned -> (B,T,d) causal hiddens."""
        T = emb.shape[1]
        x = emb + self.seq_pos[:, :T]
        pad = torch.arange(T, device=emb.device)[None] >= lens[:, None]
        for blk in self.blocks:
            x = blk.forward_seq(x, pad)
        return self.ln_f(x)

    def step(self, ints, floats, cache, idx):
        """One decision for streams idx; returns (k,d) hidden at current step."""
        pos = cache.make_room(idx)
        x = self.embed_step(ints, floats) + self.seq_pos[0, pos]
        for blk, kc, vc in zip(self.blocks, cache.k, cache.v):
            x = blk.forward_step(x, kc, vc, idx, pos)
        cache.len[idx] = pos + 1
        return self.ln_f(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = Model(d=96, e_layers=1, t_layers=3, heads=4).eval()
    B, T = 5, 17
    ints = torch.randint(0, 100, (B, T, 80), dtype=torch.int32)
    floats = torch.rand(B, T, 160)
    lens = torch.tensor([T, 11, 7, 17, 1])
    with torch.no_grad():
        emb = m.embed_step(ints.view(-1, 80), floats.view(-1, 160)).view(B, T, -1)
        h_seq = m.forward_seq(emb, lens)
        cache = m.new_cache(B, "cpu")
        errs = []
        for t in range(T):
            alive = torch.arange(B)[lens > t]
            h = m.step(ints[alive, t], floats[alive, t], cache, alive)
            errs.append((h - h_seq[alive, t]).abs().max().item())
    worst = max(errs)
    assert worst < 1e-4, f"KV-cache path diverges from seq path: {worst}"
    print(f"cache/seq equivalence OK (max abs err {worst:.2e})")
