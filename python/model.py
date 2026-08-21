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

import json
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

N_ACTIONS = 10
N_MON, MON_INTS, MON_FLOATS = 12, 6, 13  # obs v3
OBS_INTS, OBS_FLOATS = 80, 224
CTX = 128
EMBED_CHUNK = 16384  # SDPA batch-dim limit + autograd peak-memory cap (math is chunk-invariant)


def _dex_tables():
    """Static dex knowledge as tensors: the model should not have to rediscover
    the type chart, move powers, or base stats from win/loss rewards."""
    data = json.loads((pathlib.Path(__file__).parent.parent / "data" / "gen1.json").read_text())
    n_types = len(data["types"])
    move_feat = torch.zeros(166, 3 + n_types)  # [power/100, acc/100, status, type 1-hot]
    for num, m in data["moves"].items():
        i = int(num)
        move_feat[i, 0] = (m["power"] or 0) / 100.0
        move_feat[i, 1] = (m["accuracy"] or 100) / 100.0
        move_feat[i, 2] = 1.0 if m["status"] else 0.0
        move_feat[i, 3 + m["type"]] = 1.0
    species_feat = torch.zeros(152, 5 + n_types)  # [base stats/150, type multi-hot]
    move_type = torch.zeros(166, dtype=torch.long)
    sp_t1 = torch.zeros(152, dtype=torch.long)
    sp_t2 = torch.zeros(152, dtype=torch.long)
    for num, s in data["species"].items():
        i = int(num)
        for k, st in enumerate(s["stats"]):
            species_feat[i, k] = st / 150.0
        for t in s["types"]:
            species_feat[i, 5 + t] = 1.0
        sp_t1[i] = s["types"][0]
        sp_t2[i] = s["types"][-1]  # mono-type: t2 == t1... chart row 0 is used
    for num, m in data["moves"].items():
        move_type[int(num)] = m["type"]
    chart = torch.tensor(data["chart"], dtype=torch.float32)
    # dual-type effectiveness multiplies; guard the mono-type double-count
    mono = sp_t1 == sp_t2
    return move_feat, species_feat, move_type, sp_t1, sp_t2, mono, chart


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
    def __init__(self, d=384, e_layers=3, t_layers=6, heads=6, dex_feats=True):
        super().__init__()
        self.d, self.heads, self.t_layers = d, heads, t_layers
        self.dex_feats = dex_feats
        self.species = nn.Embedding(152, 48)
        self.move = nn.Embedding(166, 48)
        self.status = nn.Embedding(256, 16)
        mf, sf, mt, t1, t2, mono, chart = _dex_tables()
        if dex_feats:
            self.register_buffer("MOVE_FEAT", mf)
            self.register_buffer("SPECIES_FEAT", sf)
            self.register_buffer("MOVE_TYPE", mt)
            self.register_buffer("SP_T1", t1)
            self.register_buffer("SP_T2", t2)
            self.register_buffer("SP_MONO", mono)
            self.register_buffer("CHART", chart)
            mon_dim = (48 + sf.shape[1]) + (48 + mf.shape[1]) + 16 + MON_FLOATS
            glob_dim = OBS_FLOATS - N_MON * MON_FLOATS + 4  # + my 4 moves' effectiveness
        else:
            mon_dim = 48 + 48 + 16 + MON_FLOATS
            glob_dim = OBS_FLOATS - N_MON * MON_FLOATS
        self.mon_in = nn.Linear(mon_dim, d)
        self.global_in = nn.Linear(glob_dim, d)
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
        self.dmg = nn.Linear(d, 2)  # aux: next-step hp_frac delta (mine, theirs)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)

    def new_cache(self, n_streams, device, dtype=torch.float32):
        return ActorCache(n_streams, self.t_layers, self.heads,
                          self.d // self.heads, device, dtype)

    def embed_step(self, ints, floats):
        """(B,80) ints, (B,160) floats -> (B,d)."""
        B = ints.shape[0]
        if B > EMBED_CHUNK:  # some SDPA kernels reject very large batch dims
            return torch.cat([self.embed_step(ints[i:i + EMBED_CHUNK], floats[i:i + EMBED_CHUNK])
                              for i in range(0, B, EMBED_CHUNK)])
        mi = ints[:, :72].view(-1, N_MON, MON_INTS).long()
        mf = floats[:, :N_MON * MON_FLOATS].view(-1, N_MON, MON_FLOATS)
        sp = mi[..., 0].clamp(0, 151)
        mv = mi[..., 2:6].clamp(0, 165)
        sp_e = self.species(sp)
        mv_e = self.move(mv)
        if self.dex_feats:
            sp_e = torch.cat([sp_e, self.SPECIES_FEAT[sp]], dim=-1)
            mv_e = torch.cat([mv_e, self.MOVE_FEAT[mv]], dim=-1)
        mon = self.mon_in(torch.cat([
            sp_e, mv_e.mean(dim=2), self.status(mi[..., 1].clamp(0, 255)), mf,
        ], dim=-1))
        g = floats[:, N_MON * MON_FLOATS:]
        if self.dex_feats:
            # effectiveness of my active's 4 moves vs their active (the exact
            # quantity the max-damage baseline computes); dual types multiply
            my_moves = mi[:, 0, 2:6].clamp(0, 165)
            their = mi[:, 6, 0].clamp(0, 151)
            mt = self.MOVE_TYPE[my_moves]
            e1 = self.CHART[mt, self.SP_T1[their].unsqueeze(1)]
            e2 = self.CHART[mt, self.SP_T2[their].unsqueeze(1)]
            eff = torch.where(self.SP_MONO[their].unsqueeze(1), e1, e1 * e2)
            eff = eff * (self.MOVE_FEAT[my_moves][..., 0] > 0)  # zero for status moves
            g = torch.cat([g, eff / 4.0], dim=-1)
        glob = self.global_in(g).unsqueeze(1)
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


def load_expanded(model, old_sd):
    """Load a v1-obs (160f, MON_FLOATS=8) checkpoint into a v3-obs (224f,
    MON_FLOATS=13) model. New feature columns are zero-initialized, so the
    expanded model is function-identical to the old one until training uses
    the new inputs. Column maps follow the layout diffs:
      mon floats: v1 [hp,lvl,act,rev,pp4] -> v3 same 8 + [maxhp, used4] new
      globals:    v1 extras cols 0..58 identical (incl turn), 59.. was pad;
                  v3 adds event flags at 59..65; eff features move 64:68->68:72
    """
    sd = model.state_dict()
    loaded = set()
    for k, v in old_sd.items():
        if k == "mon_in.weight":
            w = torch.zeros_like(sd[k])
            pre = v.shape[1] - 8  # embeddings/dex-feat columns, unchanged
            w[:, :pre] = v[:, :pre]
            w[:, pre:pre + 8] = v[:, pre:]
            sd[k] = w
        elif k == "global_in.weight":
            w = torch.zeros_like(sd[k])
            w[:, :59] = v[:, :59]
            w[:, 68:72] = v[:, 64:68]
            sd[k] = w
        elif k in sd and sd[k].shape == v.shape:
            sd[k] = v
        else:
            raise ValueError(f"cannot map {k} {tuple(v.shape)} into expanded model")
        loaded.add(k)
    missing = set(sd) - loaded
    assert not missing, f"old checkpoint lacks {missing}"
    model.load_state_dict(sd)
    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    m = Model(d=96, e_layers=1, t_layers=3, heads=4).eval()
    B, T = 5, 17
    ints = torch.randint(0, 100, (B, T, OBS_INTS), dtype=torch.int32)
    floats = torch.rand(B, T, OBS_FLOATS)
    lens = torch.tensor([T, 11, 7, 17, 1])
    with torch.no_grad():
        emb = m.embed_step(ints.view(-1, OBS_INTS), floats.view(-1, OBS_FLOATS)).view(B, T, -1)
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
