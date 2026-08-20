"""Run #3: partial-observation sequence PPO with belief head (gen1randombattle).

Differences vs train_ppo.py (run #2):
  - policy sees MASKED obs (ladder-visible info only, from env m_ints/m_floats)
  - two-level model: entity transformer per turn -> causal temporal transformer
    over the episode's decision sequence (sliding window CTX=128)
  - auxiliary belief head predicts the opponent's true species/moves per slot
    (ground truth free in self-play); trains the encoder to carry belief state
  - training on complete episodes only (carried across iteration boundaries),
    returns are exact terminal +-1/0, advantage = G - V (fixed pre-pass)

Usage: python3 python/train_seq.py --iters 2000 --envs 512 \
           --wandb pokemon-showdown-rl --run-name run3-seq-belief-18M
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gen1env import Gen1Env, N_ACTIONS  # noqa: E402
from eval_v0 import MaxDamagePolicy, RandomPolicy  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
POOL = str(ROOT / "teams" / "gen1-pool.bin")

N_MON, MON_INTS, MON_FLOATS = 12, 6, 8
CTX = 128  # temporal context window (decisions)

# obs index helpers (layout v1, see env/src/main.zig)
def belief_targets(ints_row):
    """(80,) truth ints -> opponent species (6,), opponent moves (24,)."""
    opp = ints_row[36:72].reshape(6, 6)
    return opp[:, 0], opp[:, 2:6].reshape(-1)


class Model(nn.Module):
    def __init__(self, d=384, e_layers=3, t_layers=6, heads=6):
        super().__init__()
        self.d = d
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
        self.temporal = nn.TransformerEncoder(
            mk(), t_layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.seq_pos = nn.Parameter(torch.randn(1, CTX, d) * 0.02)
        self.pi = nn.Linear(d, N_ACTIONS)
        self.v = nn.Linear(d, 1)
        self.belief_sp = nn.Linear(d, 6 * 152)
        self.belief_mv = nn.Linear(d, 24 * 166)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)

    def embed_step(self, ints, floats):
        """(B,80) masked ints, (B,160) masked floats -> (B,d) turn embedding."""
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

    def temporal_all(self, seq, lens):
        """(B,T,d) left-aligned turn embeds -> (B,T,d) causal hidden states."""
        T = seq.shape[1]
        x = seq + self.seq_pos[:, :T]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=seq.device), 1)
        pad = torch.arange(T, device=seq.device)[None] >= lens[:, None]
        return self.temporal(x, mask=causal, src_key_padding_mask=pad)

    def temporal_last(self, seq, lens):
        h = self.temporal_all(seq, lens)
        return h[torch.arange(len(lens), device=seq.device), lens - 1]


class Ctx:
    """Per-stream sliding window of turn embeddings on GPU."""

    def __init__(self, n, d, device):
        self.buf = torch.zeros(n, CTX, d, device=device)
        self.len = torch.zeros(n, dtype=torch.long, device=device)

    def append(self, idx, emb):
        full = idx[self.len[idx] >= CTX]
        if len(full):
            self.buf[full] = torch.roll(self.buf[full], -1, dims=1)
            self.len[full] = CTX - 1
        pos = self.len[idx]
        self.buf[idx, pos] = emb
        self.len[idx] = pos + 1

    def reset(self, idx):
        self.len[idx] = 0


def masked_dist(logits, mask):
    return torch.distributions.Categorical(logits=logits.masked_fill(mask == 0, -1e9))


class Stream:
    """Accumulates one episode's decision records for one (env, player)."""

    __slots__ = ("mi", "mf", "sp", "mv", "rev", "mask", "act", "logp")

    def __init__(self):
        for k in self.__slots__:
            setattr(self, k, [])

    def add(self, mi, mf, mask, act, logp, truth_ints, m_floats):
        sp, mv = belief_targets(truth_ints)
        self.mi.append(mi.copy())
        self.mf.append(mf.copy())
        self.sp.append(sp.copy())
        self.mv.append(mv.copy())
        self.rev.append(m_floats[48:96].reshape(6, 8)[:, 3].copy())
        self.mask.append(mask.copy())
        self.act.append(act)
        self.logp.append(logp)

    def pack(self, ret):
        n = len(self.act)
        s = slice(max(0, n - CTX), n)  # keep last CTX decisions
        ep = {k: np.array(getattr(self, k)[s]) for k in self.__slots__}
        ep["ret"] = ret
        for k in self.__slots__:
            getattr(self, k).clear()
        return ep


@torch.no_grad()
def evaluate(model, device, opponent, episodes=300, n_envs=128, seed=123):
    env = Gen1Env(POOL, n=n_envs, seed=seed)
    ctx = Ctx(n_envs, model.d, device)
    wins = total = 0
    while total < episodes:
        rows = np.flatnonzero(env.needs[:, 0])
        if len(rows):
            ints = torch.from_numpy(env.m_ints[rows, 0]).to(device)
            floats = torch.from_numpy(env.m_floats[rows, 0]).to(device)
            mask = torch.from_numpy(env.masks[rows, 0]).to(device)
            idx = torch.from_numpy(rows).to(device)
            ctx.append(idx, model.embed_step(ints, floats))
            h = model.temporal_last(ctx.buf[idx], ctx.len[idx])
            env.actions[rows, 0] = masked_dist(model.pi(h), mask).sample().cpu().numpy()
        for i in np.flatnonzero(env.needs[:, 1]):
            env.actions[i, 1] = opponent.act(env.m_ints[i, 1], env.m_floats[i, 1], env.masks[i, 1])
        env.step()
        done = np.flatnonzero(env.dones)
        if len(done):
            total += len(done)
            wins += int((env.rewards[done, 0] == 1).sum())
            ctx.reset(torch.from_numpy(done).to(device))
    env.close()
    return wins / total


def batches_of(eps, token_budget):
    eps = sorted(eps, key=lambda e: -len(e["act"]))
    out, cur, cur_max = [], [], 0
    for e in eps:
        t = max(cur_max, len(e["act"]))
        if cur and t * (len(cur) + 1) > token_budget:
            out.append(cur)
            cur, cur_max = [], 0
        cur.append(e)
        cur_max = max(cur_max, len(e["act"]))
    if cur:
        out.append(cur)
    return out


def pad_batch(batch, device):
    B, T = len(batch), max(len(e["act"]) for e in batch)
    t = {}
    t["mi"] = torch.zeros(B, T, 80, dtype=torch.int32)
    t["mf"] = torch.zeros(B, T, 160)
    t["sp"] = torch.zeros(B, T, 6, dtype=torch.long)
    t["mv"] = torch.zeros(B, T, 24, dtype=torch.long)
    t["rev"] = torch.zeros(B, T, 6)
    t["mask"] = torch.zeros(B, T, N_ACTIONS, dtype=torch.uint8)
    t["act"] = torch.zeros(B, T, dtype=torch.long)
    t["logp"] = torch.zeros(B, T)
    lens = torch.tensor([len(e["act"]) for e in batch])
    ret = torch.tensor([e["ret"] for e in batch])
    for b, e in enumerate(batch):
        n = len(e["act"])
        for k in t:
            t[k][b, :n] = torch.from_numpy(np.asarray(e[k]))
    valid = torch.arange(T)[None] < lens[:, None]
    return ({k: v.to(device) for k, v in t.items()},
            lens.to(device), ret.to(device), valid.to(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--rows", type=int, default=65536)
    ap.add_argument("--token-budget", type=int, default=24576)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--belief", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--out", default="runs/r3")
    ap.add_argument("--wandb", default=None)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    model = Model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params / 1e6:.2f}M envs={args.envs}", flush=True)

    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(project=args.wandb, name=args.run_name,
                        config={**vars(args), "params": n_params, "ctx": CTX})

    env = Gen1Env(POOL, n=args.envs, seed=1)
    n_streams = args.envs * 2
    ctx = Ctx(n_streams, model.d, device)
    streams = [Stream() for _ in range(n_streams)]
    completed = []
    ep_turns = []

    for it in range(1, args.iters + 1):
        t0 = time.time()
        rows_new = 0
        model.eval()
        while rows_new < args.rows:
            need = env.needs.reshape(-1)
            rows = np.flatnonzero(need)
            if len(rows):
                mi_np = env.m_ints.reshape(-1, 80)[rows]
                mf_np = env.m_floats.reshape(-1, 160)[rows]
                mk_np = env.masks.reshape(-1, N_ACTIONS)[rows]
                ti_np = env.ints.reshape(-1, 80)[rows]
                with torch.no_grad():
                    idx = torch.from_numpy(rows).to(device)
                    emb = model.embed_step(
                        torch.from_numpy(mi_np).to(device),
                        torch.from_numpy(mf_np).to(device))
                    ctx.append(idx, emb)
                    h = model.temporal_last(ctx.buf[idx], ctx.len[idx])
                    dist = masked_dist(model.pi(h), torch.from_numpy(mk_np).to(device))
                    acts = dist.sample()
                    logp = dist.log_prob(acts).cpu().numpy()
                    acts = acts.cpu().numpy()
                env.actions.reshape(-1)[rows] = acts
                for k, r in enumerate(rows):
                    streams[r].add(mi_np[k], mf_np[k], mk_np[k], acts[k], logp[k],
                                   ti_np[k], mf_np[k])
                rows_new += len(rows)
            env.step()
            done = np.flatnonzero(env.dones)
            if len(done):
                for i in done:
                    ep_turns.append(env.ep_turns[i])
                    for p in (0, 1):
                        s = streams[i * 2 + p]
                        if s.act:
                            completed.append(s.pack(float(env.rewards[i, p])))
                sids = np.concatenate([done * 2, done * 2 + 1])
                ctx.reset(torch.from_numpy(sids).to(device))
        collect_dt = time.time() - t0

        eps, completed = completed, []
        n_rows = sum(len(e["act"]) for e in eps)
        batches = batches_of(eps, args.token_budget)

        # fixed advantages from a value pre-pass
        with torch.no_grad():
            for batch in batches:
                t, lens, ret, valid = pad_batch(batch, device)
                emb = model.embed_step(t["mi"].view(-1, 80), t["mf"].view(-1, 160))
                h = model.temporal_all(emb.view(len(batch), -1, model.d), lens)
                val = model.v(h).squeeze(-1)
                adv = (ret[:, None] - val) * valid
                for b, e in enumerate(batch):
                    e["adv"] = adv[b, :len(e["act"])].cpu().numpy()
        all_adv = np.concatenate([e["adv"] for e in eps])
        mu, sd = all_adv.mean(), all_adv.std() + 1e-8
        for e in eps:
            e["adv"] = (e["adv"] - mu) / sd

        model.train()
        agg = dict(pg=0.0, vf=0.0, ent=0.0, kl=0.0, sp=0.0, mv=0.0, spacc=0.0, nb=0)
        for _ in range(args.epochs):
            for batch in np.random.permutation(np.array(batches, dtype=object)):
                t, lens, ret, valid = pad_batch(list(batch), device)
                adv = torch.zeros_like(t["logp"])
                for b, e in enumerate(batch):
                    adv[b, :len(e["act"])] = torch.from_numpy(e["adv"]).to(device)
                B, T = t["act"].shape
                emb = model.embed_step(t["mi"].view(-1, 80), t["mf"].view(-1, 160))
                h = model.temporal_all(emb.view(B, T, model.d), lens)
                nvalid = valid.sum().clamp(min=1)

                dist = masked_dist(model.pi(h), t["mask"])
                logp = dist.log_prob(t["act"])
                ratio = (logp - t["logp"]).exp()
                pg = -(torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv)
                       * valid).sum() / nvalid
                val = model.v(h).squeeze(-1)
                vf = (F.mse_loss(val, ret[:, None].expand_as(val), reduction="none")
                      * valid).sum() / nvalid
                ent = (dist.entropy() * valid).sum() / nvalid

                sp_logits = model.belief_sp(h).view(B, T, 6, 152)
                mv_logits = model.belief_mv(h).view(B, T, 24, 166)
                sp_t = t["sp"].masked_fill(~valid[..., None], -100)
                sp = F.cross_entropy(sp_logits.reshape(-1, 152), sp_t.reshape(-1),
                                     ignore_index=-100)
                mv_t = t["mv"].masked_fill(~valid[..., None], 0)
                mv = F.cross_entropy(mv_logits.reshape(-1, 166), mv_t.reshape(-1),
                                     ignore_index=0)

                loss = (pg + args.vf * vf - args.ent * ent
                        + args.belief * 0.5 * (sp + mv))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

                with torch.no_grad():
                    unrev = valid[..., None] & (t["rev"] == 0) & (t["sp"] > 0)
                    if unrev.any():
                        agg["spacc"] += (sp_logits.argmax(-1) == t["sp"])[unrev].float().mean().item()
                    agg["kl"] += ((t["logp"] - logp) * valid).sum().item() / nvalid.item()
                agg["pg"] += pg.item()
                agg["vf"] += vf.item()
                agg["ent"] += ent.item()
                agg["sp"] += sp.item()
                agg["mv"] += mv.item()
                agg["nb"] += 1

        nb = agg["nb"]
        metrics = {
            "rows": n_rows, "episodes": len(eps), "rows_per_s": n_rows / collect_dt,
            "ep_turns": float(np.mean(ep_turns[-500:])),
            "loss/pg": agg["pg"] / nb, "loss/vf": agg["vf"] / nb,
            "loss/entropy": agg["ent"] / nb, "loss/kl": agg["kl"] / nb,
            "belief/sp_loss": agg["sp"] / nb, "belief/mv_loss": agg["mv"] / nb,
            "belief/sp_acc_unrevealed": agg["spacc"] / nb,
        }
        msg = (f"it {it} rows {n_rows} ({metrics['rows_per_s']:.0f}/s) eps {len(eps)} "
               f"pg {metrics['loss/pg']:.4f} vf {metrics['loss/vf']:.4f} "
               f"ent {metrics['loss/entropy']:.3f} spacc {metrics['belief/sp_acc_unrevealed']:.3f}")
        if it % args.eval_every == 0 or it == args.iters:
            model.eval()
            wr = evaluate(model, device, RandomPolicy(1))
            wm = evaluate(model, device, MaxDamagePolicy())
            metrics.update({"eval/vs_random": wr, "eval/vs_maxdamage": wm})
            msg += f" | vs_random {wr:.3f} vs_maxdmg {wm:.3f}"
            torch.save({"model": model.state_dict(), "iter": it,
                        "config": vars(args)}, out / f"ckpt_{it:05d}.pt")
        if wb:
            wb.log(metrics, step=it)
        print(msg, flush=True)


if __name__ == "__main__":
    main()
