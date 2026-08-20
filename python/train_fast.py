"""Run #4 trainer: fast-path sequence PPO (same algorithm as train_seq.py).

Speed changes vs train_seq.py:
  - KV-cache incremental rollout (model.step) instead of O(T) window recompute
  - slab observation buffers: vectorized slice writes per step + one gather per
    episode (train_seq did ~8 numpy copies per decision in Python)
  - bf16 autocast for rollout and training on CUDA
  - default 4096 envs (8192 streams)

Usage: python3 python/train_fast.py --iters 2000 --envs 4096 \
           --wandb pokemon-showdown-rl --run-name run4-fast
"""

import argparse
import pathlib
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gen1env import Gen1Env, N_ACTIONS  # noqa: E402
from eval_v0 import MaxDamagePolicy, RandomPolicy  # noqa: E402
from model import CTX, Model  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
POOL = str(ROOT / "teams" / "gen1-pool.bin")

FIELDS = [  # slab name, per-row shape, dtype
    ("mi", (80,), np.int32), ("mf", (160,), np.float32),
    ("sp", (6,), np.int16), ("mv", (24,), np.int16), ("rev", (6,), np.uint8),
    ("mask", (N_ACTIONS,), np.uint8), ("act", (), np.int64), ("logp", (), np.float32),
]


def masked_dist(logits, mask):
    return torch.distributions.Categorical(
        logits=logits.float().masked_fill(mask == 0, -1e9))


class Slab:
    def __init__(self, cap):
        self.a = {k: np.zeros((cap, *shp), dt) for k, shp, dt in FIELDS}
        self.ptr = 0

    def gather(self, idx):
        return {k: v[idx] for k, v in self.a.items()}


@torch.no_grad()
def evaluate(model, device, amp, opponent, episodes=300, n_envs=256, seed=123):
    env = Gen1Env(POOL, n=n_envs, seed=seed)
    cache = model.new_cache(n_envs, device, torch.bfloat16 if amp else torch.float32)
    wins = total = 0
    while total < episodes:
        rows = np.flatnonzero(env.needs[:, 0])
        if len(rows):
            idx = torch.from_numpy(rows).to(device)
            with torch.autocast("cuda", torch.bfloat16) if amp else nullcontext():
                h = model.step(torch.from_numpy(env.m_ints[rows, 0]).to(device),
                               torch.from_numpy(env.m_floats[rows, 0]).to(device),
                               cache, idx)
                dist = masked_dist(model.pi(h), torch.from_numpy(env.masks[rows, 0]).to(device))
            env.actions[rows, 0] = dist.sample().cpu().numpy()
        for i in np.flatnonzero(env.needs[:, 1]):
            env.actions[i, 1] = opponent.act(env.m_ints[i, 1], env.m_floats[i, 1], env.masks[i, 1])
        env.step()
        done = np.flatnonzero(env.dones)
        if len(done):
            total += len(done)
            wins += int((env.rewards[done, 0] == 1).sum())
            cache.reset(torch.from_numpy(done).to(device))
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
    t = {k: torch.zeros((B, T, *shp),
                        dtype=torch.from_numpy(np.zeros(1, dt)).dtype)
         for k, shp, dt in FIELDS}
    for b, e in enumerate(batch):
        n = len(e["act"])
        for k in t:
            t[k][b, :n] = torch.from_numpy(np.ascontiguousarray(e[k]))
    lens = torch.tensor([len(e["act"]) for e in batch])
    ret = torch.tensor([e["ret"] for e in batch], dtype=torch.float32)
    valid = torch.arange(T)[None] < lens[:, None]
    return ({k: v.to(device, non_blocking=True) for k, v in t.items()},
            lens.to(device), ret.to(device), valid.to(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--rows", type=int, default=131072)
    ap.add_argument("--token-budget", type=int, default=32768)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--belief", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--e-layers", type=int, default=3)
    ap.add_argument("--t-layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--out", default="runs/r4")
    ap.add_argument("--wandb", default=None)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    autocast = (lambda: torch.autocast("cuda", torch.bfloat16)) if amp else nullcontext
    torch.backends.cuda.matmul.allow_tf32 = True
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    model = Model(args.d, args.e_layers, args.t_layers, args.heads).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params / 1e6:.2f}M envs={args.envs} amp={amp}", flush=True)

    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(project=args.wandb, name=args.run_name,
                        config={**vars(args), "params": n_params, "ctx": CTX})

    env = Gen1Env(POOL, n=args.envs, seed=1)
    n_streams = args.envs * 2
    cache = model.new_cache(n_streams, device, torch.bfloat16 if amp else torch.float32)
    stream_rows = [[] for _ in range(n_streams)]
    carry = [None] * n_streams
    completed, ep_turns = [], []

    def finish_episode(s, slab, ret):
        idx = np.asarray(stream_rows[s], np.int64)
        parts = [carry[s]] if carry[s] is not None else []
        parts.append(slab.gather(idx))
        ep = {k: np.concatenate([p[k] for p in parts]) for k in slab.a}
        ep = {k: v[-CTX:] for k, v in ep.items()}
        ep["ret"] = ret
        stream_rows[s].clear()
        carry[s] = None
        return ep

    for it in range(1, args.iters + 1):
        t0 = time.time()
        slab = Slab(args.rows + n_streams + 8)
        model.eval()
        while slab.ptr < args.rows:
            rows = np.flatnonzero(env.needs.reshape(-1))
            if len(rows):
                k = len(rows)
                mi = env.m_ints.reshape(-1, 80)[rows]
                mf = env.m_floats.reshape(-1, 160)[rows]
                mk = env.masks.reshape(-1, N_ACTIONS)[rows]
                ti = env.ints.reshape(-1, 80)[rows]
                with torch.no_grad(), autocast():
                    idx = torch.from_numpy(rows).to(device)
                    h = model.step(torch.from_numpy(mi).to(device),
                                   torch.from_numpy(mf).to(device), cache, idx)
                    dist = masked_dist(model.pi(h), torch.from_numpy(mk).to(device))
                    acts_t = dist.sample()
                    logp = dist.log_prob(acts_t).cpu().numpy()
                    acts = acts_t.cpu().numpy()
                env.actions.reshape(-1)[rows] = acts
                sl = slice(slab.ptr, slab.ptr + k)
                opp = ti[:, 36:72].reshape(k, 6, 6)
                slab.a["mi"][sl] = mi
                slab.a["mf"][sl] = mf
                slab.a["sp"][sl] = opp[:, :, 0]
                slab.a["mv"][sl] = opp[:, :, 2:6].reshape(k, 24)
                slab.a["rev"][sl] = mf[:, 48:96].reshape(k, 6, 8)[:, :, 3]
                slab.a["mask"][sl] = mk
                slab.a["act"][sl] = acts
                slab.a["logp"][sl] = logp
                for j, r in enumerate(rows):
                    stream_rows[r].append(slab.ptr + j)
                slab.ptr += k
            env.step()
            done = np.flatnonzero(env.dones)
            if len(done):
                for i in done:
                    ep_turns.append(env.ep_turns[i])
                    for p in (0, 1):
                        s = i * 2 + p
                        if stream_rows[s] or carry[s] is not None:
                            completed.append(finish_episode(s, slab, float(env.rewards[i, p])))
                cache.reset(torch.from_numpy(
                    np.concatenate([done * 2, done * 2 + 1])).to(device))
        # carry open episodes' rows across the iteration boundary
        for s in range(n_streams):
            if stream_rows[s]:
                idx = np.asarray(stream_rows[s], np.int64)
                seg = slab.gather(idx)
                if carry[s] is not None:
                    seg = {k: np.concatenate([carry[s][k], seg[k]])[-CTX:] for k in seg}
                carry[s] = seg
                stream_rows[s].clear()
        collect_dt = time.time() - t0

        eps, completed = completed, []
        if not eps:  # all episodes still open (tiny-config startup transient)
            print(f"it {it}: no completed episodes yet, collecting more", flush=True)
            continue
        n_rows = sum(len(e["act"]) for e in eps)
        batches = batches_of(eps, args.token_budget)

        with torch.no_grad():
            for batch in batches:
                t, lens, ret, valid = pad_batch(batch, device)
                with autocast():
                    emb = model.embed_step(t["mi"].view(-1, 80), t["mf"].view(-1, 160))
                    h = model.forward_seq(emb.view(len(batch), -1, model.d), lens)
                    val = model.v(h).squeeze(-1).float()
                adv = (ret[:, None] - val) * valid
                for b, e in enumerate(batch):
                    e["adv"] = adv[b, :len(e["act"])].cpu().numpy()
        all_adv = np.concatenate([e["adv"] for e in eps])
        mu, sd = all_adv.mean(), all_adv.std() + 1e-8
        for e in eps:
            e["adv"] = ((e["adv"] - mu) / sd).astype(np.float32)

        model.train()
        agg = dict(pg=0.0, vf=0.0, ent=0.0, kl=0.0, sp=0.0, mv=0.0, spacc=0.0, nb=0)
        t_train = time.time()
        for _ in range(args.epochs):
            order = np.random.permutation(len(batches))
            for bi in order:
                batch = batches[bi]
                t, lens, ret, valid = pad_batch(batch, device)
                adv = torch.zeros_like(t["logp"])
                for b, e in enumerate(batch):
                    adv[b, :len(e["act"])] = torch.from_numpy(e["adv"]).to(device)
                B, T = t["act"].shape
                nvalid = valid.sum().clamp(min=1)
                with autocast():
                    emb = model.embed_step(t["mi"].view(-1, 80), t["mf"].view(-1, 160))
                    h = model.forward_seq(emb.view(B, T, model.d), lens)
                    dist = masked_dist(model.pi(h), t["mask"])
                    val = model.v(h).squeeze(-1).float()
                    sp_logits = model.belief_sp(h).view(B, T, 6, 152).float()
                    mv_logits = model.belief_mv(h).view(B, T, 24, 166).float()
                logp = dist.log_prob(t["act"])
                ratio = (logp - t["logp"]).exp()
                pg = -(torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv)
                       * valid).sum() / nvalid
                vf = (F.mse_loss(val, ret[:, None].expand_as(val), reduction="none")
                      * valid).sum() / nvalid
                ent = (dist.entropy() * valid).sum() / nvalid
                sp_t = t["sp"].long().masked_fill(~valid[..., None], -100)
                sp = F.cross_entropy(sp_logits.reshape(-1, 152), sp_t.reshape(-1),
                                     ignore_index=-100)
                mv_t = t["mv"].long().masked_fill(~valid[..., None], 0)
                mv = F.cross_entropy(mv_logits.reshape(-1, 166), mv_t.reshape(-1),
                                     ignore_index=0)
                loss = pg + args.vf * vf - args.ent * ent + args.belief * 0.5 * (sp + mv)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    unrev = valid[..., None] & (t["rev"] == 0) & (t["sp"] > 0)
                    if unrev.any():
                        agg["spacc"] += (sp_logits.argmax(-1) == t["sp"])[unrev].float().mean().item()
                    agg["kl"] += ((t["logp"] - logp) * valid).sum().item() / nvalid.item()
                for key, tv in (("pg", pg), ("vf", vf), ("ent", ent), ("sp", sp), ("mv", mv)):
                    agg[key] += tv.item()
                agg["nb"] += 1

        nb = max(agg["nb"], 1)
        metrics = {
            "rows": n_rows, "episodes": len(eps),
            "rows_per_s": slab.ptr / collect_dt, "train_s": time.time() - t_train,
            "ep_turns": float(np.mean(ep_turns[-500:])) if ep_turns else 0.0,
            "loss/pg": agg["pg"] / nb, "loss/vf": agg["vf"] / nb,
            "loss/entropy": agg["ent"] / nb, "loss/kl": agg["kl"] / nb,
            "belief/sp_loss": agg["sp"] / nb, "belief/mv_loss": agg["mv"] / nb,
            "belief/sp_acc_unrevealed": agg["spacc"] / nb,
        }
        msg = (f"it {it} rows {slab.ptr} ({metrics['rows_per_s']:.0f}/s) "
               f"train {metrics['train_s']:.1f}s eps {len(eps)} "
               f"pg {metrics['loss/pg']:.4f} vf {metrics['loss/vf']:.4f} "
               f"ent {metrics['loss/entropy']:.3f} spacc {metrics['belief/sp_acc_unrevealed']:.3f}")
        if it % args.eval_every == 0 or it == args.iters:
            model.eval()
            wr = evaluate(model, device, amp, RandomPolicy(1))
            wm = evaluate(model, device, amp, MaxDamagePolicy())
            metrics.update({"eval/vs_random": wr, "eval/vs_maxdamage": wm})
            msg += f" | vs_random {wr:.3f} vs_maxdmg {wm:.3f}"
            torch.save({"model": model.state_dict(), "iter": it, "config": vars(args)},
                       out / f"ckpt_{it:05d}.pt")
        if wb:
            wb.log(metrics, step=it)
        print(msg, flush=True)


if __name__ == "__main__":
    main()
