"""Behavior cloning on scraped human replays (output of replay_to_traj.mjs).

Trains the same Model as train_fast.py: masked action cross-entropy + value
regression on the recorded outcome + belief auxiliary losses. The checkpoint
is a drop-in init for train_fast.py --init (run #5 warm start).

Usage: python3 python/train_bc.py --traj data/traj --epochs 4 \
           --wandb pokemon-showdown-rl --run-name bc-400k
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
from model import CTX, Model  # noqa: E402
from train_fast import batches_of, evaluate, masked_dist, pad_batch  # noqa: E402
from eval_v0 import MaxDamagePolicy, RandomPolicy  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

DT = np.dtype([
    ("bid", "<u4"), ("player", "u1"), ("done", "u1"), ("ret", "i1"), ("p0", "u1"),
    ("act", "u1"), ("mask", "u1", 10), ("p1", "u1"),
    ("rev", "u1", 6), ("p2", "u1", 2), ("sp", "<i2", 6), ("mv", "<i2", 24),
    ("mi", "<i4", 80), ("mf", "<f4", 160),
])
assert DT.itemsize == 1048


def load_episodes(traj_dir):
    rows = np.concatenate([np.fromfile(f, dtype=DT)
                           for f in sorted(pathlib.Path(traj_dir).glob("*.traj"))])
    order = np.argsort(rows["bid"].astype(np.int64) * 2 + rows["player"], kind="stable")
    rows = rows[order]
    key = rows["bid"].astype(np.int64) * 2 + rows["player"]
    bounds = np.flatnonzero(np.diff(key)) + 1
    episodes = []
    for chunk in np.split(rows, bounds):
        chunk = chunk[-CTX:]
        episodes.append({
            "mi": chunk["mi"], "mf": chunk["mf"], "mask": chunk["mask"],
            "act": chunk["act"].astype(np.int64), "sp": chunk["sp"], "mv": chunk["mv"],
            "rev": chunk["rev"], "logp": np.zeros(len(chunk), np.float32),  # unused
            "ret": float(chunk["ret"][-1]), "bid": int(chunk["bid"][0]),
        })
    return episodes


def run_batches(model, batches, device, args, opt=None):
    """One pass; returns aggregated metrics. Trains if opt is given."""
    agg = dict(pol=0.0, vf=0.0, sp=0.0, mv=0.0, acc=0.0, spacc=0.0, n=0)
    amp = device == "cuda"
    for batch in batches:
        t, lens, ret, valid = pad_batch(batch, device)
        B, T = t["act"].shape
        nvalid = valid.sum().clamp(min=1)
        with torch.autocast("cuda", torch.bfloat16, enabled=amp):
            emb = model.embed_step(t["mi"].view(-1, 80), t["mf"].view(-1, 160))
            h = model.forward_seq(emb.view(B, T, model.d), lens)
            dist = masked_dist(model.pi(h), t["mask"])
            val = model.v(h).squeeze(-1).float()
            sp_logits = model.belief_sp(h).view(B, T, 6, 152).float()
            mv_logits = model.belief_mv(h).view(B, T, 24, 166).float()
        pol = -(dist.log_prob(t["act"]) * valid).sum() / nvalid
        vf = (F.mse_loss(val, ret[:, None].expand_as(val), reduction="none")
              * valid).sum() / nvalid
        sp_t = t["sp"].long().masked_fill(~valid[..., None] | (t["sp"] < 1), -100)
        sp = F.cross_entropy(sp_logits.reshape(-1, 152), sp_t.reshape(-1), ignore_index=-100)
        mv_t = t["mv"].long().masked_fill(~valid[..., None], 0)
        mv = F.cross_entropy(mv_logits.reshape(-1, 166), mv_t.reshape(-1), ignore_index=0)
        loss = pol + args.vf * vf + args.belief * 0.5 * (sp + mv)
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        with torch.no_grad():
            logits = model.pi(h).float().masked_fill(t["mask"] == 0, -1e9)
            agg["acc"] += ((logits.argmax(-1) == t["act"]) * valid).sum().item() / nvalid.item()
            unrev = valid[..., None] & (t["rev"] == 0) & (t["sp"] > 0)
            if unrev.any():
                agg["spacc"] += (sp_logits.argmax(-1) == t["sp"])[unrev].float().mean().item()
        for k, v in (("pol", pol), ("vf", vf), ("sp", sp), ("mv", mv)):
            agg[k] += v.item()
        agg["n"] += 1
    return {k: v / max(agg["n"], 1) for k, v in agg.items() if k != "n"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="data/traj")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--belief", type=float, default=0.5)
    ap.add_argument("--token-budget", type=int, default=32768)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--e-layers", type=int, default=3)
    ap.add_argument("--t-layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--out", default="runs/bc")
    ap.add_argument("--wandb", default=None)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    eps = load_episodes(ROOT / args.traj)
    val_eps = [e for e in eps if e["bid"] % 20 == 0]
    train_eps = [e for e in eps if e["bid"] % 20 != 0]
    print(f"episodes: {len(train_eps)} train / {len(val_eps)} val "
          f"({sum(len(e['act']) for e in eps)} decisions)", flush=True)

    model = Model(args.d, args.e_layers, args.t_layers, args.heads).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params / 1e6:.2f}M", flush=True)

    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(project=args.wandb, name=args.run_name,
                        config={**vars(args), "params": n_params,
                                "episodes": len(eps)})

    val_batches = batches_of(val_eps, args.token_budget)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        order = np.random.permutation(len(train_eps))
        train_batches = batches_of([train_eps[i] for i in order], args.token_budget)
        np.random.shuffle(train_batches)
        m = run_batches(model, train_batches, device, args, opt)
        model.eval()
        with torch.no_grad():
            v = run_batches(model, val_batches, device, args, opt=None)
        wr = evaluate(model, device, device == "cuda", RandomPolicy(1))
        wm = evaluate(model, device, device == "cuda", MaxDamagePolicy())
        metrics = {f"train/{k}": x for k, x in m.items()}
        metrics.update({f"val/{k}": x for k, x in v.items()})
        metrics.update({"eval/vs_random": wr, "eval/vs_maxdamage": wm})
        if wb:
            wb.log(metrics, step=ep)
        print(f"epoch {ep} ({time.time() - t0:.0f}s) "
              f"train acc {m['acc']:.3f} val acc {v['acc']:.3f} "
              f"val spacc {v['spacc']:.3f} | vs_random {wr:.3f} vs_maxdmg {wm:.3f}",
              flush=True)
        torch.save({"model": model.state_dict(), "epoch": ep, "config": vars(args)},
                   out / f"bc_{ep:02d}.pt")


if __name__ == "__main__":
    main()
