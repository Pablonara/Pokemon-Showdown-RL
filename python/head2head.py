"""Head-to-head evaluation between two checkpoints (seed of the eval gauntlet).

Plays both side assignments (A as P1, then A as P2) for team-luck fairness.

Usage: python3 python/head2head.py <ckptA> <ckptB> [--episodes 400] [--greedy]
"""

import argparse
import math
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gen1env import Gen1Env  # noqa: E402
from model import Model, load_expanded  # noqa: E402
from train_fast import masked_dist  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
POOL = str(ROOT / "teams" / "gen1-pool.bin")


def load(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ckpt.get("config", {})
    m = Model(cfg.get("d", 384), cfg.get("e_layers", 3), cfg.get("t_layers", 6),
              cfg.get("heads", 6), dex_feats=cfg.get("dex_feats", True)).to(device)
    if ckpt["model"]["mon_in.weight"].shape == m.mon_in.weight.shape:
        missing, unexpected = m.load_state_dict(ckpt["model"], strict=False)
        assert not unexpected and all(k.startswith("dmg.") for k in missing)
    else:  # v1-obs checkpoint: expand (function-preserving)
        load_expanded(m, ckpt["model"])
    return m.eval()


@torch.no_grad()
def play(models, device, episodes, greedy, n_envs=256, seed=42):
    """models[0] plays P1, models[1] plays P2; returns models[0] wins/draws."""
    env = Gen1Env(POOL, n=n_envs, seed=seed)
    caches = [m.new_cache(n_envs, device) for m in models]
    wins = draws = total = 0
    while total < episodes:
        for p, (m, cache) in enumerate(zip(models, caches)):
            rows = np.flatnonzero(env.needs[:, p])
            if not len(rows):
                continue
            idx = torch.from_numpy(rows).to(device)
            h = m.step(torch.from_numpy(env.m_ints[rows, p]).to(device),
                       torch.from_numpy(env.m_floats[rows, p]).to(device), cache, idx)
            dist = masked_dist(m.pi(h), torch.from_numpy(env.masks[rows, p]).to(device))
            acts = dist.probs.argmax(-1) if greedy else dist.sample()
            env.actions[rows, p] = acts.cpu().numpy()
        env.step()
        done = np.flatnonzero(env.dones)
        if len(done):
            total += len(done)
            wins += int((env.rewards[done, 0] == 1).sum())
            draws += int((env.rewards[done, 0] == 0).sum())
            t = torch.from_numpy(done).to(device)
            for cache in caches:
                cache.reset(t)
    env.close()
    return wins, draws, total


def wilson(w, n, z=1.96):
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    A, B = load(args.a, device), load(args.b, device)

    w1, d1, n1 = play([A, B], device, args.episodes // 2, args.greedy, seed=42)
    w2, d2, n2 = play([B, A], device, args.episodes // 2, args.greedy, seed=42)
    wins = w1 + (n2 - w2 - d2)  # A's wins as P1 + A's wins as P2
    n = n1 + n2
    lo, hi = wilson(wins, n)
    print(f"{pathlib.Path(args.a).name} vs {pathlib.Path(args.b).name} "
          f"({'greedy' if args.greedy else 'sampled'}, n={n}): "
          f"A wins {wins / n:.3f} [{lo:.3f}, {hi:.3f}], draws {(d1 + d2) / n:.3f}")


if __name__ == "__main__":
    main()
