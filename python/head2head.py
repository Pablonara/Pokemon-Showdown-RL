"""Head-to-head evaluation between two checkpoints (seed of the eval gauntlet).

Plays both side assignments (A as P1, then A as P2) for team-luck fairness.

Usage: python3 python/head2head.py <ckptA> <ckptB> [--episodes 400] [--greedy]
"""

import argparse
import json
import math
import pathlib
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_v0 import MaxDamagePolicy  # noqa: E402
from gen1env import Gen1Env  # noqa: E402
from model import Model, smart_load  # noqa: E402
from train_fast import masked_dist  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
POOL = str(ROOT / "teams" / "gen1-pool.bin")


def load(path, device):
    if path == "maxdmg":
        return MaxDamagePolicy()
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ckpt.get("config", {})
    m = Model(cfg.get("d", 384), cfg.get("e_layers", 3), cfg.get("t_layers", 6),
              cfg.get("heads", 6), dex_feats=cfg.get("dex_feats", True)).to(device)
    smart_load(m, ckpt["model"])  # any vintage, function-preserving
    return m.eval()


class Stats:
    """Blind-spot report accumulator: what does A do in wins vs losses?"""

    def __init__(self):
        self.ep = {}
        self.agg = {k: {"n": 0, "turns": [], "am": Counter(), "bm": Counter(),
                        "asw": 0, "amv": 0, "crit_in": 0, "miss": 0, "froz": 0,
                        "crit_eps": 0} for k in ("win", "loss", "draw")}

    def _mid(self, env, i, p, slot):
        mf = env.m_floats[i, p]
        j = int(np.argmax([mf[b * 13 + 2] for b in range(6)]))  # active block
        return int(env.m_ints[i, p, j * 6 + 2 + slot])

    def record(self, env, p, a_player, rows, acts):
        for i, a in zip(rows, acts):
            e = self.ep.setdefault(int(i), {"am": Counter(), "bm": Counter(),
                                            "asw": 0, "amv": 0, "turn": 0.0,
                                            "crit_in": 0, "miss": 0, "froz": False})
            e["turn"] = float(env.m_floats[i, p, 214]) * 500  # env resets on done
            if p == a_player:
                fl = env.m_floats[i, p]
                e["crit_in"] += int(fl[220])   # their crit landed on me
                e["miss"] += int(fl[216])      # my move missed
                st = env.m_ints[i, p, [j * 6 + 1 for j in range(6)]]
                if (st & 32).any():            # any of my mons frozen
                    e["froz"] = True
            if p == a_player:
                if a < 4:
                    e["am"][self._mid(env, i, p, int(a))] += 1
                    e["amv"] += 1
                else:
                    e["asw"] += 1
            elif a < 4:
                e["bm"][self._mid(env, i, p, int(a))] += 1

    def done(self, env, i, a_player):
        e = self.ep.pop(int(i), None)
        if e is None:
            return
        r = env.rewards[i, a_player]
        k = "win" if r == 1 else ("draw" if r == 0 else "loss")
        g = self.agg[k]
        g["n"] += 1
        g["turns"].append(e["turn"])
        g["am"].update(e["am"])
        g["bm"].update(e["bm"])
        g["asw"] += e["asw"]
        g["amv"] += e["amv"]
        g["crit_in"] += e["crit_in"]
        g["crit_eps"] += int(e["crit_in"] > 0)
        g["miss"] += e["miss"]
        g["froz"] += int(e["froz"])

    def report(self, names):
        def top(c, n=6):
            tot = max(1, sum(c.values()))
            return ", ".join(f"{names.get(m, m)} {100 * v / tot:.0f}%"
                             for m, v in c.most_common(n))
        for k in ("win", "loss", "draw"):
            g = self.agg[k]
            if not g["n"]:
                print(f"A-{k}: n=0")
                continue
            t = np.array(g["turns"])
            sw = g["asw"] / max(1, g["asw"] + g["amv"])
            n = g["n"]
            print(f"A-{k}: n={n} turns p50/p90 {np.median(t):.0f}/"
                  f"{np.percentile(t, 90):.0f} switch-rate {sw:.2f} | "
                  f"crits-taken/ep {g['crit_in'] / n:.2f} "
                  f"(any: {100 * g['crit_eps'] / n:.0f}%) "
                  f"frozen {100 * g['froz'] / n:.0f}% misses/ep {g['miss'] / n:.2f}\n"
                  f"  A used: {top(g['am'])}\n  B used: {top(g['bm'])}")


@torch.no_grad()
def play(models, device, episodes, greedy, n_envs=256, seed=42,
         stats=None, a_player=0):
    """models[0] plays P1, models[1] plays P2; returns models[0] wins/draws."""
    env = Gen1Env(POOL, n=n_envs, seed=seed)
    caches = [None if isinstance(m, MaxDamagePolicy) else m.new_cache(n_envs, device)
              for m in models]
    wins = draws = total = 0
    while total < episodes:
        for p, (m, cache) in enumerate(zip(models, caches)):
            rows = np.flatnonzero(env.needs[:, p])
            if not len(rows):
                continue
            if isinstance(m, MaxDamagePolicy):
                for i in rows:
                    env.actions[i, p] = m.act(env.m_ints[i, p], env.m_floats[i, p],
                                              env.masks[i, p])
                if stats is not None:
                    stats.record(env, p, a_player, rows, env.actions[rows, p])
                continue
            idx = torch.from_numpy(rows).to(device)
            h = m.step(torch.from_numpy(env.m_ints[rows, p]).to(device),
                       torch.from_numpy(env.m_floats[rows, p]).to(device), cache, idx)
            dist = masked_dist(m.pi(h), torch.from_numpy(env.masks[rows, p]).to(device))
            acts = dist.probs.argmax(-1) if greedy else dist.sample()
            acts = acts.cpu().numpy()
            env.actions[rows, p] = acts
            if stats is not None:
                stats.record(env, p, a_player, rows, acts)
        env.step()
        done = np.flatnonzero(env.dones)
        if len(done):
            total += len(done)
            wins += int((env.rewards[done, 0] == 1).sum())
            draws += int((env.rewards[done, 0] == 0).sum())
            if stats is not None:
                for i in done:
                    stats.done(env, i, a_player)
            t = torch.from_numpy(done).to(device)
            for cache in caches:
                if cache is not None:
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
    ap.add_argument("--stats", action="store_true",
                    help="blind-spot report: moves/lengths split by outcome")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    A, B = load(args.a, device), load(args.b, device)
    st = Stats() if args.stats else None

    w1, d1, n1 = play([A, B], device, args.episodes // 2, args.greedy, seed=42,
                      stats=st, a_player=0)
    w2, d2, n2 = play([B, A], device, args.episodes // 2, args.greedy, seed=42,
                      stats=st, a_player=1)
    wins = w1 + (n2 - w2 - d2)  # A's wins as P1 + A's wins as P2
    n = n1 + n2
    lo, hi = wilson(wins, n)
    print(f"{pathlib.Path(args.a).name} vs {pathlib.Path(args.b).name} "
          f"({'greedy' if args.greedy else 'sampled'}, n={n}): "
          f"A wins {wins / n:.3f} [{lo:.3f}, {hi:.3f}], draws {(d1 + d2) / n:.3f}")
    if st is not None:
        names = {int(k): v["name"] for k, v in
                 json.load(open(ROOT / "data" / "gen1.json"))["moves"].items()}
        st.report(names)


if __name__ == "__main__":
    main()
