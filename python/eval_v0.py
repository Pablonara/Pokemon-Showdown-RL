"""Evaluation harness v0: paired, side-swapped matches between simple policies.

Each matchup runs twice with the same seed (identical team-pair sequence, since
the env assigns pool teams by global battle counter): once with A as P1, once
swapped. Outcomes are paired by battle index, which cancels team-luck variance.

Policies:
    random      uniform over legal actions
    max-damage  best base_power x STAB x type-effectiveness move vs their active;
                random legal switch if no move is legal

Usage: python3 python/eval_v0.py [n_battles=2000] [pool=teams/gen1-pool.bin]
"""

import json
import math
import pathlib
import sys

import numpy as np

from gen1env import Gen1Env

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "data" / "gen1.json").read_text())
MOVES = {int(k): v for k, v in DATA["moves"].items()}
SPECIES = {int(k): v for k, v in DATA["species"].items()}
CHART = np.array(DATA["chart"])


class RandomPolicy:
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, ints, floats, mask):
        legal = np.flatnonzero(mask)
        return int(self.rng.choice(legal))


class MaxDamagePolicy:
    name = "max-damage"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, ints, floats, mask):
        my_species, their_species = int(ints[0]), int(ints[36])
        my_types = SPECIES[my_species]["types"] if my_species else []
        their_types = SPECIES[their_species]["types"] if their_species else []
        best, best_score = -1, -1.0
        for a in range(4):
            if not mask[a]:
                continue
            move = MOVES.get(int(ints[2 + a]))
            if move is None:
                continue
            score = 0.0
            if not move["status"] and move["power"] > 0:
                eff = 1.0
                for t in their_types:
                    eff *= CHART[move["type"], t]
                stab = 1.5 if move["type"] in my_types else 1.0
                score = move["power"] * eff * stab
            if score > best_score:
                best, best_score = a, score
        if best >= 0:
            return best
        if mask[9]:  # locked move / struggle
            return 9
        return int(self.rng.choice(np.flatnonzero(mask)))


def run(pol_p1, pol_p2, n_battles, pool, seed, n_envs=64):
    """Plays until n_battles complete; returns {battle_idx: (p1_reward, turns)}."""
    env = Gen1Env(pool, n=n_envs, seed=seed)
    out = {}
    pols = (pol_p1, pol_p2)
    while len(out) < n_battles:
        for i in range(env.n):
            for p in (0, 1):
                if env.needs[i, p]:
                    env.actions[i, p] = pols[p].act(
                        env.ints[i, p], env.floats[i, p], env.masks[i, p])
        env.step()
        for i in np.flatnonzero(env.dones):
            idx = int(env.ep_idx[i])
            if idx not in out:
                out[idx] = (float(env.rewards[i, 0]), int(env.ep_turns[i]))
    env.close()
    return out


def wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def paired_eval(pol_a, pol_b, n_battles, pool, seed=7):
    fwd = run(pol_a, pol_b, n_battles, pool, seed)
    rev = run(pol_b, pol_a, n_battles, pool, seed)
    common = sorted(fwd.keys() & rev.keys())
    w = d = 0
    turns = []
    for k in common:
        for r, t in ((fwd[k][0], fwd[k][1]), (-rev[k][0], rev[k][1])):
            w += r == 1
            d += r == 0
            turns.append(t)
    n = 2 * len(common)
    lo, hi = wilson(w, n)
    print(f"{pol_a.name} vs {pol_b.name}: {w}/{n} wins = {w / n:.3f} "
          f"[{lo:.3f}, {hi:.3f}] (draws {d / n:.3f}, avg {np.mean(turns):.0f} turns, "
          f"{len(common)} paired battles x2)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    pool = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "teams" / "gen1-pool.bin")
    paired_eval(RandomPolicy(1), RandomPolicy(2), n, pool)
    paired_eval(MaxDamagePolicy(), RandomPolicy(1), n, pool)
