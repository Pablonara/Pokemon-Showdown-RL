"""Fast-path sequence PPO trainer (runs #4+; run #5 adds the league).

Speed (run #4): KV-cache rollout, slab buffers, bf16, 4096 envs.
League (run #5):
  - opponent mix: mirror self-play (fixed share) + scripted bots (max-damage,
    random) sampled PFSP-style (probability ~ how much we lose to them) +
    an optional frozen-checkpoint block (--league)
  - only the learner's side trains when the opponent is scripted/frozen
  - GAE(gamma, lam) advantages; draw penalty (stalling to the turn cap is bad)
  - R-NaD-style anchor: KL(pi || pi_ref) penalty, pi_ref refreshed periodically
  - eval reports sampled + greedy and draw rates

Usage: python3 python/train_fast.py --iters 2000 --envs 4096 \
           --init runs/bc/bc_05.pt --league runs/r4/ckpt_latest.pt \
           --wandb pokemon-showdown-rl --run-name run5-league
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
from gen1env import Gen1Env, N_ACTIONS, OBS_INTS, OBS_FLOATS  # noqa: E402
from eval_v0 import MaxDamagePolicy, RandomPolicy  # noqa: E402
from model import CTX, Model  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
POOL = str(ROOT / "teams" / "gen1-pool.bin")

FIELDS = [  # slab name, per-row shape, dtype
    ("mi", (80,), np.int32), ("mf", (224,), np.float32),
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
def evaluate(model, device, amp, opponent, episodes=300, n_envs=256, seed=123, greedy=False):
    env = Gen1Env(POOL, n=n_envs, seed=seed)
    cache = model.new_cache(n_envs, device, torch.bfloat16 if amp else torch.float32)
    wins = draws = total = 0
    while total < episodes:
        rows = np.flatnonzero(env.needs[:, 0])
        if len(rows):
            idx = torch.from_numpy(rows).to(device)
            with torch.autocast("cuda", torch.bfloat16) if amp else nullcontext():
                h = model.step(torch.from_numpy(env.m_ints[rows, 0]).to(device),
                               torch.from_numpy(env.m_floats[rows, 0]).to(device),
                               cache, idx)
                dist = masked_dist(model.pi(h), torch.from_numpy(env.masks[rows, 0]).to(device))
            acts = dist.probs.argmax(-1) if greedy else dist.sample()
            env.actions[rows, 0] = acts.cpu().numpy()
        for i in np.flatnonzero(env.needs[:, 1]):
            env.actions[i, 1] = opponent.act(env.m_ints[i, 1], env.m_floats[i, 1], env.masks[i, 1])
        env.step()
        done = np.flatnonzero(env.dones)
        if len(done):
            total += len(done)
            wins += int((env.rewards[done, 0] == 1).sum())
            draws += int((env.rewards[done, 0] == 0).sum())
            cache.reset(torch.from_numpy(done).to(device))
    env.close()
    return wins / total, draws / total


class Opponents:
    """Per-env opponent assignment with PFSP-lite adaptive sampling.

    opp[i] == -1: mirror self-play (both players train).
    opp[i] >= 0: pool index; player 1 is driven by that opponent, only player 0
    trains. Scripted opponents are resampled per episode with probability
    ~ (1 - winrate_ema + eps): we practice most against what beats us.
    Frozen-checkpoint opponents (if any) own a fixed env block (KV-cache
    memory is per-block, not per-model-per-all-envs).
    """

    def __init__(self, args, device, amp, n_envs):
        self.device, self.amp = device, amp
        self.pool = [("maxdmg", MaxDamagePolicy()), ("random", RandomPolicy(7))]
        self.wr = np.full(len(self.pool), 0.5)  # learner winrate EMA per opponent
        self.self_share = args.self_share
        self.hot_frac = getattr(args, "hot_frac", 0.0)
        self.hot = np.zeros(n_envs, bool)
        self.opp = np.full(n_envs, -1, dtype=np.int64)
        self.frozen = None
        self.block = np.zeros(n_envs, dtype=bool)
        if args.league:
            ckpt = torch.load(args.league, map_location=device, weights_only=True)
            cfg = ckpt.get("config", {})
            self.frozen = Model(cfg.get("d", 384), cfg.get("e_layers", 3),
                                cfg.get("t_layers", 6), cfg.get("heads", 6),
                                dex_feats=cfg.get("dex_feats", False)).to(device)
            if ckpt["model"]["mon_in.weight"].shape == self.frozen.mon_in.weight.shape:
                # older checkpoints predate the dmg aux head (unused when acting)
                missing, unexpected = self.frozen.load_state_dict(ckpt["model"], strict=False)
                assert not unexpected and all(k.startswith("dmg.") for k in missing), \
                    (missing, unexpected)
            else:  # v1-obs checkpoint: expand (function-preserving)
                from model import load_expanded
                load_expanded(self.frozen, ckpt["model"])
            self.frozen.eval()
            n_block = max(1, int(n_envs * args.league_share))
            self.block[-n_block:] = True
            self.block_base = n_envs - n_block
            self.fcache = self.frozen.new_cache(
                n_block, device, torch.bfloat16 if amp else torch.float32)
            self.wr_frozen = 0.5
        for i in range(n_envs):
            self.resample(i)

    def resample(self, i):
        if self.block[i]:
            self.opp[i] = 100  # frozen block marker
            self.hot[i] = False
            return
        if np.random.random() < self.self_share:
            self.opp[i] = -1
            # hot lane: some mirror games sample at high temperature to widen
            # the visited-state distribution (IS-corrected via stored logp)
            self.hot[i] = np.random.random() < self.hot_frac
            return
        self.opp[i] = np.random.choice(len(self.pool), p=(w := 1.05 - self.wr) / w.sum())
        self.hot[i] = False

    def record(self, i, learner_reward):
        win = float(learner_reward > 0)
        if self.block[i]:
            self.wr_frozen += 0.005 * (win - self.wr_frozen)
        elif self.opp[i] >= 0:
            k = self.opp[i]
            self.wr[k] += 0.005 * (win - self.wr[k])
        self.resample(i)

    def act(self, env):
        """Fills player-1 actions for all non-mirror envs that need one."""
        need1 = env.needs[:, 1] > 0
        for k, (_, pol) in enumerate(self.pool):
            for i in np.flatnonzero(need1 & (self.opp == k) & ~self.block):
                env.actions[i, 1] = pol.act(env.m_ints[i, 1], env.m_floats[i, 1],
                                            env.masks[i, 1])
        if self.frozen is not None:
            rows = np.flatnonzero(need1 & self.block)
            if len(rows):
                bidx = torch.from_numpy(rows - self.block_base).to(self.device)
                with torch.no_grad(), \
                     (torch.autocast("cuda", torch.bfloat16) if self.amp else nullcontext()):
                    h = self.frozen.step(
                        torch.from_numpy(env.m_ints[rows, 1]).to(self.device),
                        torch.from_numpy(env.m_floats[rows, 1]).to(self.device),
                        self.fcache, bidx)
                    dist = masked_dist(self.frozen.pi(h),
                                       torch.from_numpy(env.masks[rows, 1]).to(self.device))
                env.actions[rows, 1] = dist.sample().cpu().numpy()

    def on_done(self, done):
        if self.frozen is not None:
            b = done[self.block[done]]
            if len(b):
                self.fcache.reset(torch.from_numpy(b - self.block_base).to(self.device))

    def trains(self, stream):
        """Does this stream (env*2+p) produce training rows?"""
        i, p = stream // 2, stream % 2
        return p == 0 or self.opp[i] == -1

    def metrics(self):
        m = {f"league/wr_{name}": self.wr[k] for k, (name, _) in enumerate(self.pool)}
        m["league/mirror_frac"] = float((self.opp == -1).mean())
        m["league/hot_frac"] = float(self.hot.mean())
        if self.frozen is not None:
            m["league/wr_frozen"] = self.wr_frozen
        return m


def gae_batch(vals, rets, lens, gamma, lam):
    """Vectorized terminal-reward GAE. vals (B,T) padded, rets (B,), lens (B,).
    Returns adv (B,T) with zeros beyond each episode's length."""
    B, T = vals.shape
    valid = np.arange(T)[None, :] < lens[:, None]
    last_t = lens - 1
    v_next = np.zeros_like(vals)
    v_next[:, :-1] = vals[:, 1:]
    v_next[np.arange(B), last_t] = 0.0  # terminal
    r = np.zeros_like(vals)
    r[np.arange(B), last_t] = rets
    delta = (r + gamma * v_next - vals) * valid
    adv = np.zeros_like(vals)
    last = np.zeros(B, dtype=vals.dtype)
    for t in range(T - 1, -1, -1):  # scan over T (<=128), vectorized over B
        last = delta[:, t] + gamma * lam * last * valid[:, t]
        adv[:, t] = last
    return adv * valid


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
    ap.add_argument("--token-budget", type=int, default=65536)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--belief", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--draw-penalty", type=float, default=0.3)
    ap.add_argument("--self-share", type=float, default=0.7)
    ap.add_argument("--league", default=None, help="frozen checkpoint opponent")
    ap.add_argument("--league-share", type=float, default=0.1)
    ap.add_argument("--anchor-kl", type=float, default=0.02)
    ap.add_argument("--anchor-every", type=int, default=150)
    ap.add_argument("--dmg", type=float, default=1.0, help="damage-prediction aux weight")
    ap.add_argument("--hot-frac", type=float, default=0.15,
                    help="fraction of mirror envs sampled at high temperature (state diversity)")
    ap.add_argument("--hot-temp", type=float, default=1.5)
    ap.add_argument("--dex-feats", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--e-layers", type=int, default=3)
    ap.add_argument("--t-layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--out", default="runs/r4")
    ap.add_argument("--init", default=None, help="checkpoint to initialize from (e.g. BC)")
    ap.add_argument("--init-expand", default=None,
                    help="v1-obs checkpoint to load via zero-init column expansion")
    ap.add_argument("--wandb", default=None)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    autocast = (lambda: torch.autocast("cuda", torch.bfloat16)) if amp else nullcontext
    torch.backends.cuda.matmul.allow_tf32 = True
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    model = Model(args.d, args.e_layers, args.t_layers, args.heads,
                  dex_feats=args.dex_feats).to(device)
    if args.init:
        ckpt = torch.load(args.init, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"initialized from {args.init}", flush=True)
    elif args.init_expand:
        from model import load_expanded
        ckpt = torch.load(args.init_expand, map_location=device, weights_only=True)
        load_expanded(model, ckpt["model"])
        print(f"initialized from {args.init_expand} (expanded to obs v3)", flush=True)
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
    opps = Opponents(args, device, amp, args.envs)
    anchor = None
    if args.anchor_kl > 0:
        import copy
        anchor = copy.deepcopy(model).eval()
        for p in anchor.parameters():
            p.requires_grad_(False)

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
            opps.act(env)  # scripted/frozen opponents answer their decisions
            all_rows = np.flatnonzero(env.needs.reshape(-1))
            rows = np.array([r for r in all_rows if opps.trains(r)], dtype=np.int64)
            if len(rows):
                k = len(rows)
                mi = env.m_ints.reshape(-1, OBS_INTS)[rows]
                mf = env.m_floats.reshape(-1, OBS_FLOATS)[rows]
                mk = env.masks.reshape(-1, N_ACTIONS)[rows]
                ti = env.ints.reshape(-1, OBS_INTS)[rows]
                with torch.no_grad(), autocast():
                    idx = torch.from_numpy(rows).to(device)
                    h = model.step(torch.from_numpy(mi).to(device),
                                   torch.from_numpy(mf).to(device), cache, idx)
                    logits = model.pi(h)
                    if args.hot_frac > 0:
                        hot = torch.from_numpy(opps.hot[rows // 2]).to(device)
                        logits = logits / torch.where(hot, args.hot_temp, 1.0)[:, None]
                    dist = masked_dist(logits, torch.from_numpy(mk).to(device))
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
                slab.a["rev"][sl] = mf[:, 78:156].reshape(k, 6, 13)[:, :, 3]  # v3 layout
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
                            r = float(env.rewards[i, p])
                            if r == 0:
                                r = -args.draw_penalty  # stalling is not safety
                            completed.append(finish_episode(s, slab, r))
                    opps.record(i, float(env.rewards[i, 0]))
                opps.on_done(done)
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
        # pad once, reuse for the value pre-pass and every epoch (we are
        # overhead-bound, not FLOPs-bound; re-padding per epoch dominated)
        padded = [pad_batch(b, device) for b in batches]

        with torch.no_grad():
            adv_sum, adv_sq, adv_n = 0.0, 0.0, 0
            for batch, (t, lens, ret, valid) in zip(batches, padded):
                B = len(batch)
                with autocast():
                    emb = model.embed_step(t["mi"].view(-1, OBS_INTS), t["mf"].view(-1, OBS_FLOATS))
                    h = model.forward_seq(emb.view(B, -1, model.d), lens)
                    val = model.v(h).squeeze(-1).float()
                    if anchor is not None:  # frozen within the iteration: compute once
                        a_emb = anchor.embed_step(
                            t["mi"].view(-1, OBS_INTS), t["mf"].view(-1, OBS_FLOATS))
                        a_h = anchor.forward_seq(a_emb.view(B, -1, model.d), lens)
                        a_logits = anchor.pi(a_h).float().masked_fill(t["mask"] == 0, -1e9)
                        t["anchor_logp"] = F.log_softmax(a_logits, -1)
                v_np = val.cpu().numpy()
                lens_np = lens.cpu().numpy()
                rets_np = ret.cpu().numpy().astype(np.float32)
                adv = gae_batch(v_np, rets_np, lens_np, args.gamma, args.lam)
                t["adv"] = torch.from_numpy(adv).to(device)
                t["vtarget"] = (t["adv"] + val) * valid
                adv_sum += float(adv.sum())
                adv_sq += float((adv * adv).sum())
                adv_n += int(lens_np.sum())
        mu = adv_sum / max(adv_n, 1)
        sd = (max(adv_sq / max(adv_n, 1) - mu * mu, 1e-12)) ** 0.5 + 1e-8
        for _, (t, lens, ret, valid) in zip(batches, padded):
            t["adv"] = ((t["adv"] - mu) / sd) * valid

        model.train()
        # metric accumulators stay on-GPU; a single host sync per iteration
        agg = {k: torch.zeros((), device=device)
               for k in ("pg", "vf", "ent", "kl", "sp", "mv", "spacc", "dmg")}
        nb = 0
        t_train = time.time()
        for _ in range(args.epochs):
            order = np.random.permutation(len(padded))
            for bi in order:
                t, lens, ret, valid = padded[bi]
                adv, vtarget = t["adv"], t["vtarget"]
                B, T = t["act"].shape
                nvalid = valid.sum().clamp(min=1)
                with autocast():
                    emb = model.embed_step(t["mi"].view(-1, OBS_INTS), t["mf"].view(-1, OBS_FLOATS))
                    h = model.forward_seq(emb.view(B, T, model.d), lens)
                    dist = masked_dist(model.pi(h), t["mask"])
                    val = model.v(h).squeeze(-1).float()
                    sp_logits = model.belief_sp(h).view(B, T, 6, 152).float()
                    mv_logits = model.belief_mv(h).view(B, T, 24, 166).float()
                    dmg_pred = model.dmg(h).float()
                logp = dist.log_prob(t["act"])
                ratio = (logp - t["logp"]).exp()
                pg = -(torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv)
                       * valid).sum() / nvalid
                vf = (F.mse_loss(val, vtarget, reduction="none") * valid).sum() / nvalid
                ent = (dist.entropy() * valid).sum() / nvalid
                akl = torch.zeros((), device=device)
                if anchor is not None:  # anchor log-probs precomputed in the pre-pass
                    kl_ref = (dist.probs * (dist.logits - t["anchor_logp"])).sum(-1)
                    akl = (kl_ref * valid).sum() / nvalid
                sp_t = t["sp"].long().masked_fill(~valid[..., None] | (t["sp"] < 1), -100)
                sp = F.cross_entropy(sp_logits.reshape(-1, 152), sp_t.reshape(-1),
                                     ignore_index=-100)
                mv_t = t["mv"].long().masked_fill(~valid[..., None], 0)
                mv = F.cross_entropy(mv_logits.reshape(-1, 166), mv_t.reshape(-1),
                                     ignore_index=0)
                # damage aux: next-step hp_frac deltas of both actives; only
                # where t+1 is valid and neither active switched
                hp_me, hp_them = t["mf"][:, :, 0], t["mf"][:, :, 78]  # v3: their active block @78
                same = ((t["mi"][:, 1:, 0] == t["mi"][:, :-1, 0])
                        & (t["mi"][:, 1:, 36] == t["mi"][:, :-1, 36])
                        & valid[:, 1:])
                d_tgt = torch.stack([hp_me[:, 1:] - hp_me[:, :-1],
                                     hp_them[:, 1:] - hp_them[:, :-1]], dim=-1)
                dmg = (F.mse_loss(dmg_pred[:, :-1], d_tgt, reduction="none").sum(-1)
                       * same).sum() / same.sum().clamp(min=1)
                loss = (pg + args.vf * vf - args.ent * ent
                        + args.belief * 0.5 * (sp + mv) + args.anchor_kl * akl
                        + args.dmg * dmg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    unrev = valid[..., None] & (t["rev"] == 0) & (t["sp"] > 0)
                    if unrev.any():
                        agg["spacc"] += (sp_logits.argmax(-1) == t["sp"])[unrev].float().mean()
                    agg["kl"] += ((t["logp"] - logp) * valid).sum() / nvalid
                    for key, tv in (("pg", pg), ("vf", vf), ("ent", ent),
                                    ("sp", sp), ("mv", mv), ("dmg", dmg)):
                        agg[key] += tv.detach()
                nb += 1

        if anchor is not None and it % args.anchor_every == 0:
            anchor.load_state_dict(model.state_dict())  # move the magnet

        agg = {k: float(v) for k, v in agg.items()}  # single host sync
        nb = max(nb, 1)
        metrics = {
            **opps.metrics(),
            "rows": n_rows, "episodes": len(eps),
            "rows_per_s": slab.ptr / collect_dt, "train_s": time.time() - t_train,
            "ep_turns": float(np.mean(ep_turns[-500:])) if ep_turns else 0.0,
            "loss/pg": agg["pg"] / nb, "loss/vf": agg["vf"] / nb,
            "loss/entropy": agg["ent"] / nb, "loss/kl": agg["kl"] / nb,
            "belief/sp_loss": agg["sp"] / nb, "belief/mv_loss": agg["mv"] / nb,
            "belief/sp_acc_unrevealed": agg["spacc"] / nb,
            "loss/dmg": agg.get("dmg", 0.0) / nb,
            # plasticity tripwire: are the obs-v3 feature columns being used?
            # (mon block cols 8..13 = maxhp+usage; global cols 59..65 = events)
            "surgery/new_col_norm": float(
                (model.mon_in.weight[:, -5:].norm()
                 + model.global_in.weight[:, 59:65].norm()).detach()),
        }
        msg = (f"it {it} rows {slab.ptr} ({metrics['rows_per_s']:.0f}/s) "
               f"train {metrics['train_s']:.1f}s eps {len(eps)} "
               f"pg {metrics['loss/pg']:.4f} vf {metrics['loss/vf']:.4f} "
               f"ent {metrics['loss/entropy']:.3f} spacc {metrics['belief/sp_acc_unrevealed']:.3f}")
        if it % args.eval_every == 0 or it == args.iters:
            model.eval()
            wr, _ = evaluate(model, device, amp, RandomPolicy(1))
            wm, dm = evaluate(model, device, amp, MaxDamagePolicy())
            wg, dg = evaluate(model, device, amp, MaxDamagePolicy(), greedy=True)
            metrics.update({"eval/vs_random": wr, "eval/vs_maxdamage": wm,
                            "eval/vs_maxdamage_greedy": wg,
                            "eval/draws_maxdamage": dm, "eval/draws_maxdamage_greedy": dg})
            msg += f" | vs_random {wr:.3f} vs_maxdmg {wm:.3f} greedy {wg:.3f}"
            torch.save({"model": model.state_dict(), "iter": it, "config": vars(args)},
                       out / f"ckpt_{it:05d}.pt")
        if wb:
            wb.log(metrics, step=it)
        print(msg, flush=True)


if __name__ == "__main__":
    main()
