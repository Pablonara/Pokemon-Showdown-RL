"""Cold-start self-play PPO for gen1randombattle (smoke-scale, single GPU).

Mirror self-play: one policy plays both sides; every decision point (env,
player) is a training row with the episode's terminal reward (+1/-1/0) as its
return (gamma=1); rows from episodes still open at rollout end bootstrap with
V(s_now). Masked categorical policy over the fixed 10-action space.

Usage:
    python3 python/train_ppo.py [--iters 200] [--envs 512] [--eval-every 10]
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
GLOBAL_FLOATS = 160 - N_MON * MON_FLOATS  # 64


class Policy(nn.Module):
    def __init__(self, d=192, layers=3, heads=4):
        super().__init__()
        self.species = nn.Embedding(152, 32)
        self.move = nn.Embedding(166, 32)
        self.status = nn.Embedding(256, 16)
        self.mon_in = nn.Linear(32 + 32 + 16 + MON_FLOATS, d)
        self.global_in = nn.Linear(GLOBAL_FLOATS, d)
        self.pos = nn.Parameter(torch.randn(1, N_MON + 1, d) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d, heads, dim_feedforward=4 * d, batch_first=True,
            norm_first=True, activation="gelu", dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.pi = nn.Linear(d, N_ACTIONS)
        self.v = nn.Linear(d, 1)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)

    def forward(self, ints, floats):
        # ints (B,80) int32, floats (B,160) f32
        mi = ints[:, :72].view(-1, N_MON, MON_INTS).long()
        mf = floats[:, :96].view(-1, N_MON, MON_FLOATS)
        mon = self.mon_in(torch.cat([
            self.species(mi[..., 0]),
            self.move(mi[..., 2:6]).mean(dim=2),
            self.status(mi[..., 1].clamp(0, 255)),
            mf,
        ], dim=-1))
        glob = self.global_in(floats[:, 96:]).unsqueeze(1)
        x = torch.cat([glob, mon], dim=1) + self.pos
        x = self.encoder(x)[:, 0]
        return self.pi(x), self.v(x).squeeze(-1)


def masked_dist(logits, mask):
    return torch.distributions.Categorical(logits=logits.masked_fill(mask == 0, -1e9))


class Rollout:
    """Flat row storage for decision points across (env, player) pairs."""

    def __init__(self, cap):
        self.ints = np.zeros((cap, 80), np.int32)
        self.floats = np.zeros((cap, 160), np.float32)
        self.masks = np.zeros((cap, N_ACTIONS), np.uint8)
        self.actions = np.zeros(cap, np.int64)
        self.logp = np.zeros(cap, np.float32)
        self.value = np.zeros(cap, np.float32)
        self.ret = np.zeros(cap, np.float32)
        self.closed = np.zeros(cap, bool)
        self.n = 0


@torch.no_grad()
def evaluate(model, device, opponent, episodes=300, n_envs=128, seed=123):
    env = Gen1Env(POOL, n=n_envs, seed=seed)
    wins = draws = total = 0
    turns = []
    while total < episodes:
        rows = np.flatnonzero(env.needs[:, 0])
        if len(rows):
            ints = torch.from_numpy(env.ints[rows, 0]).to(device)
            floats = torch.from_numpy(env.floats[rows, 0]).to(device)
            mask = torch.from_numpy(env.masks[rows, 0]).to(device)
            logits, _ = model(ints, floats)
            acts = masked_dist(logits, mask).sample().cpu().numpy()
            env.actions[rows, 0] = acts
        for i in np.flatnonzero(env.needs[:, 1]):
            env.actions[i, 1] = opponent.act(env.ints[i, 1], env.floats[i, 1], env.masks[i, 1])
        env.step()
        done = np.flatnonzero(env.dones)
        for i in done:
            total += 1
            wins += env.rewards[i, 0] == 1
            draws += env.rewards[i, 0] == 0
            turns.append(env.ep_turns[i])
    env.close()
    return wins / total, draws / total, float(np.mean(turns))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--rows", type=int, default=65536, help="rows per PPO iteration")
    ap.add_argument("--minibatch", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--out", default="runs/ppo_smoke")
    ap.add_argument("--wandb", default=None, help="wandb project name; enables logging")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    log_f = (out / "log.csv").open("a")

    model = Policy().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params / 1e6:.2f}M envs={args.envs}", flush=True)

    wb = None
    if args.wandb:
        import wandb  # key via WANDB_API_KEY env var; never in the repo
        wb = wandb.init(project=args.wandb, name=args.run_name,
                        config={**vars(args), "params": n_params})

    env = Gen1Env(POOL, n=args.envs, seed=1)
    open_rows = [[[] for _ in range(2)] for _ in range(args.envs)]
    ep_turns_hist = []

    for it in range(1, args.iters + 1):
        buf = Rollout(args.rows + args.envs * 2 + 8)
        t0 = time.time()
        model.eval()
        while buf.n < args.rows:
            need = env.needs.reshape(-1)
            rows = np.flatnonzero(need)
            if len(rows):
                flat_i, flat_p = rows // 2, rows % 2
                with torch.no_grad():
                    ints = torch.from_numpy(env.ints.reshape(-1, 80)[rows]).to(device)
                    floats = torch.from_numpy(env.floats.reshape(-1, 160)[rows]).to(device)
                    mask = torch.from_numpy(env.masks.reshape(-1, N_ACTIONS)[rows]).to(device)
                    logits, value = model(ints, floats)
                    dist = masked_dist(logits, mask)
                    acts = dist.sample()
                    logp = dist.log_prob(acts)
                a = acts.cpu().numpy()
                env.actions.reshape(-1)[rows] = a
                base = buf.n
                sl = slice(base, base + len(rows))
                buf.ints[sl] = env.ints.reshape(-1, 80)[rows]
                buf.floats[sl] = env.floats.reshape(-1, 160)[rows]
                buf.masks[sl] = env.masks.reshape(-1, N_ACTIONS)[rows]
                buf.actions[sl] = a
                buf.logp[sl] = logp.cpu().numpy()
                buf.value[sl] = value.cpu().numpy()
                buf.n += len(rows)
                for k, (i, p) in enumerate(zip(flat_i, flat_p)):
                    open_rows[i][p].append(base + k)
            env.step()
            for i in np.flatnonzero(env.dones):
                ep_turns_hist.append(env.ep_turns[i])
                for p in (0, 1):
                    r = env.rewards[i, p]
                    idx = open_rows[i][p]
                    buf.ret[idx] = r
                    buf.closed[idx] = True
                    open_rows[i][p].clear()

        # bootstrap rows from episodes cut off by rollout end with V(s_now)
        with torch.no_grad():
            ints = torch.from_numpy(env.ints.reshape(-1, 80)).to(device)
            floats = torch.from_numpy(env.floats.reshape(-1, 160)).to(device)
            _, v_now = model(ints, floats)
            v_now = v_now.cpu().numpy()
        for i in range(args.envs):
            for p in (0, 1):
                idx = open_rows[i][p]
                if idx:
                    buf.ret[idx] = v_now[i * 2 + p]
                    open_rows[i][p].clear()  # rows consumed; episode tail next iter

        n = buf.n
        collect_dt = time.time() - t0
        adv = buf.ret[:n] - buf.value[:n]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        t_ints = torch.from_numpy(buf.ints[:n]).to(device)
        t_floats = torch.from_numpy(buf.floats[:n]).to(device)
        t_masks = torch.from_numpy(buf.masks[:n]).to(device)
        t_acts = torch.from_numpy(buf.actions[:n]).to(device)
        t_logp = torch.from_numpy(buf.logp[:n]).to(device)
        t_ret = torch.from_numpy(buf.ret[:n]).to(device)
        t_adv = torch.from_numpy(adv).to(device)

        model.train()
        pl_s = vl_s = ent_s = kl_s = 0.0
        nb = 0
        for _ in range(args.epochs):
            perm = torch.randperm(n, device=device)
            for s in range(0, n - args.minibatch + 1, args.minibatch):
                mb = perm[s:s + args.minibatch]
                logits, value = model(t_ints[mb], t_floats[mb])
                dist = masked_dist(logits, t_masks[mb])
                logp = dist.log_prob(t_acts[mb])
                ratio = (logp - t_logp[mb]).exp()
                pg = -torch.min(
                    ratio * t_adv[mb],
                    ratio.clamp(1 - args.clip, 1 + args.clip) * t_adv[mb]).mean()
                vl = F.mse_loss(value, t_ret[mb])
                ent = dist.entropy().mean()
                loss = pg + args.vf * vl - args.ent * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    kl_s += (t_logp[mb] - logp).mean().abs().item()
                pl_s += pg.item()
                vl_s += vl.item()
                ent_s += ent.item()
                nb += 1

        metrics = {
            "rows": n, "rows_per_s": n / collect_dt,
            "ep_turns": float(np.mean(ep_turns_hist[-500:])),
            "loss/pg": pl_s / nb, "loss/vf": vl_s / nb,
            "loss/entropy": ent_s / nb, "loss/kl": kl_s / nb,
        }
        msg = (f"it {it} rows {n} ({metrics['rows_per_s']:.0f}/s) "
               f"ep_turns {metrics['ep_turns']:.1f} "
               f"pg {pl_s / nb:.4f} vf {vl_s / nb:.4f} ent {ent_s / nb:.3f} kl {kl_s / nb:.4f}")
        if it % args.eval_every == 0 or it == args.iters:
            model.eval()
            wr, dr, t_r = evaluate(model, device, RandomPolicy(1))
            wm, dm, t_m = evaluate(model, device, MaxDamagePolicy())
            metrics.update({"eval/vs_random": wr, "eval/vs_maxdamage": wm,
                            "eval/draws_vs_random": dr, "eval/turns_vs_maxdamage": t_m})
            msg += f" | vs_random {wr:.3f} vs_maxdmg {wm:.3f} (turns {t_m:.0f})"
            torch.save({"model": model.state_dict(), "iter": it}, out / f"ckpt_{it:05d}.pt")
        if wb:
            wb.log(metrics, step=it)
        print(msg, flush=True)
        log_f.write(msg.replace(" ", ",") + "\n")
        log_f.flush()


if __name__ == "__main__":
    main()
