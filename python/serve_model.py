"""Minimal HTTP inference server for playing on Pokémon Showdown.

Keeps a KV-cache slot per active battle (temporal context), frees it when the
bot reports the battle done.

    POST /act  {"battle": id, "ints": [80], "floats": [160], "mask": [10]}
               -> {"action": 0..9}
    POST /end  {"battle": id}

Usage: python3 python/serve_model.py <checkpoint> [--port 8765] [--greedy]
"""

import argparse
import json
import pathlib
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import OBS_FLOATS, Model, smart_load  # noqa: E402
from train_fast import masked_dist  # noqa: E402


def pad_obs_v1(floats):
    """Map a v1 (160f) observation from the not-yet-updated bot into the v3
    layout (new features zeroed) - mirrors model.load_expanded's column map."""
    out = np.zeros(OBS_FLOATS, np.float32)
    old = np.asarray(floats, np.float32)
    for j in range(12):
        out[j * 13:j * 13 + 8] = old[j * 8:j * 8 + 8]
    out[156:156 + 59] = old[96:96 + 59]
    return out

MAX_BATTLES = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--greedy", action="store_true",
                    help="argmax actions (strongest vs one-shot/static foes; predictable + loop-prone)")
    ap.add_argument("--min-p", type=float, default=0.15,
                    help="drop actions below this fraction of the top action's "
                         "probability (kills 1%%-accident samples, keeps near-ties); "
                         "0 disables")
    ap.add_argument("--temp", type=float, default=None,
                    help="sampling temperature; ~0.5 = near-greedy strength, breaks loops, "
                         "less predictable vs repeat/adaptive opponents")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = ckpt.get("config", {})
    model = Model(cfg.get("d", 384), cfg.get("e_layers", 3), cfg.get("t_layers", 6),
                  cfg.get("heads", 6), dex_feats=cfg.get("dex_feats", True)).to(device)
    smart_load(model, ckpt["model"])  # any vintage, function-preserving
    model.eval()
    cache = model.new_cache(MAX_BATTLES, device)
    slots = {}  # battle id -> cache slot
    free = list(range(MAX_BATTLES))
    print(f"serving {args.checkpoint} on :{args.port} "
          f"({device}, {'greedy' if args.greedy else 'sampled'})", flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bid = str(req.get("battle"))
            if self.path == "/end":
                if bid in slots:
                    slot = slots.pop(bid)
                    cache.reset(torch.tensor([slot], device=device))
                    free.append(slot)
                return self._json({"ok": True})
            if self.path != "/act":
                return self._json({"error": "unknown endpoint"}, 404)
            if bid not in slots:
                if not free:
                    return self._json({"error": "no free battle slots"}, 503)
                slots[bid] = free.pop()
            slot = slots[bid]
            f = np.asarray(req["floats"], np.float32)
            if len(f) != OBS_FLOATS:
                f = pad_obs_v1(f)  # bot still sends v1 obs
            ints = torch.tensor(np.asarray(req["ints"], np.int32)[None], device=device)
            floats = torch.tensor(f[None], device=device)
            mask = torch.tensor(np.asarray(req["mask"], np.uint8)[None], device=device)
            with torch.no_grad():
                h = model.step(ints, floats, cache, torch.tensor([slot], device=device))
                logits = model.pi(h)
                if args.temp:
                    logits = logits / args.temp
                dist = masked_dist(logits, mask)
                if args.min_p > 0 and not args.greedy:
                    p = dist.probs.clone()
                    p[p < args.min_p * p.max()] = 0.0
                    dist = torch.distributions.Categorical(probs=p)
                action = int(dist.probs.argmax(-1) if args.greedy and not args.temp
                             else dist.sample())
                win_prob = float(model.v(h))
            self._json({"action": action, "value": win_prob,
                        "probs": [round(p, 4) for p in dist.probs[0].tolist()]})

    HTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
