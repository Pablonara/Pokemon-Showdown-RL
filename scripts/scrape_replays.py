#!/usr/bin/env python3
"""Scrape the full public replay history for a Showdown format (default
gen1randombattle) into JSONL shards. Resumable, deduplicating, polite.

- Pages replay.pokemonshowdown.com/search.json backwards via `before=`.
- Fetches each replay's full log from /<id>.json.
- State in <out>/state.json (oldest uploadtime seen + downloaded ids).
- ~4 req/s with backoff on errors; safe to stop and re-run anytime.

Usage: python3 scripts/scrape_replays.py [format] [out_dir]
"""

import json
import pathlib
import sys
import time
import urllib.request

FORMAT = sys.argv[1] if len(sys.argv) > 1 else "gen1randombattle"
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else f"data/replays/{FORMAT}")
BASE = "https://replay.pokemonshowdown.com"
DELAY = 0.25  # seconds between requests
SHARD_SIZE = 2000

OUT.mkdir(parents=True, exist_ok=True)
state_path = OUT / "state.json"
state = json.loads(state_path.read_text()) if state_path.exists() else {"before": None, "done": []}
done = set(state["done"])


def get(url, tries=5):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pokemon-showdown-rl research scraper (contact: repo)"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001 - backoff on any transient error
            wait = 2 ** attempt
            print(f"retry {url} in {wait}s ({e})", flush=True)
            time.sleep(wait)
    return None


def shard_file():
    n = len(done) // SHARD_SIZE
    return OUT / f"shard-{n:05d}.jsonl"


def save_state(before):
    state["before"] = before
    state["done"] = sorted(done)
    state_path.write_text(json.dumps(state))


total0 = len(done)
t0 = time.time()
before = state["before"]
while True:
    url = f"{BASE}/search.json?format={FORMAT}" + (f"&before={before}" if before else "")
    page = get(url)
    time.sleep(DELAY)
    if page is None:
        print("search failed repeatedly; stopping", flush=True)
        break
    if not page:
        print("reached end of archive", flush=True)
        break
    for item in page:
        rid = item["id"]
        if rid in done:
            continue
        replay = get(f"{BASE}/{rid}.json")
        time.sleep(DELAY)
        if replay is None:
            continue
        with shard_file().open("a") as f:
            f.write(json.dumps(replay, separators=(",", ":")) + "\n")
        done.add(rid)
    before = page[-1]["uploadtime"]
    save_state(before)
    rate = (len(done) - total0) / max(time.time() - t0, 1)
    print(f"{len(done)} replays (oldest {time.strftime('%Y-%m-%d', time.localtime(before))}, "
          f"{rate:.1f}/s)", flush=True)
    if len(page) < 51 and len(page) < 10:  # short page = near the end
        print("archive exhausted", flush=True)
        break
print(f"done: {len(done)} replays in {OUT}", flush=True)
