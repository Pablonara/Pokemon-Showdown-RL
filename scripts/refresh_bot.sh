#!/bin/bash
# Pull the newest checkpoint from the training box, restart the local model
# server + RanchuBot on the local Showdown server.
#
#   PSRL_BOX=user@host scripts/refresh_bot.sh [run_dir=r7]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:-r7}"
: "${PSRL_BOX:?set PSRL_BOX=user@host of the training box}"

CKPT=$(ssh -p 22 "$PSRL_BOX" "ls ~/psrl/runs/$RUN/ | tail -1")
echo "fetching $RUN/$CKPT"
scp -q -P 22 "$PSRL_BOX:psrl/runs/$RUN/$CKPT" runs/serving.pt

pkill -f "serve_mode[l]" 2>/dev/null || true
pkill -f "showdown_bo[t]" 2>/dev/null || true
sleep 1
nohup python3 python/serve_model.py runs/serving.pt --greedy > /tmp/serve.log 2>&1 &
sleep 8
nohup node scripts/showdown_bot.mjs \
  --server ws://localhost:8000/showdown/websocket \
  --name RanchuBot --accept > /tmp/bot.log 2>&1 &
sleep 4
tail -1 /tmp/serve.log
tail -1 /tmp/bot.log
echo "RanchuBot refreshed to $RUN/$CKPT"
