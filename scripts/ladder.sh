#!/bin/bash
# Play rated games on the official Pokémon Showdown ladder.
#
#   PS_USER=BotAccountName PS_PASS=secret scripts/ladder.sh [games=10] [extra bot args]
#
# Uses the local model server (starts one on runs/latest_r6.pt if not running).
# Credentials come from the environment only - never stored in the repo.
# Add e.g. `--narrate YourMainAccount` to get per-turn thinking as PMs.
set -euo pipefail
cd "$(dirname "$0")/.."

GAMES="${1:-10}"
shift || true
# credentials: env vars, or a local .ladder.env file (gitignored, chmod 600)
#   PS_USER=name
#   PS_PASS=password
[ -f .ladder.env ] && set -a && . .ladder.env && set +a
: "${PS_USER:?set PS_USER to the bot account name}"
: "${PS_PASS:?set PS_PASS to the bot account password}"

mkdir -p logs
CKPT="${MODEL_CKPT:-runs/serving.pt}" # refresh_bot.sh keeps this at the newest run
if ! pgrep -f "serve_mode[l]" > /dev/null; then
  echo "starting model server ($CKPT, greedy)..."
  nohup python3 python/serve_model.py "$CKPT" --greedy > logs/serve.log 2>&1 &
  sleep 8
fi

LOG="logs/ladder-$(date +%Y%m%d-%H%M%S).log"
echo "laddering $GAMES games as $PS_USER -> $LOG"
# auto-resume: if the connection drops mid-session, replay only the remainder
DONE=0
TRIES=0
while [ "$DONE" -lt "$GAMES" ] && [ "$TRIES" -lt 5 ]; do
  node scripts/showdown_bot.mjs \
    --server wss://sim3.psim.us/showdown/websocket \
    --name "$PS_USER" --pass "$PS_PASS" \
    --search "$((GAMES - DONE))" --timer "$@" 2>&1 | tee -a "$LOG" || true
  DONE=$(grep -cE ": (win|tie)" "$LOG" || true)
  TRIES=$((TRIES + 1))
  if [ "$DONE" -lt "$GAMES" ]; then
    echo "session interrupted ($DONE/$GAMES games done), resuming in 10s..."
    sleep 10
  fi
done
echo "session complete: $DONE/$GAMES games (see $LOG)"
