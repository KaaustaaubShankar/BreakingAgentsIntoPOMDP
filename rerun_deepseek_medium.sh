#!/usr/bin/env bash
# KA59-Simple camera-ready: DeepSeek-V4-Pro medium cells, one cell per command.
#
# WHY THE EARLIER SWEEPS DIED
#   DeepSeek at medium draws 8,000-10,000 output tokens per turn on real KA59
#   states (measured: 8,101 and 9,934/turn). A cap near that mean truncates the
#   tail, and a truncated body returns content=None. The direct API at 4,096
#   lost 81% of turns; OpenRouter at 16,384 lost the tail. The cap is now
#   65,536 (ka59_game/llm_client.py:OPENROUTER_MAX_TOKENS) and a truncated body
#   is retried up to 3 times instead of voiding the trial.
#
# COST -- READ BEFORE RUNNING
#   ~1.2M output tokens per full trial at $1.74/M => roughly $2-4 per trial.
#   A 20-trial cell is therefore ~$40-80, and all five cells ~$200-400.
#   Check credit first; each cell is a separate command so you can stop anywhere.
set -euo pipefail
cd "$(dirname "$0")"

PY=venv/bin/python
MODEL="deepseek/deepseek-v4-pro"
SORT="throughput"   # no allow-list: route to the fastest provider available
TARGET=20
MAX_INFRA_ERRORS=5

# The OPENROUTER_API_KEY in ./.env is dead (401 "User not found"); the live key
# is in bp35/.env. Exported here to override what dotenv loads from the root.
export OPENROUTER_API_KEY="$(grep -oE '^OPENROUTER_API_KEY=.*' bp35/.env | cut -d= -f2- | tr -d "\"' ")"
[ -n "${OPENROUTER_API_KEY}" ] || { echo "no OpenRouter key in bp35/.env" >&2; exit 1; }

credit () {
  curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  | $PY -c 'import json,sys; d=json.load(sys.stdin)["data"]; print("credit remaining: $%.2f" % d["limit_remaining"])'
}

cell () {  # $1 = config
  credit
  echo "=== medium / $1  (target N=$TARGET) ==="
  $PY -m scripts.run_ka59_camera_ready \
    --provider openrouter --model "$MODEL" --upstream-sort "$SORT" \
    --reasoning-effort medium --config "$1" \
    --target-n "$TARGET" --max-infrastructure-errors "$MAX_INFRA_ERRORS" --resume
  $PY -m scripts.run_ka59_camera_ready --index
  credit
}

case "${1:-}" in
  plan)   $PY -m scripts.run_ka59_camera_ready --provider openrouter --model "$MODEL" \
            --upstream-sort "$SORT" --reasoning-effort medium --target-n "$TARGET" --plan ;;
  credit) credit ;;
  baseline|world_hard|mechanics_hard|mechanics_hard_format_only|feedback_hard) cell "$1" ;;
  all)    for c in baseline world_hard mechanics_hard mechanics_hard_format_only feedback_hard; do
            cell "$c"
          done ;;
  *) cat <<USAGE
usage: $0 {plan|credit|<cell>|all}

  plan     per-cell deficits as JSON; zero model calls
  credit   remaining OpenRouter balance
  <cell>   run one cell to N=$TARGET, resume-safe. One of:
             baseline  world_hard  mechanics_hard  mechanics_hard_format_only  feedback_hard
  all      every cell in order (~\$200-400 -- check credit first)

One cell at a time is the recommended way: a complete cell is usable evidence,
five partial cells are not. Infrastructure failures are saved as excluded
evidence, never scored as losses, and halt the run after $MAX_INFRA_ERRORS.
USAGE
     exit 1 ;;
esac
