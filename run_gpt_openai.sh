#!/usr/bin/env bash
# KA59-Simple camera-ready: GPT-5.2 through the DIRECT OpenAI API.
#
# POOLING
#   Accepted May-13 OpenRouter trials are pooled with new direct-API trials, so
#   only the deficit runs. Recorded in the manifest under cross_transport_pooling
#   with its known differences -- notably that the accepted trials ran at
#   max_tokens=1024 and logged no token counts, so unlike the DeepSeek case the
#   cap cannot be shown to have been non-binding. Disclose in the paper.
#
#   mechanics_hard is already at 0/24 (pooled with the historical fallthrough
#   trials) and runs nothing. mechanics_hard_format_only is the ported control
#   with a different prompt hash, so it correctly inherits no history.
set -euo pipefail
cd "$(dirname "$0")"

PY=venv/bin/python
MODEL="gpt-5.2"          # direct-API slug; "openai/gpt-5.2" is an invalid model ID here
TARGET=20
MAX_INFRA_ERRORS=5

[ -n "${OPENAI_API_KEY:-}" ] || export OPENAI_API_KEY="$(grep -oE '^OPENAI_API_KEY=.*' .env | cut -d= -f2- | tr -d "\"' ")"
[ -n "${OPENAI_API_KEY}" ] || { echo "no OPENAI_API_KEY in .env" >&2; exit 1; }

cell () {  # $1 = effort, $2 = config
  echo "=== $1 / $2  (target N=$TARGET) ==="
  $PY -m scripts.run_ka59_camera_ready \
    --provider openai --model "$MODEL" --reasoning-effort "$1" --config "$2" \
    --target-n "$TARGET" --max-infrastructure-errors "$MAX_INFRA_ERRORS" --resume
  $PY -m scripts.run_ka59_camera_ready --index
}

effort_all () {  # $1 = effort
  for c in baseline world_hard mechanics_hard mechanics_hard_format_only feedback_hard; do
    cell "$1" "$c"
  done
}

case "${1:-}" in
  plan)   for e in none medium; do
            $PY -m scripts.run_ka59_camera_ready --provider openai --model "$MODEL" \
              --reasoning-effort "$e" --target-n "$TARGET" --plan
          done ;;
  none)   effort_all none ;;
  medium) effort_all medium ;;
  cell)   cell "${2:?effort}" "${3:?config}" ;;
  *) cat <<USAGE
usage: $0 {plan|none|medium|cell <effort> <config>}

  plan                      per-cell deficits, zero model calls
  none                      all five cells at effort none   (65 trials to run)
  medium                    all five cells at effort medium (100 trials)
  cell <effort> <config>    a single cell, e.g. cell none baseline

Configs: baseline world_hard mechanics_hard mechanics_hard_format_only feedback_hard
Every cell is resume-safe; re-run the same command to continue.
USAGE
     exit 1 ;;
esac
