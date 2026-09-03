#!/usr/bin/env bash
#
# KA59-Simple camera-ready: single-matrix driver (5 configs x N, one cell at a time).
#
# Run two of these SIMULTANEOUSLY, one per provider - they use different APIs
# and write to different protocol-hash directories, so they never collide:
#
#   PROVIDER=openrouter MODEL=openai/gpt-5.6-luna EFFORT=medium ./run_matrix.sh
#   PROVIDER=openai     MODEL=gpt-5.2             EFFORT=medium ./run_matrix.sh
#
# Do NOT run two workers on the SAME provider+model+effort+config: --resume
# counts the deficit once at startup and never re-checks, so both would run the
# full deficit and overshoot the target silently.
#
# Safe to kill and re-run: every cell uses --resume and runs only the deficit.
#
set -uo pipefail

REPO="${REPO:-/Users/kaaustaaubshankar/Documents/Coding/BreakingAgentsIntoPOMDP}"
cd "$REPO" || { echo "FATAL: cannot cd to $REPO"; exit 1; }

PY="./venv/bin/python"
PROVIDER="${PROVIDER:-openrouter}"
MODEL="${MODEL:?set MODEL, e.g. openai/gpt-5.6-luna (openrouter) or gpt-5.2 (openai)}"
EFFORT="${EFFORT:-medium}"
TARGET_N="${TARGET_N:-10}"
MAX_INFRA="${MAX_INFRA:-5}"
MIN_REMAINING="${MIN_REMAINING:-25}"   # openrouter credit floor (USD)
MAX_WAIT="${MAX_WAIT:-7200}"
CONFIGS=(baseline world_hard mechanics_hard mechanics_hard_format_only feedback_hard)

LOGDIR="${LOGDIR:-$REPO/camera_ready/logs}"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%dT%H%M%S)"
TAG="$(echo "${PROVIDER}_${MODEL}_${EFFORT}" | tr '/' '_')"
MAIN_LOG="$LOGDIR/matrix_${TAG}_${STAMP}.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"; }

# ---------------------------------------------------------------- API key ----
# NOTE: OPENROUTER_API_KEY in ./.env is DEAD (401). The live one is in bp35/.env.
if [ "$PROVIDER" = "openrouter" ]; then
  export OPENROUTER_API_KEY="$(grep -oE '^OPENROUTER_API_KEY=.*' bp35/.env 2>/dev/null | cut -d= -f2- | tr -d "\"' ")"
  [ -z "${OPENROUTER_API_KEY:-}" ] && { echo "FATAL: no OPENROUTER_API_KEY in bp35/.env"; exit 1; }
else
  export OPENAI_API_KEY="$(grep -ohE '^OPENAI_API_KEY=.*' bp35/.env .env 2>/dev/null | head -1 | cut -d= -f2- | tr -d "\"' ")"
  [ -z "${OPENAI_API_KEY:-}" ] && { echo "FATAL: no OPENAI_API_KEY found"; exit 1; }
fi

# Remaining OpenRouter credit, or "" if not applicable / undeterminable.
key_remaining() {
  [ "$PROVIDER" != "openrouter" ] && { echo ""; return; }
  curl -s --max-time 30 https://openrouter.ai/api/v1/key \
    -H "Authorization: Bearer $OPENROUTER_API_KEY" 2>/dev/null \
  | "$PY" -c 'import json,sys
try:
    r = json.load(sys.stdin)["data"].get("limit_remaining")
    print("" if r is None else round(float(r), 4))
except Exception:
    print("")'
}

# Fail fast on a dead key before spending anything.
if [ "$PROVIDER" = "openrouter" ]; then
  R="$(key_remaining)"
  [ -z "$R" ] && { echo "FATAL: OpenRouter key check failed (dead key / network)."; exit 1; }
  log "openrouter key OK - remaining \$$R"
else
  # NOTE: keys come from bp35/.env FIRST - ./.env holds dead keys for both
  # OpenAI and OpenRouter. Also, /v1/models returns 200 even with a dead balance, so it is NOT a
  # sufficient preflight. Issue a 1-token completion: that surfaces
  # insufficient_quota / credit_balance_exhausted before we burn a whole cell
  # on excluded infrastructure errors.
  RESP="$(curl -s --max-time 30 https://api.openai.com/v1/chat/completions \
    -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
    -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}],"max_tokens":1}')"
  if echo "$RESP" | grep -q '"error"'; then
    echo "FATAL: OpenAI preflight failed:"
    echo "$RESP" | head -c 400; echo
    exit 1
  fi
  log "openai key OK (completion preflight passed)"
fi

# -------------------------------------------------- same-provider collision --
# Only same-provider runners contend for a rate limit AND can double-count a
# shared cell. A run on the other provider is not a collision.
RUNNER_PAT="scripts.run_ka59_camera_ready.*--provider ${PROVIDER}\b"
waited=0
while [ "${SKIP_RUNNER_CHECK:-0}" != "1" ] && pgrep -f "$RUNNER_PAT" >/dev/null 2>&1; do
  if [ "$waited" -ge "$MAX_WAIT" ]; then
    log "FATAL: another $PROVIDER runner still active after ${MAX_WAIT}s. Aborting."; exit 1
  fi
  [ "$waited" -eq 0 ] && log "another $PROVIDER runner is active; waiting..."
  sleep 30; waited=$((waited + 30))
done

# upstream-sort is an OpenRouter routing concept and is part of the identity
# hash - only pass it there, and pass it on EVERY openrouter cell so the hashes
# stay consistent across a matrix.
UPSTREAM=()
[ "$PROVIDER" = "openrouter" ] && UPSTREAM=(--upstream-sort throughput)

log "=== START $PROVIDER | $MODEL @ $EFFORT | target_n=$TARGET_N | log $MAIN_LOG"
for cfg in "${CONFIGS[@]}"; do
  R="$(key_remaining)"
  if [ -n "$R" ] && [ "$("$PY" -c "print(1 if float('$R')<float('$MIN_REMAINING') else 0)")" = "1" ]; then
    log "STOPPING: credit \$$R below \$$MIN_REMAINING floor. Not starting $cfg."; exit 3
  fi
  CELL_LOG="$LOGDIR/${TAG}_${STAMP}_${cfg}.log"
  log ">>> CELL START $cfg ${R:+(credit \$$R)} -> $CELL_LOG"
  start=$(date +%s)
  "$PY" -u -m scripts.run_ka59_camera_ready \
    --provider "$PROVIDER" --model "$MODEL" --reasoning-effort "$EFFORT" \
    --config "$cfg" --target-n "$TARGET_N" \
    --max-infrastructure-errors "$MAX_INFRA" \
    ${UPSTREAM[@]+"${UPSTREAM[@]}"} --resume --verbose 2>&1 | tee -a "$CELL_LOG"
  rc="${PIPESTATUS[0]}"
  el=$(( $(date +%s) - start ))
  log "<<< CELL END $cfg rc=$rc | new: $(grep -c '^VALID WIN' "$CELL_LOG")W/$(grep -c '^VALID LOSS' "$CELL_LOG")L infra=$(grep -c '^ERROR (excluded' "$CELL_LOG") | $((el/60))m"
  if [ "$rc" -ne 0 ]; then
    log "HALTED on $cfg (rc=$rc). Stopping; error limit NOT raised. Completed cells stay valid."
    exit "$rc"
  fi
  "$PY" -m scripts.run_ka59_camera_ready --index >> "$MAIN_LOG" 2>&1
done
log "=== MATRIX COMPLETE: $PROVIDER $MODEL @ $EFFORT ${R:+| credit \$$(key_remaining)}"
