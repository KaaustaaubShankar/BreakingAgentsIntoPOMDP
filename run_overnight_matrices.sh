#!/usr/bin/env bash
#
# KA59-Simple camera-ready: overnight driver.
#
#   Matrix 1: openai/gpt-5.6-luna  @ reasoning effort "medium"
#   Matrix 2: openai/gpt-5.2       @ reasoning effort "medium"
#
# Each matrix is 5 configs x N=20, run STRICTLY one cell at a time.
#
# Usage (from the repo root):
#   ./run_overnight_matrices.sh                  # run in the foreground
#   nohup ./run_overnight_matrices.sh > /tmp/overnight.log 2>&1 &   # detached
#
# Safe to kill and re-run at any time: every cell uses --resume, so the runner
# counts what is already on disk and runs only the deficit. Nothing duplicates.
#
# Tunables (override via env):
#   TARGET_N=20 MAX_INFRA=5 MIN_REMAINING=25 ./run_overnight_matrices.sh
#
set -uo pipefail

REPO="${REPO:-/Users/kaaustaaubshankar/Documents/Coding/BreakingAgentsIntoPOMDP}"
cd "$REPO" || { echo "FATAL: cannot cd to $REPO"; exit 1; }

PY="./venv/bin/python"
TARGET_N="${TARGET_N:-10}"
MAX_INFRA="${MAX_INFRA:-5}"
# Abort before starting a new cell if OpenRouter credit falls below this (USD).
MIN_REMAINING="${MIN_REMAINING:-25}"
# Max seconds to wait for another camera-ready runner to finish before giving up.
MAX_WAIT="${MAX_WAIT:-7200}"

CONFIGS=(baseline world_hard mechanics_hard mechanics_hard_format_only feedback_hard)
# "<model> <reasoning_effort>", run in this order.
MATRICES=(
  "openai/gpt-5.6-luna medium"
  "openai/gpt-5.2 medium"
)

LOGDIR="${LOGDIR:-$REPO/camera_ready/logs}"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%dT%H%M%S)"
MAIN_LOG="$LOGDIR/overnight_${STAMP}.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"; }

# ---------------------------------------------------------------- API key ----
# The OPENROUTER_API_KEY in ./.env is DEAD (401 "User not found").
# The live key lives in bp35/.env.
if [ ! -f bp35/.env ]; then
  echo "FATAL: bp35/.env not found; that is where the live key lives."; exit 1
fi
export OPENROUTER_API_KEY="$(grep -oE '^OPENROUTER_API_KEY=.*' bp35/.env | cut -d= -f2- | tr -d "\"' ")"
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "FATAL: could not read OPENROUTER_API_KEY from bp35/.env"; exit 1
fi

# Returns remaining credit in USD, or empty string if it cannot be determined.
key_remaining() {
  curl -s --max-time 30 https://openrouter.ai/api/v1/key \
    -H "Authorization: Bearer $OPENROUTER_API_KEY" 2>/dev/null \
  | "$PY" -c 'import json,sys
try:
    d = json.load(sys.stdin)["data"]
    r = d.get("limit_remaining")
    print("" if r is None else round(float(r), 4))
except Exception:
    print("")'
}

REMAINING="$(key_remaining)"
if [ -z "$REMAINING" ]; then
  echo "FATAL: key check failed. The key is dead, rate-limited, or the network is down."
  echo "       Verify with:"
  echo "       curl -s https://openrouter.ai/api/v1/key -H \"Authorization: Bearer \$OPENROUTER_API_KEY\""
  exit 1
fi
log "key OK - remaining credit: \$$REMAINING"

# ------------------------------------------- don't collide with another run --
# Two concurrent OPENROUTER runners share a rate limit; 429s land as
# infrastructure failures and can halt a cell that was otherwise healthy.
# Runners on other providers (e.g. --provider foundry) use a different API and
# quota entirely, so they are not a collision - match openrouter runs only.
# Set SKIP_RUNNER_CHECK=1 to bypass this guard altogether.
RUNNER_PAT="${RUNNER_PAT:-scripts.run_ka59_camera_ready.*--provider openrouter}"
waited=0
while [ "${SKIP_RUNNER_CHECK:-0}" != "1" ] && pgrep -f "$RUNNER_PAT" > /dev/null 2>&1; do
  if [ "$waited" -ge "$MAX_WAIT" ]; then
    log "FATAL: another camera-ready runner still active after ${MAX_WAIT}s. Aborting."
    exit 1
  fi
  [ "$waited" -eq 0 ] && log "another camera-ready runner is active; waiting for it to finish..."
  sleep 30
  waited=$((waited + 30))
done
[ "$waited" -gt 0 ] && log "other runner finished after ${waited}s; proceeding."

# ------------------------------------------------------------------- run -----
log "=== START overnight driver | target_n=$TARGET_N max_infra=$MAX_INFRA min_remaining=\$$MIN_REMAINING"
log "=== log: $MAIN_LOG"

for matrix in "${MATRICES[@]}"; do
  MODEL="${matrix%% *}"
  EFFORT="${matrix##* }"
  log "########## MATRIX: $MODEL @ effort=$EFFORT ##########"

  for cfg in "${CONFIGS[@]}"; do
    REMAINING="$(key_remaining)"
    if [ -n "$REMAINING" ]; then
      below="$("$PY" -c "print(1 if float('$REMAINING') < float('$MIN_REMAINING') else 0)")"
      if [ "$below" = "1" ]; then
        log "STOPPING: credit \$$REMAINING is below the \$$MIN_REMAINING floor. Not starting $cfg."
        log "Re-run this script after topping up; --resume picks up exactly where it left off."
        exit 3
      fi
    fi

    CELL_LOG="$LOGDIR/${STAMP}_$(echo "$MODEL" | tr '/' '_')_${EFFORT}_${cfg}.log"
    log ">>> CELL START $MODEL $EFFORT $cfg (credit \$$REMAINING) -> $CELL_LOG"
    start=$(date +%s)

    # -u so VALID/ERROR lines appear in the log immediately instead of sitting
    # in a block buffer. --upstream-sort throughput is part of the protocol
    # identity hash, so it must match across every cell of a matrix.
    "$PY" -u -m scripts.run_ka59_camera_ready \
      --provider openrouter --model "$MODEL" --reasoning-effort "$EFFORT" \
      --config "$cfg" --target-n "$TARGET_N" \
      --max-infrastructure-errors "$MAX_INFRA" \
      --upstream-sort throughput --resume --verbose 2>&1 | tee -a "$CELL_LOG"
    rc="${PIPESTATUS[0]}"

    elapsed=$(( $(date +%s) - start ))
    wins="$(grep -c '^VALID WIN' "$CELL_LOG" 2>/dev/null || echo 0)"
    losses="$(grep -c '^VALID LOSS' "$CELL_LOG" 2>/dev/null || echo 0)"
    infra="$(grep -c '^ERROR (excluded' "$CELL_LOG" 2>/dev/null || echo 0)"
    log "<<< CELL END $cfg rc=$rc | new this run: ${wins}W/${losses}L, excluded_infra=${infra} | $((elapsed/60))m ${elapsed}s"

    if [ "$rc" -ne 0 ]; then
      # rc=2 means the runner stopped after MAX_INFRA infrastructure failures.
      # Do NOT raise the limit to push through - stop and report.
      log "HALTED on $cfg (rc=$rc). Stopping the chain; the error limit was NOT raised."
      log "Completed cells remain valid. Investigate, then re-run this script to resume."
      exit "$rc"
    fi

    "$PY" -m scripts.run_ka59_camera_ready --index >> "$MAIN_LOG" 2>&1
  done

  log "########## MATRIX COMPLETE: $MODEL @ $EFFORT ##########"
done

REMAINING="$(key_remaining)"
log "=== ALL MATRICES COMPLETE | remaining credit: \$$REMAINING"
"$PY" -m scripts.run_ka59_camera_ready --index | tee -a "$MAIN_LOG"
