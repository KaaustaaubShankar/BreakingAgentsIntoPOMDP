#!/bin/bash
# DeepSeek full knockout matrix — KA59-Simple (addresses Sean review points 1,3,5)
#
# BUDGET REALITY: $150 total available. N=20 per cell is non-negotiable per Sean.
# We MUST be extremely disciplined on reasoning efforts.
#
# CURRENT STATUS (from our runs + review):
# - Only none + medium so far.
# - Medium is dramatically more expensive than none (reasoning tokens explode).
# - Real OpenRouter rates for deepseek/deepseek-v4-pro are ~$0.30/M input, $0.90/M output
#   (the old JSON "costs" of $1.56 / $7.68 were using Grok defaults and are fake for this model).
#
# RECOMMENDATION (my strong advice):
# - Stick to **none + medium only** for the full N=20 matrix.
# - Do NOT add high or xhigh right now.
#   Reasons:
#     * Budget: high/xhigh will easily double or triple the medium cost. We don't have room if we want N=20 across 5-6 configs.
#     * Science: none vs medium already gave us the interesting non-monotonic and cross-model differences in the past GPT/Grok data.
#       High/xhigh are more "aspirational" and risk turning DeepSeek into another expensive non-diagnostic row (review point 5).
#     * Time: higher efforts make every turn slower.
#   Only consider high/xhigh later if:
#     - After seeing the none+medium N=20 data, one cell is mysteriously bad and higher effort might rescue it.
#     - Budget has headroom after N=20 none+medium is secured.
#     - The meeting tomorrow explicitly asks for it.
#
# BATCHING STRATEGY (per meeting):
# Run batches of 20 trials per cell (N=20 per cell for statistical power per Sean's feedback).
# Current n=5 batches are insufficient; we are expanding to 20.
# We can merge multiple batches using scripts/merge_ablation_batches.py if needed.
# Run none first (cheap), then medium if budget allows. Goal axis removed per meeting (keep feedback).
#
# CONFIGS (goal axis removed; 5 configs total):
# baseline + world_hard + mechanics_hard + mechanics_hard_format_only (to address the confound) + feedback_hard
#
# USAGE (parallel for better wall time):
#   bash run_deepseek_full_knockout.sh none     # do this first (cheap, fast) -- runs 5 configs in PARALLEL
#   bash run_deepseek_full_knockout.sh medium   # only after reviewing none + meeting -- also parallel
#   bash run_deepseek_full_knockout.sh both     # (advanced) both reasonings fully parallel (monitor costs and budget!)
#
# The script now launches one background process per config (for the chosen reasoning).
# This should cut the wall-clock time for a full "none" matrix from many hours down to ~time of the slowest single cell.
# Separate per-config logs: /tmp/deepseek_<effort>_<config>.log
# PIDs and wait logic are used so it still waits for everything to finish before exiting.
#
# After any batch, send me the new ablation JSON(s) and I'll merge + give you clean CSV rows
# with correct real costs.

set -euo pipefail

# Prefer the project's .venv if it exists (this is how previous successful runs were done)
if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
    echo "Using project .venv python: $PYTHON"
else
    PYTHON="python3"
    echo "Using system python3 (you may need to 'source .venv/bin/activate' first if arc_agi is missing)"
fi

REASONING="${1:-none}"
shift || true

# Use realistic DeepSeek rates so the printed $ numbers are trustworthy
IN_RATE=0.30
OUT_RATE=0.90

COMMON_BASE=(
  --env ka59simple
  --provider openrouter
  --model deepseek/deepseek-v4-pro
  --trials 20
  --input-cost-per-m "$IN_RATE"
  --output-cost-per-m "$OUT_RATE"
)

# Safely initialize arrays (critical under set -u / nounset)
declare -a EXTRA_ARGS=()
declare -a CUSTOM_CONFIGS=()

# Parse remaining args (support --configs foo bar for targeted runs)
while [ $# -gt 0 ]; do
  if [ "$1" = "--configs" ]; then
    shift
    while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
      CUSTOM_CONFIGS+=("$1")
      shift
    done
  else
    EXTRA_ARGS+=("$1")
    shift
  fi
done

if [ ${#CUSTOM_CONFIGS[@]} -gt 0 ]; then
  CONFIGS_LIST=("${CUSTOM_CONFIGS[@]}")
else
  CONFIGS_LIST=(baseline world_hard mechanics_hard mechanics_hard_format_only feedback_hard)
fi

# Build COMMON safely (avoids "unbound variable" on empty array under set -u)
COMMON=("${COMMON_BASE[@]}")
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  COMMON+=("${EXTRA_ARGS[@]}")
fi

run_one_config() {
    local cfg="$1"
    local effort="$2"
    local log="/tmp/deepseek_${effort}_${cfg}.log"
    echo "  [parallel] Starting ${effort}::${cfg} -> $log"
    "$PYTHON" -m scripts.run_real_ablation "${COMMON[@]}" \
      --reasoning-effort "$effort" \
      --configs "$cfg" \
      > "$log" 2>&1 &
    echo $! > "/tmp/deepseek_${effort}_${cfg}.pid"
}

wait_for_configs() {
    local effort="$1"
    echo "Waiting for all ${effort} configs to finish..."
    for cfg in "${CONFIGS_LIST[@]}"; do
        pidfile="/tmp/deepseek_${effort}_${cfg}.pid"
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                wait "$pid" 2>/dev/null || true
            fi
            rm -f "$pidfile"
        fi
        log="/tmp/deepseek_${effort}_${cfg}.log"
        if [ -f "$log" ]; then
            echo "  [done] ${effort}::${cfg} (see $log)"
            # show last line of summary if present
            tail -5 "$log" | grep -E "(wins|→|SUMMARY)" | tail -1 || true
        fi
    done
}

case "$REASONING" in
  none)
    if printf '%s\n' "${COMMON[@]}" | grep -q -- '--configs'; then
      echo "=== Launching DeepSeek — reasoning=none (targeted configs from args, single process, n=20 per cell) ==="
      "$PYTHON" -m scripts.run_real_ablation "${COMMON[@]}" \
        --reasoning-effort none
    else
      echo "=== Launching DeepSeek full matrix — reasoning=none only, n=20 per cell (PARALLEL mode) ==="
      echo "This is the cheap/fast half. Good for getting real token numbers before the meeting."
      echo "Running all 5 configs in parallel (one background process per config) for much better wall time."
      for cfg in "${CONFIGS_LIST[@]}"; do
          run_one_config "$cfg" "none"
      done
      wait_for_configs "none"
      echo "All 'none' configs finished. Check individual logs or the sidecars in results/ka59simple_real_ablation/"
    fi
    ;;
  medium)
    if printf '%s\n' "${COMMON[@]}" | grep -q -- '--configs'; then
      echo "=== Launching DeepSeek — reasoning=medium (targeted configs from args, single process, n=20 per cell) ==="
      echo "WARNING: Expect 4-8x the cost and time of 'none' because of reasoning tokens."
      "$PYTHON" -m scripts.run_real_ablation "${COMMON[@]}" \
        --reasoning-effort medium
    else
      echo "=== Launching DeepSeek full matrix — reasoning=medium, n=20 per cell (PARALLEL mode) ==="
      echo "WARNING: Expect 4-8x the cost and time of 'none' because of reasoning tokens."
      echo "Only do this if budget and the meeting allow it."
      echo "Running all 5 configs in parallel for better wall time."
      for cfg in "${CONFIGS_LIST[@]}"; do
          run_one_config "$cfg" "medium"
      done
      wait_for_configs "medium"
      echo "All 'medium' configs finished."
    fi
    ;;
  both)
    echo "=== Launching BOTH none + medium in parallel (each with internal parallel configs) ==="
    echo "This will use significant parallelism. Monitor costs carefully."
    for cfg in "${CONFIGS_LIST[@]}"; do
        run_one_config "$cfg" "none" &
    done
    for cfg in "${CONFIGS_LIST[@]}"; do
        run_one_config "$cfg" "medium" &
    done
    wait
    echo "All done."
    ;;
  high|xhigh)
    echo "ERROR: high/xhigh not recommended right now."
    echo "See the top of this script for the budget + review reasoning."
    echo "If the meeting tomorrow explicitly wants it, we can add a targeted version (e.g. only baseline + mechanics_hard)."
    exit 1
    ;;
  *)
    echo "Usage: $0 [none|medium]"
    echo "Run none first. Decide on medium (and the remaining 15 trials to N=20) after the meeting."
    exit 1
    ;;
esac

echo
echo "Batch complete. New data in results/ka59simple_real_ablation/"
echo "Run the same command again later with more trials (or the other reasoning) and we'll merge."
