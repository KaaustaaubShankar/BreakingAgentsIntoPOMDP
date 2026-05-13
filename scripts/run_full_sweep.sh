#!/usr/bin/env bash
# run_full_sweep.sh — turnkey driver for the JKJ knockout-matrix sweep.
#
# Usage:
#   ./scripts/run_full_sweep.sh ENV
#
# where ENV is one of: ka59 ka59simple bp35 ls20
#
# Runs the full attributed comparison for one environment:
#   GPT-5.2 (no-reasoning, medium-reasoning) x 5 configs x 5 trials
#   Grok-4.1-fast (no-reasoning, medium-reasoning) x 5 configs x 5 trials
#   Plus the uniform-random baseline (50 trials).
#
# Prereqs:
#   export OPENROUTER_API_KEY=sk-or-...
#   pip install openai dotenv  (already in requirements)
#
# Total runtime estimate at current per-trial latency (~1-3s/turn):
#   ka59simple: ~30-45 min wall-clock if cells run in parallel
#   ka59:       ~60-90 min (50 turns vs 32, 7 levels vs 1)
#   bp35/ls20:  comparable to ka59simple
#
# Output: results/<env>_real_ablation/ablation_*.json
#         results/<env>_game/run_*.json (per-trial event histories)
#         results/<env>_real_ablation/random_baseline_*.json

set -euo pipefail

ENV="${1:-}"
if [[ -z "$ENV" ]]; then
  echo "usage: $0 ENV" >&2
  echo "  ENV ∈ {ka59, ka59simple, bp35, ls20}" >&2
  exit 1
fi

case "$ENV" in
  ka59)        TURNS=50 ;;
  ka59simple)  TURNS=32 ;;
  bp35|ls20)   TURNS=64 ;;
  *)
    echo "unknown env: $ENV" >&2
    exit 1
    ;;
esac

CONFIGS="baseline world_hard goal_hard mechanics_hard feedback_hard"
TRIALS=5
LOG_DIR="logs/$ENV-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG_DIR"

echo "=== Sweep: env=$ENV  turns=$TURNS  trials=$TRIALS  log_dir=$LOG_DIR ==="

# --- 1. Random baseline (no LLM, fast) ---
echo "[1/5] Random baseline ($TRIALS=50 trials)..."
python3 -m scripts.run_random_baseline --env "$ENV" --trials 50 \
        --max-turns "$TURNS" --seed 0 \
        > "$LOG_DIR/random.log" 2>&1 &
PID_RAND=$!

# --- 2-5. Four LLM cells in parallel ---
launch_cell() {
  local cell="$1"; local model="$2"; local effort="$3"
  echo "[2-5] Launching $cell ($model, $effort)..."
  python3 -u -m scripts.run_real_ablation \
    --env "$ENV" --provider openrouter \
    --model "$model" --reasoning-effort "$effort" \
    --trials "$TRIALS" --max-turns "$TURNS" \
    --configs $CONFIGS \
    > "$LOG_DIR/$cell.log" 2>&1 &
}

launch_cell "gpt52_no"  "openai/gpt-5.2"     "none"
launch_cell "gpt52_med" "openai/gpt-5.2"     "medium"
launch_cell "grok_no"   "x-ai/grok-4.1-fast" "none"
launch_cell "grok_med"  "x-ai/grok-4.1-fast" "medium"

echo
echo "All processes launched. Waiting for completion..."
echo "  Random baseline PID: $PID_RAND"
echo "  Tail any cell live:  tail -F $LOG_DIR/<cell>.log"
echo

wait

echo
echo "=== Sweep complete ==="
echo "Results in: results/${ENV}_real_ablation/"
echo "Per-trial JSONs in: results/${ENV}_game/"
echo "Logs in: $LOG_DIR/"
echo
echo "Next step: python3 scripts/wilson_cis.py  (computes CIs across all cells)"
