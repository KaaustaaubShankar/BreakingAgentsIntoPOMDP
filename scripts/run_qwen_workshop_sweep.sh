#!/usr/bin/env bash
# Qwen-MLX workshop sweep.
#
# Phase 1 — primary rows for the paper's existing table (n=10 by default).
# These slot in next to the GPT-5.2 / GPT-4.1 / Grok / Claude rows.
#   - Instruct-2507 @ none      Qwen open-weights, no-reasoning
#   - Thinking-2507 @ medium    Qwen open-weights, with reasoning
#
# Phase 2 — supporting isolation analysis (n=5 by default, supporting only).
# Separates thinking-capability from token-budget.
#   - Instruct-2507 @ medium    null cell: same budget as Phase 1 thinking,
#                               but model can't think — isolates budget effect
#   - Thinking-2507 @ high      tests whether more reasoning budget moves
#                               win-rate beyond medium
#
# Usage:
#   ./scripts/run_qwen_workshop_sweep.sh \
#       [--env ka59simple|ka59] \
#       [--primary-trials N] [--supporting-trials N] \
#       [--skip-phase1] [--skip-phase2]
#
# Output:
#   results/qwen_workshop/<timestamp>/<phase>__<model>__<effort>__nN.log
#   results/<env>_real_ablation/ablation_qwen-mlx_*.json    (written by the runner)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_ID="ka59simple"
PRIMARY_TRIALS=10
SUPPORTING_TRIALS=5
SKIP_PHASE1=0
SKIP_PHASE2=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env) ENV_ID="$2"; shift 2 ;;
        --primary-trials) PRIMARY_TRIALS="$2"; shift 2 ;;
        --supporting-trials) SUPPORTING_TRIALS="$2"; shift 2 ;;
        --skip-phase1) SKIP_PHASE1=1; shift ;;
        --skip-phase2) SKIP_PHASE2=1; shift ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

MODEL_INSTRUCT="lmstudio-community/Qwen3-4B-Instruct-2507-MLX-8bit"
MODEL_THINKING="lmstudio-community/Qwen3-4B-Thinking-2507-MLX-8bit"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="results/qwen_workshop/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo " Qwen-MLX workshop sweep"
echo "   env:                 $ENV_ID"
echo "   primary trials:      $PRIMARY_TRIALS  (the two paper-table rows)"
echo "   supporting trials:   $SUPPORTING_TRIALS  (isolation analysis)"
echo "   log dir:             $LOG_DIR"
echo "   start:               $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

source .venv/bin/activate || { echo "FATAL: .venv missing — run 'uv venv .venv && uv pip install mlx-lm openai anthropic python-dotenv tqdm pydantic'"; exit 1; }

run_phase () {
    local label="$1"
    local model="$2"
    local efforts="$3"  # space-separated list
    local trials="$4"
    local log_name
    log_name="$(echo "${label}__${model##*/}__$(echo "$efforts" | tr ' ' '-')__n${trials}" | tr '/' '_')"
    local log_path="${LOG_DIR}/${log_name}.log"
    echo ""
    echo "--- $label :: $model :: efforts=[$efforts] :: n=${trials} ---"
    echo "    log → $log_path"
    local t0=$(date +%s)
    # shellcheck disable=SC2086
    python -m scripts.run_real_ablation \
        --env "$ENV_ID" \
        --provider qwen-mlx \
        --model "$model" \
        --trials "$trials" \
        --reasoning-effort $efforts \
        >>"$log_path" 2>&1
    local rc=$?
    local elapsed=$(( $(date +%s) - t0 ))
    if [[ $rc -eq 0 ]]; then
        echo "    DONE  (${elapsed}s)"
    else
        echo "    FAILED rc=$rc  (${elapsed}s) — see $log_path; continuing to next phase"
    fi
}

if [[ $SKIP_PHASE1 -eq 0 ]]; then
    echo ""
    echo "============ PHASE 1 — primary rows (paper table, n=$PRIMARY_TRIALS) ============"
    # Two cells that slot into the existing table next to GPT-5.2 / Grok / Claude.
    run_phase "phase1-instruct-none"      "$MODEL_INSTRUCT"  "none"    "$PRIMARY_TRIALS"
    run_phase "phase1-thinking-medium"    "$MODEL_THINKING"  "medium"  "$PRIMARY_TRIALS"
fi

if [[ $SKIP_PHASE2 -eq 0 ]]; then
    echo ""
    echo "========= PHASE 2 — supporting isolation (n=$SUPPORTING_TRIALS) ========="
    # Isolates thinking-capability from token-budget. Instruct @ medium is the
    # null cell (model can't think, same budget as Phase 1 thinking row).
    # Thinking @ high tests whether more headroom moves win-rate at all.
    run_phase "phase2-instruct-medium"    "$MODEL_INSTRUCT"  "medium"  "$SUPPORTING_TRIALS"
    run_phase "phase2-thinking-high"      "$MODEL_THINKING"  "high"    "$SUPPORTING_TRIALS"
fi

echo ""
echo "================================================================"
echo " SWEEP COMPLETE"
echo "   end:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "   logs:      $LOG_DIR"
echo "   ablations: results/${ENV_ID}_real_ablation/"
echo "================================================================"
