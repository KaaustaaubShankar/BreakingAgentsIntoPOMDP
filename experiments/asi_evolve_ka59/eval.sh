#!/bin/bash
# eval.sh — ASI-Evolve evaluation wrapper for KA59
#
# Called by ASI-Evolve as: bash eval.sh (no args)
# cwd = experiment directory
# Reads: ./code  (candidate agent_step function written by ASI-Evolve)
# Writes: ./results.json  (must include eval_score + success fields)

set -e

EXPERIMENT_DIR="$(pwd)"
CODE_FILE="${EXPERIMENT_DIR}/code"
RESULT_JSON="${EXPERIMENT_DIR}/results.json"
LOG_FILE="${EXPERIMENT_DIR}/eval.log"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EVALUATOR="${REPO_ROOT}/experiments/asi_evolve_ka59/evaluator.py"

error_exit() {
    cat > "$RESULT_JSON" << EOF
{
    "success": false,
    "eval_score": 0.0,
    "win_rate": 0.0,
    "avg_levels_completed": 0.0,
    "temp": {
        "error": "$1"
    }
}
EOF
    exit 0
}

if [ ! -f "$CODE_FILE" ]; then
    error_exit "candidate code not found at ${CODE_FILE}"
fi

echo "=== KA59 Evaluation ===" > "$LOG_FILE"
echo "Candidate: ${CODE_FILE}" >> "$LOG_FILE"
echo "Evaluator: ${EVALUATOR}" >> "$LOG_FILE"

# Run evaluator — outputs JSON to stdout
EVAL_OUTPUT=$(cd "$REPO_ROOT" && python3 "$EVALUATOR" "$CODE_FILE" 2>>"$LOG_FILE")
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ] || [ -z "$EVAL_OUTPUT" ]; then
    error_exit "evaluator failed (exit $EXIT_CODE)"
fi

# Parse score from evaluator stdout and write results.json
# Evaluator emits INFO lines before JSON — extract last line only
EVAL_JSON=$(echo "$EVAL_OUTPUT" | tail -1)
python3 - "$EVAL_JSON" "$RESULT_JSON" << 'PYEOF'
import sys, json

raw = sys.argv[1]
out_path = sys.argv[2]

try:
    data = json.loads(raw)
    score = float(data.get("score", 0.0))
    metrics = data.get("metrics", {})
    result = {
        "success": True,  # eval ran cleanly; score=0 is valid (agent lost)
        "eval_score": score,
        "win_rate": score,
        "avg_levels_completed": metrics.get("avg_levels_completed", 0.0),
        "trials": metrics.get("trials", []),
        "temp": {}
    }
except Exception as e:
    result = {
        "success": False,
        "eval_score": 0.0,
        "win_rate": 0.0,
        "avg_levels_completed": 0.0,
        "temp": {"error": str(e), "raw": raw[:200]}
    }

with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
PYEOF

echo "eval_score: $(python3 -c "import json; print(json.load(open('$RESULT_JSON')).get('eval_score', 0.0))")" >> "$LOG_FILE"
exit 0
