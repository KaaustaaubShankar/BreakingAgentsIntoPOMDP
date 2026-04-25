#!/bin/bash
# eval.sh — ASI-Evolve evaluation wrapper for KA59
# Called by ASI-Evolve as: eval.sh <candidate_path>
# Must print a JSON object with "score" key to stdout.

set -e
cd "$(dirname "$0")/../.."  # repo root

CANDIDATE="$1"
if [ -z "$CANDIDATE" ]; then
    echo '{"score": 0.0, "error": "no candidate path"}'
    exit 1
fi

python experiments/asi_evolve_ka59/evaluator.py "$CANDIDATE"
