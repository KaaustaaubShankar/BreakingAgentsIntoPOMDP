#!/bin/bash
# Run ASI-Evolve evolution on KA59.
# Usage: ./experiments/asi_evolve_ka59/run.sh [--steps 30]
#
# Reads OPENROUTER_API_KEY from .env in the jkj repo root.

set -e
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ASI_EVOLVE="$HOME/ASI-Evolve"

# Load .env from repo root
if [ -f "$REPO_ROOT/.env" ]; then
    export $(grep -v '^#' "$REPO_ROOT/.env" | xargs)
fi

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY not set"
    exit 1
fi

STEPS="${1:---steps}"
if [ "$STEPS" = "--steps" ]; then
    SHIFT_ARGS=""
else
    SHIFT_ARGS=""
fi

cd "$ASI_EVOLVE"
exec python3 main.py \
    --experiment ka59_pomdp \
    --config "$REPO_ROOT/experiments/asi_evolve_ka59/config.yaml" \
    --eval-script "$REPO_ROOT/experiments/asi_evolve_ka59/eval.sh" \
    "$@"
