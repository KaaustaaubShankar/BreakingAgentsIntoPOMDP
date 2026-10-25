#!/usr/bin/env bash
# KA59-Simple camera-ready: DeepSeek-V4-Pro via OpenRouter, upstream pinned.
#
# Cells already satisfied by pooled accepted evidence are NOT re-run:
#   none/baseline 12/20, none/world_hard 0/20, none/feedback_hard 15/20,
#   none/mechanics_hard 8/39 (pooled with the historical fallthrough trials).
# What remains: the ported format-only control at none, and all five at medium.
#
# Resume-safe: re-running the same stage continues where it stopped. Trials that
# fail on infrastructure are saved as excluded evidence and never counted.
set -euo pipefail
cd "$(dirname "$0")"

PY=venv/bin/python
MODEL="deepseek/deepseek-v4-pro"
UPSTREAM="DigitalOcean,StreamLake,GMICloud"
TARGET=20
MAX_INFRA_ERRORS=5

# The OPENROUTER_API_KEY in ./.env is dead (401 "User not found"). The live key
# is in bp35/.env. Export it here so dotenv's root .env value is overridden.
export OPENROUTER_API_KEY="$(grep -oE '^OPENROUTER_API_KEY=.*' bp35/.env | cut -d= -f2- | tr -d "\"' ")"
if [ -z "${OPENROUTER_API_KEY}" ]; then echo "no OpenRouter key found in bp35/.env" >&2; exit 1; fi

run_cell () {  # effort, config
  local effort="$1" config="$2"
  echo "=== ${effort} / ${config} ==="
  $PY -m scripts.run_ka59_camera_ready \
    --provider openrouter --model "$MODEL" --upstream-provider "$UPSTREAM" \
    --reasoning-effort "$effort" --config "$config" \
    --target-n "$TARGET" --max-infrastructure-errors "$MAX_INFRA_ERRORS" --resume
}

plan_all () {
  for effort in none medium; do
    $PY -m scripts.run_ka59_camera_ready \
      --provider openrouter --model "$MODEL" --upstream-provider "$UPSTREAM" \
      --reasoning-effort "$effort" --target-n "$TARGET" --plan
  done
}

case "${1:-}" in
  plan)    plan_all ;;
  probe)   run_cell none  mechanics_hard_format_only ;;   # 20 trials, ~1h, ~$4
  medium)  for c in baseline world_hard mechanics_hard mechanics_hard_format_only feedback_hard; do
             run_cell medium "$c"
           done ;;
  all)     "$0" probe && "$0" medium ;;
  *) cat <<USAGE
usage: $0 {plan|probe|medium|all}

  plan    zero model calls; prints the per-cell deficit as JSON
  probe   stage 1 -- none/format-only control, 20 trials  (~1 h,  ~\$4)
  medium  stage 2 -- all five cells at medium, 100 trials (SLOW: >=100 s/turn
          measured on DigitalOcean; re-probe before committing. ~\$94)
  all     both stages, in order

Check remaining credit before stage 2:
  curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer \$OPENROUTER_API_KEY"
USAGE
     exit 1 ;;
esac
