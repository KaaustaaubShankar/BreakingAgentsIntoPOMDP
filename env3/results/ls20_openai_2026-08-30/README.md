# LS20 knockout batches — OpenAI direct API (2026-08-30)

Two LS20 (env3) single-axis knockout batches run against `api.openai.com`
directly rather than through OpenRouter.

| Batch | Model | N/cell | Cells | Trials | Results dir |
|---|---|---|---|---|---|
| A | `gpt-5.6-luna` | 20 | 4 configs x {none, medium} | 160 | `env3/results/ls20_openai_2026-08-30/gpt-5.6-luna/` |
| B | `gpt-5.2` | 10 | 4 configs x {none, medium} | 80 | `env3/results/ls20_openai_2026-08-30/gpt-5.2/` |

`goal_hard` was excluded, matching `DEFAULT_CONFIGS` in `env3/ablation.py` and
the ka59simple knockout. Configs run: `baseline`, `world_hard`,
`mechanics_hard`, `feedback_hard`.

Aggregate (win rates, Wilson CIs, tokens, cost): `ls20_openai_aggregate.json`.

## Layout

```
env3/results/ls20_openai_2026-08-30/
├── ls20_openai_aggregate.json          both batches, win rates + Wilson CIs + cost
├── dashboard_rows.csv                  rows in tracking-sheet format
├── gpt-5.6-luna/
│   ├── ls20_FINAL_matrix_gpt-5.6-luna.json
│   ├── summaries/                      26 per-cell ablation summaries
│   └── raw/                            160 per-trial run JSONs (full turn history)
└── gpt-5.2/
    ├── ls20_FINAL_matrix_gpt-5.2.json
    ├── summaries/                      9 per-cell ablation summaries
    └── raw/                            80 per-trial run JSONs
```

The FINAL matrix files mirror the shape of
`results/ls20_real_ablation/ls20_FINAL_matrix_deepseek-v4-pro.json`. Unlike the
DeepSeek run, the per-trial JSONs are included so the win rates are checkable
from raw episodes.

Figure rows were appended to `env3/results/figure_data/winrate_ci.csv`.
The new GPT-5.2 rows are labelled
**`GPT-5.2 (OpenAI API)`** to keep them distinct from the existing n=5
`GPT-5.2` rows collected through OpenRouter — those two sets disagree
substantially on `world_hard` and must not be pooled without a decision:

| Cell | Existing (OpenRouter, n=5) | New (OpenAI API, n=10) |
|---|---|---|
| none::world_hard | 60% | 10% |
| medium::world_hard | 0% | 40% |
| none::mechanics_hard | 20% | 0% |
| medium::mechanics_hard | 100% | 100% |

## Reproducing

```bash
# 1. Put the key in the repo-root .env (gitignored)
echo 'OPENAI_API_KEY=sk-proj-...' >> .env

# 2. Launch every cell as a detached worker
python3 scripts/launch_ls20_openai_batches.py

# 3. Fill any cell short of its target N (safe to repeat; run with no workers alive)
python3 scripts/topup_ls20_openai_batches.py --chunk 3

# 4. Aggregate
python3 scripts/summarize_ls20_openai_batches.py --json out.json
```

A single cell directly:

```bash
cd env3
python3 ablation.py --provider openai --model gpt-5.6-luna \
  --trials 20 --configs world_hard --reasoning-effort medium \
  --input-cost-per-m 1.00 --output-cost-per-m 6.00 \
  --results-dir results_ls20_gpt56luna_n20 --quiet
```

## Notes for interpretation

- **New provider.** `env3/llm_client.py` previously supported OpenRouter and
  qwen-local only. An `openai` provider was added: it reads `OPENAI_API_KEY`,
  passes `reasoning_effort` as a top-level parameter (not OpenRouter's
  `extra_body.reasoning`), and uses `max_completion_tokens` because reasoning
  models on the OpenAI chat surface reject `max_tokens`. Verified that
  `effort=none` yields `reasoning_tokens=0` and `effort=medium` yields non-zero.
- **Endpoint change vs earlier GPT-5.2 data.** Prior LS20 GPT-5.2 rows were
  collected through OpenRouter as `openai/gpt-5.2` at n=5. These 10 trials come
  from the OpenAI API directly as `gpt-5.2`. Pooling the two sets mixes
  endpoints, so they are kept in a separate directory rather than merged.
- **Turn budget.** `max_levels=1` and `TURNS_PER_LEVEL=50`, so `avg_turns=50.0`
  means every trial in that cell exhausted its budget.
- **Errors.** GPT-5.2: 0 errors across 80 trials. gpt-5.6-luna: 23 turn-level
  errors across 160 trials (~8000 turns) — 22 were HTTP 429 rate limits that
  outlasted the client's retry budget while 28 workers ran concurrently, plus
  one invalid action (`MOVE_NORTHWEST`). A failed turn is logged and the episode
  continues, so the effect is ~0.3% of turns lost, concentrated in no single cell.
- **Sharding.** Trials for a cell were split across several workers to cut wall
  clock (`medium::world_hard` costs ~19 min/trial). Every filename carries the
  worker PID and a microsecond timestamp, so parallel writers do not collide.
  The aggregate counts per-trial run files rather than trusting any single
  per-cell summary, which is why there are more `ablation_summary_*.json` files
  than cells.
