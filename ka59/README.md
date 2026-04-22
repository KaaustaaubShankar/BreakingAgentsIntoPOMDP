# Multi-Env Ablation Runner (ka59/)

Thin dispatcher that mirrors the Kaus env4 ablation for **LS20 / KA59 / BP35**
and produces a unified Josh table:

```
  Win%  AvgTurns  AvgLevels  InvClk  Flips  RelDiff
```

## Quick start

```bash
# Show pre-baked sample table (no LLM calls)
python ablation.py --sample

# Live run — all envs, 3 trials, free Llama model
python ablation.py

# BP35 only, 5 trials, Claude Sonnet
python ablation.py --envs bp35 --trials 5 --model anthropic/claude-sonnet-4-6

# 64-turn proactive KA59 runs (Flash / Sonnet — real KA59 0% → lift?)
python ablation.py --envs ka59 --model google/gemini-2.0-flash --trials 3
python ablation.py --envs ka59 --model anthropic/claude-sonnet-4-6 --trials 3
```

## Pytest

```bash
# From repo root
python -m pytest ka59/tests/ -v

# From ka59/
python -m pytest tests/ -v
```

## Column legend

| Column    | BP35              | LS20                   | KA59                    |
|-----------|-------------------|------------------------|-------------------------|
| InvClk    | invalid_actions   | wall_collisions        | blocked_count           |
| Flips     | gravity_flips     | goals_ever_activated   | passable_walls_found    |
| RelDiff   | avg_turns / baseline_avg_turns (within env × model group) |

## Ablation configs

Same 5-axis structure as env3/env4:

| Config        | World | Goal | Mechanics | Feedback |
|---------------|-------|------|-----------|----------|
| baseline      | EASY  | EASY | EASY      | EASY     |
| world_hard    | HARD  | EASY | EASY      | EASY     |
| goal_hard     | EASY  | HARD | EASY      | EASY     |
| mechanics_hard| EASY  | EASY | HARD      | EASY     |
| feedback_hard | EASY  | EASY | EASY      | HARD     |

## Fixes shipped in this module

- **BP35 baseline (4cbacf8 broken)** — runner uses env4's env4-chirality path
  with v2 obs (action_affordances / valid_targets) as default format.
- **v2 obs** — `world_easy_format=v2` is now the default for BP35 runs.
- **KA59 LLM** — `ka59_llm_runner.py` lets Flash/Sonnet play the
  mechanics-discovery probe (64-turn budget, transfer_wall_push scenario).
- **Cross-game hypo traces** — each UnifiedRunResult.hypo_trace captures
  env-specific discovery signals (gravity-flip turns / goal-activation turns /
  passable-wall IDs) for cross-env hypothesis ruling.

## Environment setup

Requires a virtualenv with `arc-agi`:

```bash
cd jkj-breaking-agents/ka59
python3.12 -m venv .venv
source .venv/bin/activate
pip install arc-agi openai python-dotenv numpy pillow
```

Copy (or symlink) `environment_files/` from `~/tmp/kaus-jkj/env4/` — already
done if you cloned via the setup script.  Both `bp35` and `ls20` game files
must be present:

```
ka59/environment_files/
  bp35/
  ls20/
```
