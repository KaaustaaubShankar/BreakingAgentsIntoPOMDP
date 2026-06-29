# GPT-5.2 KA59-Simple — per-trial files

This directory holds the GPT-5.2 KA59-Simple runs in two forms:

- **Aggregates / sidecars** (`ablation_*.json`, `sidecar_*.json`) — the original
  bundled outputs the figure was built from.
- **Per-trial files** (`run_*_<config>_<effort>_t<n>.json`) — one file per trial,
  with full turn-by-turn `history`, matching the convention used for BP35 and the
  DeepSeek KA59 dirs.

## Provenance of the per-trial files

The per-trial histories were always saved (in `results/ka59simple_game/`); they
just weren't packaged here. They are recovered verbatim from those originals — no
data is synthesised.

**Effort tag.** Per-trial `reasoning_effort` logging was only added on 2026-06-13;
these runs predate it, so the on-disk originals carry `reasoning_effort: null`.
The tag was recovered as follows (each restored file is marked
`"_effort_recovered": true`):

- **Main configs** (baseline / world_hard / mechanics_hard / feedback_hard) come
  from the single 2026-05-13 ablation (`ablation_..._20260513T055440`), which ran
  **N=5 per effort**. Each trial was matched back to that aggregate's per-cell
  `trial_data`:
  - **Winning trials** match by exact fingerprint (won + turns + wall_transfers +
    forced_reframes + max_goals_occupied + …) → effort tag is exact.
  - **Losing trials** are interchangeable within a config (identical config,
    `won:false`, identical metrics), so they fill the per-effort loss quota. Which
    individual loss is labelled `none` vs `medium` is reconstructed, but every
    cell's win/loss count and all metrics are preserved exactly.
- **norules** (`mechanics_hard_format_only`) effort is unambiguous by run date:
  `none` = 2026-06-01, `medium` = 2026-06-20. All trials are losses (0%), so the
  10-per-effort selection here is lossless.

Every (config, effort) cell reproduces the aggregate / figure win rate exactly:

| config | none | medium |
|--------|------|--------|
| baseline | 5/5 | 4/5 |
| world_hard | 0/5 | 0/5 |
| mechanics_hard | 0/5 | 0/5 |
| feedback_hard | 3/5 | 5/5 |
| mechanics_hard_format_only (norules) | 0/10 | 0/10 |

## Note on N

The main configs have **N=5 per effort** of real per-trial data (one 2026-05-13
run). The submission figure presents these rows at N=10; that is a presentation
choice at the paper level, not a second batch of trials — only N=5/effort exists
on disk. norules is N=10 per effort.
