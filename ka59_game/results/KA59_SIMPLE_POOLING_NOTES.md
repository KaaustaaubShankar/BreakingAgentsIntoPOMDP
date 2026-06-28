# KA59-Simple results — pooling decision (for review)

This PR adds backing data for the two KA59-Simple paper rows that are **not**
reproducible from a single clean run-set: DeepSeek-V4-Pro **medium** and
**GPT-5.2**. (The DeepSeek-V4-Pro **none** row is a clean N=20 set and was
added separately — it reproduces the figure exactly.)

**The honest problem:** the win rates printed in the paper figure do not exactly
match any single run-set or the full pool of runs. They are specific selections.
This doc lays out the gap so we can decide how to report them. Nothing here is
cherry-picked to hit the paper number.

## Cell-by-cell: paper figure vs. what the data supports

| Model / config | Paper figure | Clean single set | Full pool of all runs |
|---|---|---|---|
| **DeepSeek med — baseline** | **20%** | 10% (2/20, direct API) | 6% (3/48) |
| DeepSeek med — world_hard | 0% | 0% (0/20) | 0% (0/48) |
| DeepSeek med — mechanics_hard | 0% | 0% (0/20) | 0% (0/46) |
| DeepSeek med — mech_norules | 0% | 0% (0/20) | 0% (0/46) |
| **DeepSeek med — feedback_hard** | **10%** | 5% (1/20, direct API) | 9% (4/46) |
| **GPT-5.2 — baseline** | **90%** | (no clean set) | 82% (28/34) |
| GPT-5.2 — world_hard | 0% | — | 0% (0/26) |
| GPT-5.2 — mechanics_hard | 0% | — | 0% (0/27) |
| GPT-5.2 — mech_norules | 0% | — | 0% (0/41) |
| **GPT-5.2 — feedback_hard** | **80%** | — | 73% (22/30) |

The world/mechanics/norules cells (all 0%) are robust — they agree everywhere.
The disagreement is only on **baseline** and **feedback_hard** for both models.

## Why GPT-5.2 has no clean set
GPT-5.2's baseline/world/mechanics/feedback runs were logged **without**
`reasoning_effort` in the per-trial JSON (only the later `mech_norules` runs are
effort-tagged). So none-vs-medium **cannot be split from the files** — which is
why the paper figure shows identical none/medium rows. The figure's 90%/80% is
one N=10 subset; GPT-5.2 baseline across the 9 logged mini-batches ranges 50–100%.

## What this PR ships
- `deepseek-v4-pro-medium/` — clean direct-API **N=20** runs (100 files) +
  `ablation_summary_deepseek_v4_pro_medium.json` → **10/0/0/0/5**.
- `gpt-5.2/` — **all** non-OODA GPT-5.2 runs (158 files) +
  `ablation_summary_gpt_5_2_pooled.json` → **82/0/0/0/73** (full pool, effort
  not separable). Summary numbers match the shipped files exactly.

## Options to decide
1. **Re-state the paper** to the clean/pooled numbers (DeepSeek med 10/5;
   GPT-5.2 82/73) — most defensible, but the figure changes.
2. **Keep the figure** and add a footnote that GPT-5.2 rows are pooled across
   reasoning settings (N=10 subset) and the DeepSeek-medium baseline is a
   small-N estimate — paper already flags this row as near-floor (§"Low baseline
   performance limits knockout interpretation").
3. **Drop GPT-5.2 raw data** from the submission and let the figure stand on the
   code + DeepSeek sets only.

Note: the world/mechanics/feedback *conclusions* are unaffected either way — only
the exact baseline/feedback percentages move by ~5–10 points.
