# KA59-Simple figure — corrected to clean run-sets (A + B)

The earlier figure pooled runs in ways that matched no single experiment. This
corrects the KA59-Simple heatmap so **every cell is backed by one clean run-set**
shipped in this directory — no cross-run pooling, real Wilson CIs.

## Corrected numbers (now == the shipped data)

| Model / reasoning | N | baseline | world_hard | mech_hard | mech_norules | feedback_hard |
|---|---|---|---|---|---|---|
| GPT-5.2 none | 5 | 100% (5/5) | 0% | 0% | 0% | 60% (3/5) |
| GPT-5.2 medium | 5 | 80% (4/5) | 0% | 0% | 0% | 100% (5/5) |
| DeepSeek-V4-Pro none | 20 | 60% (12/20) | 0% | 20% (4/20) | 20% (4/20) | 75% (15/20) |
| DeepSeek-V4-Pro medium | 20 | 10% (2/20) | 0% | 0% | 0% | 5% (1/20) |

Backing data (this PR + PR #16):
- `gpt-5.2/` — N=5 none+medium matrix (`...20260513T055440...`) + mech_norules runs.
  GPT-5.2 per-trial JSONs don't store `reasoning_effort`, so the effort split lives
  in the aggregate file (shipped) rather than per-trial filenames.
- `deepseek-v4-pro-medium/` — clean direct-API N=20 (100 runs).
- DeepSeek-V4-Pro none N=20 is in **PR #16**.
- `ablation_summary_*` — per-config summaries (env3/env4 array format).
- `plots/plot.py` updated + `plots/ka59_simple_heatmap.{pdf,png}` regenerated.

## What changed vs. the previous figure
- GPT-5.2 was shown pooled as N=10 `[90,0,0,0,80]` for both efforts. That was an
  approximation; the real data is N=5 **per effort**: none `100/0/0/0/60`,
  medium `80/0/0/0/100`.
- DeepSeek-medium was shown `20/0/0/0/10`; the clean N=20 set is `10/0/0/0/5`.
- World / mechanics / norules cells (all 0%) are unchanged — robust everywhere.

## Paper text edits this implies (suggestions for the authors — not applied)
The current paper (`backup(2).tex`) was written against the old pooled numbers:
1. **§Models and Metrics ("each config is evaluated for 20 trials")** — true for
   DeepSeek (N=20) but **GPT-5.2 KA59-Simple is N=5**. Either re-run GPT-5.2 to
   N=20 or state the N=5 (the figure now shows the wider CIs honestly).
2. **§"Feedback knockouts" ("GPT-5.2 configurations fall from 90% to 80%")** —
   with the split: GPT-5.2 *none* goes 100→60 under feedback_hard; *medium* is
   80→100 (i.e. flat within N=5 noise). Reword to the per-effort numbers.
3. **§"Feedback knockouts" ("Deepseek medium falls from 20% to 10%")** — now
   baseline 10% → feedback 5%.
4. **§"Low baseline performance limits knockout interpretation" (Deepseek medium
   "20% baseline")** — now **10%**. The capability-threshold argument is
   unaffected (10% is still near-floor vs GPT-5.2's 80–100%).
5. Swap the figure image (`ka59_simple_heatmap_2.png`) for the regenerated one.

Net: the World/Mechanics/Feedback *conclusions* do not change — only a few exact
percentages and the GPT-5.2 N move.
