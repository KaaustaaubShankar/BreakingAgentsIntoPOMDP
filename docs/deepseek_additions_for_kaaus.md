# DeepSeek additions for the COLM paper — for Kaaus (from Edward)

These are **optional, data-driven additions** to drop into your paper as *you* see
fit — I'm not rewriting your prose. I reverted the working copy back to your
version. Below is exactly what's new in the data and where it could go, plus the
rationale, so you can decide what to include.

Source of truth: the **Google sheet** (overview). Raw per-trial JSON is in the
repo for agent analysis only. Precomputed CIs: `results/figure_data/winrate_ci.csv`.

---

## 1. DeepSeek-V4-Pro is now run (it wasn't in the table)
N=20 per cell, 128-turn budget. KA59-Simple medium pools real trials across the
OpenRouter + direct API runs (n=14–23; empty/errored trials excluded).

Table rows you could add (cols: Model & Reason & BASELINE & WORLD & GOAL & MECH & FEEDBACK):
```
DeepSeek V4 Pro & none   & 60\%  & 0\% & --- & 20\% & 75\%  \\   % KA59-Simple
DeepSeek V4 Pro & medium & 19\%  & 0\% & --- & 0\%  & 22\%  \\   % KA59-Simple
DeepSeek V4 Pro & none   & 100\% & 5\% & --- & 45\% & 95\%  \\   % LS20
DeepSeek V4 Pro & medium & 100\% & 5\% & --- & 0\%  & 100\% \\   % LS20
```
(Goal dropped per the meeting → `---`. BP35 DeepSeek not run.)

## 2. These rows support real significance (n=20)
The GPT/Grok cells are n=5 so they stay qualitative, but DeepSeek is large enough
for Wilson 95% CIs + Fisher's exact (no vs medium reasoning):
- KA59-Simple BASELINE 60%→19% (p=0.019), FEEDBACK_HARD 75%→22% (p=0.003)
- LS20 MECHANICS_HARD 45%→0% (p=0.001)
Interesting wrinkle: medium reasoning **hurts** DeepSeek where it **helped** GPT —
worth a sentence if you want it. Full CI table LaTeX is ready in
`results/figure_data/` notes if useful.

## 3. Figures
`results/figure_data/winrate_ci.csv` has every cell with `k, n, win%, ci_lo, ci_hi`
(Wilson 95%) for error-bar bars. `figure_data/figure_specs.md` has the plot spec.

## 4. Data issue to flag (NOT a rewrite — your call)
The "0% verbal discovery" line may be an instrumentation artifact: the strict
keyword filter checked a channel that was mostly empty. A judge pass over 689
trials gives **hypothesize 30–60% / confirm 0.6%** (κ=0.90 across two judges;
see `docs/colm_verbal_discovery_rewrite.md`). Flagging so you can decide whether
to keep, correct, or cut that claim — entirely your call.

## Optional drafts available if you want them
- A Background "Problem Setting" subsection (POMDP formalism + axis→component map)
- A prompt-templates appendix (the EASY/HARD axis templates verbatim)
Both are drafted; say the word and I'll send them as standalone snippets to adapt.
