# KA59-Simple results — reality, validity, and options

This PR archives the actual KA59-Simple run data behind the paper figure and
applies one confirmed correction to the figure.

## Fix applied in this PR
**GPT-5.2 (medium) row was mis-entered with the *none* row's numbers.** The
hand-entered figure (commit `1c36111`) showed GPT-5.2 medium = `90/0/0/0/80`,
identical to none. The Google Sheet, the `513-medium` run, and the per-effort
data all say medium = **`80/0/0/0/100`**. `plots/plot.py` + the regenerated
`ka59_simple_heatmap.pdf` now show GPT-5.2 med = 80/0/0/0/100 (distinct from none
90/0/0/0/80). The none row was already correct.

## The short version
- **DeepSeek-V4-Pro was the only model run at a full clean N=20.** GPT-5.2 on
  KA59-Simple is N=10 per reasoning setting (2 runs of N=5 each; effort inferred
  from token/time since it predates per-trial effort logging).
- The paper figure's KA59 numbers were **hand-entered** (commit `1c36111`,
  2026-06-20 "cleaning up plot") from the Google Sheet, not computed from the run
  files. `plots/plot.py` stores percentages and back-calculates counts
  (`k = round(pct/100 * N)`), so the displayed fractions (e.g. "4/20") are derived
  from the percentage, not real win tallies.

## What the run data actually supports

| Model | reasoning | N | baseline | world | mech | norules | feedback | source |
|---|---|---|---|---|---|---|---|---|
| DeepSeek-V4-Pro | none | **20** | 60% (12/20) | 0% | 20% (4/20) | 20% (4/20) | 75% (15/20) | clean direct API `20260612` |
| DeepSeek-V4-Pro | medium | **20** | **10% (2/20)** | 0% | 0% | 0% | **5% (1/20)** | clean direct API `20260617` |
| GPT-5.2 | none | 5 | 100% (5/5) | 0% | 0% | 0% | 60% (3/5) | run `20260513` + norules batches |
| GPT-5.2 | medium | 5 | 80% (4/5) | 0% | 0% | 0% | 100% (5/5) | run `20260513` + norules batches |
| GPT-5.2 | **pooled** | **10** | **90% (9/10)** | 0% | 0% | 0% | **80% (8/10)** | `20260513` none+medium combined |

## Three findings

**1. GPT-5.2 N=10 = 90/0/0/0/80 is REPRODUCIBLE and legitimate.** It is the single
`20260513` run pooled across its two reasoning settings (none 5/5 + medium 4/5 =
9/10 baseline; none 3/5 + medium 5/5 = 8/10 feedback). KA59 GPT was run at N=5 per
effort — fewer than BP35 (N=15) and LS20 (N=10) — and pooled across reasoning to
reach N=10, comparable to the other environments. **One caveat:** the figure shows
this as *two identical* per-effort rows (med and none both 90/0/0/0/80), which reads
as N=10 *per effort*. It is really one N=10 pool; the genuine per-effort split is
none `100/0/0/60` vs medium `80/0/0/100` (N=5 each).

**2. DeepSeek-medium baseline is the one real divergence.** The figure shows 20%
(rendered 4/20), but at the full clean N=20 the result is **10% (2/20)**. Only 3
winning medium-baseline trials exist in the entire dataset, so 20% is a favorable
small-N cut (~3/16), not the N=20 number. Same direction for feedback: figure 10%
vs clean N=20 **5%**. Since DeepSeek was the only model run to a full N=20, its
clean N=20 is the authoritative DeepSeek number.

**3. Everything else matches.** DeepSeek-none (60/0/20/20/75) and all the 0% cells
(world, mechanics) reproduce exactly across every run.

## Does this affect paper validity?
**No conclusion is overturned.** World→0%, Mechanics→KA59 collapse (except
DeepSeek-none 20%), Feedback survivable, DeepSeek-med near-floor — all robust. The
issues are accuracy/presentation, not findings:
- Methods says "each config is evaluated for 20 trials." True for DeepSeek (N=20),
  **not for GPT-5.2** (KA59 N=10 pooled / N=5 per effort; BP35 N=15; LS20 N=10).
- The GPT KA59 figure presents a pooled N=10 number as two per-effort rows.
- DeepSeek-medium 20%/10% is a sub-N=20 cut; the full N=20 is 10%/5%.

## Options (re-running GPT-5.2 is too expensive, so both are no-new-data)

**Option 1 — keep the figure matching the paper, fix only the wording (minimal).**
- Leave the figure as-is (this PR does this).
- Methods: state the real N — "GPT-5.2: N=10 on KA59-Simple (pooled across reasoning),
  N=15 BP35, N=10 LS20; DeepSeek-V4-Pro: N=20." Drop the blanket "20 trials."
- Either relabel the two identical GPT rows as one "GPT-5.2 (N=10)" row, or add a
  footnote that they are pooled across reasoning.
- DeepSeek-medium: keep 20%/10% only if reported as the small-N cut with N stated;
  otherwise move to the clean N=20 (10%/5%).

**Option 2 — correct the figure to the clean data (most defensible).**
- GPT-5.2: one pooled "N=10" row (90/0/0/0/80) **or** the per-effort N=5 split
  (none 100/0/0/60, med 80/0/0/100).
- DeepSeek-medium: 10%/5% (full clean N=20).
- Update the three sentences in §Results that cite the changed numbers.

Either way the central story is unchanged; Option 1 keeps Kaaus's figure and only
needs the Methods N sentence plus the DeepSeek-medium caveat.
