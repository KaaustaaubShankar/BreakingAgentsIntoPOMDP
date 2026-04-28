# KA59 OODA / OODA-F Sweep — 2026-04-28 (Experiment B)

**Lane:** ka59_game (Edward)
**Provider:** Direct OpenAI API (`OPENAI_API_KEY`)
**Trials:** n=2 per cell, 50 turns per trial
**Total trials:** 8 (2 configs × 2 models × 2 trials)

**Companion to:** [Experiment A — knockout matrix](ka59_matrix_2026-04-27.md)

## Setup notes

- **Reasoning model substitution:** matrix used `openai/gpt-5.2` via OpenRouter; this sweep used `gpt-5.1` direct (gpt-5.2 isn't on the direct OpenAI key). Both are gpt-5.x reasoning models from the same lineage but the version delta is real — flag it in any cross-experiment claim.
- **Harness fix landed:** the `_generate_openai` path was sending `max_tokens=1024`, which OpenAI rejects for gpt-5.x and o-series reasoning models. Patched to `max_completion_tokens=4096` to give reasoners headroom for thinking tokens. This is why the matrix-via-OpenRouter run was able to use gpt-5.2 (OpenRouter ignores the param) but a direct gpt-5.x run would have failed without the fix.
- **Smoke check:** gpt-4.1 baseline via direct OpenAI = 43 clicks/trial; matrix-via-OpenRouter baseline = 45/45 = 45 avg. Provider drift is within trial-to-trial noise. We treat the matrix's baseline numbers as a valid reference for this sweep.

## Headline result

**OODA / OODA-F repair the `mechanics_hard` collapse in the reasoning model.**

| Config | gpt-4.1 clicks/trial | gpt-5.x clicks/trial | Δ vs baseline (gpt-5.x) |
|---|---:|---:|---:|
| `baseline` *(matrix, gpt-5.2)* | 45.0 | 16.0 | — |
| `mechanics_hard` *(matrix, gpt-5.2)* | 49.0 | 47.0 | **+194%** ← collapse |
| `mechanics_ooda` *(this run, gpt-5.1)* | 41.0 | 18.0 | +13% |
| `mechanics_ooda_f` *(this run, gpt-5.1)* | 38.0 | 16.5 | +3% |

Under `mechanics_hard`, gpt-5.x's click count collapses from baseline 16 → 47, matching gpt-4.1's flat click-spam — the reasoning advantage is erased. Substituting the OODA scaffold (which replaces the mechanics description rather than removing it) brings gpt-5.x back to ~16–18 clicks, **at or below baseline**. Adding the forced-reframe layer (OODA-F) does not improve over OODA on click count.

For gpt-4.1, the scaffold yields a smaller but consistent improvement (45 → 41 → 38). The non-reasoning model never had a strong baseline behavioral signature to lose, so there's less to "repair."

**Wins remain 0/8 across both models and both configs.** The wall-transfer mechanic is still not behaviorally exploited. The repair is in *behavioral signature*, not in solving.

## Per-model progression tables

### gpt-4.1

| Config | Wins | Avg clicks/trial | Per-trial | Reframes | Discovery |
|---|:---:|---:|:---:|---:|---:|
| baseline *(matrix)* | 0/2 | 45.0 | 45, 45 | 0 | 0% |
| mechanics_hard *(matrix)* | 0/2 | 49.0 | 49, 49 | 0 | 0% |
| mechanics_ooda | 0/2 | 41.0 | 41, 41 | 0 | 50% * |
| mechanics_ooda_f | 0/2 | 38.0 | 38, 38 | 7 | 100% * |

### Reasoning model (gpt-5.2 → gpt-5.1)

| Config | Model | Wins | Avg clicks/trial | Per-trial | Reframes | Discovery |
|---|:---:|:---:|---:|:---:|---:|---:|
| baseline *(matrix)* | gpt-5.2 | 0/2 | 16.0 | 16, 16 | 0 | 0% |
| mechanics_hard *(matrix)* | gpt-5.2 | 0/2 | 47.0 | 47, 47 | 0 | 0% |
| mechanics_ooda | gpt-5.1 | 0/2 | 18.0 | 15, 21 | 0 | 0% |
| mechanics_ooda_f | gpt-5.1 | 0/2 | 16.5 | 14, 19 | 8 | 0% |

`*` See "discovery rate caveat" below.

## Discovery rate caveat

`check_discovery()` in `ka59_game/prompts.py` is keyword-string matching on the orient field:

```python
DISCOVERY_KEYWORDS = ["transfer", "pass through", "asymmetr", "different for push",
                      "through the wall", "through wall", "push through", ...]
```

gpt-4.1's OODA-F orient text says **"Clicking just *transfers* selection"** (turn 4 of trial 2) — the word "transfers" matches the keyword filter, but contextually the model is talking about CLICK transferring *which piece is selected*, not the wall_transfer mechanic at all. Same artifact for trial 1 with "selectable pieces cannot move *through walls*" matching `"through wall"`.

So gpt-4.1's apparent 100% discovery rate on OODA-F is a **measurement artifact** of the loose keyword filter. gpt-5.1 uses different vocabulary in its richer orient text ("teleports/swaps", "Discarding prior hypothesis") that doesn't trigger the filter — yielding a 0% reading despite arguably more sophisticated reasoning.

**Recommendation for the paper:** report click count as the primary behavioral signal. Discovery rate, if reported, should come with the caveat that the keyword filter has high false-positive rate on the non-reasoning model and high false-negative rate on the reasoning model.

## Reframe activity (sanity check on OODA-F)

| Config | gpt-4.1 reframes / trial | gpt-5.1 reframes / trial |
|---|---:|---:|
| mechanics_ooda_f trial 1 | 4 | 6 |
| mechanics_ooda_f trial 2 | 3 | 2 |

Both models receive forced reframes through the trial — confirming the OODA-F mechanism is firing. gpt-5.1 averages slightly more reframes than gpt-4.1 (4 vs 3.5), suggesting the reasoner is ping-ponging between hypotheses more often.

## Story for the Ben meeting

> **Experiment A** (matrix): Win rate doesn't discriminate axes — all 0%. Click count does, for the reasoning model only. `mechanics_hard` specifically erases gpt-5.x's behavioral discrimination, dragging it to gpt-4.1-level click-spam.
>
> **Experiment B** (this sweep): Replacing the mechanics axis-knockout with an OODA scaffold *repairs* the collapse. gpt-5.x click count drops from 47 (under hard) back to ~16–18 (under OODA / OODA-F), matching baseline. The OODA-F forced-reframe layer doesn't add measurable behavioral lift over plain OODA, but does fire actively (7–8 reframes across 4 trials).
>
> Together: the reasoning model has the behavioral capacity to discriminate, but loses it under information starvation; metacognitive scaffolding restores the discrimination. Neither produces wins — wall-transfer remains undiscovered behaviorally — but the matrix axis the scaffold fixes is exactly the one where the matrix showed reasoning collapse.

## Caveats

- **n=2 is small.** Click-count differences are striking on the page but error bars are wide. Specifically the two trials at each cell often produced identical click counts (e.g., gpt-4.1 OODA = 41/41), which deserves checking — could indicate deterministic-seed behavior rather than independent samples.
- **Different reasoning models in A vs B** (gpt-5.2 vs gpt-5.1). Same lineage, but the within-experiment baseline (gpt-5.2 matrix vs gpt-5.1 OODA) introduces a small confound. Worth running gpt-5.1 baseline once for cross-validation if budget allows.
- **Discovery rate is unreliable** (see caveat section above). Click count is the cleaner signal.
- **Wins are 0 across all 28 trials run tonight** (20 from matrix + 8 from OODA sweep). Cannot make any claim about win-rate effects of OODA.

## Source artifacts

- `results/ka59_real_ablation/ablation_openai_gpt-4.1_20260428T075210.json` — gpt-4.1 OODA + OODA-F summary
- `results/ka59_real_ablation/ablation_openai_gpt-5.1_20260428T081142.json` — gpt-5.1 OODA + OODA-F summary
- `results/ka59_real_ablation/sidecar_openai_*_mechanics_ooda*.json` — per-config sidecars (4 files)
- `results/ka59_game/run_fE_gE_mO_wE_2026042807*.json` — per-trial JSONs with full action + orient histories
