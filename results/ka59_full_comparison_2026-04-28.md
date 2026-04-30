# KA59 — Full Model Comparison

All runs use the corrected MECHANICS_EASY prompt ("*attempts to shift…interactions may produce unexpected results*"). 50 turns per trial, 1 level. Discovery rate computed under the strict keyword filter (push/move-through-wall context required, CLICK-selection language vetoed).

**Wins: 0/80 across every cell shown. ObjPush = 0, WallTransfer = 0 across all models. The wall_transfer mechanic is not behaviorally exploited by any model in any condition.** The signal lives in click count and discovery-rate-when-OODA-instrumented.

## Click count per trial (avg)

| Model · n | baseline | world_hard | goal_hard | mechanics_hard | feedback_hard | mechanics_ooda | mechanics_ooda_f |
|---|---:|---:|---:|---:|---:|---:|---:|
| **gpt-4.1**           | 46.8 | 39.6 | 45.2 | 48.6 | 46.6 | 52.4 | 47.6 |
| **gpt-5.1**           | 39.0 | — | — | — | — | 24.5 | 24.5 |
| **claude-haiku-4-5**  | 33.2 | 37.8 | 39.8 | 34.2 | 28.0 | 51.0¹ | 52.0¹ |
| **gpt-5.2 (no-R)²**  | 28.0 | 3.2 | 9.8 | 48.4 | 28.0 | — | — |
| **gpt-5.2 (med-R)³** | 9.4 | 1.8 | 11.8 | 42.0 | 10.0 | — | — |

¹ haiku OODA cells are n=1 pilot only; haiku matrix axes are n=5.
gpt-4.1 matrix axes and baseline-of-matrix are n=5; gpt-4.1 OODA cells (and OODA-baseline=60.0) are from the n=5 OODA rerun.
gpt-5.1 cells are n=2.
² gpt-5.2 (no-R) = no explicit reasoning effort; T040301 batch; n=5 per axis.
³ gpt-5.2 (med-R) = medium reasoning effort; baseline/world/goal from T062343 (n=5); mech_hard from T211019+T002328 (n=5 composite); feedback_hard from T002436+bonus (n=6). Corrected 2026-04-30.

## OODA scaffold — discovery rate (strict filter)

| Model · n | mechanics_ooda | mechanics_ooda_f | OODA-F reframes/trial |
|---|:---:|:---:|---:|
| **gpt-4.1 · n=5** | 1/5 (20%) | 0/5 (0%) | 7.4 |
| **gpt-5.1 · n=2** | 0/2 (0%)  | 0/2 (0%) | 3.5 |
| **claude-haiku-4-5 · n=1** | 1/1 (pilot) | 1/1 (pilot) | 9.0 |

## What the numbers actually say

- **gpt-4.1 matrix is flat** (39.6–48.6 across all 5 axes). The non-reasoning model doesn't behaviorally discriminate the knockouts; it click-spams at the same rate regardless of which axis is hardened. `world_hard` shows the only meaningful drop (-15% vs baseline) — degraded observation slightly suppresses clicking.
- **haiku matrix shows axis-specific suppression** (28–40 range). `feedback_hard` drops haiku to 28 clicks (-16% vs baseline) — when feedback is degraded, the model runs fewer probing experiments. `goal_hard` is highest at 40 (+20%) — unclear directionality, possibly searching harder when the goal is ambiguous.
- **gpt-5.2 (no reasoning) has the most dramatic axis discrimination.** `world_hard` crushes clicking to 3.2 (-89%) with avg 14.6 blocked moves — degraded observation causes the model to attempt movement but get stuck. `mech_hard` spikes to 48.4 (+73%) — the model click-spams more when mechanics are hidden. `feedback_hard` = neutral (0% vs baseline).
- **gpt-5.2 (medium reasoning) shows a different pattern from no-R.** `mech_hard` causes extreme click-spam (+347%, avg 42.0). `world_hard` again dominant suppressor (-81%). Crucially: `feedback_hard` shows LOW clicks (10.0) but HIGH blocked moves (28.2) — the model is moving and getting stuck rather than clicking. This is a qualitatively different failure mode from click-spam.
- **OODA scaffold reduces clicking sharply for the reasoning model** (gpt-5.1: 39 → 24.5, -37%). Smaller effect for gpt-4.1 (60 → 52.4, -13% within the OODA rerun). Asking the model to articulate observe/orient/decide makes it more deliberate, not just verbose.
- **Forced reframes fire but don't translate.** gpt-4.1 OODA-F averaged 7.4 reframes/trial, but discovery dropped to 0% under the strict filter (the apparent 100% before recompute was all CLICK-selection false positives). The metacognitive intervention is firing but not guiding the model toward wall_transfer.
- **Only one genuine discovery across 23 OODA-instrumented cells under the strict filter.** gpt-4.1 mechanics_ooda trial 3 articulated push-through-wall language. The "model can articulate the hidden mechanic" signal is rare even under the favorable scaffold.

## What's missing for the paper

- ~~gpt-5.x on the matrix axes~~ **DONE** — gpt-5.2 (no-R) and gpt-5.2 (med-R) full matrices added 2026-04-30. Note: the original openrouter runs used the broken prompt and remain invalid; the T040301 (no-R) and T062343+corrected (med-R) batches are canonical.
- **Sonnet, opus** — claude-proxy now works for haiku at 5.2s/turn, so a full sweep is feasible (~110 min/model). Pending.
- **Cross-environment.** Josh's LS20 results (gpt-4.1 + gpt-5.2) showed `world_hard` as the discriminating axis there, vs `mechanics_hard` (under the old prompt) on KA59. Re-running LS20 under the new prompt would close the cross-env story.

## Methodology footnotes

- Strict filter: `_DISCOVERY_PATTERNS` requires explicit push/move-through-wall context or "wall is passable" framing. `_VETO_PATTERNS` excludes CLICK-teleport language and negation phrases like "no path through the walls." See `ka59_game/prompts.py:check_discovery`.
- Click count = total CLICK actions per trial. CLICK is the action verb the model uses to select between pieces; on KA59, click-spam is the default failure mode (the model uses CLICK as a no-op when it can't figure out a movement).
- Reframes = `forced_reframe` events fired by the OODA-F harness when the model gets stuck (no progress for N consecutive turns). Only fires under `mechanics_ooda_f`.
- All trials at max_turns=50 (1 level). Per Kaaus: KA59 is configured as 50 turns/level, 1 level for these ablation runs.
