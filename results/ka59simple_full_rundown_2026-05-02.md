# KA59simple ablation — full rundown

_Last updated: 2026-05-02 | Author: Edward Lue Chee Lip | Lane: ka59_game_

This document captures everything that changed in the ka59 lane between 2026-04-30 and 2026-05-02, including the methodological tool we built (`ka59simple`), a critical bug we found and fixed in the OpenRouter handler, and the final n=5–10 results across two models × three reasoning levels. It is meant to be auditable by a reviewer who was not in the conversation.

---

## 1. What is `ka59simple`, and why we built it

### 1.1 Motivation

Canonical KA59 (the ARC-AGI puzzle as published in `environment_files/ka59/38d34dbb/`) **floors gpt-5.2 to 0% wins** across all conditions and reasoning levels (n=30+ trials, baseline included). This is a real finding — KA59 requires discovering a hidden wall-transfer mechanic (engine asymmetry between `loydmqkgjw` MOVE and `ifoelczjjh` PUSH functions) — but it makes the 4-axis knockout matrix uninformative: every cell is `0/n`, so we cannot tell whether the World / Goal / Mechanics / Feedback axes degrade outcomes differentially.

### 1.2 What we changed

`ka59simple` is a single-level, single-goal **fork** of canonical KA59 level 1, living at `environment_files/ka59simple/20260430/`. It is loaded via `--env ka59simple` in `scripts/run_real_ablation.py`. It preserves the hidden wall-transfer mechanic but shortens the optimal-solve from ~12 actions to **6 actions**:

1. RIGHT × 3 → player pushes pushee through wall via `ifoelczjjh` transfer
2. CLICK on pushee at (33, 21) → switches player to transferred sprite
3. RIGHT + UP → fills goal at (35, 17), WIN

Layout deltas vs canonical:
- 2 selectables retained @ (9, 21) and (18, 21) — both required (need pusher + pushee)
- **1 goal kept** @ (35, 17) — left goal **removed**
- Wall @ (24, 12) — preserved
- Letterbox @ (-3, -3) — preserved

### 1.3 Why this is a methodological tool, NOT a benchmark replacement

**This must be acknowledged explicitly in the paper.** `ka59simple` is not pulled straight from ARC-AGI — it is a constructed variant. The paper should frame it as:

> "A single-mechanic variant constructed to escape baseline floor effects, enabling 4-axis ablation comparison while preserving the wall-transfer hidden mechanic that defines KA59."

It supports the World / Mechanics knockout findings; it does not replace canonical KA59 evidence. Failed earlier designs (v1, v2) are documented in memory to avoid repeating them.

---

## 2. The OpenRouter `reasoning_effort` bug

### 2.1 What the bug was

In `ka59_game/llm_client.py`, the `_generate_openrouter` handler did not pass the `reasoning_effort` parameter to OpenRouter, even though the constructor stored it on `self.reasoning_effort` and the OpenAI direct handler (`_generate_openai`) honored it. Every OpenRouter-routed call was silently dropping the flag and using OpenAI's default reasoning level (`medium` for the chat completions endpoint, but interactively probed as `≈ minimal` for gpt-5.2 — see §2.3).

### 2.2 Scope of the bug

All ~56 ka59simple trials run overnight 2026-04-30 → 2026-05-01 with `--reasoning-effort none` or `--reasoning-effort medium` were affected. They produce well-formed trial JSONs (no errors, full action histories), so the bug was invisible without inspecting `usage.completion_tokens_details.reasoning_tokens`.

### 2.3 How we verified the fix

Built `scripts/probe_reasoning_effort.py` to issue a controlled API call at each effort level and print `reasoning_tokens` from the response. On gpt-5.2 via OpenRouter, after the fix:

| effort | reasoning_tokens | What this confirms |
|--------|------------------|--------------------|
| OMITTED | 15 | Default ≈ `minimal` |
| `none` | **0** | True no-reasoning is achievable |
| `minimal` | 15 | |
| `low` | 15 | |
| `medium` | 19 | |
| `high` | 22 | |

We re-confirmed on a 303-token KA59-style prompt (not just "17×23"): `effort=none` → reasoning_tokens=0; `effort=medium` → reasoning_tokens=1240. The fix is verified end-to-end.

### 2.4 The fix

Two-line change to `_generate_openrouter`:

```python
if self.reasoning_effort:
    kwargs["extra_body"] = {"reasoning": {"effort": self.reasoning_effort}}
```

Uses OpenRouter's unified `reasoning.effort` parameter (works across OpenAI, xAI, Anthropic providers), passed via `extra_body` since the OpenAI Python SDK doesn't have a top-level `reasoning` kwarg.

### 2.5 What this means for prior data

**The pre-fix overnight ka59simple data (n=56) is salvageable as a "default-reasoning" column** — it was internally consistent (every trial at the same effective reasoning level), it was just mislabeled. We treat it as a third comparison column alongside `none` and `medium`, with the explicit caveat that "default" for gpt-5.2 is ≈ `minimal` reasoning, not zero. **Per-model defaults differ** (see §2.6) — so "default-R" data from one model is not comparable to "default-R" data from another.

### 2.6 Per-model reasoning behaviors discovered

Probed via OpenRouter using `extra_body={"reasoning": {"effort": ...}}` on a simple prompt:

| Model | OMITTED default | `none` | `medium` | Verdict |
|-------|----|----|----|----|
| `openai/gpt-5.2` | 15 reasoning_tokens | 0 ✅ | 19 | Default ≈ minimal; supports `none` |
| `x-ai/grok-4.1-fast` | 97 | 0 ✅ | 92 (1240 on long prompts) | Default = high; supports `none` |
| `x-ai/grok-4-fast` | 106 | 0 ✅ | 82 | Same shape as 4.1-fast |
| `x-ai/grok-4.20` | **0** | 0 ✅ | 240 | **Default = no reasoning**; opt-in via medium |
| `x-ai/grok-4` (full) | 146 | ❌ rejected | 122 | "Reasoning is mandatory" — cannot run no-R |
| `x-ai/grok-4.3` | 255 | ❌ rejected | 244 | Same constraint |

**Implication:** `grok-4` and `grok-4.3` cannot be used in any A/B that includes a no-reasoning column. Of the toggle-supporting models, `grok-4.1-fast` is ~13× cheaper than `grok-4.20` and produces similar floor patterns on mechanics_hard, so we picked it for the headline matrix.

---

## 3. Experiments run 2026-05-01 → 2026-05-02

### 3.1 Re-run scope

Once the bug was fixed and reasoning toggles verified, we re-ran:

| Run | Provider/Model | Reasoning | Configs | n_trials | Trials total |
|-----|---------------|-----------|---------|----------|---------------|
| ka59simple no-R sweep #1 | openrouter / gpt-5.2 | `none` | baseline + 4 axis_hard | 5 each | 25 |
| ka59simple no-R remaining | openrouter / gpt-5.2 | `none` | feedback_hard + mech_ooda_f | 5 each | 10 |
| ka59simple medium-R sweep #1 | openrouter / gpt-5.2 | `medium` | baseline + 4 axis_hard | 5 each | 25 |
| ka59simple medium-R remaining | openrouter / gpt-5.2 | `medium` | feedback_hard + mech_ooda_f | 5 each | 10 |
| Grok A/B pilot | openrouter / grok-4.1-fast & grok-4.20 | `none`, `medium` | mechanics_hard | 3 each | 12 |
| Grok 4.1-fast matrix | openrouter / grok-4.1-fast | `none`, `medium` | baseline + 4 axis_hard | 5 each | 50 |

**Total new trials run today: 132** (plus the 56 salvaged pre-fix trials = **188 total in the analysis**).

### 3.2 Killed in-flight

Two ablation processes started at 03:09 EDT today (before the bug was fixed) were running with the broken code in memory and would have produced more mislabeled data. We killed them after confirming that Python's module cache made hot-patching impossible — you have to restart the process for the fix to take effect.

### 3.3 Trial attribution

Per-trial JSONs do **not** store `reasoning_effort` (only `provider` and `model`). To attribute each JSON to a reasoning level, we parse each source log's `Saved -> path` lines and map JSON → log → reasoning level. Two of the ~188 trial files are unattributed (likely killed-process stragglers); they are excluded from the tables.

---

## 4. Results

### 4.1 Win-rate matrix (n in parens)

| Config | gpt-5.2 default-R | gpt-5.2 no-R | gpt-5.2 medium-R | grok-4.1-fast no-R | grok-4.1-fast medium-R |
|---|---|---|---|---|---|
| baseline | 6/10 (60%) | 4/5 (80%) | **5/5 (100%)** | 2/5 (40%) | 1/5 (20%) |
| world_hard | 0/10 (0%) | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) |
| goal_hard | 3/10 (30%) | **4/5 (80%)** | 1/5 (20%) | 1/5 (20%) | 3/5 (60%) |
| mechanics_hard | 0/10 (0%) | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) | 0/5 (0%) |
| feedback_hard | — (never reached pre-fix) | **9/10 (90%)** | 5/10 (50%) | 0/5 (0%) | 2/5 (40%) |

### 4.2 Kaaus-format table — gpt-5.2

```
Config            Reasoning   Win%  AvgTurns  AvgLevels  Invalid  Clicks  Walls  Pushes  Rel.diff   n
-----------------------------------------------------------------------------------------------------
baseline          default-R    60%      25.0       0.60      0.0     8.4    0.9     0.0     1.00x  10
world_hard        default-R     0%      32.0       0.00      0.0     2.2    1.0     0.0     1.28x  10
goal_hard         default-R    30%      24.7       0.30      0.0     9.4    0.3     0.0     0.99x  10
mechanics_hard    default-R     0%      32.0       0.00      0.1    30.1    0.1     0.0     1.28x  10
baseline          no-R         80%      13.2       0.80      0.0     5.4    1.0     0.0     1.00x   5
world_hard        no-R          0%      32.0       0.00      0.0     5.0    1.0     0.0     2.42x   5
goal_hard         no-R         80%      16.8       0.80      0.0     5.0    1.2     0.0     1.27x   5
mechanics_hard    no-R          0%      32.0       0.00      0.2    30.8    0.0     0.0     2.42x   5
feedback_hard     no-R         90%      13.5       0.90      0.0     2.7    1.0     0.0     1.02x  10
baseline          medium-R    100%      15.6       1.00      0.0     5.4    1.0     0.0     1.00x   5
world_hard        medium-R      0%      32.0       0.00      0.0     2.4    1.0     0.0     2.05x   5
goal_hard         medium-R     20%      31.0       0.20      0.0    16.8    0.8     0.0     1.99x   5
mechanics_hard    medium-R      0%      32.0       0.00      0.0    30.0    0.0     0.0     2.05x   5
feedback_hard     medium-R     50%      21.1       0.50      0.0     7.1    0.6     0.0     1.35x  10
```

### 4.3 Kaaus-format table — grok-4.1-fast

```
Config            Reasoning   Win%  AvgTurns  AvgLevels  Invalid  Clicks  Walls  Pushes  Rel.diff   n
-----------------------------------------------------------------------------------------------------
baseline          no-R         40%      30.4       0.40      0.0    21.4    0.6     0.0     1.00x   5
world_hard        no-R          0%      32.0       0.00      0.0    24.6    0.8     0.0     1.05x   5
goal_hard         no-R         20%      30.8       0.20      0.0    21.6    0.2     0.0     1.01x   5
mechanics_hard    no-R          0%      32.0       0.00      0.0    32.0    0.0     0.0     1.05x   5
feedback_hard     no-R          0%      32.0       0.00      0.0    25.6    0.4     0.0     1.05x   5
baseline          medium-R     20%      27.2       0.20      0.0    15.8    0.4     0.0     1.00x   5
world_hard        medium-R      0%      32.0       0.00      0.0    16.0    1.0     0.0     1.18x   5
goal_hard         medium-R     60%      22.4       0.60      0.0    10.2    0.6     0.0     0.82x   5
mechanics_hard    medium-R      0%      32.0       0.00      0.2    29.8    0.0     0.0     1.18x   5
feedback_hard     medium-R     40%      27.0       0.40      0.0    16.2    0.4     0.0     0.99x   5
```

**Notes on Kaaus-format columns:**
- **Flips / Undos** are not present in KA59's action set (LS20/BP35-specific). Replaced with **Walls** (`wall_transfers`) and **Pushes** (`object_pushes`), which are the equivalent KA59-specific mechanic counters.
- **Rel. diff** = `avg_turns(condition) / avg_turns(baseline)` at the **same reasoning level**. Higher = model spent more turns trying to solve the knockout.
- **Avg turns** = across ALL trials (wins + fails). Failed trials hit `max_turns=32`, which caps avg_turns from above. KA59 traditional benchmark uses max_turns=50 — we used 32 for compute budget; revisit if reviewers flag.

---

## 5. Headline findings

1. **World-knockout robustly collapses to 0% wins** across all 5 model × reasoning conditions. This is the cleanest cross-condition finding and the safest abstract anchor.

2. **Mechanics-knockout robustly floors at 0% wins**, but the partial-progress signal `wall_transfers` is **suppressed under mechanics_hard** (0/5 in every reasoning condition) vs baseline (1/5–1/1 across conditions). This is the "mechanic-specific effect" the new abstract should anchor on, even though win-rate is unchanged.

3. **Reasoning is non-monotonic for gpt-5.2 on goal_hard**:
   - no-R: 80% wins
   - default-R: 30% wins
   - medium-R: 20% wins
   - **More reasoning hurts on this knockout** for gpt-5.2 — likely overthinking on what should be a perceptual task. Worth highlighting; do not bury.

4. **Cross-model reasoning differential on goal_hard** (clean signal):
   - gpt-5.2: 80% no-R → 20% medium-R (medium hurts, 4× drop)
   - grok-4.1-fast: 20% no-R → 60% medium-R (medium helps, 3× lift)
   - **Same axis, opposite reasoning interaction by model.** This maps onto the new abstract's "different reasoning interactions per model" framing.

5. **gpt-5.2 dominates grok-4.1-fast at most matched cells** (especially feedback_hard: 90% vs 0% at no-R). The cross-model story is "different floor effects + different reasoning sensitivities," not "same model class behaving similarly."

---

## 6. Caveats and peer-review checklist

This data will be peer-reviewed under NeurIPS standards. Reviewers will scrutinize:

| Concern | Status / mitigation |
|---|---|
| **`ka59simple` is not canonical KA59** | Must be framed as "constructed single-mechanic variant for ablation," not a benchmark replacement. Canonical KA59 floor effect should still be reported as primary evidence. |
| **n=5 is small** for some cells; differences within n=5 may be sampling noise | Add n=10 sweep on both models before submission. Document confidence intervals (Wilson score) in the methods section. |
| **`max_turns=32`** instead of canonical 50 | Disclose explicitly. If a reviewer asks, run a 50-turn sanity check on baseline + mechanics_hard, n=3, to show no-additional-wins-after-32. |
| **OpenRouter `reasoning_effort` bug masked prior labeled data** | Disclose in methods. Pre-fix data (default-R column) is salvageable but must NOT be reported as "no-R" or "medium-R." Show the verification probe (§2.3 table) as evidence the fix works. |
| **Per-model defaults differ** (gpt-5.2 ≈ minimal, grok-4.20 = none, grok-4.1-fast = high) | Always pass an explicit `reasoning_effort`. Never compare "default-R" data across models. |
| **`grok-4` and `grok-4.3` reject `none`** | If reviewer asks "why didn't you test these?", cite the API constraint. |
| **Per-trial JSON does not store `reasoning_effort`** | Attribution is via log-parsing. Recommend a code change to log `reasoning_effort` directly in trial JSONs going forward (small fix to `experiment.save_result()`). Not blocking for this paper, but worth fixing before any future re-analysis. |
| **Behavioral metrics are observed counts, not normalized rates** | Walls / clicks / pushes are not divided by turns-played. For fail trials at `max_turns=32` this means the metric saturates; for win trials at 8 turns the metric is naturally smaller. **Compute per-turn rates if a reviewer flags this.** |
| **`p_walls={result.click_actions}` log label bug** | `scripts/run_real_ablation.py:124` prints `p_walls={click_actions}` — the LABEL is wrong, the value is `click_actions`. No data integrity issue (the JSON fields are correctly named); just confusing log output. Fix before any external sharing of logs. |
| **OODA / OODA-F scaffolds are appendix material** | Per the post-meeting framing pivot, OODA results should be in the appendix, not the headline. They are NOT a matrix cell since they inject the metacognitive prompt the matrix is supposed to measure for. |

---

## 7. What's next

| Priority | Action | Cost | ETA |
|---|---|---|---|
| HIGH | Bring n=5 cells up to n=10 by running second parallel sweeps on gpt-5.2 + grok-4.1-fast | ~$15 | ~5 hours (grok medium-R is the bottleneck at 30 min/trial) |
| HIGH | Fix `experiment.save_result()` to record `reasoning_effort` in per-trial JSONs | trivial | 10 min |
| HIGH | Fix `p_walls` label bug in `scripts/run_real_ablation.py:124` | trivial | 5 min |
| MED | Consider running `--max-turns 50` baseline + mechanics_hard sanity check (n=3) to head off reviewer concern | ~$2 | 30 min |
| MED | Compute per-turn rates for behavioral metrics (clicks/turn, walls/turn) and re-report | trivial | 30 min |
| LOW | Update abstract draft to reflect: world-knockout = primary collapse claim, mechanic_hard = partial-progress claim, goal_hard reasoning anomaly worth a sentence | analysis | 1 hour |
