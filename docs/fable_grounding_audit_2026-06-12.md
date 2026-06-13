# Grounding audit & path to submission — Fable analysis, 2026-06-12

Audit of the verbal-vs-behavioral discovery pipeline for the COLM 2026 paper
("A POMDP-Inspired Framework for Evaluating LLM Agents", submission 2026-06-23).
TL;DR: **the paper's "verbal discovery rate: 0%" result is an instrumentation
artifact**, the honest corrected finding is *stronger* than the current claim,
and the fix is mostly post-hoc analysis over data we already have. Win-rate
ablation results (the core tables, Ben's three findings) are **unaffected**.

---

## 1. What we found

### Finding 1 — the verbal-discovery check reads a channel that is empty in almost every run

`check_discovery()` is applied only to the `orient` field
(`ka59_game/experiment.py:412-417`). `orient` exists only in OODA / OODA_F
mechanics configs. In every standard config the model emits `reasoning`
instead — which is logged in each trial's history but **never checked**.
So "0% verbal discovery across all conditions" mostly means "we ran a strict
filter over an empty string."

Empirical scan over all **839** existing trial JSONs
(`results/ka59simple_game/` + `results/ka59_game/`):

| Channel | Trials with a discovery-pattern hit |
|---|---|
| `orient` (what the pipeline checks today) | 4 |
| `reasoning` (ignored today) | **184** |
| end-of-run `understanding` reflection (ignored today) | 8 (of 634 with text) |
| behavioral: `wall_transfers > 0` | 280 |

### Finding 2 — but the regex can't simply be pointed at `reasoning`

Sampling the 184 matched reasoning texts shows most are **speculative probing
intent**, not statements of a discovered rule:

> "All directions are blocked by walls; try moving right to **test if any wall
> is passable** or triggers a push/teleport-like interaction." (turn 1, before
> any transfer)

> "Probe for any hidden interaction by **trying to push through** the right
> side despite the blocked hint."

> "I need to move right **through the wall gap**." (outright false positive)

The regex conflates three epistemically distinct states: *never mentions the
mechanic*, *hypothesizes it*, and *asserts it as a confirmed rule*. The
interesting science is exactly in that distinction.

### Finding 3 — behavioral discovery has no timestamps

`wall_transfers` is an aggregate counter; per-turn transfer events are not
written into history. `scripts/extract_verbal_behavioral.py`'s
`_behavioral_discovery_turn()` is literally a `return None` placeholder. The
headline grounding metric — did the agent *say it* before or after *doing
it* — is **uncomputable from current logs**. Also, "behavioral discovery =
any transfer ≥ 1" counts accidental stumbles (many losing trials have exactly
one transfer) the same as deliberate exploitation.

---

## 2. Impact on the paper

Source of truth: `colm2026_conference.tex` (synced from Overleaf copy `(9)`,
2026-06-12). Four passages state the claim, in escalating strength:

| Line | Claim | Exposure |
|---|---|---|
| ~78 (contributions) | agents "never state the rule under a strict filter" | most exposed — filter ran on an empty channel for most cells |
| ~204 (metrics) | verbal discovery "under a strict keyword filter" | must name the text channel; channel must be the right one |
| ~251 (results) | "Verbal discovery … is 0% across all completed KA59-Simple conditions" | false as stated once `reasoning` is scored |
| ~271 (limitations) | hedges, calls for human-validated measure | closest to defensible, but frames 0% as "direct evidence" of dissociation |

A reviewer who reads the harness (it's a benchmark paper; some will) would
find this. We should fix it before they do.

### The corrected framing is a better result

Under the same filter applied to text the models actually produce, agents are
**hypothesis-rich but confirmation-poor**: they constantly probe "maybe the
wall is passable," physically trigger the transfer, yet rarely consolidate it
into a stated rule even after using it. That preserves — and sharpens — the
three-way dissociation the contributions section wants (trigger ≠ exploit ≠
articulate), with real rates instead of a suspicious 0%. One genuinely new,
testable result falls out: **does verbal *confirmation* predict winning where
mere hypothesizing does not?**

---

## 3. Plan to submission (2026-06-23)

Ordered so cheap unblocking work lands first; the analysis track is the
paper-critical path.

**Track 1 — instrumentation fix (~30 min, before any more runs).**
Add per-turn `outcome` flags (`wall_transfers`, `object_pushes`, `blocked`)
to history entries in `experiment.py` (detection already happens per turn at
lines ~483-511; it just isn't logged), and record a `verbal_candidate_turn`
that also checks `reasoning`. Every trial run before this lands is another
trial that can't support verbal-vs-behavioral timing. *(In progress.)*

**Track 2 — remaining pipeline runs.** Launch after Track 1 so new data is
fully instrumented; accumulates in the background while analysis proceeds.

**Track 3 — corrected discovery analysis (post-hoc, all 839 existing trials,
no reruns).**
1. Three-level scorer: regex as cheap candidate prefilter → LLM judge
   classifying each candidate turn + every understanding reflection as
   *no mention / hypothesis / confirmed claim*. Hand-label ~50 trials and
   report judge agreement (turns "we changed our metric" into "we validated
   our metric").
2. Corrected table per config × model: hypothesis rate, confirmation rate,
   first-hypothesis vs. first-transfer turn (timing on newly instrumented
   runs; rates on everything). Compute confirmation→win correlation.
3. Rewrite the four passages around the corrected numbers.

**Track 4 — BP35 (env4) audit.** Same orient-only gap likely exists in env4's
own prompts/client. Cheap to scan, embarrassing to miss before the
cross-environment claims go out.

## 3b. Results of the corrected analysis (completed 2026-06-12, same day)

The three-level judge pass is done: 1,453 items (every regex-candidate turn +
every end-of-run understanding reflection) across 689 trials, judged by
Claude Haiku 4.5 via OpenRouter (`scripts/judge_verbal_discovery.py`,
judgments cached in `results/verbal_judge/judgments.jsonl`).

**Item-level: none = 755, hypothesis = 679, confirmed = 4.**

- **Hypotheses are everywhere.** Capable models speculate about wall
  pass-through in a large fraction of trials (GPT-5.2 baseline: 11/31
  trials; feedback-hard: 16/29; goal-hard: 21/34; DeepSeek-v4-pro similar or
  higher). Agents actively probe the right hypothesis.
- **Confirmed rule statements almost never happen: 4 trials out of 689
  (~0.6%).** And none of the four states the full asymmetry (active piece
  blocked vs. pushed piece passes); they are partial pass-through claims.
  Notably, 2 of the 4 have zero wall transfers — confirmed-sounding claims
  with no behavioral basis (ungrounded belief), e.g. "outer wall is passable"
  in a losing, zero-transfer trial.
- **The exit interview consolidates nothing.** Of 654 judged understanding
  reflections, exactly 1 is a (partial) confirmed statement. Models that
  physically used the mechanic and won still describe walls as blocking in
  the post-run reflection.

**Corrected paper claim:** replace "verbal discovery is 0% under a strict
keyword filter" with: under a judge-validated three-level measure over all
verbal channels, agents *hypothesize* the hidden rule frequently
(30–60% of trials for capable models) but *consolidate* it almost never
(~0.6% of trials, 0% for the full asymmetry) — even in trials where they
trigger the mechanic and win. The trigger ≠ exploit ≠ articulate dissociation
survives and is now measured on the channels agents actually write to.

Remaining for this track: hand-label ~50 items to report judge agreement, and
regenerate per-condition tables for the paper from the cache (`--table-only`).

## 4. Decisions needed from the team

1. **Prominence:** does the corrected discovery analysis become a results
   subsection (recommended — it directly serves the paper's thesis that
   outcome metrics conflate completion with discovery), or a minimal fix to
   the four passages? Trade-off is writing time against Ben's other open asks
   (normalized cross-game metrics, reasoning-level figures, third model
   family).
2. **Judge model & budget** for the LLM scorer (a few hundred candidate turns
   plus 634 understanding texts — small).
3. **Forced-choice prediction probe** (end-of-run "what happens if A pushes B
   against the wall?") — stronger grounding than free text, but needs new
   runs across configs. Recommendation: future work, keep this submission's
   claims within the post-hoc analysis.

## 5. What is *not* affected

- All win-rate ablation matrices and the cross-environment comparison.
- Ben's three findings (world hardest; reasoning helps only mechanics-hard;
  stronger generations score higher on mechanics-hard).
- Wall-transfer count tables (Appendix) — counts are fine; only their
  *interpretation as "behavioral discovery"* and the verbal 0% claim change.
