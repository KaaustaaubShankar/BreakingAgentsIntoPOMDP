# JKJ Abstract — Working Draft
_Last updated: 2026-04-30 | Status: working draft, NOT final_

---

## Current draft

```latex
\begin{abstract}
Can AI agents infer hidden interaction rules they were never told?
We introduce a four-axis knockout benchmark (World, Goal, Mechanics,
Feedback) for evaluating hidden-mechanic discovery across three partially
observable environments of varying difficulty (KA59, BP35, LS20).
By systematically degrading each observability axis, we reveal
environment-specific bottleneck axes: world degradation collapses
performance on LS20 (5.26$\times$ behavioral amplification) and BP35
(2.90$\times$), while mechanics degradation is the binding constraint
on KA59---the hardest environment, where no model achieves wins even
at baseline. Critically, model capability modulates \emph{which} axes
induce failure: on BP35, GPT-5.2 remains robust under mechanics
degradation (100\% wins) while GPT-4.1 collapses (0\%, 2.57$\times$
amplification)---same axis, divergent outcomes. An OODA-F metacognitive
scaffold (7.4 forced reframes/trial) fails to translate verbal mechanic
discovery into behavioral policy change. Together, these results
demonstrate that hidden-mechanic discovery is a capability axis on which
models fail distinctly by environment and by model capacity, and that
behavioral benchmarks---not prompt interventions---are the appropriate
diagnostic tool.
\end{abstract}
```

---

## Pending changes before finalizing

### Model lineup (MUST FIX before submission)
Current draft references gpt-4.1 — **this is being dropped**.

Final model set TBD, likely:
- **GPT-5.2 no-reasoning** (explicit `reasoning_effort=none` or equivalent — verify this isn't just default=medium)
- **GPT-5.2 medium-reasoning** (explicit `reasoning_effort=medium`)
- **Grok 4.1 or 4.2** — 4.2 more expensive, evaluate if budget allows
- **Anthropic models** (Haiku/Opus/Sonnet) — expensive, use sparingly; may drop if budget is the constraint

Action: re-run LS20 and BP35 with the final model set, then update all numbers.

### Numbers that will change
- LS20 behavioral amplification values (currently from gpt-4.1 vs gpt-5.2 — will change with Grok comparison)
- BP35 behavioral amplification values (same reason)
- The BP35 divergence finding (gpt-4.1 vs gpt-5.2 on mechanics_hard) — will need equivalent framing once gpt-4.1 is replaced with Grok
- `[N] trials` placeholder — fill in once reruns complete

### Behavioral amplification — definition note
"Behavioral amplification" = `Rel.diff` from the comparison table = (metric_in_condition / metric_at_baseline).
- Metric = clicks (KA59) or wall hits (LS20/BP35) depending on environment
- Make sure methodology section defines this clearly per-environment.

### KA59 wall transfer finding
Edward noted: mech_hard on KA59 showed **increase in wall transfer mechanic attempts** — this is the +73% (no-R) / +347% (med-R) click amplification and may include actual wall transfer events.
- Verify: were any wall_transfer events logged under mech_hard? Check run JSONs.
- If yes, this is a notable secondary finding: mech degradation causes models to probe walls more, but still not exploit the mechanic.

### KA59 canonical baseline floor effect (2026-05-01)
Empirical confirmation: gpt-5.2 on canonical KA59 baseline scores 0% across reasoning levels:
- no-reasoning:    0/13 wins, 0 wall_transfers across all trials
- medium-reasoning: 0/1 valid wins, 0 wall_transfers (n=1 due to OpenAI quota)
- high-reasoning:   0/3 wins, 0 wall_transfers, max_goals_occupied=1 (greedy left-fill)

Reasoning effort does not rescue mechanic discovery on canonical KA59 — failure mode is consistent across reasoning levels.

### KA59 simplified variant (`ka59simple`) — methodological tool
Built 2026-05-01 to escape the canonical floor effect. Single-goal fork of canonical level 1 (keeps both selectables + wall + right goal, drops left goal). Solvable in 6 actions; same 4-axis ablation framework applies. Pilot (n=2, gpt-5.2 no-R, baseline only): 2/2 wins, 1-2 wall_transfers per trial — escapes the floor. Full matrix run in flight (no-R + med-R, all 7 configs) for benchmark column data.

Methods-section framing: "single-mechanic variant constructed to escape baseline floor effects, enabling 4-axis ablation comparison." NOT a co-headline; supports the existing world-knockout / mechanics-degradation framing.

### Consistent findings (hold across environments — Edward's observation)
- `world_hard` → **always collapses to 0% wins** across all three environments and models
- `mechanics_hard` → **always degrades behavioral metric** (direction varies by env)
These are the two robust claims. Abstract should anchor on these.

---

## Open experiment decisions
| Question | Decision needed by |
|---|---|
| Grok 4.1 vs 4.2? | ASAP — affects budget planning |
| Run Anthropic full matrix or pilot only? | ASAP |
| Verify no-R = truly no reasoning (not default=medium)? | Before finalizing model labels |
| Re-run LS20 + BP35 with final model set? | Before May 3 (need time to update abstract) |
