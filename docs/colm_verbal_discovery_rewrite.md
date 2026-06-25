# COLM rewrite — verbal-discovery passages (drop-in LaTeX)

Replacements for the four passages in `colm2026_conference.tex` affected by
the grounding audit (`docs/fable_grounding_audit_2026-06-12.md`). Numbers come
from the judge pass over **all KA59/KA59-Simple trials on disk** (689 trials
with ≥1 judged item; caches in `results/verbal_judge/`).

> **Scoping caveat before pasting:** the paper's win-rate tables use specific
> n=5 cells; the verbal numbers below pool every historical trial (all models,
> efforts, dates). Either (a) present the verbal analysis as a separate
> pooled analysis with its n stated — as drafted below — or (b) recompute on
> exactly the canonical table trials. Don't mix the two silently.
>
> **Keep it KA59-specific.** The BP35 audit (`scripts/audit_bp35_verbal.py`,
> audit doc §3c) confirms verbal discovery does not transfer: BP35 states its
> mechanics in the easy prompt and has no per-turn verbal channel under
> mechanics_hard, so there is no analogous "discovery" to measure. Do not
> phrase the verbal finding as cross-environment. The defensible cross-env
> verbal result is the opposite direction — BP35 agents *re-infer* withheld
> rules (100% gravity, 82% breakable) and win, where KA59 agents do neither —
> which belongs in the inferability-spectrum discussion, not the discovery
> metric definition.

---

## 1. Contributions bullet (line ~78)

**Replace:**
> ...agents trigger the KA59 wall-transfer rule without winning, win without
> articulating it, and never state the rule under a strict filter...

**With:**

```latex
\item Discovery metrics showing that triggering a hidden mechanic, exploiting
it to win, and verbally consolidating it are separable: agents trigger the
KA59 wall-transfer rule without winning, frequently \emph{hypothesize} wall
pass-through in their stated reasoning (30--60\% of trials for capable
models), yet assert it as a confirmed rule in under 1\% of trials---and never
state the full push-vs-move asymmetry (Section~\ref{sec:results}).
```

## 2. Metric definition (line ~204)

**Replace** the "verbal discovery rate ... strict keyword filter" clause
**with:**

```latex
...and verbal discovery, measured at three levels over the agent's stated
per-turn reasoning, OODA orient text, and end-of-run reflection: \emph{none}
(no claim of wall pass-through), \emph{hypothesis} (speculates about or
proposes testing pass-through), and \emph{confirmed} (asserts pass-through as
an observed or known rule). A keyword filter proposes candidate passages and
an LLM judge assigns the level; judge reliability is assessed with a second
judge model and a hand-labeled subset (Appendix~\ref{app:judge}). These
metrics distinguish solving the task, physically triggering the hidden
mechanic, hypothesizing the rule, and consolidating it.
```

## 3. Results paragraph (line ~251)

**Replace** "Verbal discovery, meanwhile, is 0\% across all completed
KA59-Simple conditions under the strict filter---no model reliably states the
push-through-wall rule even when it uses the mechanic behaviorally."
**with:**

```latex
Verbal discovery splits into two very different quantities. Agents
\emph{hypothesize} wall pass-through frequently---GPT-5.2 verbalizes a
pass-through hypothesis in 11 of 31 pooled baseline trials, and in every
condition where both occur, hypothesizing trials win at least as often as
non-hypothesizing ones (e.g.\ baseline 50\% vs.\ 43\%, goal-hard 47\% vs.\
33\%)---but \emph{confirmed} rule statements are nearly absent: 4 of 689
pooled trials (0.6\%) across all models and conditions, none stating the
full push-vs-move asymmetry. Exactly one end-of-run reflection out of 654
asserts even partial pass-through: models exploit the mechanic and win while
still describing walls as blocking in their post-run reflections.
```

## 4. Limitations (line ~271)

**Replace** the final "verbal discovery metric fires at 0\%..." sentence
**with:**

```latex
Finally, verbal discovery is judged by an LLM over candidate passages
proposed by a keyword filter; two judge models (Claude Haiku 4.5,
Grok 4.3) agree at $\kappa = 0.90$ on the three-level labels
(Appendix~\ref{app:judge}), but the measure inherits the usual caveats of
model-graded evaluation, and candidate proposal by keyword filter may miss
paraphrased rule statements outside the end-of-run reflection, which is
judged exhaustively. The near-absence of confirmed rule statements
(0.6--2.1\% of trials depending on judge) is therefore an estimate of rule
\emph{consolidation in stated reasoning}, not proof that no internal rule
representation exists; a
forced-choice prediction probe at the end of each run would test
consolidation more directly and is left to future work.
```

---

## Numbers source

| Quantity | Value | Source |
|---|---|---|
| Trials with ≥1 judged item | 689 | `judge_verbal_discovery.py --table-only` |
| Item labels (none/hyp/conf) | 755 / 679 / 4 | `results/verbal_judge/judgments.jsonl` |
| Confirmed trials | 4 (0.6%) | same; 2 of 4 have zero wall transfers |
| Understanding reflections judged | 654 (1 confirmed) | same |
| Verbal level × win cross-tab | see file | `results/verbal_judge/verbal_level_vs_win.txt` |
| Per-condition table | see file | `results/verbal_judge/summary_table.txt` |
| Judge agreement (κ) | **0.896** three-level / 0.914 collapsed; 94.7% raw (n=1389) | `judgments_grok.jsonl` (Grok 4.3) vs `judgments.jsonl` (Haiku 4.5) |
| Confirmed-trial rate, judge sensitivity | Haiku 0.6% / Grok 2.1% — both <2.1% | per-judge rollup |
| Hypothesis-or-better rate (robust) | Haiku 33% / Grok 31% | per-judge rollup |
| Hand-label sheet (blind, 50 items) | for team | `results/verbal_judge/handlabel_sample_blind.csv` (+`_key.csv`) |
