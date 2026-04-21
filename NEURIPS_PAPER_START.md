# NeurIPS paper start for JKJ benchmark work

## Short answer

Yes, the template you downloaded is fine.

The bottleneck is **not** the LaTeX structure, it is the **paper claim** and how the current environments fit that claim.

Use the NeurIPS template as-is, and build the paper around this framing.

## Current strongest paper framing

**Working claim:**

Current agent evaluations under-measure whether agents can **discover hidden mechanics through interaction under partial observability**. A factorized benchmark over **World / Goal / Mechanics / Feedback**, plus **epistemic traces**, exposes failures that final score alone hides.

That is stronger than:
- "we made a few game benchmarks"
- "we tested agents on ARC-like environments"

## What the current environments contribute

- **KA59**: smallest clean mechanics-discovery case
- **BP35**: harder planning / interaction case with clicks, gravity, and forward consequences
- **LS20**: another environment in the broader benchmark set

The paper should not pretend all environments do the same thing. They support the same thesis from different angles.

## Suggested title directions

### More benchmark-forward
- **Breaking Agents into POMDPs: Evaluating Hidden-Mechanic Discovery Under Partial Observability**
- **Beyond Final Score: Measuring Hidden-Rule Discovery in Agent Benchmarks**
- **Evaluating Mechanistic Discovery in Interactive Agent Benchmarks**

### More measurement-forward
- **Epistemic Traces for Evaluating Hidden-Mechanic Discovery in Interactive Environments**
- **What Win Rate Misses: Measuring Latent Rule Discovery in Agent Benchmarks**

## Strongest current abstract shape

One-paragraph structure:

1. **Problem**
   Existing agent evaluations focus on task completion or final score, which can hide whether an agent actually inferred the latent mechanics of the environment.

2. **Approach**
   We introduce a benchmark framing that factorizes difficulty along World / Goal / Mechanics / Feedback, and we track epistemic traces of what the agent appears to have discovered through interaction.

3. **Evidence**
   Across environments including KA59, BP35, and LS20, we compare baseline agents and show that identical or near-identical final outcomes can conceal large differences in hidden-rule discovery.

4. **Significance**
   The benchmark reveals a failure mode that standard outcome-only evaluation misses, giving a more faithful picture of interactive reasoning under partial observability.

## Section-by-section mapping onto your NeurIPS template

## 1. Introduction

Goal of intro:
- explain why outcome-only evaluation is insufficient
- define the hidden-mechanic discovery problem
- preview the factorized benchmark idea
- preview epistemic traces

Minimum contributions list should look something like:
- a benchmark framing over **World / Goal / Mechanics / Feedback**
- multiple environments instantiating parts of that framing
- epistemic-trace analysis that complements outcome metrics
- empirical evidence that discovery and success can diverge

## 2. Related Work

Likely buckets:
- POMDPs / partial observability
- agent benchmarks / ARC-like environments / interactive evaluation
- model-based reasoning / latent dynamics / exploration
- interpretability or behavior-trace evaluation if relevant

Important: the paper should distinguish itself from generic benchmark papers by saying it is not only measuring success, but also **mechanic discovery**.

## 3. Method

### 3.1 Problem setup
Define:
- environment family with latent transition rules
- agent interaction protocol
- observable state vs hidden mechanics
- benchmark axes: World / Goal / Mechanics / Feedback
- epistemic trace concept

### 3.2 Approach
Describe:
- how environments are instantiated
- how ablations / probes are constructed
- what metrics are collected
- what the epistemic trace records

Keep this section crisp. Do not bury the reader in implementation details too early.

## 4. Experiments

This section matters most.

### Experimental setup
Need:
- environments used: LS20, BP35, KA59
- agent set: at least simple baselines plus stronger agents later
- evaluation protocol
- number of runs / seeds if applicable
- metrics

### Main results
Need at least some combination of:
- task success / score
- discovery metric
- epistemic-trace metric
- per-axis or per-environment breakdown

### Ablations and analysis
Natural things to include:
- World observability changes
- Mechanics contrastive pairs
- where win/success and discovery diverge
- case studies from KA59 / BP35

## What KA59 already gives you

KA59 already supports a clean small result:
- hidden mechanics are discoverable through interaction
- simple baselines differ behaviorally
- reduced observability suppresses discovery

That makes KA59 a strong **analysis figure / motivating result**, but probably not the whole NeurIPS paper alone.

## What probably still needs to exist for a serious main-track submission

- cross-environment evidence, not just KA59
- stronger agents than only toy hand-built baselines
- one clean primary metric table that supports the main claim
- one or two case studies that make the epistemic-trace story intuitive
- clear limitations section

## Good result table shape

Instead of generic Metric 1 / Metric 2, use something more like:

| Environment | Agent | Success | Discovery | Epistemic signal |
|---|---|---:|---:|---:|
| KA59 | NaiveRight | ... | ... | ... |
| KA59 | RotateOnBlock | ... | ... | ... |
| KA59 | MinimalHypothesis | ... | ... | ... |
| BP35 | ... | ... | ... | ... |
| LS20 | ... | ... | ... | ... |

Where:
- **Success** = completion / score / progress
- **Discovery** = whether hidden mechanic was behaviorally inferred
- **Epistemic signal** = trace-derived summary metric

## Limitation section should be explicit

Be honest about:
- current environment count
- current agent strength
- whether the epistemic metric is hand-designed
- which benchmark axes are only partially instantiated
- whether results are benchmark-specific

That honesty helps more than pretending completeness.

## Practical recommendation

Use the downloaded NeurIPS template unchanged.

Then start writing into it with this structure:
- intro claim
- benchmark framing
- environment mapping
- experiments table
- limitations

Do **not** spend time polishing LaTeX right now.
Spend time clarifying:
- exact claim
- exact metrics
- exact cross-environment evidence needed
