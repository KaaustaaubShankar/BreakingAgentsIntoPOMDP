# NeurIPS paper start for JKJ / BreakingAgentsIntoPOMDP

## What the project actually is right now

This project is no longer best described as just "a few game benchmarks" or just the older Zendo implementation.

The real project direction is:

- evaluate whether agents can **discover hidden rules / latent mechanics through interaction**
- do that under **partial specification / partial observability**
- factor difficulty along the four benchmark axes from the proposal and Monday meeting:
  - **World**
  - **Goal**
  - **Mechanics**
  - **Feedback**
- measure not only final outcome, but also the agent's **epistemic trajectory** while it is discovering the environment

That is the actual scientific core.

## Important project history to preserve accurately

There are really **two layers of history** here:

### 1. Earlier repo state
The repo already had earlier Zendo work, including a difficulty-ladder style experiment setup.
That work is useful, but it is **not identical** to the newer framing.

### 2. Current project trajectory
The newer framing, reflected in the Monday matrix notes and JKJ archaeology, is moving toward:
- a **knockout-style benchmark framing** rather than just a difficulty ladder
- multiple environments rather than one game only
- a benchmark question about **hidden-mechanic discovery**, not just success rate
- a paper contribution that includes **belief tracking / epistemic traces**, not only final win/loss

So the paper should present the project as an evolution:
- from single-environment / difficulty-style experiments
- toward a broader benchmark for discovery under partial observability

That is much more accurate than pretending the project started in its final form.

## Where each environment fits today

### Zendo
- important earlier implementation work
- already had axis-style experimentation and ablation ideas
- useful for historical continuity and prior benchmark framing
- but should not be presented as the only or final environment

### BP35 / `env4`
- currently the more game-facing, interaction-heavy environment
- click-based, gravity-based, forward-planning-heavy
- good for showing that discovery is not just symbolic rule induction, but also interactive control with hidden consequences
- currently implementation-heavy and still being debugged

### LS20
- another environment in the broader benchmark set
- helps prevent the story from collapsing into one bespoke game
- important for cross-environment evidence

### KA59
- the cleanest current mechanics-discovery slice
- smallest controlled case
- already has clear baselines, scenarios, benchmark output, and an epistemic-style result
- best current environment for a readable motivating example / case study

## Current strongest paper claim

### Conservative version
Current agent evaluations often fail to distinguish **task completion** from **mechanic discovery**. We propose a benchmark framing for environments with hidden rules under partial specification, and show that epistemic traces can reveal failures that final outcome metrics miss.

### Stronger version
We introduce a cross-environment benchmark for **hidden-mechanic discovery under partial observability**, factorized along World / Goal / Mechanics / Feedback, and show that agents with similar outcome scores can differ sharply in whether they actually discover the latent transition structure of the task.

The stronger version is the one you want if the experiments support it.

## What the current codebase already supports

### Already real
- environment-specific benchmark work in Zendo, BP35, LS20, and KA59
- benchmark-axis language from the proposal / Monday framing
- observability / belief-tracking trajectory in the project history
- KA59 reportable baselines and a mechanics-discovery result

### Not fully real yet
- one fully unified benchmark harness across all environments
- one fully standardized epistemic metric shared across all environments
- one polished final experiment table across LS20 / BP35 / KA59

So the paper should be ambitious in framing, but honest in what is already implemented versus still being unified.

## Current best project-specific framing for the paper

> Existing agent benchmarks over-index on final score and under-measure whether an agent actually discovered the hidden mechanics of an environment. We study this problem across interactive tasks with partial specifications, factorizing difficulty along World, Goal, Mechanics, and Feedback, and using epistemic traces to measure how agent beliefs change during interaction.

That is much closer to the real project than generic benchmark language.

## What KA59 already gives the paper concretely

KA59 is the strongest current small example because it already shows a clean result.

Current result shape:
- `NaiveRightAgent`, `RotateOnBlockAgent`, `MinimalHypothesisAgent` all run on the same canonical scenarios
- in `tw_push`, only `MinimalHypothesisAgent` discovers the hidden wall-transfer asymmetry
- in `tw_push_world_blind`, that discovery disappears under reduced observability

Representative numbers:

### `tw_push`
- `NaiveRight`: `moved=1 blocked=9 p_walls=0`
- `RotateOnBlock`: `moved=9 blocked=1 p_walls=0`
- `MinimalHypothesis`: `moved=7 blocked=3 p_walls=1`

### `tw_push_world_blind`
- `NaiveRight`: `moved=1 blocked=9 p_walls=0`
- `RotateOnBlock`: `moved=9 blocked=1 p_walls=0`
- `MinimalHypothesis`: `moved=9 blocked=1 p_walls=0`

This gives a nice paper-facing statement:
- the discovery-capable agent finds the hidden mechanic in the fully observed mechanics probe
- that same discovery vanishes when observability is degraded

That is not the whole paper, but it is already a credible figure / motivating result.

## What the paper should say about benchmark axes

The paper should not claim that every environment fully instantiates every axis equally.

Instead, say something like:

- the benchmark family is organized around **World / Goal / Mechanics / Feedback**
- different environments and conditions instantiate different slices of that space
- some environments are currently stronger for certain axes than others
- the benchmark is being built as a **family**, not a single monolithic task

That is both true and stronger intellectually.

## Actual template guidance for this project

The NeurIPS template itself is fine.

The real work is filling it with the right story:
- why this project exists
- how it evolved from earlier Zendo-only work
- why the knockout / factorized framing matters
- how cross-environment evidence supports the claim
- why epistemic traces are necessary beyond outcome metrics

## Draft abstract extract (v0)

```text
Current agent evaluations often emphasize final success or score while under-measuring whether an agent actually discovered the latent mechanics of the task. We study hidden-mechanic discovery in interactive environments with partial specification, framing evaluation along four separable axes: World, Goal, Mechanics, and Feedback. We build a benchmark family spanning multiple environments, including controlled mechanics probes and more interaction-heavy game settings, and augment outcome metrics with epistemic traces that summarize what the agent appears to have inferred during interaction. In a clean mechanics-discovery case, we show that a discovery-capable baseline can identify a hidden transition asymmetry that simpler reactive baselines miss, and that this discovery signal disappears when observability is degraded. These results illustrate how similar task outcomes can mask large differences in latent-rule discovery, motivating benchmark designs that evaluate belief formation in addition to task completion.
```

## Draft introduction opening extract (v0)

```text
Many recent agent evaluations focus on whether a model eventually solves a task, reaches a goal, or achieves a high final score. However, in interactive environments with hidden rules, final outcome alone is often an incomplete measure of competence. An agent may succeed through brittle heuristics without discovering the true mechanics of the environment, or fail despite forming a partially correct internal model of the task. If the goal is to understand interactive reasoning under partial observability, then we need evaluations that distinguish task completion from latent-mechanic discovery.

This paper studies that gap. We frame interactive agent evaluation as a family of partially specified environments in which different aspects of the task can be weakened or withheld: the observable world state, the objective, the transition mechanics, or the feedback signal. This yields a benchmark perspective organized around four capability axes, World, Goal, Mechanics, and Feedback. Across environments in this family, we ask not only whether an agent succeeds, but also what it appears to have learned while interacting.

To make this measurable, we pair standard outcome metrics with epistemic traces: compact summaries of how an agent's working hypotheses or inferred structure evolve over time. This lets us detect cases where agents with similar outcomes differ substantially in what they have actually discovered about the environment.
```

## Draft contribution bullets (v0)

```text
Our contributions are:
(1) a benchmark framing for hidden-mechanic discovery under partial observability, organized along World, Goal, Mechanics, and Feedback;
(2) a multi-environment benchmark family spanning environments with different interaction and discovery profiles;
(3) epistemic-trace measurements that complement standard outcome metrics by tracking discovery behavior during interaction; and
(4) empirical evidence that outcome metrics alone can hide major differences in latent-rule discovery.
```

## Draft method / problem setup extract (v0)

```text
We consider interactive environments in which the agent observes only a partial specification of the task. Some aspects of the environment may be directly observable, while others, including parts of the transition dynamics, objective specification, or evaluative feedback, may be withheld or weakened. We organize these sources of uncertainty into four benchmark axes: World, Goal, Mechanics, and Feedback. A benchmark condition is defined by an environment together with a choice of which axes are fully specified, weakened, or partially hidden.

In this setting, evaluation should capture more than terminal success. We therefore separate outcome metrics from discovery metrics. Outcome metrics measure task completion, progress, or score. Discovery metrics instead ask whether the agent behaviorally demonstrates knowledge of a hidden rule or latent transition pattern. In addition, we collect epistemic traces, lightweight summaries of the agent's evolving hypotheses or discovered structure over the course of an episode.
```

## Draft experiments setup extract (v0)

```text
We evaluate agents across multiple environments that instantiate different discovery demands. These include a controlled mechanics probe, KA59, which isolates hidden transition asymmetries in a minimal setting; BP35, a more interaction-heavy environment involving clicks, gravity, and forward planning; and LS20, which broadens the benchmark beyond a single bespoke game. For each environment, we define conditions that map onto the benchmark axes where possible, and compare baseline agents with differing levels of adaptivity and hypothesis tracking.

Our evaluation reports both standard outcome metrics and discovery-oriented measurements. In KA59, for example, we measure whether an agent discovers the wall-transfer asymmetry and how this changes under degraded observability. More generally, we test whether similar task outcomes can conceal different levels of latent-rule discovery across environments and conditions.
```

## Draft limitations paragraph (v0)

```text
This work is currently limited by the number and maturity of benchmark environments, as well as by the strength and diversity of the evaluated agents. Not all environments instantiate all four benchmark axes equally cleanly, and some axis mappings are currently stronger proxies than full binary removals. In addition, our current epistemic-trace measurements are partly hand-designed and environment-dependent, rather than fully standardized across the entire benchmark family. We therefore view the present results as evidence for the usefulness of discovery-oriented evaluation, rather than as a final complete benchmark.
```

## What the experiments section should concretely show

At minimum, the paper needs:

1. **One clean motivating result**
   - KA59 is currently best for this
2. **Cross-environment evidence**
   - LS20 + BP35 + KA59 should support the same higher-level claim
3. **A table where outcome and discovery diverge**
   - this is the paper's key empirical moment
4. **At least one epistemic-trace case study**
   - ideally one clean figure

## A more accurate table shape than the generic template

| Environment | Condition | Agent | Outcome | Discovery | Epistemic signal |
|---|---|---:|---:|---:|---:|
| KA59 | Mechanics probe | NaiveRight | ... | ... | ... |
| KA59 | Mechanics probe | MinimalHypothesis | ... | ... | ... |
| KA59 | World-degraded | MinimalHypothesis | ... | ... | ... |
| BP35 | ... | ... | ... | ... | ... |
| LS20 | ... | ... | ... | ... | ... |

## What still needs to be true for a serious NeurIPS main-track push

- stronger agents than only toy handcrafted baselines
- consistent cross-environment evidence
- a crisp statement of what prior evaluation misses
- a clear discovery metric and epistemic metric story
- disciplined limitations section

So the strongest honest position is:
- **direction:** strong
- **current artifacts:** promising but not yet enough by themselves
- **next step:** convert the current environment work into one coherent cross-environment experiment story

## Practical recommendation

Use the downloaded NeurIPS template unchanged, but fill it from this project-specific framing rather than generic benchmark language.

If you want to start writing immediately, the best order is:
1. abstract
2. introduction
3. contributions
4. experimental setup table
5. limitations

That will force the actual claim to become precise before the rest of the draft fills in.
