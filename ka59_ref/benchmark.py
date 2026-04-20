"""
KA59 Micro-Benchmark Evaluator
================================
Runs an agent factory across a set of named level specs and returns a compact,
structured summary.  Thin wrapper over run_episode; no engine internals exposed.

Usage
-----
    from ka59_ref.scenarios import SCENARIOS
    from ka59_ref.benchmark  import evaluate_agent
    from ka59_ref.discovery  import MinimalHypothesisAgent

    result = evaluate_agent(MinimalHypothesisAgent, SCENARIOS, max_steps=50)

    for r in result.results:
        print(r.scenario_name, r.termination, r.moved_count, r.passable_walls_found)

    # Or look up by name:
    r = result.by_name("transfer_wall_push")

Design notes
------------
- agent_factory is called once per scenario to produce a fresh agent.
- The agent must expose an act(obs) -> Action method.
- passable_walls_found is extracted via getattr(agent, 'passable_walls', set())
  so the evaluator works for any policy callable that has the attribute,
  and gracefully returns 0 for plain lambdas that don't.
- No engine tag names, no internal KA59State references appear in outputs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .runner import run_episode, EpisodeTrace
from .env    import ObjectView


# ---------------------------------------------------------------------------
# EpistemicSummary: agent belief state at episode end (engine-internal-free)
# ---------------------------------------------------------------------------

@dataclass
class EpistemicSummary:
    """
    Minimal summary of an agent's hypothesis/belief state at the end of an
    episode.  Extracted from the agent object after run_episode completes.

    For agents that expose readable hypothesis state (e.g. MinimalHypothesisAgent):
        has_beliefs=True; blocked_dirs and push_success_dirs are populated.

    For agents without such state (NaiveRightAgent, RotateOnBlockAgent, lambdas):
        has_beliefs=False; all dicts are empty.

    No engine-internal tag strings appear in any field.  Direction keys are
    always from {"right", "left", "up", "down"}.

    Fields
    ------
    has_beliefs          : True iff agent exposed structured hypothesis state
    blocked_dirs         : direction → count of times that direction was blocked
    push_success_dirs    : direction → count of push-assisted move successes
    passable_walls_found : distinct wall IDs observed to be passable by pushed blocks
    """
    has_beliefs:          bool
    blocked_dirs:         Dict[str, int]
    push_success_dirs:    Dict[str, int]
    passable_walls_found: int


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioResult:
    """
    Compact outcome summary for one agent-scenario run.

    Fields
    ------
    scenario_name       : name from the scenarios dict
    steps_taken         : number of steps the runner executed
    termination         : "done" | "max_steps"
    final_selected_pos  : (x, y) of the selected piece in the final observation
    moved_count         : steps where the selected piece actually moved
    blocked_count       : steps where the selected piece did not move
                          (= steps_taken − moved_count)
    passable_walls_found: len(agent.passable_walls) if agent has that attr, else 0
                          Non-zero only when a pushed block was observed at a
                          wall position → empirical WALL_TRANSFER discovery.
    """
    scenario_name:       str
    steps_taken:         int
    termination:         str            # "done" | "max_steps"
    final_selected_pos:  Tuple[int, int]
    moved_count:         int
    blocked_count:       int
    passable_walls_found: int


@dataclass
class BenchmarkResult:
    """
    Aggregated results for one agent evaluated across multiple scenarios.

    Fields
    ------
    agent_name : class name of the agent (or "unknown")
    results    : tuple of ScenarioResult, one per scenario, in input order
    epistemic  : dict mapping scenario_name → EpistemicSummary
                 Contains the agent's belief/hypothesis state at episode end
                 for each scenario.  Engine-internal-free.
    """
    agent_name: str
    results:    Tuple[ScenarioResult, ...]
    epistemic:  Dict[str, EpistemicSummary] = field(default_factory=dict)

    def by_name(self, name: str) -> ScenarioResult:
        """Return the ScenarioResult for the given scenario name."""
        for r in self.results:
            if r.scenario_name == name:
                return r
        raise KeyError(f"No result for scenario {name!r}")

    def epistemic_for(self, name: str) -> EpistemicSummary:
        """Return the EpistemicSummary for the given scenario name."""
        if name not in self.epistemic:
            raise KeyError(f"No epistemic summary for scenario {name!r}")
        return self.epistemic[name]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def evaluate_agent(
    agent_factory: Callable[[], Any],
    scenarios:     Dict[str, dict],
    max_steps:     int = 50,
) -> BenchmarkResult:
    """
    Run *agent_factory()* on each scenario and collect structured results.

    Parameters
    ----------
    agent_factory : callable() -> agent
        Called once per scenario.  The returned object must have
        act(obs: list[ObjectView]) -> Action.
    scenarios     : dict mapping scenario name → level spec dict
    max_steps     : hard cap on steps per episode (passed to run_episode)

    Returns
    -------
    BenchmarkResult with one ScenarioResult per scenario.
    """
    scenario_results:  List[ScenarioResult]          = []
    epistemic_map:     Dict[str, EpistemicSummary]   = {}
    inferred_name:     Optional[str]                 = None

    for scenario_name, level_spec in scenarios.items():
        agent = agent_factory()

        # Infer agent class name from first instance
        if inferred_name is None:
            inferred_name = type(agent).__name__

        trace = run_episode(level_spec, agent.act, max_steps=max_steps)
        scenario_results.append(_make_result(scenario_name, trace, agent))
        epistemic_map[scenario_name] = _extract_epistemic(agent)

    return BenchmarkResult(
        agent_name = inferred_name or "unknown",
        results    = tuple(scenario_results),
        epistemic  = epistemic_map,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _final_selected_pos(trace: EpisodeTrace) -> Tuple[int, int]:
    """Extract (x, y) of the selected piece from the last available observation."""
    # Use last step's obs if available; fall back to initial_obs
    obs = trace.steps[-1].obs if trace.steps else trace.initial_obs
    for view in obs:
        if view.is_selected:
            return (view.x, view.y)
    # Fallback: selected piece not found (should not happen in valid levels)
    return (0, 0)


def _extract_epistemic(agent: Any) -> EpistemicSummary:
    """
    Extract a minimal epistemic summary from the agent after a run.

    Agents that expose hypothesis state (blocked_count, push_success,
    passable_walls attributes) get has_beliefs=True with populated dicts.
    All other agents get has_beliefs=False with empty dicts.
    Engine-internal tag strings never appear in the output.
    """
    has_beliefs = (
        hasattr(agent, "blocked_count")
        and hasattr(agent, "push_success")
    )
    if has_beliefs:
        return EpistemicSummary(
            has_beliefs          = True,
            blocked_dirs         = dict(getattr(agent, "blocked_count", {})),
            push_success_dirs    = dict(getattr(agent, "push_success", {})),
            passable_walls_found = len(getattr(agent, "passable_walls", set())),
        )
    return EpistemicSummary(
        has_beliefs          = False,
        blocked_dirs         = {},
        push_success_dirs    = {},
        passable_walls_found = len(getattr(agent, "passable_walls", set())),
    )


def _make_result(
    scenario_name: str,
    trace:         EpisodeTrace,
    agent:         Any,
) -> ScenarioResult:
    """Derive a ScenarioResult from a completed EpisodeTrace and agent state."""
    moved_count  = sum(1 for s in trace.steps if s.moved)
    blocked_count = sum(1 for s in trace.steps if not s.moved)

    passable_walls_found = len(getattr(agent, "passable_walls", set()))

    return ScenarioResult(
        scenario_name       = scenario_name,
        steps_taken         = len(trace.steps),
        termination         = trace.termination,
        final_selected_pos  = _final_selected_pos(trace),
        moved_count         = moved_count,
        blocked_count       = blocked_count,
        passable_walls_found= passable_walls_found,
    )
