"""
ka59_llm_runner.py — Thin adapter: delegates to ka59_ref.experiment.run_agent().

Replaces the original hand-rolled runner that had two critical bugs:
  1. Weak observation: raw pixel coordinates with no semantic grid.
     ka59_ref uses get_structured_state() → JSON with indexed grid, movement
     hints, and step budget — same structured-state shape as LS20/BP35 easy.
  2. Heuristic win: "passable_walls_found > 0 AND move_rate > 0.4".
     ka59_ref checks game_state == "WIN" (real engine win condition).

Unified metric mapping (unchanged interface for runner.py):
  invalid_clicks ← blocked_count  (= ka59_ref.moves_blocked)
  flips          ← passable_walls_found  (= ka59_ref.push_events, push-event
                   proxy for wall-transfer discoveries)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).parents[2]  # jkj-breaking-agents/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ka59_ref.experiment import run_agent as _run_ref  # type: ignore

DEFAULT_SCENARIO = "transfer_wall_push"
DEFAULT_MAX_STEPS = 64


@dataclass
class RunResult:
    """Result wrapper that exposes the field names runner.py expects."""
    config: dict[str, str]
    scenario: str
    won: bool
    turns: int
    levels_completed: int
    provider: str = ""
    model: str = ""
    errors: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    understanding: dict[str, str] = field(default_factory=dict)
    timestamp: str = ""
    blocked_count: int = 0         # moves_blocked from ka59_ref
    passable_walls_found: int = 0  # push_events from ka59_ref (proxy)
    select_actions: int = 0
    moved_count: int = 0


def run_agent(
    world_level: str = "EASY",
    goal_level: str = "EASY",
    mechanics_level: str = "EASY",
    feedback_level: str = "EASY",
    provider: str = "openrouter",
    model: str = "meta-llama/llama-3.3-70b-instruct:free",
    scenario_name: str = DEFAULT_SCENARIO,
    max_steps: int = DEFAULT_MAX_STEPS,
    verbose: bool = True,
) -> RunResult:
    ref = _run_ref(
        scenario=scenario_name,
        world_level=world_level,
        goal_level=goal_level,
        mechanics_level=mechanics_level,
        feedback_level=feedback_level,
        provider=provider,
        model=model,
        max_turns=max_steps,
        verbose=verbose,
    )
    moved = max(0, ref.turns - ref.moves_blocked - ref.invalid_actions)
    return RunResult(
        config={"world": world_level, "goal": goal_level,
                "mechanics": mechanics_level, "feedback": feedback_level},
        scenario=scenario_name,
        won=ref.won,
        turns=ref.turns,
        levels_completed=ref.levels_completed,
        provider=ref.provider,
        model=ref.model,
        errors=list(ref.errors),
        history=list(ref.history),
        understanding=dict(ref.understanding),
        timestamp=ref.timestamp,
        blocked_count=ref.moves_blocked,
        passable_walls_found=ref.push_events,
        select_actions=ref.select_actions,
        moved_count=moved,
    )
