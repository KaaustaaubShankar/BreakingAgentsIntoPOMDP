"""
ka59_llm_runner.py — Thin adapter: delegates to ka59_game.experiment.run_agent().

Uses the REAL KA59 arc_agi game (multi-level, up to 7 levels) instead of the
ka59_ref synthetic micro-scenarios. Matches the pattern of ls20_runner.py and
bp35_runner.py — real game, real win condition, real difficulty curve.

Unified metric mapping:
  invalid_clicks ← blocked_count  (= ka59_game.moves_blocked)
  flips          ← passable_walls_found  (= ka59_game.click_actions, proxy for
                   wall-transfer mechanic engagement)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).parents[2]  # jkj-breaking-agents/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ka59_game.experiment import run_agent as _run_real  # type: ignore

DEFAULT_MAX_TURNS = 200  # real game levels need more headroom than synthetic scenarios


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
    blocked_count: int = 0         # moves_blocked from ka59_game
    passable_walls_found: int = 0  # click_actions proxy
    select_actions: int = 0
    moved_count: int = 0


def run_agent(
    world_level: str = "EASY",
    goal_level: str = "EASY",
    mechanics_level: str = "EASY",
    feedback_level: str = "EASY",
    provider: str = "openrouter",
    model: str = "meta-llama/llama-3.3-70b-instruct:free",
    scenario_name: str = "ka59",   # ignored — real game has fixed levels
    max_steps: int = DEFAULT_MAX_TURNS,
    verbose: bool = True,
) -> RunResult:
    ref = _run_real(
        world_level=world_level,
        goal_level=goal_level,
        mechanics_level=mechanics_level,
        feedback_level=feedback_level,
        provider=provider,
        model=model,
        max_levels=1,
        turns_per_level=max_steps,
        verbose=verbose,
    )
    moved = max(0, ref.turns - ref.moves_blocked - ref.invalid_actions)
    return RunResult(
        config={"world": world_level, "goal": goal_level,
                "mechanics": mechanics_level, "feedback": feedback_level},
        scenario="ka59_game",
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
        passable_walls_found=ref.click_actions,
        select_actions=ref.click_actions,
        moved_count=moved,
    )
