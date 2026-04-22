"""
runner.py — Thin multi-env dispatcher.

Usage:
    from ka59.runner import run_one
    result = run_one("bp35", "baseline", {"world": "EASY", ...}, provider, model)

Supported envs: "bp35" | "ls20" | "ka59"

Each call:
  1. Lazy-imports the right adapter
  2. Calls run_agent() for that environment
  3. Returns a UnifiedRunResult with consistent Josh-table columns

No arc_agi import at module level — safe to import for testing without the
arc_agi package installed.
"""

from __future__ import annotations

from typing import Optional

from unified_result import UnifiedRunResult

SUPPORTED_ENVS = ("bp35", "ls20", "ka59")

# Default scenario for KA59 LLM runs (mechanics-discovery probe)
KA59_DEFAULT_SCENARIO = "transfer_wall_push"
KA59_DEFAULT_MAX_STEPS = 64


def run_one(
    env_name: str,
    config_name: str,
    config: dict[str, str],
    provider: str,
    model: str,
    *,
    max_turns: Optional[int] = None,
    world_easy_format: str = "v2",
    ka59_scenario: str = KA59_DEFAULT_SCENARIO,
    verbose: bool = False,
) -> UnifiedRunResult:
    """
    Run one episode in the specified environment and return a UnifiedRunResult.

    Parameters
    ----------
    env_name         : "bp35" | "ls20" | "ka59"
    config_name      : ablation label (e.g. "baseline", "world_hard")
    config           : dict with world/goal/mechanics/feedback → "EASY"|"HARD"
    provider         : LLM provider (only "openrouter" currently)
    model            : model identifier (OpenRouter format)
    max_turns        : hard turn cap; overrides per-env default if set
    world_easy_format: "v1"|"v2" (BP35 only; v2 adds action_affordances)
    ka59_scenario    : KA59 scenario name (default: transfer_wall_push)
    verbose          : print per-turn events to stdout
    """
    env_name = env_name.lower().strip()
    if env_name not in SUPPORTED_ENVS:
        raise ValueError(f"Unknown env {env_name!r}; supported: {SUPPORTED_ENVS}")

    if env_name == "bp35":
        return _run_bp35(
            config_name, config, provider, model,
            max_turns=max_turns,
            world_easy_format=world_easy_format,
            verbose=verbose,
        )
    elif env_name == "ls20":
        return _run_ls20(
            config_name, config, provider, model,
            max_turns=max_turns,
            verbose=verbose,
        )
    elif env_name == "ka59":
        return _run_ka59(
            config_name, config, provider, model,
            max_steps=max_turns or KA59_DEFAULT_MAX_STEPS,
            scenario_name=ka59_scenario,
            verbose=verbose,
        )
    else:
        raise RuntimeError(f"Unhandled env: {env_name!r}")  # unreachable


# ---------------------------------------------------------------------------
# Private dispatch targets
# ---------------------------------------------------------------------------

def _run_bp35(
    config_name: str,
    config: dict[str, str],
    provider: str,
    model: str,
    *,
    max_turns: Optional[int],
    world_easy_format: str,
    verbose: bool,
) -> UnifiedRunResult:
    from envs.bp35_runner import run_agent, RunResult  # lazy import

    result: RunResult = run_agent(
        world_level=config.get("world", "EASY"),
        goal_level=config.get("goal", "EASY"),
        mechanics_level=config.get("mechanics", "EASY"),
        feedback_level=config.get("feedback", "EASY"),
        provider=provider,
        model=model,
        world_easy_format=world_easy_format,
        max_levels=1,
        turns_per_level=max_turns or 128,
        verbose=verbose,
    )

    hypo = _bp35_hypo_trace(result)
    return UnifiedRunResult(
        env="bp35",
        config_name=config_name,
        config=config,
        provider=provider,
        model=model,
        won=result.won,
        turns=result.turns,
        levels_completed=result.levels_completed,
        invalid_clicks=result.invalid_actions,
        flips=result.gravity_flips,
        errors=list(result.errors),
        raw=result,
        hypo_trace=hypo,
    )


def _run_ls20(
    config_name: str,
    config: dict[str, str],
    provider: str,
    model: str,
    *,
    max_turns: Optional[int],
    verbose: bool,
) -> UnifiedRunResult:
    from envs.ls20_runner import run_agent, RunResult  # lazy import

    result: RunResult = run_agent(
        world_level=config.get("world", "EASY"),
        goal_level=config.get("goal", "EASY"),
        mechanics_level=config.get("mechanics", "EASY"),
        feedback_level=config.get("feedback", "EASY"),
        provider=provider,
        model=model,
        max_levels=1,
        turns_per_level=max_turns or 100,
        verbose=verbose,
    )

    hypo = _ls20_hypo_trace(result)
    return UnifiedRunResult(
        env="ls20",
        config_name=config_name,
        config=config,
        provider=provider,
        model=model,
        won=result.won,
        turns=result.turns,
        levels_completed=result.levels_completed,
        invalid_clicks=result.wall_collisions,
        flips=result.goals_ever_activated,
        errors=list(result.errors),
        raw=result,
        hypo_trace=hypo,
    )


def _run_ka59(
    config_name: str,
    config: dict[str, str],
    provider: str,
    model: str,
    *,
    max_steps: int,
    scenario_name: str,
    verbose: bool,
) -> UnifiedRunResult:
    from envs.ka59_llm_runner import run_agent, RunResult  # lazy import

    result: RunResult = run_agent(
        world_level=config.get("world", "EASY"),
        goal_level=config.get("goal", "EASY"),
        mechanics_level=config.get("mechanics", "EASY"),
        feedback_level=config.get("feedback", "EASY"),
        provider=provider,
        model=model,
        scenario_name=scenario_name,
        max_steps=max_steps,
        verbose=verbose,
    )

    hypo = _ka59_hypo_trace(result)
    return UnifiedRunResult(
        env="ka59",
        config_name=config_name,
        config=config,
        provider=provider,
        model=model,
        won=result.won,
        turns=result.turns,
        levels_completed=result.levels_completed,
        invalid_clicks=result.blocked_count,
        flips=result.passable_walls_found,
        errors=list(result.errors),
        raw=result,
        hypo_trace=hypo,
    )


# ---------------------------------------------------------------------------
# Cross-game hypothesis-ruling trace extractors
# ---------------------------------------------------------------------------

def _bp35_hypo_trace(result) -> dict:
    """
    BP35 hypo trace: gravity-flip sequence and invalid-click clusters.
    These signal whether the agent discovered gravity-switch mechanics.
    """
    flip_turns: list[int] = []
    invalid_turns: list[int] = []
    action_histogram: dict[str, int] = {}

    prev_gravity = None
    for event in getattr(result, "history", []):
        t = event.get("type", "")
        turn = event.get("turn")
        if t == "action":
            action = event.get("action", "")
            action_histogram[action] = action_histogram.get(action, 0) + 1
            state = event.get("state_before", {})
            g = state.get("player", {}).get("gravity")
            if prev_gravity is not None and g != prev_gravity:
                if turn:
                    flip_turns.append(turn)
            if g is not None:
                prev_gravity = g
        elif t == "invalid_action" and turn:
            invalid_turns.append(turn)

    return {
        "env": "bp35",
        "gravity_flip_turns": flip_turns,
        "gravity_flip_count": len(flip_turns),
        "invalid_click_turns": invalid_turns,
        "action_histogram": action_histogram,
        "hypothesis": "gravity-flip mechanic discovered" if flip_turns else "no gravity-flip discovered",
    }


def _ls20_hypo_trace(result) -> dict:
    """
    LS20 hypo trace: goal-activation sequence.
    Signals whether the agent discovered shape/color/rotation modifier tiles.
    """
    goal_activation_turns: list[int] = []
    prev_activated_count = 0

    for event in getattr(result, "history", []):
        t = event.get("type", "")
        turn = event.get("turn")
        if t == "action":
            state = event.get("state_before", {})
            activated = state.get("goals", {}).get("activated_list", [])
            count = sum(1 for x in activated if x)
            if count > prev_activated_count and turn:
                goal_activation_turns.append(turn)
            prev_activated_count = count

    return {
        "env": "ls20",
        "goal_activation_turns": goal_activation_turns,
        "goals_activated_total": getattr(result, "goals_ever_activated", 0),
        "hypothesis": (
            "modifier tiles discovered" if goal_activation_turns else
            "no modifier-tile discovery observed"
        ),
    }


def _ka59_hypo_trace(result) -> dict:
    """
    KA59 hypo trace: wall-transfer discovery signal.
    The canonical mechanic: does the LLM discover passable vs solid walls?
    """
    return {
        "env": "ka59",
        "passable_walls_found": getattr(result, "passable_walls_found", 0),
        "blocked_count": getattr(result, "blocked_count", 0),
        "moved_count": getattr(result, "moved_count", 0),
        "select_actions": getattr(result, "select_actions", 0),
        "understanding": getattr(result, "understanding", {}),
        "hypothesis": (
            "wall-transfer asymmetry discovered" if getattr(result, "passable_walls_found", 0) > 0
            else "wall-transfer asymmetry NOT discovered"
        ),
    }
