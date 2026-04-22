"""
bp35_adapter.py — BP35 (env4) adapter for the multi-env runner.

Imports bp35-specific code from the vendored copy of env4's experiment.py
support files and delegates to run_agent() directly.

Unified metric mapping:
  invalid_clicks ← RunResult.invalid_actions
  flips          ← RunResult.gravity_flips

Cross-game hypothesis-ruling trace:
  Stored in UnifiedRunResult.hypo_trace as a summary dict derived from the
  agent's action history (gravity-flip sequences, invalid-click clusters).
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Optional, Any

# ---------------------------------------------------------------------------
# Path setup — make vendor/bp35 and ka59_ref importable
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parents[2]          # jkj-breaking-agents/
_VENDOR_BP35 = Path(__file__).parents[1] / "vendor" / "bp35"
_ENV_FILES_DIR = Path(__file__).parents[1] / "environment_files"

# Insert vendor path so imports resolve against our copy
if str(_VENDOR_BP35) not in sys.path:
    sys.path.insert(0, str(_VENDOR_BP35))


# ---------------------------------------------------------------------------
# Lazy import guard — arc_agi may not be available in test environments
# ---------------------------------------------------------------------------

def _import_experiment():
    """
    Import the BP35 experiment module, pointing it at our environment_files.

    The module reads environment_files relative to cwd; we temporarily set
    cwd to the ka59 directory which contains our environment_files copy.
    """
    try:
        # Patch _make_env to use our environment_files path
        import importlib
        import game_interface  # noqa: F401 — triggers path resolution
        import llm_client      # noqa: F401
        import prompts         # noqa: F401

        # Inline the run_agent logic to avoid arc_agi cwd dependency
        from bp35_runner import run_agent, RunResult  # type: ignore
        return run_agent, RunResult
    except ImportError:
        raise


def _build_hypo_trace(result: Any) -> dict:
    """
    Derive a lightweight hypothesis-ruling trace from the raw BP35 RunResult.

    Extracts:
      - gravity_flip_turns: turns where gravity reversed
      - invalid_action_turns: turns with invalid actions
      - action_histogram: count of each action type used
    """
    trace: dict[str, Any] = {
        "env": "bp35",
        "gravity_flip_turns": [],
        "invalid_action_turns": [],
        "action_histogram": {},
    }
    if not hasattr(result, "history"):
        return trace

    for event in result.history:
        t = event.get("type", "")
        turn = event.get("turn")
        if t == "action":
            action = event.get("action", "")
            trace["action_histogram"][action] = trace["action_histogram"].get(action, 0) + 1
        elif t == "invalid_action":
            if turn is not None:
                trace["invalid_action_turns"].append(turn)
        elif t == "gravity_flip" or (
            t == "action"
            and event.get("action", "") == "CLICK"
            and event.get("gravity_flip", False)
        ):
            if turn is not None:
                trace["gravity_flip_turns"].append(turn)

    # gravity flip turns can also be reconstructed from state_before diffs
    flip_turns = []
    prev_gravity = None
    for event in result.history:
        if event.get("type") == "action":
            state = event.get("state_before", {})
            g = state.get("player", {}).get("gravity")
            if prev_gravity is not None and g != prev_gravity:
                flip_turns.append(event.get("turn", 0))
            if g is not None:
                prev_gravity = g
    if flip_turns:
        trace["gravity_flip_turns"] = flip_turns

    return trace


# ---------------------------------------------------------------------------
# Public adapter function
# ---------------------------------------------------------------------------

def run_bp35(
    config_name: str,
    config: dict[str, str],
    provider: str,
    model: str,
    max_turns: Optional[int] = None,
    world_easy_format: str = "v2",
    verbose: bool = False,
) -> "UnifiedRunResult":  # noqa: F821 — imported at call site
    """
    Run one BP35 episode and return a UnifiedRunResult.

    Parameters
    ----------
    config_name :  ablation config label (e.g. "baseline", "world_hard")
    config      :  dict with keys world/goal/mechanics/feedback → "EASY"|"HARD"
    provider    :  LLM provider (only "openrouter" supported)
    model       :  model identifier
    max_turns   :  hard turn cap; overrides per-level budget if set
    world_easy_format: "v1" or "v2" (v2 includes action_affordances/valid_targets)
    """
    from unified_result import UnifiedRunResult  # noqa — relative
    sys.path.insert(0, str(_REPO_ROOT))

    # Temporarily change cwd so arc_agi finds environment_files
    ka59_dir = Path(__file__).parents[1]
    original_cwd = os.getcwd()
    os.chdir(ka59_dir)

    try:
        # Import freshly from vendor/bp35 each time to avoid stale state
        _reload_vendor_imports()
        from bp35_runner import run_agent  # type: ignore
        result = run_agent(
            world_level=config["world"],
            goal_level=config["goal"],
            mechanics_level=config["mechanics"],
            feedback_level=config["feedback"],
            provider=provider,
            model=model,
            world_easy_format=world_easy_format,
            max_levels=1,
            verbose=verbose,
        )
    finally:
        os.chdir(original_cwd)

    hypo = _build_hypo_trace(result)
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


def _reload_vendor_imports() -> None:
    """Ensure vendor/bp35 modules are loaded from the right path."""
    import importlib
    for mod_name in ("game_interface", "llm_client", "prompts"):
        if mod_name in sys.modules:
            del sys.modules[mod_name]
