"""
ls20_runner.py — LS20 agent loop adapted for the ka59 multi-env module.

Mirrors env3/experiment.py with two changes:
  1. Imports game_interface / llm_client / prompts from vendor/ls20/
  2. Resolves environment_files relative to the ka59/ directory

Unified metric mapping:
  invalid_clicks ← RunResult.wall_collisions
  flips          ← RunResult.goals_ever_activated

Cross-game hypothesis trace:
  hypo_trace captures the shape/color/rotation sequence the agent used,
  which goals were activated and in which order, and whether the agent
  discovered modifier-tile effects (implicit in goals_ever_activated trend).
"""

from __future__ import annotations

import json
import sys
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ── vendor path setup ─────────────────────────────────────────────────────────

_KA59_DIR = Path(__file__).parents[1]
_VENDOR_LS20 = _KA59_DIR / "vendor" / "ls20"

for _p in (str(_VENDOR_LS20),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import from vendor/ls20 — must happen after path setup
from game_interface import (  # type: ignore
    ASCII_LEGEND,
    build_feedback_easy,
    frame_to_ascii,
    frame_to_base64_png,
    get_structured_state,
)
from llm_client import LLMClient  # type: ignore
from prompts import FEEDBACK_HARD, UNDERSTANDING_PROMPT, build_system_prompt  # type: ignore

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

# ── constants ─────────────────────────────────────────────────────────────────

RESULTS_DIR = _KA59_DIR / "results" / "ls20"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TURNS_PER_LEVEL = 100
ACTION_MAP: dict[str, GameAction] = {
    "MOVE_NORTH": GameAction.ACTION1,
    "MOVE_SOUTH": GameAction.ACTION2,
    "MOVE_WEST":  GameAction.ACTION3,
    "MOVE_EAST":  GameAction.ACTION4,
}


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RunResult:
    config: dict[str, str]
    won: bool
    turns: int
    levels_completed: int
    provider: str = ""
    model: str = ""
    vision: bool = False
    errors: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    understanding: dict[str, str] = field(default_factory=dict)
    timestamp: str = ""
    wall_collisions: int = 0
    goals_ever_activated: int = 0


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_env():
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    return arcade.make("ls20", render_mode=None)


# ── main run_agent ────────────────────────────────────────────────────────────

def run_agent(
    world_level: str = "EASY",
    goal_level: str = "EASY",
    mechanics_level: str = "EASY",
    feedback_level: str = "EASY",
    provider: str = "openrouter",
    model: str = "meta-llama/llama-3.3-70b-instruct:free",
    vision: bool = False,
    turns_per_level: int = TURNS_PER_LEVEL,
    max_levels: Optional[int] = None,
    verbose: bool = True,
) -> RunResult:
    config = {
        "world": world_level,
        "goal": goal_level,
        "mechanics": mechanics_level,
        "feedback": feedback_level,
    }
    result = RunResult(
        config=config,
        won=False,
        turns=0,
        levels_completed=0,
        provider=provider,
        model=model,
        vision=vision,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    client = LLMClient(provider=provider, model=model)
    system_prompt = build_system_prompt(goal_level, mechanics_level)

    env = _make_env()
    frame_data = env.observation_space
    if frame_data is None:
        result.errors.append("Failed to initialise LS20 game.")
        return result

    levels_to_play = max_levels if max_levels is not None else int(frame_data.win_levels)
    max_turns = levels_to_play * turns_per_level

    history: list[dict[str, Any]] = []
    action_history: list[str] = []
    activated_goal_ids: set = set()
    prev_state: Optional[dict[str, Any]] = None
    last_action_name = ""
    recent_actions: list[str] = []

    def log(event: dict[str, Any]) -> None:
        history.append({"timestamp": datetime.now(timezone.utc).isoformat(), **event})
        if verbose:
            print(f"  [{event.get('type', '?')}] {event.get('summary', '')}")

    log({"type": "game_start", "summary": "LS20 episode started.", "config": config})

    for turn in range(1, max_turns + 1):
        result.turns = turn
        curr_state = get_structured_state(env, frame_data)
        use_vision = world_level == "HARD" and vision and bool(frame_data.frame)

        if prev_state is not None:
            # Wall collision: position unchanged after directional move
            if (
                last_action_name in ("MOVE_NORTH", "MOVE_SOUTH", "MOVE_WEST", "MOVE_EAST")
                and prev_state.get("player", {}).get("position")
                == curr_state.get("player", {}).get("position")
            ):
                result.wall_collisions += 1

            # Goals ever activated
            for i, activated in enumerate(curr_state.get("goals", {}).get("activated_list", [])):
                if activated:
                    activated_goal_ids.add(i)
            result.goals_ever_activated = len(activated_goal_ids)

            if feedback_level == "EASY":
                feedback_text = build_feedback_easy(prev_state, curr_state, last_action_name)
            else:
                feedback_text = FEEDBACK_HARD
            action_history.append(f"Turn {turn - 1}: {last_action_name}\n  Result: {feedback_text}")

        if world_level == "EASY":
            obs_block = (
                f"CURRENT STATE (turn {turn}/{max_turns}):\n"
                f"{json.dumps(curr_state, indent=2)}"
            )
        elif use_vision:
            obs_block = f"Turn {turn}/{max_turns}. The current LS20 frame is shown in the image."
        else:
            ascii_grid = frame_to_ascii(frame_data.frame[-1]) if frame_data.frame else ""
            obs_block = f"CURRENT FRAME (turn {turn}/{max_turns}):\n{ascii_grid}"
            if turn == 1:
                obs_block += f"\n\nCharacter reference: {ASCII_LEGEND}"

        recent_history = action_history[-10:]
        history_block = (
            "RECENT ACTIONS (last 10 turns):\n" + "\n".join(recent_history) + "\n"
            if recent_history else ""
        )

        warnings: list[str] = []
        if len(recent_actions) >= 8 and len(set(recent_actions[-8:])) == 1:
            warnings.append(
                f"WARNING: You repeated {recent_actions[-1]} for 8 turns. Try a different action."
            )

        user_prompt = (
            ("\n".join(warnings) + "\n" if warnings else "")
            + history_block
            + "\n"
            + obs_block
            + '\n\nRespond with a single JSON object only: '
            '{"reasoning": "<plan>", "action": "MOVE_NORTH"}'
        )

        retry_prompt = (
            "You forgot valid JSON. Reply with ONLY one JSON object.\n"
            'Valid actions: MOVE_NORTH, MOVE_SOUTH, MOVE_WEST, MOVE_EAST\n'
            '{"reasoning": "moving east to reach modifier", "action": "MOVE_EAST"}'
        )

        try:
            if use_vision:
                b64 = frame_to_base64_png(frame_data.frame[-1])
                reply = client.generate_with_image(system_prompt, user_prompt, b64)
            else:
                reply = client.generate(system_prompt, user_prompt)
            try:
                parsed = client.parse_json(reply)
            except ValueError:
                reply = client.generate(system_prompt, retry_prompt)
                parsed = client.parse_json(reply)
        except Exception as exc:
            err = f"Turn {turn}: LLM/parse error — {exc}"
            result.errors.append(err)
            log({"type": "error", "summary": err})
            continue

        action_name = str(parsed.get("action", "")).upper().strip()
        reasoning = str(parsed.get("reasoning", "")).strip()

        raw_action = ACTION_MAP.get(action_name)
        if raw_action is None:
            err = f"Turn {turn}: unknown action '{action_name}'"
            result.errors.append(err)
            log({"type": "invalid_action", "summary": err})
            continue

        log({
            "type": "action",
            "summary": f"Turn {turn}: {action_name}" + (f" | {reasoning}" if reasoning else ""),
            "turn": turn,
            "action": action_name,
            "reasoning": reasoning,
            "state_before": curr_state,
        })

        prev_state = curr_state
        last_action_name = action_name
        recent_actions.append(last_action_name)

        frame_data = env.step(raw_action)
        if frame_data is None:
            result.errors.append(f"Turn {turn}: env.step returned None.")
            break

        result.levels_completed = frame_data.levels_completed
        if frame_data.state.name == "WIN":
            result.won = True
            log({"type": "win", "summary": f"WIN after {turn} turns."})
            break
        if frame_data.state.name == "GAME_OVER":
            log({"type": "game_over", "summary": f"GAME_OVER after {turn} turns."})
            break
        if max_levels is not None and frame_data.levels_completed >= max_levels:
            result.won = True
            log({"type": "win", "summary": f"Reached {max_levels} level(s) after {turn} turns."})
            break
    else:
        log({"type": "timeout", "summary": f"Turn budget ({max_turns}) exhausted."})

    result.history = history
    return result
