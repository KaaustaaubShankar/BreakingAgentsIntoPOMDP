"""
bp35_runner.py — BP35 agent loop adapted for the ka59 multi-env module.

Mirrors env4/experiment.py with two changes:
  1. Imports game_interface / llm_client / prompts from vendor/bp35/
  2. Resolves environment_files relative to the ka59/ directory

All ablation axis semantics (world_easy_format v2 = action_affordances +
valid_targets) are preserved from the env4 source.
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

_KA59_DIR = Path(__file__).parents[1]           # jkj-breaking-agents/ka59/
_VENDOR_BP35 = _KA59_DIR / "vendor" / "bp35"

for _p in (str(_VENDOR_BP35),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from game_interface import (   # type: ignore  — from vendor/bp35/
    ASCII_LEGEND,
    build_feedback_easy,
    frame_to_ascii,
    frame_to_base64_png,
    get_structured_state,
)
from llm_client import LLMClient  # type: ignore
from prompts import FEEDBACK_HARD, UNDERSTANDING_PROMPT, build_system_prompt  # type: ignore

# ── arc imports (require arc_agi in the active venv) ─────────────────────────

import arc_agi
from arc_agi import OperationMode
from arcengine import ActionInput, FrameDataRaw, GameAction

# ── constants ─────────────────────────────────────────────────────────────────

RESULTS_DIR = _KA59_DIR / "results" / "bp35"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TURNS_PER_LEVEL = 128
ACTION_MAP: dict[str, GameAction] = {
    "MOVE_LEFT":  GameAction.ACTION3,
    "MOVE_RIGHT": GameAction.ACTION4,
    "UNDO":       GameAction.ACTION7,
}
WORLD_EASY_FORMATS = ("v1", "v2")


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
    invalid_actions: int = 0
    click_actions: int = 0
    gravity_flips: int = 0
    undos: int = 0


# ── helpers (verbatim from env4/experiment.py) ────────────────────────────────

def _grid_to_click_coords(env: Any, position: list[int]) -> dict[str, int]:
    game = env._game.oztjzzyqoek
    grid_x, grid_y = int(position[0]), int(position[1])
    cell_w, cell_h = game.hdnrlfmyrj.qmcjluiioz
    camera_y = int(game.camera.rczgvgfsfb[1])
    return {
        "x": grid_x * cell_w + cell_w // 2,
        "y": grid_y * cell_h - camera_y + cell_h // 2,
    }


def _build_click_data(env: Any, position: list[int]) -> dict[str, int]:
    return _grid_to_click_coords(env, position)


def _sorted_positions(positions: list[list[int]]) -> list[list[int]]:
    return sorted([[int(p[0]), int(p[1])] for p in positions], key=lambda p: (p[1], p[0]))


def _semantic_grid_legend_text() -> str:
    return (
        "SEMANTIC LEGEND:\n"
        "- P=player  +=goal  x=breakable  y=expansion  g=gravity switch\n"
        "- 1/2=toggle tiles  u/v=spikes  o/m/w=support tiles\n"
        "- space=' ' means no tracked interactive object on that cell\n"
        "COORDINATES:\n"
        "- Positions are [x, y]\n"
        "- x increases to the right, y increases downward"
    )


def _indexed_semantic_grid_text(rows: list[str]) -> str:
    if not rows:
        return ""
    width = len(rows[0])
    tens = "".join(str((x // 10) % 10) if x >= 10 else " " for x in range(width))
    ones = "".join(str(x % 10) for x in range(width))
    out: list[str] = ["x-index:", f"      {tens}", f"      {ones}"]
    for y, row in enumerate(rows):
        out.append(f"y={y:02d}  {row}")
    return "\n".join(out)


def _build_world_easy_v1_observation(curr_state: dict, turn: int, max_turns: int) -> str:
    player_x, player_y = curr_state["player"]["position"]
    semantic_grid = "\n".join(curr_state["semantic_grid"])
    indexed_semantic_grid = _indexed_semantic_grid_text(curr_state["semantic_grid"])
    return (
        f"CURRENT STATE (turn {turn}/{max_turns}):\n"
        f"{json.dumps({k: v for k, v in curr_state.items() if k != 'semantic_grid'}, indent=2)}\n\n"
        "LOCATION ANCHOR:\n"
        f"- Player is at [x, y] = [{player_x}, {player_y}]\n"
        "- In the indexed grid below, find row y and column x. That cell is the player location.\n\n"
        f"{_semantic_grid_legend_text()}\n\n"
        f"SEMANTIC GRID (indexed):\n{indexed_semantic_grid}\n\n"
        f"SEMANTIC GRID (plain):\n{semantic_grid}"
    )


def _build_world_easy_v2_observation(
    curr_state: dict,
    turn: int,
    max_turns: int,
    local_radius: int = 4,
) -> str:
    """V2: adds action_affordances and valid_targets for richer grounding."""
    player_x, player_y = curr_state["player"]["position"]
    semantic_grid: list[str] = curr_state["semantic_grid"]
    grid_h = len(semantic_grid)
    grid_w = len(semantic_grid[0]) if semantic_grid else 0

    clickable_positions = _sorted_positions(
        [item["position"] for item in curr_state["objects"]["clickable_tiles"]]
    )
    goals = _sorted_positions(curr_state["objects"]["goals"])
    spikes = _sorted_positions(curr_state["objects"]["spikes"])
    supports = _sorted_positions(
        [item["position"] for item in curr_state["objects"]["safe_supports"]]
    )

    y0 = max(0, int(player_y) - local_radius)
    y1 = min(grid_h - 1, int(player_y) + local_radius) if grid_h else -1
    x0 = max(0, int(player_x) - local_radius)
    x1 = min(grid_w - 1, int(player_x) + local_radius) if grid_w else -1

    local_window_rows: list[str] = []
    for y in range(y0, y1 + 1):
        local_window_rows.append(semantic_grid[y][x0: x1 + 1])

    local_window = {
        "center": [int(player_x), int(player_y)],
        "radius": int(local_radius),
        "x_range": [x0, x1],
        "y_range": [y0, y1],
        "rows": local_window_rows,
    }

    canonical = {
        "turn": {"current": turn, "max": max_turns},
        "player": {
            "position": [int(player_x), int(player_y)],
            "gravity": curr_state["player"]["gravity"],
            "facing": curr_state["player"]["facing"],
        },
        "movement": {
            "left_target": curr_state["movement"]["left_target"],
            "right_target": curr_state["movement"]["right_target"],
            "left_blocked": bool(curr_state["movement"]["left_blocked"]),
            "right_blocked": bool(curr_state["movement"]["right_blocked"]),
            "left_tile_names": sorted(curr_state["movement"]["left_tile_names"]),
            "right_tile_names": sorted(curr_state["movement"]["right_tile_names"]),
        },
        "level": curr_state["level"],
        "resources": curr_state["resources"],
        "game_state": curr_state["game_state"],
        "objects": {
            "goals": goals,
            "spikes": spikes,
            "clickable_positions": clickable_positions,
            "safe_support_positions": supports,
            "counts": {
                "goals": len(goals),
                "spikes": len(spikes),
                "clickable_tiles": len(clickable_positions),
                "safe_supports": len(supports),
            },
        },
        "action_affordances": {
            "move_left": {
                "valid": not bool(curr_state["movement"]["left_blocked"]),
                "blocked": bool(curr_state["movement"]["left_blocked"]),
                "target": curr_state["movement"]["left_target"],
            },
            "move_right": {
                "valid": not bool(curr_state["movement"]["right_blocked"]),
                "blocked": bool(curr_state["movement"]["right_blocked"]),
                "target": curr_state["movement"]["right_target"],
            },
            "click": {
                "valid_targets": clickable_positions,
                "valid_target_count": len(clickable_positions),
            },
            "undo": {"valid": True},
        },
        "local_window": local_window,
    }

    return (
        f"WORLD_EASY_V2 (turn {turn}/{max_turns}):\n"
        f"{json.dumps(canonical, indent=2)}\n\n"
        f"{_semantic_grid_legend_text()}\n\n"
        "SEMANTIC GRID (global):\n"
        + "\n".join(semantic_grid)
    )


def _safe_env_step(
    env: Any,
    action: Any,
    data: Optional[dict[str, int]] = None,
) -> Optional[Any]:
    if isinstance(action, ActionInput):
        frame_data = env._game.perform_action(action, raw=True)
        frame_data.guid = env._guid
        frame_data.game_id = env.environment_info.game_id
        env._set_last_response(frame_data, reasoning=action.reasoning)
        return frame_data
    return env.step(action, data=data)


def _make_env():
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    return arcade.make("bp35", render_mode=None)


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
    world_easy_format: str = "v2",
    verbose: bool = True,
) -> RunResult:
    world_easy_format = world_easy_format.lower().strip()
    if world_easy_format not in WORLD_EASY_FORMATS:
        raise ValueError(f"Invalid world_easy_format={world_easy_format!r}")

    config = {
        "world": world_level,
        "goal": goal_level,
        "mechanics": mechanics_level,
        "feedback": feedback_level,
        "world_easy_format": world_easy_format,
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
        result.errors.append("Failed to initialise BP35 game.")
        return result

    levels_to_play = max_levels if max_levels is not None else int(frame_data.win_levels)
    max_turns = levels_to_play * turns_per_level

    history: list[dict[str, Any]] = []
    action_history: list[str] = []
    prev_state: Optional[dict[str, Any]] = None
    last_action_name = ""
    recent_actions: list[str] = []

    def log(event: dict[str, Any]) -> None:
        history.append({"timestamp": datetime.now(timezone.utc).isoformat(), **event})
        if verbose:
            print(f"  [{event.get('type', '?')}] {event.get('summary', '')}")

    log({"type": "game_start", "summary": "BP35 episode started.", "config": config})

    for turn in range(1, max_turns + 1):
        result.turns = turn
        curr_state = get_structured_state(env, frame_data)
        use_vision = world_level == "HARD" and vision and bool(frame_data.frame)

        if prev_state is not None:
            if feedback_level == "EASY":
                feedback_text = build_feedback_easy(prev_state, curr_state, last_action_name)
            else:
                feedback_text = FEEDBACK_HARD
            action_history.append(f"Turn {turn - 1}: {last_action_name}\n  Result: {feedback_text}")

        if prev_state is not None and prev_state["player"]["gravity"] != curr_state["player"]["gravity"]:
            result.gravity_flips += 1

        if world_level == "EASY":
            if world_easy_format == "v2":
                obs_block = _build_world_easy_v2_observation(curr_state, turn, max_turns)
            else:
                obs_block = _build_world_easy_v1_observation(curr_state, turn, max_turns)
        elif use_vision:
            obs_block = f"Turn {turn}/{max_turns}. The current BP35 frame is shown in the image."
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

        clickable_positions = [item["position"] for item in curr_state["objects"]["clickable_tiles"]]
        status_lines = [
            f"Position: {curr_state['player']['position']}",
            f"Gravity: {curr_state['player']['gravity']}",
            f"Steps remaining: {curr_state['resources']['steps_remaining']}",
            f"Goals: {curr_state['objects']['goals']}",
        ]
        if curr_state["movement"]["left_blocked"]:
            status_lines.append("LEFT is currently blocked.")
        if curr_state["movement"]["right_blocked"]:
            status_lines.append("RIGHT is currently blocked.")
        if clickable_positions:
            status_lines.append(f"Clickable tiles: {clickable_positions}")
        status_block = "STATUS:\n" + "\n".join(status_lines) + "\n"

        warnings: list[str] = []
        if len(recent_actions) >= 8 and len(set(recent_actions[-8:])) == 1:
            warnings.append(
                f"WARNING: You repeated {recent_actions[-1]} for 8 turns. Try a different move or click."
            )

        user_prompt = (
            ("\n".join(warnings) + "\n" if warnings else "")
            + history_block
            + "\n"
            + status_block
            + "\n"
            + obs_block
            + '\n\nRespond with a single JSON object only: {"reasoning": "<plan>", "action": "MOVE_LEFT"}'
        )

        retry_prompt = (
            "You forgot valid JSON. Reply with ONLY one JSON object.\n"
            'Examples:\n'
            '{"reasoning": "I should move right to line up with the goal", "action": "MOVE_RIGHT"}\n'
            '{"reasoning": "I should flip gravity", "action": "CLICK", "target_position": [4, 10]}\n'
            '{"reasoning": "This path is wrong", "action": "UNDO"}'
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
        target_position = parsed.get("target_position")

        raw_action = None
        raw_data: Optional[dict[str, int]] = None

        if action_name in ACTION_MAP:
            raw_action = ACTION_MAP[action_name]
            if action_name == "UNDO":
                result.undos += 1
        elif action_name == "CLICK":
            if not isinstance(target_position, list) or len(target_position) != 2:
                result.invalid_actions += 1
                err = f"Turn {turn}: CLICK missing target_position"
                result.errors.append(err)
                log({"type": "invalid_action", "summary": err, "raw_reply": reply})
                continue
            try:
                raw_action = GameAction.ACTION6
                raw_data = _build_click_data(env, [int(target_position[0]), int(target_position[1])])
                result.click_actions += 1
            except Exception as exc:
                result.invalid_actions += 1
                err = f"Turn {turn}: invalid click target {target_position!r} — {exc}"
                result.errors.append(err)
                log({"type": "invalid_action", "summary": err, "raw_reply": reply})
                continue
        else:
            result.invalid_actions += 1
            err = f"Turn {turn}: unknown action '{action_name}'"
            result.errors.append(err)
            log({"type": "invalid_action", "summary": err, "raw_reply": reply})
            continue

        log({
            "type": "action",
            "summary": f"Turn {turn}: {action_name}" + (f" | {reasoning}" if reasoning else ""),
            "turn": turn,
            "action": action_name,
            "target_position": target_position,
            "reasoning": reasoning,
            "state_before": curr_state,
        })

        prev_state = curr_state
        last_action_name = action_name if action_name != "CLICK" else f"CLICK {target_position}"
        recent_actions.append(last_action_name)

        frame_data = _safe_env_step(env, raw_action, data=raw_data)
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
