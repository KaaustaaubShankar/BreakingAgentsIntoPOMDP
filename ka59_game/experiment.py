"""
experiment.py — LLM agent harness for the REAL KA59 game (via arc_agi).

Parallel to env4/experiment.py (which drives BP35). Same RunResult schema,
same config {world, goal, mechanics, feedback}, same JSON response protocol.

Differences vs ka59_ref/experiment.py (faithful sim):
- Uses arc_agi.Arcade + real KA59 game source.
- Multi-level progression (up to win_levels, default 7) instead of one scenario.
- CLICK action (ACTION6) is the SELECT analog — pick a selectable piece by
  target_position in cell coordinates.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import arc_agi
from arc_agi import OperationMode
from arcengine import FrameDataRaw, GameAction

from .game_interface import (
    ACTION_MAP,
    STEP,
    build_feedback_easy,
    get_structured_state,
)
from .llm_client import LLMClient
from .prompts import FEEDBACK_HARD, UNDERSTANDING_PROMPT, build_system_prompt, check_discovery


RESULTS_DIR = Path("results") / "ka59_game"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TURNS_PER_LEVEL = 64
STUCK_THRESHOLD = 5  # same order of magnitude as env4's bp35 budget
MAX_LEVELS_HARD_CAP = 7       # arc_agi reports win_levels=7 for ka59


@dataclass
class RunResult:
    config: dict[str, str]
    won: bool
    turns: int
    levels_completed: int
    provider: str = ""
    model: str = ""
    errors: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    understanding: dict[str, str] = field(default_factory=dict)
    timestamp: str = ""
    invalid_actions: int = 0
    click_actions: int = 0
    moves_blocked: int = 0
    forced_reframes: int = 0
    orient_history: list[str] = field(default_factory=list)
    discovery_turn: Optional[int] = None


def save_result(result: RunResult, run_id: Optional[str] = None) -> Path:
    cfg = result.config
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    tag = f"f{cfg['feedback'][0]}_g{cfg['goal'][0]}_m{cfg['mechanics'][0]}_w{cfg['world'][0]}"
    path = RESULTS_DIR / f"run_{tag}_{run_id}.json"
    payload = {
        "config": result.config,
        "won": result.won,
        "turns": result.turns,
        "levels_completed": result.levels_completed,
        "provider": result.provider,
        "model": result.model,
        "timestamp": result.timestamp,
        "errors": result.errors,
        "understanding": result.understanding,
        "invalid_actions": result.invalid_actions,
        "click_actions": result.click_actions,
        "moves_blocked": result.moves_blocked,
        "forced_reframes": result.forced_reframes,
        "orient_history": result.orient_history,
        "discovery_turn": result.discovery_turn,
        "history": result.history,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"  Saved -> {path}")
    return path


def _make_env():
    from pathlib import Path
    env_dir = str(Path(__file__).parents[1] / "environment_files")
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=env_dir)
    return arcade.make("ka59", render_mode=None)


def _grid_to_click_data(env, grid_x: int, grid_y: int) -> dict[str, int]:
    """Convert game-grid (pixel) coordinates to DISPLAY coordinates for CLICK.

    The engine's ACTION6 handler passes action.data.{x,y} through
    camera.display_to_grid() before doing a get_sprite_at lookup. So we must
    submit display coords, not raw grid coords. Offset = (MAX_DIMENSION -
    viewport width) // 2, plus camera scroll.
    """
    cam = env._game.camera
    offset_x = (cam.MAX_DIMENSION - cam.width) // 2
    offset_y = (cam.MAX_DIMENSION - cam.height) // 2
    return {
        "x": int(grid_x) + int(cam.x) + offset_x,
        "y": int(grid_y) + int(cam.y) + offset_y,
    }


def _indexed_semantic_grid_text(rows: list[str]) -> str:
    if not rows:
        return "(empty grid)"
    width = len(rows[0])
    tens = "".join(str((x // 10) % 10) if x >= 10 else " " for x in range(width))
    ones = "".join(str(x % 10) for x in range(width))
    out: list[str] = [
        "x-index:",
        f"      {tens}",
        f"      {ones}",
    ]
    for y, row in enumerate(rows):
        out.append(f"y={y:02d}  {row}")
    return "\n".join(out)


def _build_world_easy_observation(
    curr_state: dict[str, Any], turn: int, max_turns: int
) -> str:
    state_no_grid = {k: v for k, v in curr_state.items() if k != "semantic_grid"}
    indexed_grid = _indexed_semantic_grid_text(curr_state["semantic_grid"])
    legend = (
        "SEMANTIC LEGEND:\n"
        "- P=selected  S=other selectable  B=block  #=wall  +=goal  .=empty\n"
        "COORDINATES:\n"
        "- Positions in `objects` and `player` are pixel units.\n"
        "- 1 cell = 3 pixels. `semantic_grid` uses cells; x-index is cell x."
    )
    return (
        f"CURRENT STATE (turn {turn}/{max_turns}):\n"
        f"{json.dumps(state_no_grid, indent=2)}\n\n"
        f"{legend}\n\n"
        f"SEMANTIC GRID (indexed):\n{indexed_grid}"
    )


def _build_world_hard_observation(
    curr_state: dict[str, Any], turn: int, max_turns: int
) -> str:
    """WORLD_HARD: strip non-selected object positions from the observation.
    Agent sees only: where the selected piece is, blocked-flags per direction,
    level/resources/game_state counts. No goal positions, no wall positions.
    """
    minimal = {
        "player": curr_state["player"],
        "movement": {
            "left_blocked": curr_state["movement"]["left_blocked"],
            "right_blocked": curr_state["movement"]["right_blocked"],
            "up_blocked": curr_state["movement"]["up_blocked"],
            "down_blocked": curr_state["movement"]["down_blocked"],
        },
        "level": curr_state["level"],
        "resources": curr_state["resources"],
        "object_counts": {
            key: len(curr_state["objects"][key])
            for key in ("walls", "blocks", "selectables", "goals")
        },
        "game_state": curr_state["game_state"],
    }
    return (
        f"CURRENT STATE (turn {turn}/{max_turns}, degraded):\n"
        f"{json.dumps(minimal, indent=2)}\n"
        "(Non-player object positions are not available this run. Use blocked-flags and action outcomes to infer structure.)"
    )


def run_agent(
    world_level: str = "EASY",
    goal_level: str = "EASY",
    mechanics_level: str = "EASY",
    feedback_level: str = "EASY",
    provider: str = "openrouter",
    model: str = "google/gemini-2.5-flash",
    max_levels: Optional[int] = None,
    turns_per_level: int = DEFAULT_TURNS_PER_LEVEL,
    verbose: bool = True,
    llm_client: Optional[LLMClient] = None,
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
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    client = llm_client if llm_client is not None else LLMClient(provider=provider, model=model)
    system_prompt = build_system_prompt(goal_level, mechanics_level)

    env = _make_env()
    frame_data: FrameDataRaw = env.observation_space
    if frame_data is None:
        result.errors.append("Failed to initialise game (observation_space is None).")
        return result

    levels_to_play = min(max_levels if max_levels is not None else int(frame_data.win_levels),
                         MAX_LEVELS_HARD_CAP)
    max_turns = levels_to_play * turns_per_level
    step_budget = turns_per_level

    history: list[dict[str, Any]] = []
    action_history: list[str] = []
    prev_state: Optional[dict[str, Any]] = None
    last_action_name = ""
    last_level_index = 0
    turn_in_level = 0
    position_history: list[tuple] = []
    consecutive_same_type_count = 0
    last_action_type_tracked = ""

    def log(event: dict[str, Any]) -> None:
        history.append({"timestamp": datetime.now(timezone.utc).isoformat(), **event})
        if verbose:
            print(f"  [{event.get('type', '?')}] {event.get('summary', '')}")

    log({"type": "game_start", "summary": f"KA59 real-game run started. levels={levels_to_play}", "config": config})

    for turn in range(1, max_turns + 1):
        result.turns = turn
        curr_level_index = int(env._game.level_index)
        if curr_level_index != last_level_index:
            turn_in_level = 0
            last_level_index = curr_level_index
        turn_in_level += 1

        curr_state = get_structured_state(env, frame_data, turn_in_level, step_budget)

        if prev_state is not None:
            if feedback_level == "EASY":
                feedback_text = build_feedback_easy(prev_state, curr_state, last_action_name)
            else:
                feedback_text = FEEDBACK_HARD
            action_history.append(f"Turn {turn - 1}: {last_action_name}\n  Result: {feedback_text}")

        # Track position history for stuck detection
        curr_pos = tuple(curr_state["player"]["position"]) if isinstance(curr_state["player"]["position"], list) else curr_state["player"]["position"]
        position_history.append(curr_pos)
        if len(position_history) > STUCK_THRESHOLD:
            position_history.pop(0)

        # Track consecutive same action-type (CLICK vs MOVE)
        if last_action_name:
            last_type = "CLICK" if "CLICK" in last_action_name else "MOVE"
            if last_type == last_action_type_tracked:
                consecutive_same_type_count += 1
            else:
                consecutive_same_type_count = 1
                last_action_type_tracked = last_type

        position_stuck = (
            len(position_history) >= STUCK_THRESHOLD
            and all(p == position_history[0] for p in position_history)
        )

        if world_level == "HARD":
            obs_block = _build_world_hard_observation(curr_state, turn, max_turns)
        else:
            obs_block = _build_world_easy_observation(curr_state, turn, max_turns)

        recent_history = action_history[-10:]
        history_block = (
            "RECENT ACTIONS (last 10 turns):\n" + "\n".join(recent_history) + "\n"
            if recent_history else ""
        )

        status_lines = [
            f"Selected: {curr_state['player']['id']} @ {curr_state['player']['position']}",
            f"Level: {curr_state['level']['current']}/{curr_state['level']['total']}",
            f"Steps remaining this level: {curr_state['resources']['steps_remaining']}",
        ]
        status_block = "STATUS:\n" + "\n".join(status_lines) + "\n"

        forced_reframe_block = ""
        if mechanics_level == "OODA_F" and turn > STUCK_THRESHOLD and (
            position_stuck or consecutive_same_type_count >= STUCK_THRESHOLD
        ):
            reason = (
                f"your position has not changed for {STUCK_THRESHOLD} consecutive turns"
                if position_stuck
                else f"you have used {last_action_type_tracked} actions {consecutive_same_type_count} times in a row"
            )
            forced_reframe_block = (
                f"[FORCED REFRAME] No progress: {reason}. "
                "Your current hypothesis is wrong. ABANDON it. Form a new hypothesis and choose a DIFFERENT action type.\n\n"
            )
            result.forced_reframes += 1
            consecutive_same_type_count = 1
            log({"type": "forced_reframe", "summary": f"Turn {turn}: forced reframe ({reason})", "turn": turn})

        user_prompt = (
            forced_reframe_block
            + history_block
            + "\n"
            + status_block
            + "\n"
            + obs_block
            + '\n\nRespond with a single JSON object only: {"reasoning": "<plan>", "action": "MOVE_RIGHT"}'
        )

        retry_prompt = (
            "You forgot valid JSON. Reply with ONLY one JSON object.\n"
            "Examples:\n"
            '{"reasoning": "I should move right", "action": "MOVE_RIGHT"}\n'
            '{"reasoning": "Switch to the other selectable", "action": "CLICK", "target_position": [11, 7]}'
        )

        try:
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

        # OODA fields
        observe_text = str(parsed.get("observe", "")).strip()
        orient_text = str(parsed.get("orient", "")).strip()
        decide_text = str(parsed.get("decide", "")).strip()
        if orient_text:
            result.orient_history.append(orient_text)
            if result.discovery_turn is None and check_discovery(orient_text):
                result.discovery_turn = turn

        game_action: Optional[GameAction] = None
        action_data: Optional[dict[str, int]] = None
        if action_name in ACTION_MAP and action_name != "CLICK":
            game_action = ACTION_MAP[action_name]
        elif action_name == "CLICK":
            if not isinstance(target_position, list) or len(target_position) != 2:
                result.invalid_actions += 1
                err = f"Turn {turn}: CLICK missing target_position"
                result.errors.append(err)
                log({"type": "invalid_action", "summary": err, "raw_reply": reply})
                continue
            try:
                gx, gy = int(target_position[0]), int(target_position[1])
                game_action = GameAction.ACTION6
                action_data = _grid_to_click_data(env, gx, gy)
                result.click_actions += 1
            except Exception as exc:
                result.invalid_actions += 1
                err = f"Turn {turn}: invalid CLICK target {target_position!r} — {exc}"
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
            "summary": f"Turn {turn}: {action_name}" + (f" | {reasoning or decide_text or orient_text}" if (reasoning or decide_text or orient_text) else ""),
            "turn": turn,
            "action": action_name,
            "target_position": target_position,
            "reasoning": reasoning,
            "observe": observe_text,
            "orient": orient_text,
            "decide": decide_text,
        })

        prev_state = curr_state
        last_action_name = action_name if action_name != "CLICK" else f"CLICK {target_position}"

        try:
            frame_data = env.step(game_action, data=action_data)
        except Exception as exc:
            err = f"Turn {turn}: env.step raised {type(exc).__name__}: {exc}"
            result.errors.append(err)
            log({"type": "env_error", "summary": err})
            continue

        if frame_data is None:
            result.errors.append(f"Turn {turn}: env.step returned None.")
            break

        result.levels_completed = int(frame_data.levels_completed)

        # Detect action outcome for moves_blocked metric.
        if action_name in ACTION_MAP and action_name != "CLICK":
            post_state = get_structured_state(env, frame_data, turn_in_level, step_budget)
            if post_state["player"]["position"] == curr_state["player"]["position"]:
                result.moves_blocked += 1

        state_name = frame_data.state.name
        if state_name == "WIN":
            result.won = True
            log({"type": "win", "summary": f"WIN after {turn} turns."})
            break
        if state_name == "GAME_OVER":
            log({"type": "game_over", "summary": f"GAME_OVER after {turn} turns."})
            break
        if max_levels is not None and int(frame_data.levels_completed) >= max_levels:
            result.won = True
            log({"type": "win", "summary": f"Reached {max_levels} level(s) after {turn} turns."})
            break

        if turn_in_level >= step_budget:
            # out of budget for this level; the game may not auto-end so we stop
            log({"type": "budget_exhausted", "summary": f"Level {curr_level_index+1} step budget exhausted."})
            break
    else:
        log({"type": "timeout", "summary": f"Turn budget ({max_turns}) exhausted."})

    understanding_prompt_text = (history_block if action_history else "") + "\n" + UNDERSTANDING_PROMPT
    try:
        understanding_reply = client.generate(system_prompt, understanding_prompt_text)
        result.understanding = {
            key: str(value) for key, value in client.parse_json(understanding_reply).items()
        }
        log({
            "type": "understanding",
            "summary": "Agent explained goal and mechanics.",
            "understanding": result.understanding,
        })
    except Exception as exc:
        result.errors.append(f"Understanding prompt failed: {exc}")

    result.history = history
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the real KA59 game with an LLM agent.")
    p.add_argument("--provider", default="openrouter", choices=["openrouter"])
    p.add_argument("--model", default="google/gemini-2.5-flash")
    p.add_argument("--world", default="EASY", choices=["EASY", "HARD"])
    p.add_argument("--goal", default="EASY", choices=["EASY", "HARD"])
    p.add_argument("--mechanics", default="EASY", choices=["EASY", "HARD", "OODA", "OODA_F"])
    p.add_argument("--feedback", default="EASY", choices=["EASY", "HARD"])
    p.add_argument("--max-levels", type=int, default=1, help="Default 1 for smoke test speed")
    p.add_argument("--turns-per-level", type=int, default=DEFAULT_TURNS_PER_LEVEL)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_agent(
        world_level=args.world,
        goal_level=args.goal,
        mechanics_level=args.mechanics,
        feedback_level=args.feedback,
        provider=args.provider,
        model=args.model,
        max_levels=args.max_levels,
        turns_per_level=args.turns_per_level,
        verbose=not args.quiet,
    )
    save_result(result)


if __name__ == "__main__":
    main()
