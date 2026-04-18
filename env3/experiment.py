"""
experiment.py — Agent and human runner for Environment 3 (Grid Navigator / ls20).

Usage:
  # Agent — baseline (all EASY), Gemini free tier
  python experiment.py --provider gemini --model gemini-2.0-flash

  # Agent — axis overrides
  python experiment.py --provider gemini --model gemini-2.0-flash --world HARD
  python experiment.py --provider gemini --model gemini-2.0-flash --world HARD --vision

  # Human — play yourself to test a configuration
  python experiment.py --human
  python experiment.py --human --world HARD --goal HARD --mechanics HARD --feedback HARD

Axis levels: EASY or HARD
  World     — EASY: structured JSON state
              HARD (text):   raw 64×64 ASCII frame
              HARD (vision): rendered 512×512 PNG sent to a vision model
  Goal      — EASY: objective explained  |  HARD: nothing stated
  Mechanics — EASY: all rules explained  |  HARD: no instructions
  Feedback  — EASY: detailed state diff  |  HARD: "Ok."

Results saved as JSON in results/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tty
import termios
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

from game_interface import (
    ASCII_LEGEND,
    build_feedback_easy,
    frame_to_ascii,
    frame_to_base64_png,
    get_structured_state,
)
from llm_client import LLMClient
from prompts import (
    FEEDBACK_HARD,
    UNDERSTANDING_PROMPT,
    build_system_prompt,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

TURNS_PER_LEVEL = 100

ACTION_MAP: dict[str, GameAction] = {
    "MOVE_NORTH": GameAction.ACTION1,
    "MOVE_SOUTH": GameAction.ACTION2,
    "MOVE_WEST":  GameAction.ACTION3,
    "MOVE_EAST":  GameAction.ACTION4,
}

CLEAR = "\033[H\033[2J"


# ── Result dataclass ──────────────────────────────────────────────────────────

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
    goals_ever_activated: int = 0  # distinct goals whose requirements were ever matched


# ── Save helper ───────────────────────────────────────────────────────────────

def save_result(result: RunResult, run_id: Optional[str] = None) -> Path:
    cfg = result.config
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    tag = f"f{cfg['feedback'][0]}_g{cfg['goal'][0]}_m{cfg['mechanics'][0]}_w{cfg['world'][0]}"
    if result.vision:
        tag += "_vision"
    path = RESULTS_DIR / f"run_{tag}_{run_id}.json"
    path.write_text(json.dumps({
        "config": result.config,
        "vision": result.vision,
        "won": result.won,
        "turns": result.turns,
        "levels_completed": result.levels_completed,
        "provider": result.provider,
        "model": result.model,
        "timestamp": result.timestamp,
        "errors": result.errors,
        "understanding": result.understanding,
        "wall_collisions": result.wall_collisions,
        "goals_ever_activated": result.goals_ever_activated,
        "history": result.history,
    }, indent=2))
    print(f"  Saved → {path}")
    return path


# ── Agent loop ────────────────────────────────────────────────────────────────

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
    """Run one episode with an LLM agent."""
    config = {
        "world": world_level, "goal": goal_level,
        "mechanics": mechanics_level, "feedback": feedback_level,
    }
    timestamp = datetime.now(timezone.utc).isoformat()
    result = RunResult(
        config=config, won=False, turns=0, levels_completed=0,
        provider=provider, model=model, vision=vision, timestamp=timestamp,
    )

    client = LLMClient(provider=provider, model=model)
    system_prompt = build_system_prompt(goal_level, mechanics_level)

    arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arc.make("ls20", render_mode=None)
    frame_data = env.observation_space
    if frame_data is None:
        result.errors.append("Failed to initialise game.")
        return result

    levels_to_play = max_levels if max_levels is not None else int(frame_data.win_levels)
    max_turns = levels_to_play * turns_per_level

    history: list[dict[str, Any]] = []
    action_history: list[str] = []   # running text log folded into each user_prompt
    prev_state: Optional[dict[str, Any]] = None
    last_action_name: str = ""
    _activated_goal_ids: set[int] = set()  # tracks unique goals ever matched
    _recent_actions: list[str] = []         # full action log for loop detection
    _position_visits: dict[str, int] = {}  # position → visit count for revisit detection

    def log(event: dict[str, Any]) -> None:
        history.append({"timestamp": datetime.now(timezone.utc).isoformat(), **event})
        if verbose:
            print(f"  [{event.get('type', '?')}] {event.get('summary', '')}")

    log({"type": "game_start", "summary": "Episode started.", "config": config})

    for turn in range(1, max_turns + 1):
        result.turns = turn
        curr_state = get_structured_state(env, frame_data)
        use_vision = (world_level == "HARD" and vision and bool(frame_data.frame))

        # ── Metrics: wall collision and glyph matching ────────────────────────
        if prev_state is not None:
            if prev_state["player"]["position"] == curr_state["player"]["position"]:
                result.wall_collisions += 1

        piece = curr_state["piece"]
        for g in curr_state["goals"]:
            if (
                not g["completed"]
                and g["id"] not in _activated_goal_ids
                and piece["shape"] == g["required_shape"]
                and piece["color"] == g["required_color"]
                and piece["rotation_degrees"] == g["required_rotation_degrees"]
            ):
                _activated_goal_ids.add(g["id"])
                result.goals_ever_activated = len(_activated_goal_ids)

        # Build current observation block
        if world_level == "EASY":
            obs_block = (
                f"CURRENT STATE (turn {turn}/{max_turns}):\n"
                + json.dumps(curr_state, indent=2)
            )
        elif use_vision:
            obs_block = f"Turn {turn}/{max_turns}. The current game frame is shown in the image."
        else:
            ascii_grid = frame_to_ascii(frame_data.frame[-1]) if frame_data.frame else ""
            obs_block = f"CURRENT GRID (turn {turn}/{max_turns}):\n{ascii_grid}"
            if turn == 1:
                obs_block += f"\n\nCharacter reference: {ASCII_LEGEND}"

        # Build feedback for the previous action
        if prev_state is not None:
            if feedback_level == "EASY":
                feedback_text = build_feedback_easy(prev_state, curr_state, last_action_name)
            else:
                feedback_text = FEEDBACK_HARD
            action_history.append(
                f"Turn {turn - 1}: {last_action_name}\n  Result: {feedback_text}"
            )

        # Assemble the full user prompt (last 10 turns of history + status + current observation)
        recent_history = action_history[-10:]
        history_block = (
            "RECENT ACTIONS (last 10 turns):\n" + "\n".join(recent_history) + "\n"
            if recent_history else ""
        )

        # Plain-English status summary so the model doesn't have to parse the JSON to know what to do
        if world_level == "EASY":
            status_lines = []
            p = curr_state["player"]
            piece = curr_state["piece"]
            status_lines.append(f"Position: {p['position']}  Stamina: {p['stamina']}  Lives: {p['lives']}")
            for g in curr_state["goals"]:
                if not g["completed"]:
                    mismatches = []
                    if piece["shape"] != g["required_shape"]:
                        mismatches.append(f"shape {piece['shape']}→{g['required_shape']}")
                    if piece["color"] != g["required_color"]:
                        mismatches.append(f"color {piece['color']}→{g['required_color']}")
                    if piece["rotation_degrees"] != g["required_rotation_degrees"]:
                        mismatches.append(f"rotation {piece['rotation_degrees']}°→{g['required_rotation_degrees']}°")
                    if mismatches:
                        status_lines.append(f"Goal {g['id']} at {g['position']}: NOT READY — fix {', '.join(mismatches)}")
                    else:
                        status_lines.append(f"Goal {g['id']} at {g['position']}: ACTIVATED — step on it to complete!")
            status_block = "STATUS:\n" + "\n".join(status_lines) + "\n"
        else:
            status_block = ""

        # ── Loop detection warnings ───────────────────────────────────────────
        warnings: list[str] = []

        # Warn if the same action dominates the last 10 moves
        if len(_recent_actions) >= 10:
            last_10 = _recent_actions[-10:]
            most_common = max(set(last_10), key=last_10.count)
            if last_10.count(most_common) >= 8:
                warnings.append(
                    f"WARNING: You have used {most_common} {last_10.count(most_common)} "
                    f"of the last 10 turns. This is not working. "
                    f"You MUST try a completely different direction to escape."
                )

        # Warn if current position has been visited many times
        pos_key = str(curr_state["player"]["position"])
        _position_visits[pos_key] = _position_visits.get(pos_key, 0) + 1
        if _position_visits[pos_key] >= 5:
            warnings.append(
                f"WARNING: You have been at position {curr_state['player']['position']} "
                f"{_position_visits[pos_key]} times. You are stuck in a loop. "
                f"Try a completely different route."
            )

        warning_block = "\n".join(warnings) + "\n" if warnings else ""

        user_prompt = (
            f"{warning_block}{history_block}\n{status_block}\n{obs_block}"
            '\n\nRespond with a single JSON object only: {"reasoning": "<your plan>", "action": "MOVE_NORTH"}'
        )

        # LLM call — retry once if the model forgets to include JSON
        _RETRY_PROMPT = (
            'You forgot to include a valid JSON object. Reply with ONLY the JSON, nothing else.\n'
            'Example: {"reasoning": "I need to go west to reach the modifier", "action": "MOVE_WEST"}\n'
            'Valid actions: MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST'
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
                # Retry once with an explicit reminder
                reply = client.generate(system_prompt, _RETRY_PROMPT)
                parsed = client.parse_json(reply)
            action_name = str(parsed.get("action", "")).upper().strip()
            # Reasoning is either in the JSON (old format) or plain text before the JSON
            reasoning = str(parsed.get("reasoning", "")).strip()
            if not reasoning:
                json_start = reply.rfind("{")
                reasoning = reply[:json_start].strip() if json_start > 0 else ""
        except Exception as exc:
            err = f"Turn {turn}: LLM/parse error — {exc}"
            result.errors.append(err)
            log({"type": "error", "summary": err})
            continue

        if action_name not in ACTION_MAP:
            err = f"Turn {turn}: unknown action '{action_name}'"
            result.errors.append(err)
            log({"type": "invalid_action", "summary": err, "raw_reply": reply})
            continue

        log({
            "type": "action",
            "summary": f"Turn {turn}: {action_name}" + (f" | {reasoning}" if reasoning else ""),
            "turn": turn, "action": action_name, "reasoning": reasoning, "state_before": curr_state,
        })

        prev_state = curr_state
        last_action_name = action_name
        _recent_actions.append(action_name)

        frame_data = env.step(ACTION_MAP[action_name])
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

    # Post-game understanding
    understanding_prompt = (
        (history_block or "") + "\n" + UNDERSTANDING_PROMPT
    )
    try:
        understanding_reply = client.generate(system_prompt, understanding_prompt)
        result.understanding = {
            k: str(v) for k, v in client.parse_json(understanding_reply).items()
        }
        log({"type": "understanding", "summary": "Agent explained goal and mechanics.",
             "understanding": result.understanding})
    except Exception as exc:
        result.errors.append(f"Understanding prompt failed: {exc}")

    result.history = history
    return result


# ── Human play mode ───────────────────────────────────────────────────────────

_KEY_TO_ACTION: dict[str, tuple[str, GameAction]] = {
    "\x1b[A": ("MOVE_NORTH", GameAction.ACTION1),
    "\x1b[B": ("MOVE_SOUTH", GameAction.ACTION2),
    "\x1b[D": ("MOVE_WEST",  GameAction.ACTION3),
    "\x1b[C": ("MOVE_EAST",  GameAction.ACTION4),
    "w": ("MOVE_NORTH", GameAction.ACTION1),
    "s": ("MOVE_SOUTH", GameAction.ACTION2),
    "a": ("MOVE_WEST",  GameAction.ACTION3),
    "d": ("MOVE_EAST",  GameAction.ACTION4),
    "k": ("MOVE_NORTH", GameAction.ACTION1),
    "j": ("MOVE_SOUTH", GameAction.ACTION2),
    "h": ("MOVE_WEST",  GameAction.ACTION3),
    "l": ("MOVE_EAST",  GameAction.ACTION4),
}


def _read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = os.read(fd, 1)
        if ch == b"\x1b":
            ch += os.read(fd, 2)
        return ch.decode("utf-8", errors="ignore")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def human_play(
    world_level: str = "EASY",
    goal_level: str = "EASY",
    mechanics_level: str = "EASY",
    feedback_level: str = "EASY",
) -> None:
    """
    Interactive human play mode.

    Shows exactly what an LLM agent would receive at each axis level so you
    can test the configuration before running the full experiment.
    Results are saved to results/ as model="human".
    Controls: arrow keys / WASD / hjkl = move   r = reset   q = quit
    """
    config = {
        "world": world_level, "goal": goal_level,
        "mechanics": mechanics_level, "feedback": feedback_level,
    }
    timestamp = datetime.now(timezone.utc).isoformat()
    history: list[dict[str, Any]] = []
    system_prompt = build_system_prompt(goal_level, mechanics_level)

    arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arc.make("ls20", render_mode=None)
    frame_data = env.observation_space

    step = 0
    prev_state: Optional[dict[str, Any]] = None
    feedback_line = ""
    first_render = True

    def render() -> None:
        nonlocal first_render
        if not frame_data:
            return
        curr = get_structured_state(env, frame_data)
        p, lvl = curr["player"], curr["level"]

        print(CLEAR, end="")
        print(
            f"=== Grid Navigator — Human Play  "
            f"[W:{world_level} G:{goal_level} M:{mechanics_level} F:{feedback_level}] ==="
        )
        print(
            f"Step {step:>4}  |  Level {lvl['current']}/{lvl['total']}  |  "
            f"Lives {p['lives']}  |  Stamina {p['stamina']}  |  {frame_data.state.name}"
        )
        print()

        if world_level == "EASY":
            print("── AGENT SEES: structured JSON ────────────────────────────────")
            print(json.dumps(curr, indent=2))
            print()

        print("── GRID (ASCII) ────────────────────────────────────────────────")
        if frame_data.frame:
            print(frame_to_ascii(frame_data.frame[-1]))
        if first_render:
            print(f"  {ASCII_LEGEND}")
            first_render = False

        if world_level == "HARD":
            print("\n  (Hard World: agent receives only the ASCII grid above, no JSON)")

        if feedback_line:
            print(f"\n── FEEDBACK ({feedback_level}) ─────────")
            print(f"  {feedback_line}")

        print("\n  Controls: arrow keys / WASD / hjkl = move   r = reset   q = quit")

    # Show system prompt so the human knows what the agent was told
    print(CLEAR, end="")
    print("=== SYSTEM PROMPT (what the LLM agent is told at the start) ===\n")
    print(system_prompt if system_prompt else "(nothing stated)")
    print("\nPress any key to start…")
    _read_key()
    render()

    while True:
        try:
            key = _read_key()
        except (KeyboardInterrupt, EOFError):
            break

        if key in ("q", "\x03"):
            break

        if key == "r":
            frame_data = env.step(GameAction.RESET)
            step = 0
            prev_state = None
            feedback_line = ""
            history.append({"type": "reset",
                             "timestamp": datetime.now(timezone.utc).isoformat()})
            render()
            continue

        entry = _KEY_TO_ACTION.get(key)
        if entry is None:
            continue

        action_name, game_action = entry
        state_before = get_structured_state(env, frame_data)
        step += 1
        frame_data = env.step(game_action)
        if frame_data is None:
            break
        state_after = get_structured_state(env, frame_data)

        feedback_line = (
            build_feedback_easy(state_before, state_after, action_name)
            if feedback_level == "EASY" else FEEDBACK_HARD
        )
        history.append({
            "type": "action",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "turn": step, "action": action_name,
            "state_before": state_before, "feedback": feedback_line,
        })
        prev_state = state_before
        render()

        if frame_data.state.name == "WIN":
            print("\n  YOU WIN! Press r to restart or q to quit.")
        elif frame_data.state.name == "GAME_OVER":
            print("\n  GAME OVER. Press r to restart or q to quit.")

    result = RunResult(
        config=config,
        won=(frame_data.state.name == "WIN") if frame_data else False,
        turns=step,
        levels_completed=frame_data.levels_completed if frame_data else 0,
        provider="human", model="human", timestamp=timestamp, history=history,
    )
    save_result(
        result,
        run_id=f"human_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
    )
    print("\nSession saved.")


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Environment 3 (Grid Navigator).")
    p.add_argument("--provider", default="openrouter",
                   choices=["openrouter"],
                   help="LLM provider (default: openrouter).")
    p.add_argument("--model",    default="meta-llama/llama-3.3-70b-instruct:free",
                   help="Model name for the chosen provider (default: gemini-2.0-flash).")
    p.add_argument("--world",     choices=["EASY", "HARD"], default="EASY")
    p.add_argument("--goal",      choices=["EASY", "HARD"], default="EASY")
    p.add_argument("--mechanics", choices=["EASY", "HARD"], default="EASY")
    p.add_argument("--feedback",  choices=["EASY", "HARD"], default="EASY")
    p.add_argument("--vision",    action="store_true",
                   help="Send rendered PNG images instead of ASCII for Hard World.")
    p.add_argument("--turns-per-level", type=int, default=TURNS_PER_LEVEL,
                   help="Turn budget per level (default: 100).")
    p.add_argument("--max-levels", type=int, default=None,
                   help="Stop after completing this many levels (default: play all levels).")
    p.add_argument("--human",     action="store_true",
                   help="Play yourself instead of running an LLM agent.")
    p.add_argument("--quiet",     action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.human:
        human_play(
            world_level=args.world,
            goal_level=args.goal,
            mechanics_level=args.mechanics,
            feedback_level=args.feedback,
        )
    else:
        print(
            f"Running: provider={args.provider}  model={args.model}  "
            f"world={args.world}  goal={args.goal}  "
            f"mechanics={args.mechanics}  feedback={args.feedback}  "
            f"vision={args.vision}"
        )
        result = run_agent(
            world_level=args.world,
            goal_level=args.goal,
            mechanics_level=args.mechanics,
            feedback_level=args.feedback,
            provider=args.provider,
            model=args.model,
            vision=args.vision,
            turns_per_level=args.turns_per_level,
            max_levels=args.max_levels,
            verbose=not args.quiet,
        )
        save_result(result)
        outcome = "WIN" if result.won else "LOSS"
        print(f"\nResult: {outcome}  |  {result.turns} turns  |  "
              f"{result.levels_completed} levels completed")
        if result.errors:
            print(f"Errors: {result.errors}")
