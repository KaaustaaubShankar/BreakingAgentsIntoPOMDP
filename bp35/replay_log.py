"""
replay_log.py - Replay a saved BP35 run log from env4/results.

Usage:
  python replay_log.py --log results/run_fE_gE_mE_wE_20260420T222144.json
  python replay_log.py --log results/run_...json --delay 0.2 --verify-state
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import arc_agi
from arc_agi import OperationMode
from arcengine import FrameDataRaw, GameAction

from game_interface import ASCII_LEGEND, frame_to_ascii, get_structured_state

CLEAR = "\033[H\033[2J"

ACTION_MAP: dict[str, GameAction] = {
    "MOVE_LEFT": GameAction.ACTION3,
    "MOVE_RIGHT": GameAction.ACTION4,
    "UNDO": GameAction.ACTION7,
    "RESET": GameAction.RESET,
}


def _grid_to_click_data(env: Any, position: list[int]) -> dict[str, int]:
    game = env._game.oztjzzyqoek
    grid_x, grid_y = int(position[0]), int(position[1])
    cell_w, cell_h = game.hdnrlfmyrj.qmcjluiioz
    camera_y = int(game.camera.rczgvgfsfb[1])
    return {
        "x": grid_x * cell_w + cell_w // 2,
        "y": grid_y * cell_h - camera_y + cell_h // 2,
    }


def _render(step: int, frame_data: FrameDataRaw, env: Any, clear: bool) -> None:
    if not frame_data.frame:
        return

    if clear:
        print(CLEAR, end="")

    state = get_structured_state(env, frame_data)
    level = state["level"]
    player = state["player"]
    resources = state["resources"]

    print(
        f"Step {step:>4}  |  State: {frame_data.state.name:<12}  |  "
        f"Level: {level['current']}/{level['total']}"
    )
    print(
        f"Player: {player['position']}  |  Gravity: {player['gravity']:<4}  |  "
        f"Steps: {resources['steps_used']}/{resources['step_budget']}"
    )
    print()
    print(frame_to_ascii(frame_data.frame[-1]))
    print()
    print("  " + ASCII_LEGEND)


def _load_actions(log_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(log_path.read_text())
    history = payload.get("history", [])
    actions = [entry for entry in history if entry.get("type") == "action"]
    actions.sort(key=lambda e: int(e.get("turn", 10**9)))
    return actions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay an env4 BP35 run log.")
    parser.add_argument("--log", required=True, help="Path to run_*.json")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.15,
        help="Seconds to wait between replayed turns (default: 0.15)",
    )
    parser.add_argument("--max-turns", type=int, default=None, help="Replay at most N turns")
    parser.add_argument(
        "--verify-state",
        action="store_true",
        help="Check replay state against logged state_before and print mismatches.",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear terminal between frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    actions = _load_actions(log_path)
    if args.max_turns is not None:
        actions = actions[: args.max_turns]
    if not actions:
        print("No action entries found in log history.")
        return

    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make("bp35", render_mode=None)
    frame_data = env.observation_space
    if frame_data is None:
        print("Failed to initialize BP35 environment.")
        return

    print(f"Replaying {len(actions)} action(s) from {log_path}")
    _render(0, frame_data, env, clear=not args.no_clear)

    for i, entry in enumerate(actions, start=1):
        action_name = str(entry.get("action", "")).upper().strip()
        if action_name == "CLICK":
            target_position = entry.get("target_position")
            if not isinstance(target_position, list) or len(target_position) != 2:
                print(f"[turn {i}] Skipping CLICK: missing/invalid target_position")
                continue
            try:
                action = GameAction.ACTION6
                data = _grid_to_click_data(env, [int(target_position[0]), int(target_position[1])])
            except Exception as exc:
                print(f"[turn {i}] Skipping CLICK at {target_position!r}: {exc}")
                continue
            print(f"[turn {i}] CLICK {target_position}")
            next_frame = env.step(action, data=data)
        else:
            action = ACTION_MAP.get(action_name)
            if action is None:
                print(f"[turn {i}] Skipping unknown action: {action_name!r}")
                continue
            print(f"[turn {i}] {action_name}")
            next_frame = env.step(action)

        if next_frame is None:
            print(f"[turn {i}] env.step returned None, stopping replay.")
            break

        if args.verify_state:
            replay_state = get_structured_state(env, next_frame)
            logged_before = entry.get("state_before")
            if isinstance(logged_before, dict):
                if replay_state["level"]["current"] < logged_before.get("level", {}).get("current", 1):
                    print(
                        f"[turn {i}] WARNING: level drift (replay={replay_state['level']['current']}, "
                        f"logged_before={logged_before.get('level', {}).get('current')})"
                    )

        frame_data = next_frame
        _render(i, frame_data, env, clear=not args.no_clear)

        if frame_data.state.name in ("WIN", "GAME_OVER"):
            print(f"Replay ended with {frame_data.state.name} at turn {i}.")
            break

        if args.delay > 0:
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
