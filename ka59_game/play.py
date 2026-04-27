"""
play.py — Manual runner for the real KA59 game.

  python -m ka59_game.play           agent stub (right-biased) for ~20 turns
  python -m ka59_game.play --human   line-mode interactive play
"""

from __future__ import annotations

import argparse
import sys

import arc_agi
from arc_agi import OperationMode
from arcengine import FrameDataRaw, GameAction

from .experiment import DEFAULT_TURNS_PER_LEVEL, _grid_to_click_data
from .game_interface import ACTION_MAP, build_feedback_easy, get_structured_state


LEGEND = (
    "  P=selected  S=other selectable  B=block  #=wall  +=goal  .=empty    "
    "Commands: l/r/u/d=move   c <x> <y>=click cell   p=print state   q=quit"
)


def _make_env():
    return arc_agi.Arcade(operation_mode=OperationMode.OFFLINE).make("ka59", render_mode=None)


def _render(state: dict, prev: dict | None, last_action: str | None) -> None:
    print("-" * 72)
    print(
        f"player={state['player']['id']}@{state['player']['position']}  "
        f"level={state['level']['current']}/{state['level']['total']}  "
        f"steps_left={state['resources']['steps_remaining']}  "
        f"game_state={state['game_state']}"
    )
    print("semantic_grid:")
    for row in state["semantic_grid"]:
        print(f"  {row}")
    if prev is not None and last_action is not None:
        print()
        print("  " + build_feedback_easy(prev, state, last_action))
    print(LEGEND)


def human_mode() -> None:
    env = _make_env()
    fd: FrameDataRaw = env.observation_space
    turn_in_level = 0
    budget = DEFAULT_TURNS_PER_LEVEL

    state = get_structured_state(env, fd, turn_in_level, budget)
    _render(state, None, None)

    last_action_name: str | None = None
    prev: dict | None = None

    while state["game_state"] == "IN_PROGRESS":
        try:
            raw = input("action> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        if raw in ("q", "quit", "exit"):
            break
        if raw in ("p", "print"):
            import json as _json
            print(_json.dumps(state, indent=2))
            continue

        action = None
        data = None
        action_name = raw
        if raw in ("l", "left"):
            action = ACTION_MAP["MOVE_LEFT"]
            action_name = "MOVE_LEFT"
        elif raw in ("r", "right"):
            action = ACTION_MAP["MOVE_RIGHT"]
            action_name = "MOVE_RIGHT"
        elif raw in ("u", "up"):
            action = ACTION_MAP["MOVE_UP"]
            action_name = "MOVE_UP"
        elif raw in ("d", "down"):
            action = ACTION_MAP["MOVE_DOWN"]
            action_name = "MOVE_DOWN"
        elif raw.startswith("c "):
            parts = raw.split()
            if len(parts) != 3:
                print("usage: c <grid_x> <grid_y>  (same units as player.position)")
                continue
            try:
                gx, gy = int(parts[1]), int(parts[2])
            except ValueError:
                print("x and y must be integers")
                continue
            action = GameAction.ACTION6
            data = _grid_to_click_data(env, gx, gy)
            action_name = f"CLICK {[gx, gy]}"
        else:
            print(f"unknown command: {raw!r}")
            continue

        fd = env.step(action, data=data)
        turn_in_level += 1
        prev = state
        last_action_name = action_name
        state = get_structured_state(env, fd, turn_in_level, budget)
        _render(state, prev, last_action_name)

    print()
    print(f"Final: game_state={state['game_state']}  levels_completed={state['level']['current']-1}")


def agent_stub() -> None:
    env = _make_env()
    fd: FrameDataRaw = env.observation_space
    turn_in_level = 0
    budget = DEFAULT_TURNS_PER_LEVEL
    state = get_structured_state(env, fd, turn_in_level, budget)
    _render(state, None, None)

    for turn in range(1, 21):
        if state["game_state"] != "IN_PROGRESS":
            break
        fd = env.step(ACTION_MAP["MOVE_RIGHT"])
        turn_in_level += 1
        prev = state
        state = get_structured_state(env, fd, turn_in_level, budget)
        print(f"\nTurn {turn}: MOVE_RIGHT")
        _render(state, prev, "MOVE_RIGHT")

    print()
    print(f"Final: game_state={state['game_state']}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-KA59 manual runner.")
    p.add_argument("--human", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.human:
        human_mode()
    else:
        agent_stub()


if __name__ == "__main__":
    main()
