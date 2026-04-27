"""
play.py — ARC-AGI BP35 runner

Modes:
  python play.py           — agent mode (runs a short stub sequence)
  python play.py --human   — human mode (keyboard control + click prompt)
"""

from __future__ import annotations

import os
import sys
import termios
import tty
from typing import Any, Optional

import arc_agi
from arc_agi import OperationMode
from arcengine import ActionInput, FrameDataRaw, GameAction

from game_interface import ASCII_LEGEND, frame_to_ascii, get_structured_state

CLEAR = "\033[H\033[2J"

LEGEND = (
    "  Controls: left/right arrows or A/D = move   u = undo   c = click   "
    "r = reset   q / Ctrl-C = quit"
)


def render_ascii(step: int, frame_data: FrameDataRaw, env, clear: bool = False) -> None:
    """Render the current BP35 frame and structured summary to stdout."""
    if not frame_data.frame:
        return

    if clear:
        print(CLEAR, end="")

    state = get_structured_state(env, frame_data)
    frame = frame_to_ascii(frame_data.frame[-1])
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
    print(frame)
    print()
    print("  " + ASCII_LEGEND)

    clickable = [item["position"] for item in state["objects"]["clickable_tiles"]]
    print(f"  Goals: {state['objects']['goals']}")
    print(f"  Clickable tiles: {clickable}")
    print(LEGEND)


def _read_key() -> str:
    """Read a single keypress or escape sequence from stdin."""
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


def _prompt_line(prompt: str) -> str:
    """Temporarily leave raw mode and read a normal input line."""
    print()
    return input(prompt)


def _grid_to_click_data(env, position: tuple[int, int]) -> dict[str, int]:
    game = env._game.oztjzzyqoek
    grid_x, grid_y = position
    cell_w, cell_h = game.hdnrlfmyrj.qmcjluiioz
    camera_y = int(game.camera.rczgvgfsfb[1])
    return {
        "x": grid_x * cell_w + cell_w // 2,
        "y": grid_y * cell_h - camera_y + cell_h // 2,
    }


def _safe_env_step(
    env: Any,
    action: GameAction | ActionInput,
    data: Optional[dict[str, int]] = None,
) -> Optional[FrameDataRaw]:
    """Mirror the experiment runner's safe handling for prebuilt ActionInput."""
    if isinstance(action, ActionInput):
        frame_data = env._game.perform_action(action, raw=True)
        frame_data.guid = env._guid
        frame_data.game_id = env.environment_info.game_id
        env._set_last_response(frame_data, reasoning=action.reasoning)
        return frame_data
    return env.step(action, data=data)


_KEY_TO_ACTION: dict[str, GameAction] = {
    "\x1b[D": GameAction.ACTION3,
    "\x1b[C": GameAction.ACTION4,
    "a": GameAction.ACTION3,
    "d": GameAction.ACTION4,
    "h": GameAction.ACTION3,
    "l": GameAction.ACTION4,
    "u": GameAction.ACTION7,
}


def human_mode(arc: arc_agi.Arcade) -> None:
    env = arc.make("bp35", render_mode=None)
    step = 0

    def show(frame_data: FrameDataRaw) -> None:
        render_ascii(step, frame_data, env, clear=True)

    if env.observation_space:
        show(env.observation_space)

    print("\nPress any movement key to begin...")

    while True:
        try:
            key = _read_key()
        except (KeyboardInterrupt, EOFError):
            break

        if key in ("q", "\x03"):
            break

        if key == "r":
            result = _safe_env_step(env, GameAction.RESET)
            step = 0
            if result:
                show(result)
            continue

        if key == "c":
            try:
                raw = _prompt_line("Click target as 'x y': ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not raw:
                show(env.observation_space)
                continue
            try:
                x_str, y_str = raw.split()
                data = _grid_to_click_data(env, (int(x_str), int(y_str)))
            except ValueError:
                print("Invalid click target. Use: x y")
                continue
            step += 1
            result = _safe_env_step(env, GameAction.ACTION6, data=data)
            if result:
                show(result)
                if result.state.name in ("WIN", "GAME_OVER"):
                    print(
                        f"\n{'You win!' if result.state.name == 'WIN' else 'Game over.'} "
                        "Press r to restart or q to quit."
                    )
            continue

        action = _KEY_TO_ACTION.get(key)
        if action is None:
            continue

        step += 1
        result = _safe_env_step(env, action)
        if result:
            show(result)
            if result.state.name in ("WIN", "GAME_OVER"):
                print(
                    f"\n{'You win!' if result.state.name == 'WIN' else 'Game over.'} "
                    "Press r to restart or q to quit."
                )


def agent_mode(arc: arc_agi.Arcade) -> None:
    env = arc.make("bp35", render_mode=None)
    stub_actions = [
        GameAction.ACTION4,
        GameAction.ACTION4,
        GameAction.ACTION3,
    ]
    result = env.observation_space
    if result:
        render_ascii(0, result, env, clear=True)
    for i, action in enumerate(stub_actions, start=1):
        result = _safe_env_step(env, action)
        if result:
            render_ascii(i, result, env, clear=True)
        if result and result.state.name in ("WIN", "GAME_OVER"):
            break


if __name__ == "__main__":
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)

    if "--human" in sys.argv:
        human_mode(arcade)
    else:
        agent_mode(arcade)
