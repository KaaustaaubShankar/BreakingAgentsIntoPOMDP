"""
play.py — ARC-AGI LS20 runner

Modes:
  python play.py           — agent mode (runs ACTION1 × 10 as a stub)
  python play.py --human   — human mode (keyboard control)
"""

import sys
import os
import tty
import termios

import arc_agi
from arc_agi import OperationMode
from arcengine import FrameDataRaw, GameAction

# Color index → ASCII character
ASCII_MAP: dict[int, str] = {
    5: " ",   # Black (background)
    4: ".",   # Off-black
    3: ":",   # Dark gray
    2: "+",   # Mid gray
    1: "=",   # Light gray
    0: "#",   # White (walls/solid)
    6: "M",   # Magenta
    7: "m",   # Light magenta
    8: "R",   # Red
    9: "B",   # Blue
    10: "b",  # Light blue
    11: "Y",  # Yellow
    12: "O",  # Orange
    13: "X",  # Maroon
    14: "G",  # Green
    15: "P",  # Purple
}

CLEAR = "\033[H\033[2J"

LEGEND = (
    "  Legend: # wall  . floor  B blue  G green  Y yellow  O orange  R red  "
    "M magenta  P purple\n"
    "  Controls: arrow keys / WASD = move   r = reset   q / Ctrl-C = quit"
)


def render_ascii(step: int, frame_data: FrameDataRaw, clear: bool = False) -> None:
    """Render the last frame of a FrameDataRaw as ASCII to stdout."""
    if not frame_data.frame:
        return

    frame = frame_data.frame[-1]
    height, width = frame.shape

    if clear:
        print(CLEAR, end="")

    state = frame_data.state.name
    levels = f"{frame_data.levels_completed}/{frame_data.win_levels}"
    print(f"Step {step:>4}  |  State: {state:<12}  |  Levels: {levels}")
    print("+" + "-" * width + "+")
    for y in range(height):
        row = "".join(ASCII_MAP.get(int(frame[y, x]), "?") for x in range(width))
        print(f"|{row}|")
    print("+" + "-" * width + "+")


# ── keyboard helpers ──────────────────────────────────────────────────────────

def _read_key() -> str:
    """Read a single keypress (or escape sequence) from stdin, raw mode."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = os.read(fd, 1)
        if ch == b"\x1b":          # possible escape sequence
            rest = os.read(fd, 2)
            ch += rest
        return ch.decode("utf-8", errors="ignore")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


_KEY_TO_ACTION: dict[str, GameAction] = {
    # Arrow keys
    "\x1b[A": GameAction.ACTION1,   # up
    "\x1b[B": GameAction.ACTION2,   # down
    "\x1b[D": GameAction.ACTION3,   # left
    "\x1b[C": GameAction.ACTION4,   # right
    # WASD
    "w": GameAction.ACTION1,
    "s": GameAction.ACTION2,
    "a": GameAction.ACTION3,
    "d": GameAction.ACTION4,
    # Vi keys
    "k": GameAction.ACTION1,
    "j": GameAction.ACTION2,
    "h": GameAction.ACTION3,
    "l": GameAction.ACTION4,
}


# ── modes ─────────────────────────────────────────────────────────────────────

def human_mode(arc: arc_agi.Arcade) -> None:
    env = arc.make("ls20", render_mode=None)

    step = 0

    def show(frame_data: FrameDataRaw) -> None:
        render_ascii(step, frame_data, clear=True)
        print(LEGEND)

    # Show initial state (reset gives us the first frame)
    if env.observation_space:
        show(env.observation_space)

    print("\nPress any movement key to begin…")

    while True:
        try:
            key = _read_key()
        except (KeyboardInterrupt, EOFError):
            break

        if key in ("q", "\x03"):       # q or Ctrl-C
            break

        if key == "r":
            result = env.step(GameAction.RESET)
            step = 0
            if result:
                show(result)
            continue

        action = _KEY_TO_ACTION.get(key)
        if action is None:
            continue

        step += 1
        result = env.step(action)
        if result:
            show(result)
            if result.state.name in ("WIN", "GAME_OVER"):
                print(f"\n{'You win!' if result.state.name == 'WIN' else 'Game over.'}"
                      "  Press r to restart or q to quit.")


def agent_mode(arc: arc_agi.Arcade) -> None:
    env = arc.make("ls20", render_mode=None, renderer=render_ascii)

    for _ in range(10):
        env.step(GameAction.ACTION1)

    print(arc.get_scorecard())


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)

    if "--human" in sys.argv:
        human_mode(arc)
    else:
        agent_mode(arc)
