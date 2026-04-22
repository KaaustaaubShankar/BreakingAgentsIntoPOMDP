"""
play.py — KA59 runner for manual sanity-checking and stub agent smoke tests.

Modes:
  python -m ka59_ref.play                      — stub agent sequence
  python -m ka59_ref.play --human              — interactive line-mode human play
  python -m ka59_ref.play --scenario <name>    — choose a scenario

This is the KA59 analogue of env4/play.py. We use line-mode input instead of
termios raw-mode because KA59 scenarios are small and step-count is tiny —
line editing is more useful than single-keystroke play.
"""

from __future__ import annotations

import argparse
import sys

from .engine import STEP
from .env import KA59BlindEnv, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, SELECT, Action
from .experiment import SCENARIO_GOALS
from .game_interface import build_feedback_easy, get_structured_state
from .scenarios import SCENARIOS


LEGEND = (
    "  P=selected  B=block  #=wall  .=empty    Commands: l/r/u/d=move   "
    "s <id>=select   p=print full state   q=quit"
)

WORD_ACTIONS: dict[str, Action] = {
    "l": MOVE_LEFT,
    "left": MOVE_LEFT,
    "MOVE_LEFT": MOVE_LEFT,
    "r": MOVE_RIGHT,
    "right": MOVE_RIGHT,
    "MOVE_RIGHT": MOVE_RIGHT,
    "u": MOVE_UP,
    "up": MOVE_UP,
    "MOVE_UP": MOVE_UP,
    "d": MOVE_DOWN,
    "down": MOVE_DOWN,
    "MOVE_DOWN": MOVE_DOWN,
}


def _render(state: dict, prev_state: dict | None, action_name: str | None) -> None:
    print("-" * 72)
    print(
        f"scenario={state['level']['scenario']}  "
        f"player={state['player']['id']}@{state['player']['position']}  "
        f"target_x={state['level']['target_x']}  "
        f"steps_remaining={state['resources']['steps_remaining']}  "
        f"game_state={state['game_state']}"
    )
    for row in state["semantic_grid"]:
        print(f"  {row}")
    if prev_state is not None and action_name is not None:
        print()
        print("  " + build_feedback_easy(prev_state, state, action_name))
    print(LEGEND)


def human_mode(scenario: str) -> None:
    if scenario not in SCENARIOS:
        print(f"Unknown scenario: {scenario}. Choices: {sorted(SCENARIOS.keys())}")
        return
    spec = SCENARIOS[scenario]
    target_x = SCENARIO_GOALS.get(scenario, STEP * 2)
    env = KA59BlindEnv()
    env.reset(spec)

    prev_state: dict | None = None
    last_action_name: str | None = None

    state = get_structured_state(env, scenario, spec["steps"], target_x)
    _render(state, None, None)

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

        action: Action | None = None
        action_name = raw
        if raw.startswith("s "):
            target_id = raw[2:].strip()
            try:
                action = SELECT(target_id)
                action_name = f"SELECT {target_id}"
            except Exception as exc:  # noqa: BLE001
                print(f"bad select: {exc}")
                continue
        elif raw in WORD_ACTIONS:
            action = WORD_ACTIONS[raw]
            action_name = raw.upper() if raw.upper().startswith("MOVE_") else f"MOVE_{raw.upper()}"
        else:
            print(f"unknown command: {raw!r}")
            continue

        try:
            env.step(action)
        except (KeyError, ValueError) as exc:
            print(f"env error: {exc}")
            continue

        prev_state = state
        last_action_name = action_name
        state = get_structured_state(env, scenario, spec["steps"], target_x)
        _render(state, prev_state, last_action_name)

    print()
    print(f"Final: game_state={state['game_state']}  steps_remaining={state['resources']['steps_remaining']}")


def agent_mode(scenario: str) -> None:
    """Stub agent: MOVE_RIGHT until win, scenario exhausted, or 20 turns."""
    if scenario not in SCENARIOS:
        print(f"Unknown scenario: {scenario}. Choices: {sorted(SCENARIOS.keys())}")
        return
    spec = SCENARIOS[scenario]
    target_x = SCENARIO_GOALS.get(scenario, STEP * 2)
    env = KA59BlindEnv()
    env.reset(spec)

    state = get_structured_state(env, scenario, spec["steps"], target_x)
    _render(state, None, None)

    for turn in range(1, min(20, spec["steps"]) + 1):
        if state["game_state"] != "IN_PROGRESS":
            break
        action_name = "MOVE_RIGHT"
        env.step(MOVE_RIGHT)
        prev = state
        state = get_structured_state(env, scenario, spec["steps"], target_x)
        print(f"\nTurn {turn}: {action_name}")
        _render(state, prev, action_name)

    print()
    print(f"Final: game_state={state['game_state']}  steps_remaining={state['resources']['steps_remaining']}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KA59 manual runner.")
    p.add_argument("--scenario", default="transfer_wall_push", choices=sorted(SCENARIOS.keys()))
    p.add_argument("--human", action="store_true", help="Interactive line-mode play")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.human:
        human_mode(args.scenario)
    else:
        agent_mode(args.scenario)


if __name__ == "__main__":
    main()
