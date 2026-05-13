"""Smoke-test env.step(GameAction.RESET) on ka59simple.

Verifies that RESET on a single-level fork restarts the level in-place
(returns a valid FrameDataRaw with the agent back at the start position)
rather than transitioning to a "game complete" state.

If RESET works: per-spec Task 2 can use env.step(RESET) between attempts.
If RESET fails: Task 2 needs to recreate the env between attempts.

Run: python3 -m scripts.smoke_test_ka59simple_reset
"""
from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parents[1]
sys.path.insert(0, str(repo_root))

from arcengine import GameAction
from ka59_game.experiment import _make_env
from ka59_game.game_interface import get_structured_state

env = _make_env("ka59simple")
frame_data = env.observation_space
state0 = get_structured_state(env, frame_data, 1, 64)
print("INITIAL state:")
print(f"  player.position = {state0['player']['position']}")
print(f"  player.id       = {state0['player']['id']}")
print(f"  level.current   = {state0['level']['current']}")
print(f"  game_state      = {state0['game_state']}")
print(f"  frame state     = {frame_data.state.name}")
print(f"  levels_completed= {frame_data.levels_completed}")

print("\nTaking 3 MOVE_LEFT actions to shift state...")
for i in range(3):
    frame_data = env.step(GameAction.ACTION3)  # MOVE_LEFT
    if frame_data is None:
        print(f"  step {i+1}: env.step returned None")
        sys.exit(1)

state_mid = get_structured_state(env, frame_data, 4, 64)
print(f"AFTER 3 LEFT moves:")
print(f"  player.position = {state_mid['player']['position']}")
print(f"  frame state     = {frame_data.state.name}")

print("\nCalling env.step(GameAction.RESET)...")
try:
    frame_data_reset = env.step(GameAction.RESET)
except Exception as exc:
    print(f"RESET raised: {type(exc).__name__}: {exc}")
    sys.exit(2)

if frame_data_reset is None:
    print("RESET returned None — recreate-env path needed in Task 2.")
    sys.exit(3)

state_after = get_structured_state(env, frame_data_reset, 1, 64)
print(f"AFTER RESET:")
print(f"  player.position = {state_after['player']['position']}")
print(f"  player.id       = {state_after['player']['id']}")
print(f"  level.current   = {state_after['level']['current']}")
print(f"  game_state      = {state_after['game_state']}")
print(f"  frame state     = {frame_data_reset.state.name}")
print(f"  levels_completed= {frame_data_reset.levels_completed}")

if state_after['player']['position'] == state0['player']['position']:
    print("\nRESET WORKS — env.step(GameAction.RESET) restores initial position.")
    print("Proceed with Task 2 using env.step(RESET) between attempts.")
    sys.exit(0)
else:
    print("\nRESET DID NOT RESTORE initial position. Task 2 needs env recreation.")
    sys.exit(4)
