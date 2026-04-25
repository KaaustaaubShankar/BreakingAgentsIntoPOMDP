"""
ASI-Evolve evaluator for KA59.

Usage (called by eval.sh):
    python evaluator.py <candidate_path>

Loads the candidate's agent_step() function, runs it against the real KA59
game via ka59_game/experiment.py infrastructure, returns a JSON score.

Score = average levels_completed across TRIALS runs (max 7 per trial).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import os
from pathlib import Path

# Ensure the repo root is on the path so ka59_game imports work
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import arc_agi
from arc_agi import OperationMode
from arcengine import FrameDataRaw, GameAction

from ka59_game.game_interface import (
    ACTION_MAP,
    build_feedback_easy,
    get_structured_state,
)
from ka59_game.experiment import DEFAULT_TURNS_PER_LEVEL, MAX_LEVELS_HARD_CAP

TRIALS = 3  # average over this many episodes
STEP_CAP = DEFAULT_TURNS_PER_LEVEL  # 64 per level


def load_candidate(path: str):
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent_step


def run_episode(agent_step_fn) -> dict:
    """Run one full KA59 game, return metrics."""
    env = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE).make("ka59", render_mode=None)
    fd: FrameDataRaw = env.observation_space

    history: list[dict] = []
    levels_completed = 0
    total_turns = 0
    won = False

    turns_this_level = 0
    state = get_structured_state(env, fd, 0, STEP_CAP)

    while state["game_state"] == "IN_PROGRESS":
        try:
            action_dict = agent_step_fn(state, history)
        except Exception as e:
            action_dict = {"action": "MOVE_RIGHT"}

        action_name = action_dict.get("action", "MOVE_RIGHT")
        prev_state = state

        if action_name == "CLICK":
            tpos = action_dict.get("target_position")
            if tpos:
                from ka59_game.experiment import _grid_to_click_data
                ga = GameAction(action_id=ACTION_MAP["CLICK"], data=_grid_to_click_data(tpos))
            else:
                ga = GameAction(action_id=ACTION_MAP.get("MOVE_RIGHT", 1))
        else:
            action_id = ACTION_MAP.get(action_name, ACTION_MAP["MOVE_RIGHT"])
            ga = GameAction(action_id=action_id)

        fd = env.step(ga)
        state = get_structured_state(env, fd, turns_this_level + 1, STEP_CAP - turns_this_level - 1)

        moved = (state["player"]["position"] != prev_state["player"]["position"])
        history.append({"state": prev_state, "action": action_dict, "moved": moved})
        total_turns += 1
        turns_this_level += 1

        game_state = state["game_state"]
        if game_state == "WIN":
            won = True
            break
        if game_state == "LEVEL_COMPLETE" or turns_this_level >= STEP_CAP:
            if game_state == "LEVEL_COMPLETE":
                levels_completed += 1
            turns_this_level = 0

        if total_turns > MAX_LEVELS_HARD_CAP * STEP_CAP + 50:
            break

    return {
        "won": won,
        "levels_completed": levels_completed + (1 if won else 0),
        "total_turns": total_turns,
        "score": float(levels_completed + (1 if won else 0)),
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"score": 0.0, "error": "no candidate path provided"}))
        sys.exit(1)

    candidate_path = sys.argv[1]
    try:
        agent_fn = load_candidate(candidate_path)
    except Exception as e:
        print(json.dumps({"score": 0.0, "error": f"load failed: {e}"}))
        sys.exit(1)

    scores = []
    all_metrics = []
    for trial in range(TRIALS):
        try:
            result = run_episode(agent_fn)
            scores.append(result["score"])
            all_metrics.append(result)
        except Exception as e:
            scores.append(0.0)
            all_metrics.append({"score": 0.0, "error": str(e)})

    avg_score = sum(scores) / len(scores)
    print(json.dumps({
        "score": avg_score,
        "metrics": {
            "trials": all_metrics,
            "avg_levels_completed": avg_score,
            "max_possible": float(MAX_LEVELS_HARD_CAP),
        }
    }))


if __name__ == "__main__":
    main()
