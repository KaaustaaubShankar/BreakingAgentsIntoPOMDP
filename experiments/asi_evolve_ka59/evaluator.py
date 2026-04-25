"""
ASI-Evolve evaluator for KA59.

Usage (called by eval.sh):
    python evaluator.py <candidate_path>

Loads the candidate's agent_step() function, runs it against the real KA59
game, returns a JSON score.

Score = average levels_completed across TRIALS runs (max 7 per trial).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

# Ensure the repo root is on the path so ka59_game imports work
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import arc_agi
from arc_agi import OperationMode

from ka59_game.game_interface import (
    ACTION_MAP,
    build_feedback_easy,
    get_structured_state,
)
from ka59_game.experiment import (
    DEFAULT_TURNS_PER_LEVEL,
    MAX_LEVELS_HARD_CAP,
    _grid_to_click_data,
    _make_env,
)

TRIALS = 3


def load_candidate(path: str):
    loader = SourceFileLoader("candidate", path)
    spec = importlib.util.spec_from_loader("candidate", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod.agent_step


def run_episode(agent_step_fn) -> dict:
    """Run one full KA59 game, return metrics."""
    env = _make_env()
    frame_data = env.observation_space
    if frame_data is None:
        return {"score": 0.0, "error": "observation_space is None"}

    levels_to_play = min(int(frame_data.win_levels), MAX_LEVELS_HARD_CAP)
    turns_per_level = DEFAULT_TURNS_PER_LEVEL
    max_turns = levels_to_play * turns_per_level

    history: list[dict] = []
    levels_completed = 0
    won = False

    last_level_index = 0
    turn_in_level = 0

    for turn in range(1, max_turns + 1):
        curr_level_index = int(env._game.level_index)
        if curr_level_index != last_level_index:
            turn_in_level = 0
            last_level_index = curr_level_index
        turn_in_level += 1

        curr_state = get_structured_state(env, frame_data, turn_in_level, turns_per_level)
        game_state = curr_state.get("game_state", "IN_PROGRESS")

        if game_state not in ("IN_PROGRESS", "LEVEL_COMPLETE"):
            if game_state == "WIN":
                won = True
                levels_completed = levels_to_play
            break

        try:
            action_dict = agent_step_fn(curr_state, history)
        except Exception:
            action_dict = {"action": "MOVE_RIGHT"}

        action_name = str(action_dict.get("action", "MOVE_RIGHT")).upper().strip()
        action_data = None
        game_action = None

        if action_name == "CLICK":
            tpos = action_dict.get("target_position")
            if isinstance(tpos, (list, tuple)) and len(tpos) == 2:
                try:
                    gx, gy = int(tpos[0]), int(tpos[1])
                    game_action = ACTION_MAP["CLICK"]
                    action_data = _grid_to_click_data(env, gx, gy)
                except Exception:
                    game_action = ACTION_MAP["MOVE_RIGHT"]
            else:
                game_action = ACTION_MAP["MOVE_RIGHT"]
        elif action_name in ACTION_MAP:
            game_action = ACTION_MAP[action_name]
        else:
            game_action = ACTION_MAP["MOVE_RIGHT"]

        prev_state = curr_state
        try:
            frame_data = env.step(game_action, data=action_data)
        except Exception:
            break

        if frame_data is None:
            break

        new_state = get_structured_state(env, frame_data, turn_in_level, turns_per_level)
        moved = (new_state["player"]["position"] != prev_state["player"]["position"])

        history.append({"state": prev_state, "action": action_dict, "moved": moved})

        new_game_state = new_state.get("game_state", "IN_PROGRESS")
        if new_game_state == "WIN":
            won = True
            levels_completed = levels_to_play
            break
        if new_game_state == "LEVEL_COMPLETE":
            levels_completed += 1
            turn_in_level = 0

    return {
        "won": won,
        "levels_completed": levels_completed,
        "total_turns": len(history),
        "score": float(levels_completed),
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
    for _ in range(TRIALS):
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
