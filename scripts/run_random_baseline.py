"""
Uniform-random agent baseline for ka59simple.

No LLM. Each turn, picks uniformly from {MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, CLICK}.
For CLICK, picks a random target_position uniformly over the cell grid.

Purpose: floor reference for the win-rate tables in the paper. A reviewer needs to know
what fraction of episodes a uniform-random policy wins to calibrate model performance.

Usage:
    python3 -m scripts.run_random_baseline --trials 50 --max-turns 32 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(__file__).parents[1]
sys.path.insert(0, str(repo_root))

import os
from arcengine import GameAction
import arc_agi
from arc_agi import OperationMode

from ka59_game.game_interface import ACTION_MAP, get_structured_state


def make_env(env_id: str = "ka59simple"):
    env_dir = repo_root / "environment_files"
    os.environ["KA59_BASE_DIR"] = str(env_dir / "ka59" / "38d34dbb")
    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE,
                            environments_dir=str(env_dir))
    return arcade.make(env_id, render_mode=None)


def grid_to_click_data(env, grid_x: int, grid_y: int) -> dict:
    cam = env._game.camera
    offset_x = (cam.MAX_DIMENSION - cam.width) // 2
    offset_y = (cam.MAX_DIMENSION - cam.height) // 2
    return {"x": int(grid_x) + int(cam.x) + offset_x,
            "y": int(grid_y) + int(cam.y) + offset_y}


def run_one_trial(env_id: str, max_turns: int, rng: random.Random) -> dict:
    env = make_env(env_id)
    frame = env.observation_space
    if frame is None:
        return {"won": False, "turns": 0, "error": "obs_space=None"}

    cam = env._game.camera
    grid_w = max(1, cam.width // 3)
    grid_h = max(1, cam.height // 3)

    move_actions = ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"]
    actions = move_actions + ["CLICK"]

    won = False
    turns = 0
    wall_transfers = 0
    object_pushes = 0
    moves_blocked = 0
    click_actions = 0
    max_goals_occupied = 0

    prev_state = get_structured_state(env, frame, 1, max_turns)
    for turn in range(1, max_turns + 1):
        turns = turn
        a = rng.choice(actions)
        if a == "CLICK":
            gx = rng.randint(0, grid_w - 1) * 3
            gy = rng.randint(0, grid_h - 1) * 3
            data = grid_to_click_data(env, gx, gy)
            game_action = GameAction.ACTION6
            click_actions += 1
        else:
            game_action = ACTION_MAP[a]
            data = None

        try:
            frame = env.step(game_action, data=data)
        except Exception as exc:
            return {"won": won, "turns": turns, "error": f"step:{exc}"}
        if frame is None:
            break

        if a in move_actions:
            post_state = get_structured_state(env, frame, turn, max_turns)
            if post_state["player"]["position"] == prev_state["player"]["position"]:
                moves_blocked += 1
            player_pos = prev_state.get("player", {}).get("position")
            for kind in ("blocks", "selectables"):
                pre_list = prev_state.get("objects", {}).get(kind, []) or []
                post_list = post_state.get("objects", {}).get(kind, []) or []
                for i in range(min(len(pre_list), len(post_list))):
                    pre_pos = pre_list[i].get("position")
                    post_pos = post_list[i].get("position")
                    if not pre_pos or not post_pos or pre_pos == post_pos:
                        continue
                    if pre_pos == player_pos:
                        continue
                    dx = abs(post_pos[0] - pre_pos[0])
                    dy = abs(post_pos[1] - pre_pos[1])
                    if (dx == 3 and dy == 0) or (dx == 0 and dy == 3):
                        object_pushes += 1
                    elif dx > 3 or dy > 3:
                        wall_transfers += 1
            goals = post_state.get("objects", {}).get("goals", []) or []
            sels = post_state.get("objects", {}).get("selectables", []) or []
            blocks = post_state.get("objects", {}).get("blocks", []) or []
            occupied = 0
            for g in goals:
                gx2, gy2 = (g.get("position") or [None, None])[:2]
                if gx2 is None:
                    continue
                gw, gh = (g.get("size") or [3, 3])[:2]
                for o in sels + blocks:
                    ox, oy = (o.get("position") or [None, None])[:2]
                    if ox is None:
                        continue
                    if gx2 <= ox < gx2 + gw and gy2 <= oy < gy2 + gh:
                        occupied += 1
                        break
            if occupied > max_goals_occupied:
                max_goals_occupied = occupied
            prev_state = post_state
        else:
            prev_state = get_structured_state(env, frame, turn, max_turns)

        state_name = frame.state.name
        if state_name == "WIN":
            won = True
            break
        if state_name == "GAME_OVER":
            break

    return {
        "won": won,
        "turns": turns,
        "wall_transfers": wall_transfers,
        "object_pushes": object_pushes,
        "moves_blocked": moves_blocked,
        "click_actions": click_actions,
        "max_goals_occupied": max_goals_occupied,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="ka59simple", choices=["ka59", "ka59simple"])
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--max-turns", type=int, default=32, dest="max_turns")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    out_dir = repo_root / "results" / f"{args.env}_real_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Random baseline: env={args.env}  trials={args.trials}  max_turns={args.max_turns}")
    trials = []
    wins = 0
    for i in range(args.trials):
        r = run_one_trial(args.env, args.max_turns, rng)
        trials.append(r)
        if r.get("won"):
            wins += 1
        print(f"  trial {i+1:>3}: won={r['won']}  turns={r.get('turns')}  "
              f"walls={r.get('wall_transfers',0)}  pushes={r.get('object_pushes',0)}  "
              f"clicks={r.get('click_actions',0)}  max_goals={r.get('max_goals_occupied',0)}")

    win_rate = wins / args.trials if args.trials else 0.0
    summary = {
        "env": args.env,
        "agent": "uniform_random",
        "trials": args.trials,
        "wins": wins,
        "win_rate": win_rate,
        "max_turns": args.max_turns,
        "seed": args.seed,
        "avg_wall_transfers": sum(t.get("wall_transfers", 0) for t in trials) / args.trials,
        "avg_object_pushes": sum(t.get("object_pushes", 0) for t in trials) / args.trials,
        "avg_max_goals_occupied": sum(t.get("max_goals_occupied", 0) for t in trials) / args.trials,
        "trial_data": trials,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"random_baseline_{args.env}_{ts}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n{'='*40}")
    print(f"Random baseline | {args.env}")
    print(f"  wins:     {wins}/{args.trials}  = {win_rate:.1%}")
    print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
