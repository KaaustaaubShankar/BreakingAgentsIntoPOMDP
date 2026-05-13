"""
Real-game KA59 ablation sweep.

Calls ka59_game.experiment.run_agent directly and writes both an aggregated
summary and a per-config sidecar JSON, plus a per-trial JSON via
experiment.save_result(). Per-trial JSON contains the full event history
(orient texts, forced reframes, action timeline) needed for the
verbal-vs-behavioral compliance benchmark.

Usage:
    python3 -m scripts.run_real_ablation --provider xai --model grok-4-1-fast --trials 2
    python3 -m scripts.run_real_ablation --env ka59simple --provider xai --model grok-4-1-fast --trials 2

Environments (--env):
    ka59        canonical 7-level KA59 (used for n=30+ baseline runs from 2026-04-28;
                floor-effects to 0% win-rate across all conditions/models)
    ka59simple  single-goal fork that removes the multi-target click sequence,
                preserving all 4 ablation axes — used to escape the floor effect

Available --configs:
    baseline, world_hard, goal_hard, mechanics_hard, feedback_hard
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(__file__).parents[1]
sys.path.insert(0, str(repo_root))

from ka59_game.experiment import run_agent

ALL_CONFIGS = {
    "baseline":       {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"},
    "world_hard":     {"world": "HARD", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"},
    "goal_hard":      {"world": "EASY", "goal": "HARD", "mechanics": "EASY", "feedback": "EASY"},
    "mechanics_hard": {"world": "EASY", "goal": "EASY", "mechanics": "HARD", "feedback": "EASY"},
    "feedback_hard":  {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "HARD"},
}

REASONING_EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh"]

ENV_CHOICES = ["ka59", "ka59simple"]


def _results_dir(env_id: str) -> Path:
    folder = "ka59_real_ablation" if env_id == "ka59" else f"{env_id}_real_ablation"
    return repo_root / "results" / folder


def run_ablation(
    provider: str,
    model: str,
    n_trials: int,
    verbose: bool = False,
    max_turns: int = 64,
    configs: list | None = None,
    reasoning_efforts: list[str] | None = None,
    env_id: str = "ka59simple",
) -> dict:
    efforts_to_run = [e.lower().strip() for e in (reasoning_efforts or ["none"])]
    invalid = [e for e in efforts_to_run if e not in REASONING_EFFORTS]
    if invalid:
        raise ValueError(
            f"Invalid reasoning_effort value(s): {invalid}. "
            f"Expected one of {REASONING_EFFORTS}."
        )

    results_dir = _results_dir(env_id)
    results_dir.mkdir(parents=True, exist_ok=True)
    # Microsecond precision so two parallel ablation processes started in the
    # same second (e.g. when launched as background jobs from the same shell)
    # don't write to the same ablation_*.json filename and clobber each other.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    summary: dict = {}  # key: f"{effort}::{cfg_name}"

    print(f"\n{'='*60}")
    print(f"KA59 REAL-GAME ABLATION  (env={env_id})")
    print(f"Provider: {provider}  Model: {model}  Trials: {n_trials}")
    print(f"Reasoning efforts: {efforts_to_run}")
    print(f"{'='*60}\n")

    run_configs = {k: v for k, v in ALL_CONFIGS.items() if configs is None or k in configs}

    for reasoning_effort in efforts_to_run:
        print(f"\n{'#' * 62}")
        print(f"  Reasoning effort: {reasoning_effort}")
        print(f"{'#' * 62}")

        for cfg_name, cfg in run_configs.items():
            wins = 0
            turns_list: list[int] = []
            levels_list: list[int] = []
            click_actions_total = 0
            moves_blocked_total = 0
            object_pushes_total = 0
            wall_transfers_total = 0
            max_goals_occupied_per_trial: list[int] = []
            forced_reframes_total = 0
            discovery_turns_observed: list[int] = []
            trials_data: list[dict] = []

            print(f"[{reasoning_effort}::{cfg_name}] running {n_trials} trials...")
            sys.stdout.flush()

            for i in range(n_trials):
                try:
                    result = run_agent(
                        world_level=cfg["world"],
                        goal_level=cfg["goal"],
                        mechanics_level=cfg["mechanics"],
                        feedback_level=cfg["feedback"],
                        provider=provider,
                        model=model,
                        max_levels=1,
                        turns_per_level=max_turns,
                        verbose=verbose,
                        reasoning_effort=reasoning_effort,
                        env_id=env_id,
                    )
                    if result.won:
                        wins += 1
                        turns_list.append(result.turns)
                    levels_list.append(result.levels_completed)
                    click_actions_total += result.click_actions
                    moves_blocked_total += result.moves_blocked
                    object_pushes_total += result.object_pushes
                    wall_transfers_total += result.wall_transfers
                    max_goals_occupied_per_trial.append(result.max_goals_occupied)
                    forced_reframes_total += result.forced_reframes
                    d_turn = result.discovery_turn
                    if d_turn is not None:
                        discovery_turns_observed.append(d_turn)

                    status = "WIN" if result.won else "FAIL"
                    print(
                        f"  trial {i+1}: {status}  turns={result.turns}  "
                        f"levels={result.levels_completed}  p_walls={result.click_actions}"
                        f"  reframes={result.forced_reframes}  d_turn={d_turn}"
                    )
                    sys.stdout.flush()

                    trials_data.append({
                        "trial": i + 1,
                        "won": result.won,
                        "turns": result.turns,
                        "levels_completed": result.levels_completed,
                        "passable_walls": result.click_actions,
                        "blocked": result.moves_blocked,
                        "object_pushes": result.object_pushes,
                        "wall_transfers": result.wall_transfers,
                        "max_goals_occupied": result.max_goals_occupied,
                        "forced_reframes": result.forced_reframes,
                        "discovery_turn": d_turn,
                    })
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"  trial {i+1}: ERROR — {e}")
                    sys.stdout.flush()
                    trials_data.append({"trial": i + 1, "error": str(e)})

            win_rate = wins / n_trials
            avg_turns = sum(turns_list) / len(turns_list) if turns_list else None
            avg_levels = sum(levels_list) / len(levels_list) if levels_list else 0
            n_completed = sum(1 for t in trials_data if "error" not in t)
            discovery_rate = (
                len(discovery_turns_observed) / n_completed if n_completed else 0.0
            )
            avg_discovery_turn = (
                sum(discovery_turns_observed) / len(discovery_turns_observed)
                if discovery_turns_observed else None
            )

            summary_key = f"{reasoning_effort}::{cfg_name}"
            summary[summary_key] = {
                "reasoning_effort": reasoning_effort,
                "config": cfg_name,
                "win_rate": win_rate,
                "wins": wins,
                "trials": n_trials,
                "avg_turns_on_win": round(avg_turns, 1) if avg_turns else None,
                "avg_levels_completed": round(avg_levels, 2),
                "passable_walls_total": click_actions_total,
                "blocked_total": moves_blocked_total,
                "object_pushes_total": object_pushes_total,
                "wall_transfers_total": wall_transfers_total,
                "avg_max_goals_occupied": (
                    round(sum(max_goals_occupied_per_trial) / len(max_goals_occupied_per_trial), 2)
                    if max_goals_occupied_per_trial else 0
                ),
                "forced_reframes_total": forced_reframes_total,
                "discovery_rate": round(discovery_rate, 3),
                "avg_discovery_turn": (
                    round(avg_discovery_turn, 1) if avg_discovery_turn else None
                ),
                "trial_data": trials_data,
            }

            try:
                sidecar = results_dir / (
                    f"sidecar_{provider}_{model.replace('/', '_')}_{timestamp}_"
                    f"{cfg_name}_{reasoning_effort}.json"
                )
                sidecar.write_text(json.dumps({
                    "provider": provider, "model": model, "env_id": env_id,
                    "config": cfg_name, "reasoning_effort": reasoning_effort,
                    "timestamp": timestamp, "n_trials": n_trials,
                    **summary[summary_key],
                }, indent=2))
            except Exception as exc:
                print(f"  (sidecar save failed: {exc})")

            print(
                f"  → {reasoning_effort}::{cfg_name}: {wins}/{n_trials} wins  "
                f"win_rate={win_rate:.0%}  avg_turns={avg_turns}  "
                f"reframes={forced_reframes_total}  "
                f"discovery_rate={discovery_rate:.0%}\n"
            )
            sys.stdout.flush()

    out = {
        "provider": provider,
        "model": model,
        "n_trials": n_trials,
        "timestamp": timestamp,
        "env": f"{env_id}_real_game",
        "env_id": env_id,
        "results": summary,
    }
    out_path = results_dir / f"ablation_{provider}_{model.replace('/', '_')}_{timestamp}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved → {out_path}")

    print(f"\n{'='*70}")
    print(f"SUMMARY — {provider}/{model}")
    print(f"{'Config':<18} {'Reason':<8} {'Wins':>6} {'Win%':>6} {'AvgTurns':>10} {'AvgLevels':>10} {'P.Walls':>8}")
    print("-" * 70)
    for key, r in summary.items():
        avg_t = f"{r['avg_turns_on_win']}" if r['avg_turns_on_win'] else "n/a"
        print(
            f"{r['config']:<18} {r['reasoning_effort']:<8} "
            f"{r['wins']:>4}/{r['trials']:<2} {r['win_rate']:>5.0%} "
            f"{avg_t:>10} {r['avg_levels_completed']:>10.2f} "
            f"{r['passable_walls_total']:>8}"
        )
    print(f"{'='*70}\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--model", default="openai/gpt-5.2")
    parser.add_argument("--env", default="ka59simple", choices=ENV_CHOICES,
                        help="Which env to ablate. Default ka59simple (single-level fork).")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=64, dest="max_turns")
    parser.add_argument("--configs", nargs="+", default=None, choices=list(ALL_CONFIGS.keys()))
    parser.add_argument("--reasoning-effort", nargs="+", default=["none", "medium"],
                        choices=REASONING_EFFORTS, dest="reasoning_efforts",
                        help="One or more reasoning effort levels to sweep in one invocation.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_ablation(args.provider, args.model, args.trials, args.verbose, args.max_turns,
                 args.configs, args.reasoning_efforts, env_id=args.env)
