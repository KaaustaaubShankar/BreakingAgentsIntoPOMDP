"""
Real-game KA59 ablation sweep.

Calls ka59_game.experiment.run_agent directly (no multi-env dispatcher) and
writes both an aggregated summary and a per-config sidecar JSON, plus a
per-trial JSON via experiment.save_result(). Per-trial JSON contains the
full event history (orient texts, forced reframes, action timeline) needed
for the verbal-vs-behavioral compliance benchmark.

Usage:
    python3 -m scripts.run_real_ablation --provider xai --model grok-4-1-fast --trials 2

Available --configs:
    baseline, world_hard, goal_hard, mechanics_hard, mechanics_ooda,
    mechanics_ooda_f, feedback_hard
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
    "baseline":         {"world": "EASY", "goal": "EASY", "mechanics": "EASY",   "feedback": "EASY"},
    "world_hard":       {"world": "HARD", "goal": "EASY", "mechanics": "EASY",   "feedback": "EASY"},
    "goal_hard":        {"world": "EASY", "goal": "HARD", "mechanics": "EASY",   "feedback": "EASY"},
    "mechanics_hard":   {"world": "EASY", "goal": "EASY", "mechanics": "HARD",   "feedback": "EASY"},
    "mechanics_ooda":   {"world": "EASY", "goal": "EASY", "mechanics": "OODA",   "feedback": "EASY"},
    "mechanics_ooda_f": {"world": "EASY", "goal": "EASY", "mechanics": "OODA_F", "feedback": "EASY"},
    "feedback_hard":    {"world": "EASY", "goal": "EASY", "mechanics": "EASY",   "feedback": "HARD"},
}

RESULTS_DIR = repo_root / "results" / "ka59_real_ablation"


def run_ablation(
    provider: str,
    model: str,
    n_trials: int,
    verbose: bool = False,
    max_turns: int = 64,
    configs: list | None = None,
) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    summary: dict = {}

    print(f"\n{'='*60}")
    print(f"KA59 REAL-GAME ABLATION")
    print(f"Provider: {provider}  Model: {model}  Trials: {n_trials}")
    print(f"{'='*60}\n")

    run_configs = {k: v for k, v in ALL_CONFIGS.items() if configs is None or k in configs}
    for cfg_name, cfg in run_configs.items():
        wins = 0
        turns_list: list[int] = []
        levels_list: list[int] = []
        click_actions_total = 0
        moves_blocked_total = 0
        forced_reframes_total = 0
        discovery_turns_observed: list[int] = []
        trials_data: list[dict] = []

        print(f"[{cfg_name}] running {n_trials} trials...")
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
                )
                if result.won:
                    wins += 1
                    turns_list.append(result.turns)
                levels_list.append(result.levels_completed)
                click_actions_total += result.click_actions
                moves_blocked_total += result.moves_blocked
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
                    "forced_reframes": result.forced_reframes,
                    "discovery_turn": d_turn,
                })
            except Exception as e:
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

        summary[cfg_name] = {
            "win_rate": win_rate,
            "wins": wins,
            "trials": n_trials,
            "avg_turns_on_win": round(avg_turns, 1) if avg_turns else None,
            "avg_levels_completed": round(avg_levels, 2),
            "passable_walls_total": click_actions_total,
            "blocked_total": moves_blocked_total,
            "forced_reframes_total": forced_reframes_total,
            "discovery_rate": round(discovery_rate, 3),
            "avg_discovery_turn": (
                round(avg_discovery_turn, 1) if avg_discovery_turn else None
            ),
            "trial_data": trials_data,
        }

        try:
            sidecar = RESULTS_DIR / f"sidecar_{provider}_{model.replace('/', '_')}_{timestamp}_{cfg_name}.json"
            sidecar.write_text(json.dumps({
                "provider": provider, "model": model, "config": cfg_name,
                "timestamp": timestamp, "n_trials": n_trials,
                **summary[cfg_name],
            }, indent=2))
        except Exception as exc:
            print(f"  (sidecar save failed: {exc})")

        print(
            f"  → {cfg_name}: {wins}/{n_trials} wins  win_rate={win_rate:.0%}  "
            f"avg_turns={avg_turns}  reframes={forced_reframes_total}  "
            f"discovery_rate={discovery_rate:.0%}\n"
        )
        sys.stdout.flush()

    out = {
        "provider": provider,
        "model": model,
        "n_trials": n_trials,
        "timestamp": timestamp,
        "env": "ka59_real_game",
        "results": summary,
    }
    out_path = RESULTS_DIR / f"ablation_{provider}_{model.replace('/', '_')}_{timestamp}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved → {out_path}")

    print(f"\n{'='*60}")
    print(f"SUMMARY — {provider}/{model}")
    print(f"{'Config':<18} {'Wins':>6} {'Win%':>6} {'AvgTurns':>10} {'AvgLevels':>10} {'P.Walls':>8}")
    print("-" * 60)
    for cfg_name, r in summary.items():
        avg_t = f"{r['avg_turns_on_win']}" if r['avg_turns_on_win'] else "n/a"
        print(
            f"{cfg_name:<18} {r['wins']:>4}/{r['trials']:<2} "
            f"{r['win_rate']:>5.0%} {avg_t:>10} {r['avg_levels_completed']:>10.2f} "
            f"{r['passable_walls_total']:>8}"
        )
    print(f"{'='*60}\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="xai")
    parser.add_argument("--model", default="grok-4-1-fast")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=64, dest="max_turns")
    parser.add_argument("--configs", nargs="+", default=None, choices=list(ALL_CONFIGS.keys()))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_ablation(args.provider, args.model, args.trials, args.verbose, args.max_turns, args.configs)
