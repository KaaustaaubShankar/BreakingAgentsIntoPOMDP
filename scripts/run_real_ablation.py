"""
Real-game KA59 ablation sweep.

Uses the actual arc_agi KA59 environment (ka59_game), not the internal
reference simulator. This is what goes in the paper.

Usage:
    python3 -m scripts.run_real_ablation --provider xai --model grok-3-mini-fast --trials 5
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(__file__).parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "ka59"))  # for unified_result

from ka59.runner import run_one

ALL_CONFIGS = {
    "baseline":       {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"},
    "world_hard":     {"world": "HARD", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"},
    "goal_hard":      {"world": "EASY", "goal": "HARD", "mechanics": "EASY", "feedback": "EASY"},
    "mechanics_hard": {"world": "EASY", "goal": "EASY", "mechanics": "HARD", "feedback": "EASY"},
    "mechanics_ooda":   {"world": "EASY", "goal": "EASY", "mechanics": "OODA",   "feedback": "EASY"},
    "mechanics_ooda_f": {"world": "EASY", "goal": "EASY", "mechanics": "OODA_F", "feedback": "EASY"},
    "feedback_hard":    {"world": "EASY", "goal": "EASY", "mechanics": "EASY",   "feedback": "HARD"},
}

RESULTS_DIR = Path(__file__).parents[1] / "results" / "ka59_real_ablation"


def run_ablation(provider: str, model: str, n_trials: int, verbose: bool = False, max_turns: int = 200, configs: list | None = None) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    summary = {}

    print(f"\n{'='*60}")
    print(f"KA59 REAL-GAME ABLATION")
    print(f"Provider: {provider}  Model: {model}  Trials: {n_trials}")
    print(f"{'='*60}\n")

    run_configs = {k: v for k, v in ALL_CONFIGS.items() if configs is None or k in configs}
    for cfg_name, cfg in run_configs.items():
        wins = 0
        turns_list = []
        levels_list = []
        passable_walls_total = 0
        blocked_total = 0
        trials_data = []

        print(f"[{cfg_name}] running {n_trials} trials...")
        sys.stdout.flush()

        for i in range(n_trials):
            try:
                result = run_one(
                    env_name="ka59",
                    config_name=cfg_name,
                    config=cfg,
                    provider=provider,
                    model=model,
                    max_turns=max_turns,
                    verbose=verbose,
                )
                if result.won:
                    wins += 1
                    turns_list.append(result.turns)
                levels_list.append(result.levels_completed)
                passable_walls_total += result.flips
                blocked_total += result.invalid_clicks

                status = "WIN" if result.won else "FAIL"
                print(f"  trial {i+1}: {status}  turns={result.turns}  levels={result.levels_completed}  p_walls={result.flips}")
                sys.stdout.flush()

                trials_data.append({
                    "trial": i + 1,
                    "won": result.won,
                    "turns": result.turns,
                    "levels_completed": result.levels_completed,
                    "passable_walls": result.flips,
                    "blocked": result.invalid_clicks,
                })
            except Exception as e:
                print(f"  trial {i+1}: ERROR — {e}")
                sys.stdout.flush()
                trials_data.append({"trial": i + 1, "error": str(e)})

        win_rate = wins / n_trials
        avg_turns = sum(turns_list) / len(turns_list) if turns_list else None
        avg_levels = sum(levels_list) / len(levels_list) if levels_list else 0

        summary[cfg_name] = {
            "win_rate": win_rate,
            "wins": wins,
            "trials": n_trials,
            "avg_turns_on_win": round(avg_turns, 1) if avg_turns else None,
            "avg_levels_completed": round(avg_levels, 2),
            "passable_walls_total": passable_walls_total,
            "blocked_total": blocked_total,
            "trial_data": trials_data,
        }

        print(f"  → {cfg_name}: {wins}/{n_trials} wins  win_rate={win_rate:.0%}  avg_turns={avg_turns}\n")
        sys.stdout.flush()

    # Save full results
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

    # Print summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY — {provider}/{model}")
    print(f"{'Config':<18} {'Wins':>6} {'Win%':>6} {'AvgTurns':>10} {'AvgLevels':>10} {'P.Walls':>8}")
    print("-" * 60)
    for cfg_name, r in summary.items():
        avg_t = f"{r['avg_turns_on_win']}" if r['avg_turns_on_win'] else "n/a"
        print(f"{cfg_name:<18} {r['wins']:>4}/{r['trials']:<2} {r['win_rate']:>5.0%} {avg_t:>10} {r['avg_levels_completed']:>10.2f} {r['passable_walls_total']:>8}")
    print(f"{'='*60}\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="xai")
    parser.add_argument("--model", default="grok-3-mini-fast")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=200, dest="max_turns")
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_ablation(args.provider, args.model, args.trials, args.verbose, args.max_turns, args.configs)
