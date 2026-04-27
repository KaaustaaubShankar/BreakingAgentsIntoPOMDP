"""
ablation.py — 5-config ablation sweep for the real KA59 game.

Same ALL_CONFIGS keys as env4/ablation.py and ka59_ref/ablation.py so the
rows align across all three environments in the paper table.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from .experiment import DEFAULT_TURNS_PER_LEVEL, RESULTS_DIR, run_agent, save_result


ALL_CONFIGS: dict[str, dict[str, str]] = {
    "baseline": {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"},
    "world_hard": {"world": "HARD", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"},
    "goal_hard": {"world": "EASY", "goal": "HARD", "mechanics": "EASY", "feedback": "EASY"},
    "mechanics_hard": {"world": "EASY", "goal": "EASY", "mechanics": "HARD", "feedback": "EASY"},
    "feedback_hard": {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "HARD"},
}


def run_ablation(
    n_trials: int = 3,
    provider: str = "openrouter",
    model: str = "google/gemini-2.5-flash",
    max_levels: int = 1,
    turns_per_level: int = DEFAULT_TURNS_PER_LEVEL,
    config_names: list[str] | None = None,
    verbose: bool = False,
) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    configs_to_run = {
        key: value for key, value in ALL_CONFIGS.items() if config_names is None or key in config_names
    }

    summary: list[dict[str, Any]] = []

    for cfg_name, cfg in configs_to_run.items():
        print(f"\n{'=' * 62}")
        print(f"  Config: {cfg_name}  {cfg}")
        print(f"{'=' * 62}")

        wins = 0
        total_turns = 0
        total_levels = 0
        total_invalid = 0
        total_clicks = 0
        total_blocked = 0
        run_files: list[str] = []

        for trial in range(1, n_trials + 1):
            print(f"\n--- Trial {trial}/{n_trials} ---")
            result = run_agent(
                world_level=cfg["world"],
                goal_level=cfg["goal"],
                mechanics_level=cfg["mechanics"],
                feedback_level=cfg["feedback"],
                provider=provider,
                model=model,
                max_levels=max_levels,
                turns_per_level=turns_per_level,
                verbose=verbose,
            )
            run_id = f"{timestamp}_{cfg_name}_t{trial}"
            path = save_result(result, run_id=run_id)
            run_files.append(str(path))

            if result.won:
                wins += 1
            total_turns += result.turns
            total_levels += result.levels_completed
            total_invalid += result.invalid_actions
            total_clicks += result.click_actions
            total_blocked += result.moves_blocked

            status = "WIN " if result.won else "LOSS"
            print(
                f"  -> {status} | turns: {result.turns} | levels: {result.levels_completed} | "
                f"invalid: {result.invalid_actions} | clicks: {result.click_actions}"
            )

        summary.append({
            "config_name": cfg_name,
            "config": cfg,
            "provider": provider,
            "model": model,
            "n_trials": n_trials,
            "wins": wins,
            "win_rate": wins / n_trials,
            "avg_turns": total_turns / n_trials,
            "avg_levels_completed": total_levels / n_trials,
            "avg_invalid_actions": total_invalid / n_trials,
            "avg_click_actions": total_clicks / n_trials,
            "avg_moves_blocked": total_blocked / n_trials,
            "run_files": run_files,
        })

    baseline = next((item for item in summary if item["config_name"] == "baseline"), None)
    baseline_turns = baseline["avg_turns"] if baseline else None
    for item in summary:
        if baseline_turns and baseline_turns > 0 and item["avg_turns"] > 0:
            item["relative_difficulty"] = round(item["avg_turns"] / baseline_turns, 3)
        else:
            item["relative_difficulty"] = None

    summary_path = RESULTS_DIR / f"ablation_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nFull summary saved -> {summary_path}")

    print("\n" + "=" * 110)
    print(
        f"{'Config':<20}  {'Win%':>5}  {'Avg turns':>9}  {'Avg levels':>10}  "
        f"{'Invalid':>7}  {'Clicks':>7}  {'Blocked':>7}  {'Rel. diff':>9}"
    )
    print("-" * 110)
    for item in summary:
        rel = f"{item['relative_difficulty']:.2f}x" if item["relative_difficulty"] is not None else "n/a"
        print(
            f"{item['config_name']:<20}  "
            f"{item['win_rate']:>5.0%}  "
            f"{item['avg_turns']:>9.1f}  "
            f"{item['avg_levels_completed']:>10.1f}  "
            f"{item['avg_invalid_actions']:>7.1f}  "
            f"{item['avg_click_actions']:>7.1f}  "
            f"{item['avg_moves_blocked']:>7.1f}  "
            f"{rel:>9}"
        )
    print("=" * 110)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run KA59 real-game ablation study.")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--provider", default="openrouter", choices=["openrouter"])
    p.add_argument("--model", default="google/gemini-2.5-flash")
    p.add_argument("--max-levels", type=int, default=1)
    p.add_argument("--turns-per-level", type=int, default=DEFAULT_TURNS_PER_LEVEL)
    p.add_argument("--configs", nargs="+", choices=list(ALL_CONFIGS.keys()))
    p.add_argument("--quiet", action="store_true", default=True, help="verbose is OFF by default for sweeps")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_ablation(
        n_trials=args.trials,
        provider=args.provider,
        model=args.model,
        max_levels=args.max_levels,
        turns_per_level=args.turns_per_level,
        config_names=args.configs,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
