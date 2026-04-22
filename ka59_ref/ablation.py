"""
ablation.py — 5-config ablation sweep for KA59, matching env4/ablation.py.

Same ALL_CONFIGS keys as env4 so KA59 and BP35 rows align in the paper table.
Per-trial metrics replace the BP35-specific ones: invalid/selects/push_events/
moves_blocked in place of invalid/clicks/gravity_flips/undos.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .experiment import RESULTS_DIR, run_agent, save_result
from .scenarios import SCENARIOS


ALL_CONFIGS: dict[str, dict[str, str]] = {
    "baseline": {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"},
    "world_hard": {"world": "HARD", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"},
    "goal_hard": {"world": "EASY", "goal": "HARD", "mechanics": "EASY", "feedback": "EASY"},
    "mechanics_hard": {"world": "EASY", "goal": "EASY", "mechanics": "HARD", "feedback": "EASY"},
    "feedback_hard": {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "HARD"},
}


def run_ablation(
    scenario: str = "transfer_wall_push",
    n_trials: int = 5,
    provider: str = "openrouter",
    model: str = "meta-llama/llama-3.3-70b-instruct:free",
    config_names: list[str] | None = None,
    verbose: bool = True,
) -> None:
    if scenario not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario {scenario!r}. Available: {sorted(SCENARIOS.keys())}"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    configs_to_run = {
        key: value for key, value in ALL_CONFIGS.items() if config_names is None or key in config_names
    }

    summary: list[dict[str, Any]] = []

    for cfg_name, cfg in configs_to_run.items():
        print(f"\n{'=' * 62}")
        print(f"  Scenario: {scenario}  |  Config: {cfg_name}")
        print(f"  {cfg}")
        print(f"{'=' * 62}")

        wins = 0
        total_turns = 0
        total_invalid_actions = 0
        total_selects = 0
        total_pushes = 0
        total_blocked_moves = 0
        run_files: list[str] = []

        for trial in range(1, n_trials + 1):
            print(f"\n--- Trial {trial}/{n_trials} ---")
            result = run_agent(
                scenario=scenario,
                world_level=cfg["world"],
                goal_level=cfg["goal"],
                mechanics_level=cfg["mechanics"],
                feedback_level=cfg["feedback"],
                provider=provider,
                model=model,
                verbose=verbose,
            )
            run_id = f"{timestamp}_{cfg_name}_t{trial}"
            path = save_result(result, run_id=run_id)
            run_files.append(str(path))

            if result.won:
                wins += 1
            total_turns += result.turns
            total_invalid_actions += result.invalid_actions
            total_selects += result.select_actions
            total_pushes += result.push_events
            total_blocked_moves += result.moves_blocked

            status = "WIN " if result.won else "LOSS"
            print(
                f"  -> {status} | turns: {result.turns} | invalid: {result.invalid_actions} | "
                f"selects: {result.select_actions} | pushes: {result.push_events}"
            )

        cfg_summary: dict[str, Any] = {
            "scenario": scenario,
            "config_name": cfg_name,
            "config": cfg,
            "provider": provider,
            "model": model,
            "n_trials": n_trials,
            "wins": wins,
            "win_rate": wins / n_trials,
            "avg_turns": total_turns / n_trials,
            "avg_invalid_actions": total_invalid_actions / n_trials,
            "avg_selects": total_selects / n_trials,
            "avg_push_events": total_pushes / n_trials,
            "avg_moves_blocked": total_blocked_moves / n_trials,
            "run_files": run_files,
        }
        summary.append(cfg_summary)

        print(
            f"\n  {cfg_name}: win_rate={cfg_summary['win_rate']:.0%} "
            f"avg_turns={cfg_summary['avg_turns']:.1f} "
            f"pushes={cfg_summary['avg_push_events']:.1f} "
            f"selects={cfg_summary['avg_selects']:.1f}"
        )

    baseline = next((item for item in summary if item["config_name"] == "baseline"), None)
    baseline_turns = baseline["avg_turns"] if baseline else None
    for item in summary:
        if baseline_turns and baseline_turns > 0 and item["avg_turns"] > 0:
            item["relative_difficulty"] = round(item["avg_turns"] / baseline_turns, 3)
        else:
            item["relative_difficulty"] = None

    summary_path = RESULTS_DIR / f"ablation_summary_{scenario}_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nFull summary saved -> {summary_path}")

    print("\n" + "=" * 110)
    print(
        f"{'Config':<20}  {'Win%':>5}  {'Avg turns':>9}  {'Invalid':>7}  "
        f"{'Selects':>7}  {'Pushes':>7}  {'Blocked':>7}  {'Rel. diff':>9}"
    )
    print("-" * 110)
    for item in summary:
        rel = f"{item['relative_difficulty']:.2f}x" if item["relative_difficulty"] is not None else "n/a"
        print(
            f"{item['config_name']:<20}  "
            f"{item['win_rate']:>5.0%}  "
            f"{item['avg_turns']:>9.1f}  "
            f"{item['avg_invalid_actions']:>7.1f}  "
            f"{item['avg_selects']:>7.1f}  "
            f"{item['avg_push_events']:>7.1f}  "
            f"{item['avg_moves_blocked']:>7.1f}  "
            f"{rel:>9}"
        )
    print("=" * 110)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run KA59 ablation study.")
    p.add_argument("--scenario", default="transfer_wall_push", choices=sorted(SCENARIOS.keys()))
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--provider", default="openrouter", choices=["openrouter"])
    p.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct:free")
    p.add_argument("--configs", nargs="+", choices=list(ALL_CONFIGS.keys()))
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_ablation(
        scenario=args.scenario,
        n_trials=args.trials,
        provider=args.provider,
        model=args.model,
        config_names=args.configs,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
