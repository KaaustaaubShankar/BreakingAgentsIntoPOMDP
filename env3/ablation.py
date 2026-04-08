"""
ablation.py — Ablation study runner for Environment 3 (Grid Navigator / ls20).

Mirrors the structure of Zendo's ablation_study.py for cross-environment
result consistency.

Configs tested:
  1. Baseline       — all axes EASY
  2. World HARD     — hard world,     rest EASY
  3. Goal HARD      — hard goal,      rest EASY
  4. Mechanics HARD — hard mechanics, rest EASY
  5. Feedback HARD  — hard feedback,  rest EASY

Usage:
  python ablation.py                        # 5 trials per config, default model
  python ablation.py --trials 10            # more trials
  python ablation.py --model openai/gpt-4o  # different model via OpenRouter
  python ablation.py --configs baseline world_hard  # run specific configs only
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiment import RunResult, run_agent, save_result

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# All configs to test: name → axis overrides (rest default to EASY)
ALL_CONFIGS: dict[str, dict[str, str]] = {
    "baseline":       {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"},
    "world_hard":     {"world": "HARD", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"},
    "goal_hard":      {"world": "EASY", "goal": "HARD", "mechanics": "EASY", "feedback": "EASY"},
    "mechanics_hard": {"world": "EASY", "goal": "EASY", "mechanics": "HARD", "feedback": "EASY"},
    "feedback_hard":  {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "HARD"},
}


def run_ablation(
    n_trials: int = 5,
    provider: str = "openrouter",
    model: str = "meta-llama/llama-3.3-70b-instruct:free",
    config_names: list[str] | None = None,
    verbose: bool = True,
) -> None:
    """
    Run the full ablation study and save individual run JSONs plus a summary.

    Args:
        n_trials:     Number of independent trials per configuration.
        model:        OpenRouter model identifier.
        config_names: Subset of config names to run (default: all).
        verbose:      Print per-turn agent actions to stdout.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    configs_to_run = {
        k: v for k, v in ALL_CONFIGS.items()
        if config_names is None or k in config_names
    }

    summary: list[dict[str, Any]] = []

    for cfg_name, cfg in configs_to_run.items():
        print(f"\n{'='*62}")
        print(f"  Config: {cfg_name}")
        print(f"  {cfg}")
        print(f"{'='*62}")

        wins = 0
        total_turns = 0
        total_levels = 0
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
                verbose=verbose,
            )
            run_id = f"{timestamp}_{cfg_name}_t{trial}"
            path = save_result(result, run_id=run_id)
            run_files.append(str(path))

            if result.won:
                wins += 1
            total_turns += result.turns
            total_levels += result.levels_completed

            status = "WIN " if result.won else "LOSS"
            print(f"  → {status} | turns: {result.turns} | "
                  f"levels: {result.levels_completed}")

        win_rate = wins / n_trials
        avg_turns = total_turns / n_trials
        avg_levels = total_levels / n_trials

        cfg_summary: dict[str, Any] = {
            "config_name": cfg_name,
            "config": cfg,
            "provider": provider,
        "model": model,
            "n_trials": n_trials,
            "wins": wins,
            "win_rate": win_rate,
            "avg_turns": avg_turns,
            "avg_levels_completed": avg_levels,
            "run_files": run_files,
        }
        summary.append(cfg_summary)

        print(f"\n  ✓ {cfg_name}: win_rate={win_rate:.0%}  "
              f"avg_turns={avg_turns:.1f}  avg_levels={avg_levels:.1f}")

    # ── Save summary ──────────────────────────────────────────────────────────
    summary_path = RESULTS_DIR / f"ablation_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nFull summary saved → {summary_path}")

    # ── Print results table ───────────────────────────────────────────────────
    print("\n" + "="*62)
    print(f"{'Config':<20}  {'Win%':>5}  {'Avg turns':>9}  {'Avg levels':>10}")
    print("-"*62)
    for s in summary:
        print(
            f"{s['config_name']:<20}  "
            f"{s['win_rate']:>5.0%}  "
            f"{s['avg_turns']:>9.1f}  "
            f"{s['avg_levels_completed']:>10.1f}"
        )
    print("="*62)


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ablation study for Environment 3.")
    p.add_argument("--trials", type=int, default=5,
                   help="Number of trials per config (default: 5).")
    p.add_argument("--provider", default="openrouter",
                   choices=["openrouter"],
                   help="LLM provider (default: openrouter).")
    p.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct:free",
                   help="OpenRouter model identifier.")
    p.add_argument("--configs", nargs="+", choices=list(ALL_CONFIGS.keys()),
                   help="Subset of configs to run (default: all).")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-turn verbose output.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ablation(
        n_trials=args.trials,
        provider=args.provider,
        model=args.model,
        config_names=args.configs,
        verbose=not args.quiet,
    )
