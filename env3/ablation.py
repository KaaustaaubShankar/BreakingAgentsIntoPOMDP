"""
ablation.py — Ablation study runner for Environment 3 (Grid Navigator / ls20).

Mirrors scripts/run_real_ablation.py (the KA59/ka59simple runner) so LS20 rows
slot into the same cross-environment "Josh table": reasoning-effort sweep,
per-trial token + cost tracking, and collision-safe output filenames.

Configs (four-axis knockout; goal_hard dropped per meeting, kept opt-in):
  1. baseline       — all axes EASY
  2. world_hard     — hard world,     rest EASY
  3. mechanics_hard — hard mechanics, rest EASY
  4. feedback_hard  — hard feedback,  rest EASY
  (goal_hard available via --configs but excluded from the default set)

Usage:
  python ablation.py                                    # default: deepseek-v4-pro, none+medium, N=20
  python ablation.py --configs baseline --reasoning-effort none   # one cell
  python ablation.py --trials 5 --model openai/gpt-5.2

Launch cells in parallel (one process per config × effort) for wall-time; the
PID in each output filename keeps parallel summaries from clobbering each other.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiment import run_agent, save_result

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

REASONING_EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh"]

# All configs: name → axis overrides (rest default to EASY).
ALL_CONFIGS: dict[str, dict[str, str]] = {
    "baseline":       {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"},
    "world_hard":     {"world": "HARD", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"},
    "goal_hard":      {"world": "EASY", "goal": "HARD", "mechanics": "EASY", "feedback": "EASY"},
    "mechanics_hard": {"world": "EASY", "goal": "EASY", "mechanics": "HARD", "feedback": "EASY"},
    "feedback_hard":  {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "HARD"},
}

# goal_hard dropped per meeting — matches the ka59simple knockout.
DEFAULT_CONFIGS = ["baseline", "world_hard", "mechanics_hard", "feedback_hard"]


def run_ablation(
    n_trials: int = 20,
    provider: str = "openrouter",
    model: str = "deepseek/deepseek-v4-pro",
    config_names: list[str] | None = None,
    reasoning_efforts: list[str] | None = None,
    input_cost_per_m: float = 0.30,
    output_cost_per_m: float = 0.90,
    verbose: bool = True,
) -> None:
    """Run the ablation sweep and save per-trial run JSONs plus a summary."""
    efforts = [e.lower().strip() for e in (reasoning_efforts or ["none", "medium"])]
    invalid = [e for e in efforts if e not in REASONING_EFFORTS]
    if invalid:
        raise ValueError(f"Invalid reasoning_effort(s): {invalid}. Expected {REASONING_EFFORTS}.")

    configs_to_run = {
        k: v for k, v in ALL_CONFIGS.items()
        if (config_names or DEFAULT_CONFIGS) and k in (config_names or DEFAULT_CONFIGS)
    }

    # Microsecond timestamp + PID so parallel cells never clobber each other.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    run_tag = f"{timestamp}_p{os.getpid()}"

    print(f"\n{'='*62}")
    print(f"LS20 (env3) ABLATION  provider={provider}  model={model}")
    print(f"efforts={efforts}  configs={list(configs_to_run)}  trials={n_trials}")
    print(f"{'='*62}")

    summary: list[dict[str, Any]] = []

    for effort in efforts:
        for cfg_name, cfg in configs_to_run.items():
            print(f"\n{'='*62}\n  {effort}::{cfg_name}  {cfg}\n{'='*62}")

            wins = 0
            total_turns = 0
            total_levels = 0
            total_wall_collisions = 0
            total_goals_activated = 0
            total_in = total_out = total_reason = 0
            total_cost = 0.0
            run_files: list[str] = []
            won_turns: list[int] = []

            for trial in range(1, n_trials + 1):
                print(f"\n--- {effort}::{cfg_name} trial {trial}/{n_trials} ---")
                result = run_agent(
                    world_level=cfg["world"],
                    goal_level=cfg["goal"],
                    mechanics_level=cfg["mechanics"],
                    feedback_level=cfg["feedback"],
                    provider=provider,
                    model=model,
                    reasoning_effort=effort,
                    max_levels=1,
                    verbose=verbose,
                )
                run_id = f"{run_tag}_{effort}_{cfg_name}_t{trial}"
                run_files.append(str(save_result(result, run_id=run_id)))

                trial_cost = (
                    result.input_tokens / 1e6 * input_cost_per_m
                    + result.output_tokens / 1e6 * output_cost_per_m
                )
                if result.won:
                    wins += 1
                    won_turns.append(result.turns)
                total_turns += result.turns
                total_levels += result.levels_completed
                total_wall_collisions += result.wall_collisions
                total_goals_activated += result.goals_ever_activated
                total_in += result.input_tokens
                total_out += result.output_tokens
                total_reason += result.reasoning_tokens
                total_cost += trial_cost

                status = "WIN " if result.won else "LOSS"
                print(f"  → {status} | turns: {result.turns} | levels: {result.levels_completed} "
                      f"| tok={result.input_tokens}in/{result.output_tokens}out"
                      f"/{result.reasoning_tokens}reason | ${trial_cost:.4f}")

            cfg_summary: dict[str, Any] = {
                "key": f"{effort}::{cfg_name}",
                "reasoning_effort": effort,
                "config_name": cfg_name,
                "config": cfg,
                "provider": provider,
                "model": model,
                "n_trials": n_trials,
                "wins": wins,
                "win_rate": wins / n_trials,
                "avg_turns": total_turns / n_trials,
                "avg_turns_on_win": (sum(won_turns) / len(won_turns)) if won_turns else None,
                "avg_levels_completed": total_levels / n_trials,
                "avg_wall_collisions": total_wall_collisions / n_trials,
                "avg_goals_activated": total_goals_activated / n_trials,
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
                "total_reasoning_tokens": total_reason,
                "total_cost_usd": round(total_cost, 4),
                "run_files": run_files,
            }
            summary.append(cfg_summary)

            print(f"\n  ✓ {effort}::{cfg_name}: win_rate={cfg_summary['win_rate']:.0%}  "
                  f"avg_turns={cfg_summary['avg_turns']:.1f}  "
                  f"cost=${cfg_summary['total_cost_usd']:.4f}")

    # ── Save summary ──────────────────────────────────────────────────────────
    summary_path = RESULTS_DIR / f"ablation_summary_{provider}_{model.replace('/', '_')}_{run_tag}.json"
    summary_path.write_text(json.dumps({
        "provider": provider, "model": model, "n_trials": n_trials,
        "timestamp": timestamp, "env": "ls20_real_game", "env_id": "ls20",
        "input_cost_per_m": input_cost_per_m, "output_cost_per_m": output_cost_per_m,
        "results": summary,
    }, indent=2))
    print(f"\nFull summary saved → {summary_path}")

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print(f"{'cell':<28}{'N':>3}{'win%':>6}{'avgTurns':>9}{'reason_tok':>11}{'cost$':>9}")
    print("-" * 96)
    grand = 0.0
    for s in summary:
        grand += s["total_cost_usd"]
        print(f"{s['key']:<28}{s['n_trials']:>3}{s['win_rate']*100:>5.0f}%"
              f"{s['avg_turns']:>9.1f}{s['total_reasoning_tokens']:>11}{s['total_cost_usd']:>9.3f}")
    print("-" * 96)
    print(f"{'TOTAL COST':>87} ${grand:.3f}")
    print("=" * 96)


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ablation study for Environment 3 (LS20).")
    p.add_argument("--trials", type=int, default=20,
                   help="Number of trials per cell (default: 20).")
    p.add_argument("--provider", default="openrouter",
                   choices=["openrouter", "qwen-local"],
                   help="LLM provider (default: openrouter).")
    p.add_argument("--model", default="deepseek/deepseek-v4-pro",
                   help="OpenRouter model identifier.")
    p.add_argument("--configs", nargs="+", choices=list(ALL_CONFIGS.keys()),
                   help=f"Subset of configs (default: {DEFAULT_CONFIGS}).")
    p.add_argument("--reasoning-effort", nargs="+", default=["none", "medium"],
                   choices=REASONING_EFFORTS, dest="reasoning_efforts",
                   help="One or more reasoning efforts to sweep (default: none medium).")
    p.add_argument("--input-cost-per-m", type=float, default=0.30, dest="input_cost_per_m",
                   help="$/M input tokens (default: 0.30, deepseek-v4-pro).")
    p.add_argument("--output-cost-per-m", type=float, default=0.90, dest="output_cost_per_m",
                   help="$/M output+reasoning tokens (default: 0.90, deepseek-v4-pro).")
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
        reasoning_efforts=args.reasoning_efforts,
        input_cost_per_m=args.input_cost_per_m,
        output_cost_per_m=args.output_cost_per_m,
        verbose=not args.quiet,
    )
