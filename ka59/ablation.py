"""
ablation.py — Multi-env ablation study runner.

Mirrors the Kaus env4 ablation structure for LS20 / KA59 / BP35 and produces
the unified Josh table (Win%/Turns/Levels/InvalidClicks/Flips/RelDiff).

Ablation configs (same 5 as env3/env4):
  baseline       — all axes EASY
  world_hard     — world HARD, rest EASY
  goal_hard      — goal HARD, rest EASY
  mechanics_hard — mechanics HARD, rest EASY
  feedback_hard  — feedback HARD, rest EASY

Usage:
  # Run all envs, 3 trials, free Llama model
  python ablation.py

  # BP35 only, 5 trials, Sonnet
  python ablation.py --envs bp35 --trials 5 --model anthropic/claude-sonnet-4-6

  # 64-turn Flash/Sonnet proactive KA59 runs
  python ablation.py --envs ka59 --model google/gemini-2.0-flash --trials 3

  # Show a pre-baked sample table (no LLM calls)
  python ablation.py --sample
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from runner import run_one, SUPPORTED_ENVS, KA59_DEFAULT_SCENARIO, KA59_DEFAULT_MAX_STEPS
from unified_result import AblationRow, UnifiedRunResult, build_rows_with_rel_diff, print_josh_table

# ── results directory ─────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── ablation configs ──────────────────────────────────────────────────────────

ALL_CONFIGS: dict[str, dict[str, str]] = {
    "baseline":       {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"},
    "world_hard":     {"world": "HARD", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"},
    "goal_hard":      {"world": "EASY", "goal": "HARD", "mechanics": "EASY", "feedback": "EASY"},
    "mechanics_hard": {"world": "EASY", "goal": "EASY", "mechanics": "HARD", "feedback": "EASY"},
    "feedback_hard":  {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "HARD"},
}


# ── sample data (for --sample flag, no LLM calls needed) ─────────────────────

_SAMPLE_ROWS_RAW = [
    # (env, cfg, model, wins, n, avg_turns, avg_levels, avg_inv, avg_flips)
    ("bp35", "baseline",       "llama-3.3-70b:free", 2, 3, 68.0, 0.67, 3.3, 1.0),
    ("bp35", "world_hard",     "llama-3.3-70b:free", 0, 3, 128.0, 0.0, 8.7, 0.0),
    ("bp35", "goal_hard",      "llama-3.3-70b:free", 1, 3, 92.0, 0.33, 4.1, 0.7),
    ("bp35", "mechanics_hard", "llama-3.3-70b:free", 0, 3, 128.0, 0.0, 12.3, 0.3),
    ("bp35", "feedback_hard",  "llama-3.3-70b:free", 1, 3, 101.0, 0.33, 5.2, 0.7),
    ("ls20", "baseline",       "llama-3.3-70b:free", 1, 3, 52.0, 0.33, 6.7, 0.7),
    ("ls20", "world_hard",     "llama-3.3-70b:free", 0, 3, 100.0, 0.0, 18.3, 0.3),
    ("ls20", "goal_hard",      "llama-3.3-70b:free", 0, 3, 88.3, 0.0, 9.0, 0.3),
    ("ls20", "mechanics_hard", "llama-3.3-70b:free", 0, 3, 100.0, 0.0, 14.0, 0.0),
    ("ls20", "feedback_hard",  "llama-3.3-70b:free", 1, 3, 61.0, 0.33, 7.3, 0.7),
    ("ka59", "baseline",       "llama-3.3-70b:free", 0, 3, 64.0, 0.0, 38.7, 0.0),
    ("ka59", "world_hard",     "llama-3.3-70b:free", 0, 3, 64.0, 0.0, 52.0, 0.0),
    ("ka59", "goal_hard",      "llama-3.3-70b:free", 0, 3, 64.0, 0.0, 42.0, 0.0),
    ("ka59", "mechanics_hard", "llama-3.3-70b:free", 0, 3, 64.0, 0.0, 60.0, 0.0),
    ("ka59", "feedback_hard",  "llama-3.3-70b:free", 0, 3, 64.0, 0.0, 40.0, 0.0),
]


def _make_sample_rows() -> list[AblationRow]:
    rows = []
    for (env, cfg, model, wins, n, avg_turns, avg_levels, avg_inv, avg_flips) in _SAMPLE_ROWS_RAW:
        rows.append(AblationRow(
            env=env,
            config_name=cfg,
            config=ALL_CONFIGS[cfg],
            provider="openrouter",
            model=model,
            n_trials=n,
            wins=wins,
            win_rate=wins / n,
            avg_turns=avg_turns,
            avg_levels=avg_levels,
            avg_invalid_clicks=avg_inv,
            avg_flips=avg_flips,
            relative_difficulty=None,
        ))
    return build_rows_with_rel_diff(rows)


# ── ablation runner ───────────────────────────────────────────────────────────

def run_ablation(
    envs: list[str],
    n_trials: int = 3,
    provider: str = "openrouter",
    model: str = "meta-llama/llama-3.3-70b-instruct:free",
    config_names: Optional[list[str]] = None,
    world_easy_format: str = "v2",
    ka59_scenario: str = KA59_DEFAULT_SCENARIO,
    ka59_max_steps: int = KA59_DEFAULT_MAX_STEPS,
    verbose: bool = False,
) -> list[AblationRow]:
    """
    Run the ablation study across the specified environments.

    Returns a list of AblationRow (one per env × config combination) with
    relative_difficulty already computed within each (env, model) group.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    configs_to_run = {
        k: v for k, v in ALL_CONFIGS.items()
        if config_names is None or k in config_names
    }

    all_runs: list[UnifiedRunResult] = []

    for env_name in envs:
        for cfg_name, cfg in configs_to_run.items():
            print(f"\n{'=' * 62}")
            print(f"  Env: {env_name}  Config: {cfg_name}  Model: {model}")
            print(f"  {cfg}")
            print(f"{'=' * 62}")

            for trial in range(1, n_trials + 1):
                print(f"\n--- Trial {trial}/{n_trials} ---")
                run = run_one(
                    env_name=env_name,
                    config_name=cfg_name,
                    config=cfg,
                    provider=provider,
                    model=model,
                    world_easy_format=world_easy_format,
                    ka59_scenario=ka59_scenario,
                    max_turns=ka59_max_steps if env_name == "ka59" else None,
                    verbose=verbose,
                )
                all_runs.append(run)
                status = "WIN " if run.won else "LOSS"
                print(
                    f"  → {status} | turns={run.turns} | levels={run.levels_completed} "
                    f"| inv={run.invalid_clicks} | flips={run.flips}"
                )

                # Save individual run
                _save_run(run, f"{timestamp}_{env_name}_{cfg_name}_t{trial}")

    # Group into AblationRows
    rows: list[AblationRow] = _aggregate_rows(all_runs, n_trials)
    rows = build_rows_with_rel_diff(rows)

    # Save summary JSON
    summary_path = RESULTS_DIR / f"ablation_summary_{timestamp}.json"
    _save_summary(rows, summary_path)
    print(f"\nSummary saved → {summary_path}")

    return rows


def _aggregate_rows(
    runs: list[UnifiedRunResult],
    n_trials: int,
) -> list[AblationRow]:
    """Group runs by (env, config_name, model) and build AblationRows."""
    from collections import defaultdict

    groups: dict[tuple, list[UnifiedRunResult]] = defaultdict(list)
    for r in runs:
        key = (r.env, r.config_name, r.model)
        groups[key].append(r)

    rows: list[AblationRow] = []
    for (env, cfg_name, model), group_runs in groups.items():
        rows.append(AblationRow.from_runs(group_runs))
    return rows


def _save_run(run: UnifiedRunResult, run_id: str) -> None:
    env_dir = RESULTS_DIR / run.env
    env_dir.mkdir(exist_ok=True)
    path = env_dir / f"run_{run_id}.json"
    payload = {
        "env": run.env,
        "config_name": run.config_name,
        "config": run.config,
        "provider": run.provider,
        "model": run.model,
        "won": run.won,
        "turns": run.turns,
        "levels_completed": run.levels_completed,
        "invalid_clicks": run.invalid_clicks,
        "flips": run.flips,
        "errors": run.errors,
        "hypo_trace": run.hypo_trace,
    }
    path.write_text(json.dumps(payload, indent=2))


def _save_summary(rows: list[AblationRow], path: Path) -> None:
    payload = [
        {
            "env": r.env,
            "config_name": r.config_name,
            "model": r.model,
            "n_trials": r.n_trials,
            "wins": r.wins,
            "win_rate": r.win_rate,
            "avg_turns": r.avg_turns,
            "avg_levels": r.avg_levels,
            "avg_invalid_clicks": r.avg_invalid_clicks,
            "avg_flips": r.avg_flips,
            "relative_difficulty": r.relative_difficulty,
        }
        for r in rows
    ]
    path.write_text(json.dumps(payload, indent=2))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-env ablation (LS20/KA59/BP35).")
    p.add_argument("--envs", nargs="+", default=list(SUPPORTED_ENVS),
                   choices=list(SUPPORTED_ENVS),
                   help="Which environments to run (default: all).")
    p.add_argument("--trials", type=int, default=3,
                   help="Trials per config (default: 3).")
    p.add_argument("--provider", default="openrouter", choices=["openrouter", "anthropic", "claude-cli"])
    p.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct:free")
    p.add_argument("--configs", nargs="+", choices=list(ALL_CONFIGS.keys()))
    p.add_argument("--world-easy-format", default="v2", choices=["v1", "v2"],
                   help="BP35 observation format (v2 adds action_affordances/valid_targets).")
    p.add_argument("--ka59-scenario", default=KA59_DEFAULT_SCENARIO,
                   help=f"KA59 scenario (default: {KA59_DEFAULT_SCENARIO}).")
    p.add_argument("--ka59-max-steps", type=int, default=KA59_DEFAULT_MAX_STEPS,
                   help=f"KA59 turn budget (default: {KA59_DEFAULT_MAX_STEPS}).")
    p.add_argument("--sample", action="store_true",
                   help="Print the pre-baked sample Josh table (no LLM calls).")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.sample:
        rows = _make_sample_rows()
        print_josh_table(rows, title="Multi-Env Ablation — SAMPLE (no live runs)")
        return

    rows = run_ablation(
        envs=args.envs,
        n_trials=args.trials,
        provider=args.provider,
        model=args.model,
        config_names=args.configs,
        world_easy_format=args.world_easy_format,
        ka59_scenario=args.ka59_scenario,
        ka59_max_steps=args.ka59_max_steps,
        verbose=not args.quiet,
    )
    print_josh_table(rows, title=f"Multi-Env Ablation — {args.model}")


if __name__ == "__main__":
    main()
