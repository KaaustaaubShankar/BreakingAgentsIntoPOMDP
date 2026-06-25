"""
ablation.py — Ablation study runner for Environment 4 (BP35).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiment import run_agent, save_result

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

REASONING_EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh"]

ALL_CONFIGS: dict[str, dict[str, str]] = {
    "baseline": {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"},
    "world_hard": {"world": "HARD", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"},
    "goal_hard": {"world": "EASY", "goal": "HARD", "mechanics": "EASY", "feedback": "EASY"},
    "mechanics_hard": {"world": "EASY", "goal": "EASY", "mechanics": "HARD", "feedback": "EASY"},
    "feedback_hard": {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "HARD"},
}


class _PrefixedWriter:
    """Line-buffering stdout wrapper that tags each line with a trial prefix.

    In parallel mode several worker processes share the terminal. Buffering
    until a newline and writing the full `"[tN] line\\n"` in one call keeps the
    per-turn logs readable and attributable instead of garbled mid-line.
    """

    def __init__(self, prefix: str, stream: Any) -> None:
        self.prefix = prefix
        self.stream = stream
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.stream.write(f"{self.prefix}{line}\n")
        self.stream.flush()
        return len(text)

    def flush(self) -> None:
        if self._buf:
            self.stream.write(f"{self.prefix}{self._buf}")
            self._buf = ""
        self.stream.flush()


def _run_single_trial(task: dict[str, Any]) -> dict[str, Any]:
    """Run one trial in isolation and persist it.

    Executed in a separate process: it builds its own env (via run_agent ->
    _make_env), saves its own result file, and returns only a small summary so
    the large `history` payload never crosses the process boundary.
    """
    cfg = task["cfg"]

    def _do() -> Any:
        result = run_agent(
            world_level=cfg["world"],
            goal_level=cfg["goal"],
            mechanics_level=cfg["mechanics"],
            feedback_level=cfg["feedback"],
            provider=task["provider"],
            model=task["model"],
            reasoning_effort=task["reasoning_effort"],
            max_levels=1,
            verbose=task["verbose"],
        )
        path = save_result(result, run_id=task["run_id"])
        return result, path

    prefix = task.get("prefix")
    if prefix:
        writer = _PrefixedWriter(prefix, sys.stdout)
        with redirect_stdout(writer):
            result, path = _do()
        writer.flush()
    else:
        result, path = _do()

    return {
        "trial": task["trial"],
        "won": result.won,
        "turns": result.turns,
        "levels_completed": result.levels_completed,
        "invalid_actions": result.invalid_actions,
        "click_actions": result.click_actions,
        "gravity_flips": result.gravity_flips,
        "undos": result.undos,
        "cost": _extract_cost(result),
        "run_file": str(path),
    }


def _print_trial(tr: dict[str, Any]) -> None:
    status = "WIN " if tr["won"] else "LOSS"
    print(
        f"  [t{tr['trial']}] -> {status} | turns: {tr['turns']} | "
        f"levels: {tr['levels_completed']} | invalid: {tr['invalid_actions']}"
    )


def run_ablation(
    n_trials: int = 5,
    provider: str = "openrouter",
    model: str = "meta-llama/llama-3.3-70b-instruct:free",
    reasoning_efforts: list[str] | None = None,
    config_names: list[str] | None = None,
    max_workers: int = 4,
    verbose: bool = True,
) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    efforts_to_run = [effort.lower().strip() for effort in (reasoning_efforts or ["none"])]
    invalid_efforts = [effort for effort in efforts_to_run if effort not in REASONING_EFFORTS]
    if invalid_efforts:
        raise ValueError(
            f"Invalid reasoning_effort value(s): {invalid_efforts}. Expected one or more of {REASONING_EFFORTS}."
        )
    configs_to_run = {
        key: value for key, value in ALL_CONFIGS.items() if config_names is None or key in config_names
    }

    summary: list[dict[str, Any]] = []
    for reasoning_effort in efforts_to_run:
        print(f"\n{'#' * 62}")
        print(f"  Reasoning effort: {reasoning_effort}")
        print(f"{('#' * 62)}")

        for cfg_name, cfg in configs_to_run.items():
            print(f"\n{'=' * 62}")
            print(f"  Config: {cfg_name}")
            print(f"  {cfg}")
            print(f"{'=' * 62}")

            # Each trial runs in its own process with its own env (no shared
            # state). Per-turn logging stays on; in parallel mode each worker's
            # output is tagged with its trial number so concurrent runs stay
            # readable instead of garbled.
            tasks = [
                {
                    "trial": trial,
                    "cfg": cfg,
                    "run_id": f"{timestamp}_{cfg_name}_{reasoning_effort}_t{trial}",
                    "provider": provider,
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "verbose": verbose,
                    "prefix": f"[t{trial}] " if max_workers > 1 else None,
                }
                for trial in range(1, n_trials + 1)
            ]

            trial_results: dict[int, dict[str, Any]] = {}
            if max_workers == 1:
                for task in tasks:
                    print(f"\n--- Trial {task['trial']}/{n_trials} ---")
                    tr = _run_single_trial(task)
                    trial_results[tr["trial"]] = tr
                    _print_trial(tr)
            else:
                print(f"\n  Running {n_trials} trials with up to {max_workers} workers...")
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_run_single_trial, t): t["trial"] for t in tasks}
                    for future in as_completed(futures):
                        trial_no = futures[future]
                        try:
                            tr = future.result()
                        except Exception as exc:
                            print(f"  [t{trial_no}] FAILED: {exc}")
                            continue
                        trial_results[tr["trial"]] = tr
                        _print_trial(tr)

            # Aggregate in trial order so the saved summary is deterministic
            # regardless of completion order.
            wins = 0
            total_turns = 0
            total_levels = 0
            total_invalid_actions = 0
            total_clicks = 0
            total_gravity_flips = 0
            total_undos = 0
            total_cost = 0.0
            run_files: list[str] = []
            run_costs: list[float] = []

            for trial in range(1, n_trials + 1):
                tr = trial_results.get(trial)
                if tr is None:
                    continue
                run_files.append(tr["run_file"])
                run_costs.append(tr["cost"])
                total_cost += tr["cost"]
                if tr["won"]:
                    wins += 1
                total_turns += tr["turns"]
                total_levels += tr["levels_completed"]
                total_invalid_actions += tr["invalid_actions"]
                total_clicks += tr["click_actions"]
                total_gravity_flips += tr["gravity_flips"]
                total_undos += tr["undos"]

            cfg_summary: dict[str, Any] = {
                "config_name": cfg_name,
                "config": cfg,
                "provider": provider,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "n_trials": n_trials,
                "wins": wins,
                "win_rate": wins / n_trials,
                "avg_turns": total_turns / n_trials,
                "avg_levels_completed": total_levels / n_trials,
                "avg_invalid_actions": total_invalid_actions / n_trials,
                "avg_click_actions": total_clicks / n_trials,
                "avg_gravity_flips": total_gravity_flips / n_trials,
                "avg_undos": total_undos / n_trials,
                "total_cost": total_cost,
                "avg_cost": total_cost / n_trials,
                "run_costs": run_costs,
                "run_files": run_files,
            }
            summary.append(cfg_summary)

            print(
                f"\n  {cfg_name}: win_rate={cfg_summary['win_rate']:.0%} "
                f"avg_turns={cfg_summary['avg_turns']:.1f} "
                f"avg_levels={cfg_summary['avg_levels_completed']:.1f} "
                f"clicks={cfg_summary['avg_click_actions']:.1f} "
                f"gravity_flips={cfg_summary['avg_gravity_flips']:.1f}"
                f" cost={cfg_summary['avg_cost']}"
            )

    baseline_turns_by_effort = {
        item["reasoning_effort"]: item["avg_turns"]
        for item in summary
        if item["config_name"] == "baseline"
    }
    for item in summary:
        baseline_turns = baseline_turns_by_effort.get(item["reasoning_effort"])
        if baseline_turns and baseline_turns > 0 and item["avg_turns"] > 0:
            item["relative_difficulty"] = round(item["avg_turns"] / baseline_turns, 3)
        else:
            item["relative_difficulty"] = None

    summary_path = RESULTS_DIR / f"ablation_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nFull summary saved -> {summary_path}")

    print("\n" + "=" * 110)
    print(
        f"{ 'Config':<20}  {'Reasoning':<10}  {'Win%':>5}  {'Avg turns':>9}  {'Avg levels':>10}  "
        f"{'Invalid':>7}  {'Clicks':>7}  {'Flips':>7}  {'Undos':>7}  {'Rel. diff':>9}  {'Cost'}"
    )
    print("-" * 110)
    for item in summary:
        rel = f"{item['relative_difficulty']:.2f}x" if item['relative_difficulty'] is not None else "n/a"
        print(
            f"{item['config_name']:<20}  "
            f"{item['reasoning_effort']:<10}  "
            f"{item['win_rate']:>5.0%}  "
            f"{item['avg_turns']:>9.1f}  "
            f"{item['avg_levels_completed']:>10.1f}  "
            f"{item['avg_invalid_actions']:>7.1f}  "
            f"{item['avg_click_actions']:>7.1f}  "
            f"{item['avg_gravity_flips']:>7.1f}  "
            f"{item['avg_undos']:>7.1f}  "
            f"{rel:>9}  "
            f"{item.get('avg_cost', 0.0)}"
        )
    print("=" * 110)


def _extract_cost(result: Any) -> float:
    """Try several common shapes to extract a numeric cost from a result object.

    Returns 0.0 when no cost can be found or conversion fails.
    """
    usage = None
    # dict-like
    try:
        if isinstance(result, dict):
            usage = result.get("usage")
        else:
            usage = getattr(result, "usage", None)
    except Exception:
        usage = None

    cost = None
    if isinstance(usage, dict):
        cost = usage.get("cost")
    else:
        cost = getattr(usage, "cost", None) if usage is not None else None

    # fallback into raw nested dicts sometimes returned by clients
    if cost is None:
        try:
            raw = result.get("raw") if isinstance(result, dict) else getattr(result, "raw", None)
            if isinstance(raw, dict):
                usage2 = raw.get("usage")
                if isinstance(usage2, dict):
                    cost = usage2.get("cost")
        except Exception:
            pass

    try:
        return float(cost) if cost is not None else 0.0
    except Exception:
        return 0.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ablation study for Environment 4.")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--provider", default="openrouter", choices=["openrouter", "qwen-local"])
    p.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct:free")
    p.add_argument(
        "--reasoning-effort",
        nargs="+",
        default=["none", "medium"],
        choices=REASONING_EFFORTS,
        help="One or more reasoning effort levels to test.",
    )
    p.add_argument("--configs", nargs="+", choices=list(ALL_CONFIGS.keys()))
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel trials per config (each in its own process/env). "
        "Use 1 for sequential. Keep modest to avoid API rate limits.",
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ablation(
        n_trials=args.trials,
        provider=args.provider,
        model=args.model,
        reasoning_efforts=args.reasoning_effort,
        config_names=args.configs,
        max_workers=args.workers,
        verbose=not args.quiet,
    )
