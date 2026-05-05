"""Merge the goal_hard re-run results with existing n=5 cells.

For each of the four (model, reasoning_effort) combos we re-ran on
2026-05-04 / 05, this script reads the per-cell ablation_*.json the harness
wrote and combines its win count with the prior n=5 from the original sweep,
producing the n=10 cell value plus a Wilson 95% CI.

This version uses the re-run log files in /tmp/jkj-rerun/ to disambiguate
which ablation_*.json belongs to which (model, reasoning) — the per-trial
JSON does not store reasoning_effort, so we have to attribute by launch.

Usage:
    python3 scripts/merge_goal_hard_rerun.py
"""
from __future__ import annotations

import glob
import json
import math
import re
from pathlib import Path

repo_root = Path(__file__).parents[1]
RES_DIR = repo_root / "results" / "ka59simple_real_ablation"
LOG_DIR = Path("/tmp/jkj-rerun")

# Existing n=5 trials from the original sweep, taken from
# `jkj results - Detailed_Results(1).csv`. These are the four cells the
# May-2026 re-run targets.
EXISTING_GOAL_HARD = {
    ("openai/gpt-5.2",     "none"):   {"wins": 4, "trials": 5},
    ("openai/gpt-5.2",     "medium"): {"wins": 1, "trials": 5},
    ("x-ai/grok-4.1-fast", "none"):   {"wins": 1, "trials": 5},
    ("x-ai/grok-4.1-fast", "medium"): {"wins": 3, "trials": 5},
}

# Maps (model, reasoning) -> log filename (without .log extension).
LAUNCH_LOG = {
    ("openai/gpt-5.2",     "none"):   "gpt52_no",
    ("openai/gpt-5.2",     "medium"): "gpt52_med",
    ("x-ai/grok-4.1-fast", "none"):   "grok_no",
    ("x-ai/grok-4.1-fast", "medium"): "grok_med",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def find_ablation_path_from_log(log_name: str) -> Path | None:
    """Each run prints the absolute path of the ablation_*.json it wrote.

    Pull it out of the log so we can attribute the JSON to a specific
    (model, reasoning) combo without relying on file mtime.
    """
    log_path = LOG_DIR / f"{log_name}.log"
    if not log_path.exists():
        return None
    text = log_path.read_text()
    m = re.search(r"Results saved → (\S+)", text)
    if not m:
        return None
    return Path(m.group(1))


def parse_cell(json_path: Path) -> dict | None:
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return None
    cell = data.get("results", {}).get("goal_hard")
    if not cell:
        return None
    return {
        "wins": cell.get("wins", 0),
        "trials": cell.get("trials", 0),
        "wall_transfers_total": cell.get("wall_transfers_total"),
        "passable_walls_total": cell.get("passable_walls_total"),
        "avg_max_goals_occupied": cell.get("avg_max_goals_occupied"),
    }


def main() -> None:
    print(f"{'Model':<22} {'R':<7} {'old k/n':>9} {'new k/n':>9} {'merged k/n':>11} {'merged %':>9} {'CI':>14}")
    print("-" * 90)

    rows: list[dict] = []
    for combo, prior in EXISTING_GOAL_HARD.items():
        log_name = LAUNCH_LOG[combo]
        path = find_ablation_path_from_log(log_name)
        if path is None or not path.exists():
            print(f"{combo[0]:<22} {combo[1]:<7} {prior['wins']}/{prior['trials']:<7} (re-run still running or no log path yet)")
            continue
        new = parse_cell(path)
        if new is None:
            print(f"{combo[0]:<22} {combo[1]:<7} {prior['wins']}/{prior['trials']:<7} (ablation_*.json missing goal_hard cell)")
            continue
        merged_wins = prior["wins"] + new["wins"]
        merged_trials = prior["trials"] + new["trials"]
        lo, hi = wilson(merged_wins, merged_trials)
        pct = merged_wins / merged_trials * 100 if merged_trials else 0.0
        ci = f"[{int(round(lo*100)):>3d}, {int(round(hi*100)):>3d}]"
        print(
            f"{combo[0]:<22} {combo[1]:<7} "
            f"{prior['wins']}/{prior['trials']:<7} "
            f"{new['wins']}/{new['trials']:<7} "
            f"{merged_wins}/{merged_trials:<9} "
            f"{pct:>7.1f}% {ci}"
        )
        rows.append({
            "combo": combo, "prior": prior, "new": new,
            "merged": (merged_wins, merged_trials), "ci": (lo, hi),
        })

    if not rows:
        return
    print()
    print("LaTeX-ready Goal-hard row for Table 1 (paste into ka59simple-wins):")
    pieces = []
    for combo in [("openai/gpt-5.2", "none"), ("openai/gpt-5.2", "medium"),
                  ("x-ai/grok-4.1-fast", "none"), ("x-ai/grok-4.1-fast", "medium")]:
        match = next((r for r in rows if r["combo"] == combo), None)
        if match:
            k, n = match["merged"]
            lo, hi = match["ci"]
            pieces.append(f"{k}/{n} [{int(round(lo*100))},{int(round(hi*100))}]")
        else:
            pieces.append("(pending)")
    print("Goal hard      &  --         & " + " & ".join(pieces) + " \\\\")


if __name__ == "__main__":
    main()
