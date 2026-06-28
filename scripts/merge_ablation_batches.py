#!/usr/bin/env python3
"""
Merge multiple ablation batch results into combined N>5 data.

Usage:
  python scripts/merge_ablation_batches.py \
    results/ka59simple_real_ablation/ablation_..._batch1.json \
    results/ka59simple_real_ablation/ablation_..._batch2.json \
    --output results/ka59simple_real_ablation/combined_n20.json

For each (reasoning::config) cell that appears in multiple inputs, it concatenates
the trial_data lists and recomputes the aggregates (wins, win_rate, totals, etc.).

This lets you run several independent n=5 batches (possibly in parallel on different
machines or days) and combine them to reach N=20 without one giant sequential run.
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

def merge_cells(cells):
    """cells: list of the per-key summary dicts from different batches"""
    if not cells:
        return None

    # Use the first as template for static fields
    base = cells[0].copy()
    all_trials = []
    for c in cells:
        all_trials.extend(c.get("trial_data", []))

    n = len(all_trials)
    if n == 0:
        return base

    wins = sum(1 for t in all_trials if t.get("won"))
    win_rate = wins / n

    turns_list = [t["turns"] for t in all_trials if t.get("won") and "turns" in t]
    avg_turns_on_win = sum(turns_list) / len(turns_list) if turns_list else None

    levels = [t.get("levels_completed", 0) for t in all_trials]
    avg_levels = sum(levels) / len(levels) if levels else 0

    # sum the counters
    for key in ("passable_walls_total", "blocked_total", "object_pushes_total",
                "wall_transfers_total", "forced_reframes_total",
                "total_input_tokens", "total_output_tokens", "total_reasoning_tokens"):
        base[key] = sum(c.get(key, 0) for c in cells)

    # avg max_goals
    max_goals = [t.get("max_goals_occupied", 0) for t in all_trials]
    base["avg_max_goals_occupied"] = sum(max_goals) / len(max_goals) if max_goals else 0

    base["wins"] = wins
    base["trials"] = n
    base["win_rate"] = win_rate
    base["avg_turns_on_win"] = round(avg_turns_on_win, 1) if avg_turns_on_win else None
    base["avg_levels_completed"] = round(avg_levels, 2)
    base["n_trials"] = n   # note: this is now the combined N
    base["trial_data"] = all_trials

    # recompute cost using the rates that were used in the inputs (assume consistent)
    # we just keep the summed cost for now; the caller can re-apply rates if needed
    base["total_cost_usd"] = round(sum(c.get("total_cost_usd", 0) for c in cells), 4)

    # discovery etc. - simple concat average
    disc_turns = [t["discovery_turn"] for t in all_trials if t.get("discovery_turn") is not None]
    base["discovery_rate"] = len(disc_turns) / n if n else 0
    base["avg_discovery_turn"] = round(sum(disc_turns) / len(disc_turns), 1) if disc_turns else None

    return base

def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="ablation_*.json files from different n=5 batches")
    p.add_argument("--output", required=True, help="path for combined json")
    args = p.parse_args()

    combined = {
        "provider": None,
        "model": None,
        "n_trials": 0,
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f"),
        "env": None,
        "env_id": None,
        "results": {},
        "merged_from": [str(Path(x).name) for x in args.inputs],
    }

    per_key = defaultdict(list)

    for inp in args.inputs:
        data = json.load(open(inp))
        if combined["provider"] is None:
            combined["provider"] = data.get("provider")
            combined["model"] = data.get("model")
            combined["env"] = data.get("env")
            combined["env_id"] = data.get("env_id")
        combined["n_trials"] += data.get("n_trials", 0)

        for key, cell in data.get("results", {}).items():
            per_key[key].append(cell)

    for key, cell_list in per_key.items():
        merged_cell = merge_cells(cell_list)
        if merged_cell:
            combined["results"][key] = merged_cell

    Path(args.output).write_text(json.dumps(combined, indent=2))
    print(f"Wrote merged file with {len(combined['results'])} cells to {args.output}")
    print(f"Total trials across batches: {combined['n_trials']}")

if __name__ == "__main__":
    main()