"""
ablation_study_chirality.py — Single-axis and two-axis knockout ablation runner for Env4 (Chirality).

Mirrors ablation_study_pt.py so all PT-based environments remain comparable.

Usage:
    # Single-axis (baseline + 4 knockouts, n=3 runs × 10 rules = 30 games per condition)
    python env4/ablation_study_chirality.py --provider openrouter --model openai/gpt-4o --runs 3

    # Two-axis combo
    python env4/ablation_study_chirality.py --provider openrouter --model openai/gpt-4o --runs 3 --combo world,feedback
"""

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env4.test_chirality_agent import run_chirality_llm_agent
from env4.prompts import AxisLevel

AXES = ["world", "goal", "mechanics", "feedback"]
ALL_RULES = list(range(1, 11))  # Rules 1-10

# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------

def make_baseline() -> dict:
    return {axis: "EASY" for axis in AXES}


def make_single_knockout(axis: str) -> dict:
    c = make_baseline()
    c[axis] = "HARD"
    return c


def make_combo_knockout(axis1: str, axis2: str) -> dict:
    c = make_baseline()
    c[axis1] = "HARD"
    c[axis2] = "HARD"
    return c


def condition_name(config: dict) -> str:
    hard_axes = [k for k, v in config.items() if v == "HARD"]
    if not hard_axes:
        return "baseline"
    return "+".join(sorted(hard_axes)) + "_HARD"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_condition(config: dict, provider: str, model: str, runs: int, max_turns: int, verbose: bool) -> list:
    results = []
    for run_idx in range(runs):
        for rule_index in ALL_RULES:
            seed = 42 + run_idx * 100 + rule_index
            print(f"  Rule {rule_index:2d} Run {run_idx+1}/{runs} | {condition_name(config)}", end=" ... ", flush=True)
            try:
                result = run_chirality_llm_agent(
                    provider=provider,
                    model=model,
                    max_turns=max_turns,
                    world=AxisLevel(config["world"]),
                    goal=AxisLevel(config["goal"]),
                    mechanics=AxisLevel(config["mechanics"]),
                    feedback=AxisLevel(config["feedback"]),
                    rule_index=rule_index,
                    seed=seed,
                    verbose=verbose,
                )
                result["condition"] = condition_name(config)
                result["run_idx"] = run_idx
                print(f"{'WIN' if result['won'] else 'LOSE'} | {result['turns_taken']} turns | ${result['usage'].get('cost', 0):.4f}")
                results.append(result)
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "won": False,
                    "turns_taken": max_turns,
                    "condition": condition_name(config),
                    "rule_index": rule_index,
                    "run_idx": run_idx,
                    "errors": [str(e)],
                    "usage": {"cost": 0.0},
                })
    return results


def print_summary_table(summaries: dict):
    baseline_turns = summaries.get("baseline", {}).get("avg_turns")
    print("\n" + "="*75)
    print(f"{'Condition':<30} {'Win%':>6} {'AvgTurns':>9} {'vs Base':>8} {'Cost/game':>10}")
    print("-"*75)
    for name, s in sorted(summaries.items(), key=lambda x: x[1].get("avg_turns", 0)):
        win_pct = s.get("win_rate", 0) * 100
        avg_t = s.get("avg_turns", 0)
        vs_base = f"{avg_t/baseline_turns:.2f}x" if baseline_turns and name != "baseline" else "—"
        cost = s.get("avg_cost_per_game", 0)
        n = s.get("n_games", 0)
        print(f"{name:<30} {win_pct:>5.0f}% {avg_t:>9.1f} {vs_base:>8} ${cost:>9.4f}  (n={n})")
    print("="*75)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--model", default="openai/gpt-4o")
    parser.add_argument("--runs", type=int, default=1, help="Runs per rule per condition")
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--combo", default=None, help="Two-axis combo e.g. world,feedback")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "logs", f"ablation_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Build conditions to run
    if args.combo:
        parts = [p.strip().lower() for p in args.combo.split(",")]
        if len(parts) != 2 or any(p not in AXES for p in parts):
            print(f"Invalid --combo. Must be two of: {AXES}")
            sys.exit(1)
        conditions = {
            "baseline": make_baseline(),
            condition_name(make_combo_knockout(*parts)): make_combo_knockout(*parts),
        }
    else:
        conditions = {"baseline": make_baseline()}
        for axis in AXES:
            c = make_single_knockout(axis)
            conditions[condition_name(c)] = c

    total_games = len(conditions) * args.runs * len(ALL_RULES)
    print(f"\nChirality Ablation Study")
    print(f"Model: {args.provider}/{args.model}")
    print(f"Conditions: {list(conditions.keys())}")
    print(f"Runs per condition: {args.runs} × {len(ALL_RULES)} rules = {args.runs * len(ALL_RULES)} games")
    print(f"Total games: {total_games}\n")

    all_results = []
    summaries = {}

    for cond_name, config in conditions.items():
        print(f"\n--- Condition: {cond_name} ---")
        results = run_condition(config, args.provider, args.model, args.runs, args.turns, args.verbose)
        all_results.extend(results)

        # Compute summary
        wins = sum(r["won"] for r in results)
        turns = [r["turns_taken"] for r in results]
        costs = [r.get("usage", {}).get("cost", 0) for r in results]
        avg_turns = sum(turns) / len(turns) if turns else 0
        avg_cost = sum(costs) / len(costs) if costs else 0

        summaries[cond_name] = {
            "n_games": len(results),
            "wins": wins,
            "win_rate": wins / len(results) if results else 0,
            "avg_turns": avg_turns,
            "avg_cost_per_game": avg_cost,
            "total_cost": sum(costs),
        }

    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print_summary_table(summaries)

    total_cost = sum(s["total_cost"] for s in summaries.values())
    print(f"\nTotal cost: ${total_cost:.4f} for {total_games} games")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
