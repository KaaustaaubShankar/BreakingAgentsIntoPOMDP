import argparse
import json
import datetime

from observability import append_run_summary, build_condition_summary, build_run_summary, write_condition_summary
from test_agent import run_llm_agent

AXES = ["world", "goal", "mechanics", "feedback"]
BASELINE = "EASY"
SINGLE_AXIS_HARD = "single-axis-hard"
MONDAY_ALIAS = "monday-alias"

def run_experiment(provider, model, turns, config, rule_index, runner=run_llm_agent):
    print(f"Running config: {config} (Rule {rule_index})")
    try:
        result = runner(
            provider=provider,
            model=model,
            max_turns=turns,
            world=config["world"],
            goal=config["goal"],
            mechanics=config["mechanics"],
            feedback=config["feedback"],
            rule_index=rule_index,
            verbose=False,
        )
        return {
            "config": config,
            "won": result["won"],
            "history_file": result["history_file"],
            "success": True,
            "turns_taken": result["turns_taken"],
            "errors": result.get("errors", []),
            "usage": result.get("usage", result.get("llm_usage")),
            "provider": result.get("provider"),
            "model": result.get("model"),
            "true_rule_name": result.get("true_rule_name"),
            "rule_index": result.get("rule_index"),
        }
    except Exception as e:
        print(f"Error running config: {config}")
        print(str(e))
        return {
            "config": config,
            "won": False,
            "history_file": None,
            "success": False,
            "error": str(e),
            "errors": [str(e)],
            "provider": provider,
            "model": model,
            "rule_index": rule_index,
        }

def main():
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser(description="Run single-axis ablation on Lithic Array")
    parser.add_argument("--provider", type=str, choices=["openrouter"], default="openrouter")
    parser.add_argument("--model", type=str, default="openai/gpt-4o", help="OpenRouter model name (e.g. openai/gpt-4o)")
    parser.add_argument("--turns", type=int, default=50, help="Max interaction turns per game")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per configuration")
    parser.add_argument("--matrix", choices=[SINGLE_AXIS_HARD, MONDAY_ALIAS], default=SINGLE_AXIS_HARD, help="single-axis-hard preserves Kaus's ladder, monday-alias uses knockout naming on the same HARD-only configs")
    parser.add_argument("--output", type=str, default=f"ablation_results_{date}.json", help="Path to save the summary results")
    
    args = parser.parse_args()
    
    # Baseline config (EASY for everything)
    base_config = {axis: BASELINE for axis in AXES}
    configs_to_run = [("baseline", base_config)]
    
    # Preserve Kaus's later structure: baseline + HARD-only single-axis runs.
    # Monday alias only changes naming, not semantics.
    for axis in AXES:
        config = base_config.copy()
        config[axis] = "HARD"
        experiment_name = f"{axis}_HARD"
        if args.matrix == MONDAY_ALIAS:
            experiment_name = f"{axis}_knockout"
        configs_to_run.append((experiment_name, config))
            
    results = []
    run_summaries = []
    
    for name, config in configs_to_run:
        print(f"\n{'='*50}")
        print(f"Experiment: {name.upper()}")
        print(f"{'='*50}")
        for rule_index in range(4):
            for i in range(args.runs):
                print(f"Rule {rule_index}, Run {i+1}/{args.runs}...")
                res = run_experiment(args.provider, args.model, args.turns, config, rule_index)
                res["experiment_name"] = name
                res["rule_index"] = rule_index
                res["run_index"] = i
                results.append(res)

                summary = build_run_summary(
                    result=res,
                    experiment_name=name,
                    config=config,
                    run_index=i,
                    environment="env1",
                )
                append_run_summary(summary)
                run_summaries.append(summary)
                
                # Print quick feedback
                status = "WON" if res.get("won") else "FAILED"
                print(f"  Result: {status} (History saved to: {res.get('history_file')})")
            
    # Save the aggregated results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    condition_summary = build_condition_summary(run_summaries)
    condition_summary_path = write_condition_summary(condition_summary)
        
    print(f"\nAblation study complete. Detailed aggregated results saved to {args.output}")
    print(f"Run summaries appended to zendo/logs/env1_run_summaries.jsonl")
    print(f"Condition summary written to {condition_summary_path}")
    
    # Print a neat summary summary
    print("\nSummary:")
    for name, _ in configs_to_run:
        runs = [r for r in results if r["experiment_name"] == name]
        wins = sum(1 for r in runs if r.get("won", False))
        total = len(runs)
        win_rate = (wins / total) * 100 if total > 0 else 0
        print(f"  {name.ljust(20)}: {wins}/{total} wins ({win_rate:.0f}%)")

if __name__ == "__main__":
    main()
