import argparse
import subprocess
import os
import glob
import json

AXES = ["world", "goal", "mechanics", "feedback"]
LEVELS = ["EASY", "MEDIUM", "HARD"]
BASELINE = "EASY"

def get_latest_history_file(log_dir="./game_logs"):
    files = glob.glob(os.path.join(log_dir, "history_*.json"))
    if not files:
        return None
    return max(files, key=os.path.getctime)

def run_experiment(provider, model, turns, config, rule_index):
    cmd = [
        "python", "test_agent.py",
        "--agent", "llm",
        "--provider", provider,
        "--model", model,
        "--turns", str(turns),
        "--world", config["world"],
        "--goal", config["goal"],
        "--mechanics", config["mechanics"],
        "--feedback", config["feedback"],
        "--rule-index", str(rule_index)
    ]
    print(f"Running config: {config} (Rule {rule_index})")
    try:
        # Run test_agent.py and capture the output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # We determine if the agent won by looking for the success message in stdout
        won = "Agent won!!" in result.stdout
        history_file = get_latest_history_file()
        
        return {
            "config": config,
            "won": won,
            "history_file": history_file,
            "success": True
        }
    except subprocess.CalledProcessError as e:
        print(f"Error running config: {config}")
        print(e.stderr)
        return {
            "config": config,
            "won": False,
            "history_file": None,
            "success": False,
            "error": e.stderr
        }

def main():
    parser = argparse.ArgumentParser(description="Run single-axis ablation on Lithic Array")
    parser.add_argument("--provider", type=str, choices=["openai", "anthropic", "gemini"], default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name (e.g. gpt-4o, claude-3-5-sonnet-20241022)")
    parser.add_argument("--turns", type=int, default=50, help="Max interaction turns per game")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per configuration")
    parser.add_argument("--output", type=str, default="ablation_results.json", help="Path to save the summary results")
    
    args = parser.parse_args()
    
    # Baseline config (EASY for everything)
    base_config = {axis: BASELINE for axis in AXES}
    configs_to_run = [("baseline", base_config)]
    
    # Single axis ablations (Scaling one axis at a time)
    for axis in AXES:
        for level in ["MEDIUM", "HARD"]:
            config = base_config.copy()
            config[axis] = level
            configs_to_run.append((f"{axis}_{level}", config))
            
    results = []
    
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
                
                # Print quick feedback
                status = "WON" if res.get("won") else "FAILED"
                print(f"  Result: {status} (History saved to: {res.get('history_file')})")
            
    # Save the aggregated results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nAblation study complete. Detailed aggregated results saved to {args.output}")
    
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
