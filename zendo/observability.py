import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Dict, List, Optional


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOGS_DIR = os.path.join(MODULE_DIR, "logs")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return mean(vals) if vals else None


def _coerce_cost(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_history_metrics(history_file: Optional[str]) -> Dict[str, Any]:
    if not history_file or not os.path.exists(history_file):
        return {
            "history_events": 0,
            "belief_trace_len": 0,
            "rule_candidate_count": 0,
            "hypothesis_revisions": None,
            "avg_belief_uncertainty": None,
            "max_belief_uncertainty": None,
            "trace_payload": None,
        }

    with open(history_file, "r") as f:
        data = json.load(f)

    events = data.get("events", []) if isinstance(data, dict) else []
    belief_trace = data.get("belief_trace", []) if isinstance(data, dict) else []
    rule_candidates = data.get("rule_candidates", []) if isinstance(data, dict) else []

    entropies = [snap.get("entropy") for snap in belief_trace if isinstance(snap, dict) and isinstance(snap.get("entropy"), (int, float))]
    revisions = 0
    for candidate in rule_candidates:
        if isinstance(candidate, dict):
            history = candidate.get("promotion_history", []) or []
            revisions += len(history)

    return {
        "history_events": len(events),
        "belief_trace_len": len(belief_trace),
        "rule_candidate_count": len(rule_candidates),
        "hypothesis_revisions": revisions if rule_candidates else None,
        "avg_belief_uncertainty": safe_mean(entropies),
        "max_belief_uncertainty": max(entropies) if entropies else None,
        "trace_payload": data,
    }


def _extract_usage(result: Dict[str, Any]) -> Dict[str, Optional[float]]:
    usage = result.get("usage") or {}
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "billed_cost_usd": _coerce_cost(usage.get("billed_cost_usd")),
        "raw_model_cost_usd": _coerce_cost(usage.get("raw_model_cost_usd")),
    }


def build_run_summary(*, result: Dict[str, Any], experiment_name: str, config: Dict[str, Any], run_index: int, logs_dir: str = DEFAULT_LOGS_DIR) -> Dict[str, Any]:
    ensure_dir(logs_dir)

    history_file = result.get("history_file")
    history_metrics = _load_history_metrics(history_file)
    usage = _extract_usage(result)

    ablated_axes = [axis for axis, level in config.items() if str(level).upper() != "EASY"]
    ablated_axis = ablated_axes[0] if len(ablated_axes) == 1 else None

    run_id = f"{experiment_name}_rule{result.get('rule_index', 0)}_run{run_index + 1:02d}"
    trace_filename = f"{run_id}_trace.json"
    trace_path = os.path.join(logs_dir, trace_filename)

    trace_payload = history_metrics.pop("trace_payload")
    if trace_payload is not None:
        with open(trace_path, "w") as f:
            json.dump(trace_payload, f, indent=2)
    else:
        trace_path = None

    total_turns = result.get("turns_taken")
    success = bool(result.get("won"))

    return {
        "run_id": run_id,
        "environment": "env1",
        "condition": experiment_name,
        "ablated_axis": ablated_axis,
        "config": config,
        "model": result.get("model"),
        "provider": result.get("provider"),
        "timestamp": utc_now_iso(),
        "turn_budget": result.get("max_turns"),
        "parameters": {},
        "success": success,
        "total_turns": total_turns,
        "turns_to_success": total_turns if success else None,
        "time_seconds": None,
        "termination_reason": "solved" if success else ("error" if result.get("errors") else "max_turns_or_failed"),
        "final_answer": result.get("understanding", {}).get("goal_understanding") if result.get("understanding") else None,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "billed_cost_usd": usage["billed_cost_usd"],
        "raw_model_cost_usd": usage["raw_model_cost_usd"],
        "hypothesis_revisions": history_metrics["hypothesis_revisions"],
        "num_queries_or_probes": history_metrics["history_events"],
        "num_propose_actions": None,
        "final_confidence": None,
        "avg_belief_uncertainty": history_metrics["avg_belief_uncertainty"],
        "max_belief_uncertainty": history_metrics["max_belief_uncertainty"],
        "trace_file": trace_path,
        "history_file": history_file,
        "errors": result.get("errors", []),
        "rule_index": result.get("rule_index"),
        "true_rule_name": result.get("true_rule_name"),
    }


def append_run_summary(summary: Dict[str, Any], *, logs_dir: str = DEFAULT_LOGS_DIR, filename: str = "env1_run_summaries.jsonl") -> str:
    ensure_dir(logs_dir)
    path = os.path.join(logs_dir, filename)
    with open(path, "a") as f:
        f.write(json.dumps(summary) + "\n")
    return path


def build_condition_summary(run_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in run_summaries:
        grouped[item["condition"]].append(item)

    conditions: Dict[str, Dict[str, Any]] = {}
    for condition, rows in grouped.items():
        success_values = [1.0 if r.get("success") else 0.0 for r in rows]
        conditions[condition] = {
            "n_runs": len(rows),
            "success_rate": safe_mean(success_values),
            "mean_total_turns": safe_mean([r.get("total_turns") for r in rows]),
            "mean_turns_to_success": safe_mean([r.get("turns_to_success") for r in rows]),
            "mean_time_seconds": safe_mean([r.get("time_seconds") for r in rows]),
            "mean_billed_cost_usd": safe_mean([r.get("billed_cost_usd") for r in rows]),
            "mean_input_tokens": safe_mean([r.get("input_tokens") for r in rows]),
            "mean_output_tokens": safe_mean([r.get("output_tokens") for r in rows]),
            "mean_hypothesis_revisions": safe_mean([r.get("hypothesis_revisions") for r in rows]),
        }

    sample = run_summaries[0] if run_summaries else {}
    return {
        "environment": sample.get("environment"),
        "model": sample.get("model"),
        "provider": sample.get("provider"),
        "generated_at": utc_now_iso(),
        "conditions": conditions,
    }


def write_condition_summary(summary: Dict[str, Any], *, logs_dir: str = DEFAULT_LOGS_DIR, filename: str = "env1_condition_summary.json") -> str:
    ensure_dir(logs_dir)
    path = os.path.join(logs_dir, filename)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path
