"""Deterministic KA59-Simple camera-ready evidence audit.

This script reads only committed raw artifacts. It never imports an LLM client,
loads credentials, or makes a network request. The JSON manifest and Markdown
truth table are generated from the same in-memory records.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "ka59_game" / "results"
GPT_UNIVERSE_PATH = REPO_ROOT / "camera_ready" / "ka59_gpt_complete_trial_universe.json"
CAMERA_READY_DIR = REPO_ROOT / "camera_ready"
MANIFEST_PATH = CAMERA_READY_DIR / "ka59_camera_ready_manifest.json"
TRUTH_PATH = CAMERA_READY_DIR / "KA59_CAMERA_READY_TRUTH.md"

PAPER_CONFIGS = (
    "baseline",
    "world_hard",
    "mechanics_hard",
    "mechanics_hard_format_only",
    "feedback_hard",
)
CONFIG_ALIASES = {
    "mech_hard": "mechanics_hard",
    "mech_norules": "mechanics_hard_format_only",
    "mechanics_no_rules": "mechanics_hard_format_only",
}

CONFIG_LEVELS: dict[str, dict[str, str]] = {
    "baseline": {
        "world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"
    },
    "world_hard": {
        "world": "HARD", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"
    },
    "mechanics_hard": {
        "world": "EASY", "goal": "EASY", "mechanics": "HARD", "feedback": "EASY"
    },
    "mechanics_hard_format_only": {
        "world": "EASY", "goal": "EASY", "mechanics": "HARD_FORMAT_ONLY",
        "feedback": "EASY"
    },
    "feedback_hard": {
        "world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "HARD"
    },
}

LEGACY_INFRASTRUCTURE_MARKERS = (
    "error code: 401",
    "error code: 402",
    "error code: 403",
    "error code: 408",
    "error code: 409",
    "error code: 429",
    "error code: 500",
    "error code: 502",
    "error code: 503",
    "error code: 504",
    "insufficient balance",
    "insufficient credits",
    "authentication",
    "unauthorized",
    "rate limit",
    "rate_limit",
    "timed out",
    "timeout",
    "connection error",
    "provider error",
    "api error",
    "openrouter returned empty",
)
INFRASTRUCTURE_MARKERS = (
    *LEGACY_INFRASTRUCTURE_MARKERS,
    "returned empty content",
    "provider empty response",
)
PARSE_MARKERS = (
    "no json object found",
    "invalid json",
    "jsondecodeerror",
    "parse error",
)
ENVIRONMENT_MARKERS = (
    "failed to initialise game",
    "env.step raised",
    "environment error",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _canonical_config(name: str) -> str:
    return CONFIG_ALIASES.get(name, name)


def _config_from_levels(config: dict[str, str]) -> str | None:
    levels = (
        config.get("world"),
        config.get("goal"),
        config.get("mechanics"),
        config.get("feedback"),
    )
    mapping = {
        ("EASY", "EASY", "EASY", "EASY"): "baseline",
        ("HARD", "EASY", "EASY", "EASY"): "world_hard",
        ("EASY", "EASY", "HARD", "EASY"): "mechanics_hard",
        ("EASY", "EASY", "HARD_FORMAT_ONLY", "EASY"):
            "mechanics_hard_format_only",
        ("EASY", "EASY", "EASY", "HARD"): "feedback_hard",
    }
    return mapping.get(levels)


def _config_from_filename(path: Path) -> str | None:
    for name in sorted(PAPER_CONFIGS, key=len, reverse=True):
        if f"_{name}_" in path.name:
            return name
    return None


def _effort_from_filename(path: Path) -> str | None:
    match = re.search(r"_(none|medium)_t\d+\.json$", path.name)
    return match.group(1) if match else None


def _normalise_for_digest(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalise_for_digest(item)
            for key, item in sorted(value.items())
            if key not in {"timestamp", "trial", "run_id"}
        }
    if isinstance(value, list):
        return [_normalise_for_digest(item) for item in value]
    return value


def _semantic_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _normalise_for_digest(payload), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fatal_errors_with_markers(
    payload: dict[str, Any], infrastructure_markers: tuple[str, ...]
) -> tuple[str | None, list[str]]:
    errors = [str(error) for error in payload.get("errors") or []]
    relevant = [
        error for error in errors
        if not error.lower().startswith("understanding prompt failed")
    ]
    lowered = "\n".join(relevant).lower()
    if any(marker in lowered for marker in ENVIRONMENT_MARKERS):
        return "environment_error", relevant
    if any(marker in lowered for marker in infrastructure_markers):
        return "infrastructure_error", relevant
    if any(marker in lowered for marker in PARSE_MARKERS):
        return "parse_error", relevant
    return None, relevant


def _fatal_errors(payload: dict[str, Any]) -> tuple[str | None, list[str]]:
    """Return the strict-error-free exclusion used by the resume-safe runner."""
    return _fatal_errors_with_markers(payload, INFRASTRUCTURE_MARKERS)


def _legacy_fatal_errors(payload: dict[str, Any]) -> tuple[str | None, list[str]]:
    """Reproduce the pre-correction classifier for an auditable migration."""
    return _fatal_errors_with_markers(payload, LEGACY_INFRASTRUCTURE_MARKERS)


def _parse_turn(error: str) -> int | None:
    match = re.match(r"Turn (\d+):", error)
    return int(match.group(1)) if match else None


def _raw_response_from_parse_error(error: str) -> str | None:
    marker = "response:\n"
    if marker not in error:
        return None
    encoded = error.split(marker, 1)[1].strip()
    try:
        value = ast.literal_eval(encoded)
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, str) else None


def _documented_json_object(text: str) -> dict[str, Any] | None:
    """Apply the JSON-object forms accepted by the historical parser."""
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    embedded = re.search(r"\{.*\}", text, re.DOTALL)
    if embedded:
        candidates.append(embedded.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_review(payload: dict[str, Any]) -> dict[str, Any]:
    """Classify parse-bearing turns without inferring absent response content."""
    _, relevant_errors = _fatal_errors(payload)
    parse_errors = [
        error for error in relevant_errors
        if any(marker in error.lower() for marker in PARSE_MARKERS)
    ]
    infrastructure_parse_errors = [
        error for error in parse_errors
        if any(marker in error.lower() for marker in INFRASTRUCTURE_MARKERS)
    ]
    actual_parse_errors = [
        error for error in parse_errors if error not in infrastructure_parse_errors
    ]
    raw_responses = [
        response
        for error in actual_parse_errors
        if (response := _raw_response_from_parse_error(error)) is not None
    ]

    parse_failure_classification: str | None = None
    raw_response_evidence: str | None = None
    if actual_parse_errors:
        if not raw_responses:
            parse_failure_classification = "indeterminate"
            raw_response_evidence = "raw response content absent from persisted error/history"
        elif all(_documented_json_object(response) is not None for response in raw_responses):
            parse_failure_classification = "harness_parse_failure"
            raw_response_evidence = "persisted response is valid under documented historical parser forms"
        elif len(raw_responses) == len(actual_parse_errors):
            parse_failure_classification = "model_protocol_failure"
            raw_response_evidence = "persisted nonempty response is invalid under documented JSON-object protocol"
        else:
            parse_failure_classification = "indeterminate"
            raw_response_evidence = "some parse-bearing turns lack persisted raw response content"

    all_parse_turns = sorted({
        turn for error in parse_errors
        if (turn := _parse_turn(error)) is not None
    })
    actual_parse_turns = sorted({
        turn for error in actual_parse_errors
        if (turn := _parse_turn(error)) is not None
    })
    action_turns = [
        int(event["turn"])
        for event in payload.get("history") or []
        if event.get("type") == "action" and isinstance(event.get("turn"), int)
    ]
    last_parse_turn = max(all_parse_turns) if all_parse_turns else None
    actions_after_last_parse = sum(
        turn > last_parse_turn for turn in action_turns
    ) if last_parse_turn is not None else 0
    disposition = None
    if infrastructure_parse_errors:
        disposition = "infrastructure_failure"
    if actual_parse_errors:
        disposition = parse_failure_classification
    return {
        "parse_error_bearing": bool(parse_errors),
        "parse_error_count": len(parse_errors),
        "parse_first_turn": min(all_parse_turns) if all_parse_turns else None,
        "parse_last_turn": last_parse_turn,
        "parse_turns": actual_parse_turns,
        "parse_review_disposition": disposition,
        "parse_failure_classification": parse_failure_classification,
        "infrastructure_parse_error_count": len(infrastructure_parse_errors),
        "actual_parse_error_count": len(actual_parse_errors),
        "raw_response_evidence": raw_response_evidence,
        "raw_response_excerpts": raw_responses[:3],
        "actions_after_last_parse": actions_after_last_parse,
        "runner_recovered_after_parse": actions_after_last_parse > 0,
    }


def _history_events(payload: dict[str, Any], event_type: str) -> list[dict[str, Any]]:
    return [
        event for event in payload.get("history") or []
        if event.get("type") == event_type
    ]


def _turn_budget(payload: dict[str, Any]) -> tuple[int, str]:
    budgets: list[int] = []
    for event in payload.get("history") or []:
        resources = (event.get("state_before") or {}).get("resources") or {}
        value = resources.get("step_budget")
        if isinstance(value, int):
            budgets.append(value)
    if budgets:
        return Counter(budgets).most_common(1)[0][0], "observed"
    return 64, "historically_recovered_from_runner_and_batch"


def _attempt_policy(payload: dict[str, Any]) -> tuple[int, str]:
    attempts = [
        int(event["attempt"])
        for event in payload.get("history") or []
        if isinstance(event.get("attempt"), int)
    ]
    if attempts:
        return max(attempts), "observed"
    return 2, "historically_recovered_from_runner_and_batch"


def _historical_code_revision(timestamp: str) -> dict[str, str]:
    # Core two-attempt implementation did not change between these result sets;
    # later instrumentation did. Raw files do not log a Git SHA, so this is
    # explicitly recovered from repository history rather than called observed.
    if timestamp < "2026-06-13":
        experiment = "da9964f"
    else:
        experiment = "604e40e"
    return {
        "experiment_revision_recovered": experiment,
        "prompt_revision_recovered": "ff0c184",
        "environment_path_recovered":
            "environment_files/ka59simple/20260430",
        "environment_revision_observed": None,
    }


def _batch_for_gpt(path: Path, config: str, effort: str) -> str:
    if config != "mechanics_hard_format_only":
        return (
            "ka59_game/results/gpt-5.2/"
            "ablation_openrouter_openai_gpt-5.2_20260513T055440_493226.json"
        )
    if effort == "none":
        return (
            "ka59_game/results/gpt-5.2/"
            "ablation_openrouter_openai_gpt-5.2_20260601T020500_243047.json"
        )
    return (
        "ka59_game/results/gpt-5.2/"
        "ablation_openrouter_openai_gpt-5.2_20260620T213830_707671_p35675.json"
    )


def _record(
    path: Path,
    *,
    config_hint: str | None = None,
    effort_hint: str | None = None,
    batch: str,
    canonical_membership: str,
) -> dict[str, Any]:
    payload = _load_json(path)
    config = _canonical_config(
        config_hint
        or _config_from_filename(path)
        or _config_from_levels(payload.get("config") or {})
        or "unknown"
    )
    effort = payload.get("reasoning_effort") or effort_hint or _effort_from_filename(path)
    effort_recovered = bool(payload.get("_effort_recovered")) or (
        payload.get("reasoning_effort") is None and effort is not None
    )
    fatal_kind, relevant_errors = _fatal_errors(payload)
    legacy_fatal_kind, _ = _legacy_fatal_errors(payload)
    parse_review = _parse_review(payload)
    actions = len(_history_events(payload, "action"))
    invalid_actions = len(_history_events(payload, "invalid_action"))
    historically_completed = (
        bool(payload.get("won")) or actions > 0 or invalid_actions > 0
    )
    completed = fatal_kind is None and historically_completed
    if fatal_kind is not None:
        status = fatal_kind
    elif not completed:
        status = "incomplete"
    else:
        status = "win" if payload.get("won") else "loss"
    turn_budget, turn_budget_provenance = _turn_budget(payload)
    attempts, attempt_provenance = _attempt_policy(payload)
    timestamp = str(payload.get("timestamp") or "")
    revision = _historical_code_revision(timestamp)
    error_samples = list(dict.fromkeys(relevant_errors))[:3]
    if fatal_kind == "environment_error":
        failure_classification = "environment_failure"
    elif fatal_kind == "infrastructure_error":
        failure_classification = "infrastructure_failure"
    elif parse_review["parse_failure_classification"] is not None:
        failure_classification = parse_review["parse_failure_classification"]
    elif not historically_completed:
        failure_classification = "incomplete"
    else:
        failure_classification = "none"
    infrastructure_clean_scored = (
        historically_completed
        and failure_classification in {"none", "model_protocol_failure"}
    )
    strict_error_free = infrastructure_clean_scored and not parse_review["parse_error_bearing"]
    return {
        "trial_id": path.stem,
        "environment": payload.get("env_id") or "ka59simple",
        **revision,
        "model": payload.get("model"),
        "provider": payload.get("provider"),
        "reasoning_effort": effort,
        "reasoning_effort_provenance": (
            "historically_recovered" if effort_recovered else "observed"
        ),
        "config": config,
        "config_levels": payload.get("config"),
        "turn_budget_per_attempt": turn_budget,
        "turn_budget_provenance": turn_budget_provenance,
        "level_attempts": attempts,
        "attempt_policy_provenance": attempt_provenance,
        "retry_semantics":
            "same level reset; prior failed attempt retained in model context",
        "timestamp": timestamp,
        "completed": completed,
        "historically_completed": historically_completed,
        "status": status,
        "won": bool(payload.get("won")) if completed else None,
        "raw_won": bool(payload.get("won")),
        "failure_classification": failure_classification,
        "prior_audit_status": legacy_fatal_kind,
        "infrastructure_clean_scored": infrastructure_clean_scored,
        "strict_error_free": strict_error_free,
        **parse_review,
        "turns_recorded": payload.get("turns"),
        "action_events": actions,
        "invalid_action_events": invalid_actions,
        "error_count": len(relevant_errors),
        "error_samples": error_samples,
        "source_raw_file": str(path.relative_to(REPO_ROOT)),
        "source_sha256": _source_digest(path),
        "semantic_sha256": _semantic_digest(payload),
        "source_batch_or_summary": batch,
        "canonical_membership": canonical_membership,
        "duplicate_of": None,
    }


def _deepseek_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    summaries = (
        RESULTS_ROOT / "ablation_summary_deepseek_v4_pro_none.json",
        RESULTS_ROOT / "ablation_summary_deepseek_v4_pro_medium.json",
    )
    for summary_path in summaries:
        summary = _load_json(summary_path)
        for cell in summary:
            config = _canonical_config(str(cell["config_name"]))
            if config not in PAPER_CONFIGS:
                continue
            for relative in cell["run_files"]:
                path = RESULTS_ROOT.parent / relative
                records.append(
                    _record(
                        path,
                        config_hint=config,
                        effort_hint=str(cell["reasoning_effort"]),
                        batch=str(summary_path.relative_to(REPO_ROOT)),
                        canonical_membership="listed_by_merged_pr18_summary",
                    )
                )
    return records


def _gpt_records() -> list[dict[str, Any]]:
    universe = _load_json(GPT_UNIVERSE_PATH)
    records: list[dict[str, Any]] = []
    for trial in universe["trials"]:
        category = str(trial["eligibility"])
        protocol_identity = str(trial["protocol_identity"])
        if trial["config"] not in PAPER_CONFIGS or trial["reasoning_effort"] not in {"none", "medium"}:
            continue
        if category not in {
            "INCLUDED_SAME_PROTOCOL",
            "EXCLUDED_INFRASTRUCTURE",
            "EXCLUDED_PROVENANCE_INSUFFICIENT",
        }:
            continue
        if category == "EXCLUDED_PROVENANCE_INSUFFICIENT" and not protocol_identity.startswith("ACCEPTED_"):
            continue

        included = category == "INCLUDED_SAME_PROTOCOL"
        infrastructure = category == "EXCLUDED_INFRASTRUCTURE"
        indeterminate = category == "EXCLUDED_PROVENANCE_INSUFFICIENT"
        parse_bearing = indeterminate or any(
            "parse error" in str(error).lower()
            for error in trial.get("infrastructure_errors", []) + trial.get("other_errors", [])
        )
        if infrastructure:
            status = "infrastructure_error"
            failure_classification = "infrastructure_failure"
        elif indeterminate:
            status = "parse_error"
            failure_classification = "indeterminate"
        else:
            status = "win" if trial["won"] else "loss"
            failure_classification = "none"
        errors = trial.get("infrastructure_errors", []) + trial.get("other_errors", [])
        records.append({
            "trial_id": trial["trial_id"],
            "environment": trial["environment_id"],
            "environment_path_recovered": "environment_files/ka59simple/20260430",
            "environment_revision_observed": None,
            "experiment_revision_recovered": "da9964f" if trial["timestamp"] < "2026-06-13" else "604e40e",
            "prompt_revision_recovered": "ff0c184",
            "model": trial["model"],
            "provider": trial["provider"],
            "reasoning_effort": trial["reasoning_effort"],
            "reasoning_effort_provenance": trial["reasoning_effort_provenance"],
            "config": trial["config"],
            "config_levels": trial["config_levels"],
            "turn_budget_per_attempt": 64,
            "turn_budget_provenance": "observed_or_batch_recovered",
            "level_attempts": 2,
            "attempt_policy_provenance": "observed_or_batch_recovered",
            "retry_semantics": trial["retry_semantics"],
            "timestamp": trial["timestamp"],
            "completed": included,
            "historically_completed": bool(trial.get("turns")),
            "status": status,
            "won": bool(trial["won"]) if included else None,
            "raw_won": bool(trial["won"]),
            "failure_classification": failure_classification,
            "prior_audit_status": "parse_error" if indeterminate else status if infrastructure else None,
            "infrastructure_clean_scored": included,
            "strict_error_free": included and not parse_bearing,
            "parse_error_bearing": parse_bearing,
            "parse_error_count": len(errors) if parse_bearing else 0,
            "parse_first_turn": None,
            "parse_last_turn": None,
            "parse_turns": [],
            "parse_review_disposition": "indeterminate" if indeterminate else "infrastructure_failure" if infrastructure and parse_bearing else None,
            "parse_failure_classification": "indeterminate" if indeterminate else None,
            "infrastructure_parse_error_count": len(errors) if infrastructure and parse_bearing else 0,
            "actual_parse_error_count": len(errors) if indeterminate else 0,
            "raw_response_evidence": "raw response content absent from persisted error/history" if indeterminate else None,
            "raw_response_excerpts": [],
            "actions_after_last_parse": 0,
            "runner_recovered_after_parse": False,
            "turns_recorded": trial.get("turns"),
            "action_events": None,
            "invalid_action_events": None,
            "error_count": len(errors),
            "error_samples": errors[:3],
            "source_raw_file": trial.get("raw_source_path") or trial["source_path"],
            "source_sha256": trial["source_semantic_sha256"],
            "semantic_sha256": trial["source_semantic_sha256"],
            "source_batch_or_summary": trial["batch"],
            "canonical_membership": "complete_historical_universe",
            "eligibility": category,
            "protocol_identity": protocol_identity,
            "duplicate_of": None,
        })
    return records


def _mark_duplicates(records: list[dict[str, Any]]) -> None:
    seen: dict[tuple[str, ...], str] = {}
    for record in sorted(records, key=lambda item: item["source_raw_file"]):
        identity = (
            str(record["environment"]),
            str(record["model"]),
            str(record["provider"]),
            str(record["reasoning_effort"]),
            str(record["config"]),
            str(record["turn_budget_per_attempt"]),
            str(record["level_attempts"]),
            str(record["semantic_sha256"]),
        )
        if identity in seen:
            record["duplicate_of"] = seen[identity]
            record["preduplicate_status"] = record["status"]
            record["status"] = "duplicate"
            record["completed"] = False
            record["won"] = None
            record["infrastructure_clean_scored"] = False
            record["strict_error_free"] = False
        else:
            seen[identity] = record["source_raw_file"]
            record["preduplicate_status"] = None


def _cell_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record["model"]),
        str(record["reasoning_effort"]),
        str(record["config"]),
    )


def _historical_display(model: str, effort: str, config: str) -> str | None:
    if model == "openai/gpt-5.2" and config != "mechanics_hard_format_only":
        if effort == "none":
            final_figure = {
                "baseline": "9/10 (90%)",
                "feedback_hard": "8/10 (80%)",
            }
            if config in final_figure:
                return final_figure[config] + " in accepted figure; pooled none+medium"
            return "0/10 (0%) in accepted figure; raw effort-specific cell is 0/5"
        final_figure = {
            "baseline": "8/10 (80%)",
            "feedback_hard": "10/10 (100%)",
        }
        if config in final_figure:
            raw = "4/5" if config == "baseline" else "5/5"
            return final_figure[config] + f" in accepted figure; rate scaled from raw {raw}"
        return "0/10 (0%) in accepted figure; rate scaled from raw 0/5"
    if model == "deepseek-v4-pro" and effort == "medium":
        display = {"baseline": "4/20 (20%)", "feedback_hard": "2/20 (10%)"}
        return display.get(config, "0/20 (0%)")
    return None


def _summarise_cells(records: list[dict[str, Any]], target_n: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_cell_key(record)].append(record)
    cells: list[dict[str, Any]] = []
    model_order = {"openai/gpt-5.2": 0, "deepseek-v4-pro": 1}
    effort_order = {"none": 0, "medium": 1}
    config_order = {name: index for index, name in enumerate(PAPER_CONFIGS)}
    for key in sorted(
        grouped,
        key=lambda item: (
            model_order.get(item[0], 9),
            effort_order.get(item[1], 9),
            config_order.get(item[2], 9),
        ),
    ):
        model, effort, config = key
        items = grouped[key]
        nonduplicates = [item for item in items if item["duplicate_of"] is None]
        infrastructure_clean = [
            item for item in nonduplicates if item["infrastructure_clean_scored"]
        ]
        strict = [item for item in nonduplicates if item["strict_error_free"]]
        infrastructure_wins = sum(item["raw_won"] for item in infrastructure_clean)
        infrastructure_losses = len(infrastructure_clean) - infrastructure_wins
        strict_wins = sum(item["raw_won"] for item in strict)
        strict_losses = len(strict) - strict_wins
        infrastructure_exclusions = sum(
            item["failure_classification"] in {
                "infrastructure_failure", "environment_failure", "incomplete"
            }
            for item in nonduplicates
        )
        model_protocol_failures = sum(
            item["failure_classification"] == "model_protocol_failure"
            for item in nonduplicates
        )
        harness_exclusions = sum(
            item["failure_classification"] == "harness_parse_failure"
            for item in nonduplicates
        )
        indeterminate_parse_exclusions = sum(
            item["failure_classification"] == "indeterminate"
            for item in nonduplicates
        )
        duplicates = sum(1 for item in items if item["status"] == "duplicate")
        metadata_modes = sorted({item["reasoning_effort_provenance"] for item in items})
        infrastructure_n = len(infrastructure_clean)
        strict_n = len(strict)
        policies_differ = (
            infrastructure_n, infrastructure_wins, infrastructure_losses
        ) != (strict_n, strict_wins, strict_losses)
        cells.append({
            "model": model,
            "reasoning_effort": effort,
            "config": config,
            "nominal_historical_n": len(items),
            "infrastructure_clean_scored_n": infrastructure_n,
            "infrastructure_clean_scored_wins": infrastructure_wins,
            "infrastructure_clean_scored_losses": infrastructure_losses,
            "infrastructure_clean_scored_rate": (
                infrastructure_wins / infrastructure_n if infrastructure_n else None
            ),
            "infrastructure_clean_scored_value": (
                f"{infrastructure_wins}/{infrastructure_n} "
                f"({infrastructure_wins / infrastructure_n:.0%})"
                if infrastructure_n else "NO SCORED TRIALS"
            ),
            "strict_error_free_n": strict_n,
            "strict_error_free_wins": strict_wins,
            "strict_error_free_losses": strict_losses,
            "strict_error_free_rate": strict_wins / strict_n if strict_n else None,
            "strict_error_free_value": (
                f"{strict_wins}/{strict_n} ({strict_wins / strict_n:.0%})"
                if strict_n else "NO STRICT TRIALS"
            ),
            "infrastructure_exclusions": infrastructure_exclusions,
            "model_protocol_failures": model_protocol_failures,
            "harness_exclusions": harness_exclusions,
            "indeterminate_parse_exclusions": indeterminate_parse_exclusions,
            "duplicates_excluded": duplicates,
            "denominator_policies_differ": policies_differ,
            "denominator_decision": (
                "NEEDS HUMAN DECISION" if policies_differ
                else "SAME SUBSTANTIVE CONCLUSION"
            ),
            # Backward-compatible strict aliases keep the existing runner's
            # plan/resume target accounting unchanged in this focused commit.
            "valid_n": strict_n,
            "wins": strict_wins,
            "losses": strict_losses,
            "errors_excluded": len(nonduplicates) - strict_n,
            "authoritative_rate": strict_wins / strict_n if strict_n else None,
            "authoritative_value": (
                f"{strict_wins}/{strict_n} ({strict_wins / strict_n:.0%})"
                if strict_n else "NO VALID TRIALS"
            ),
            "historical_display_value": _historical_display(model, effort, config),
            "metadata_provenance": metadata_modes,
            "provenance": sorted({item["source_batch_or_summary"] for item in items}),
            "infrastructure_clean_scored_raw_files": [
                item["source_raw_file"] for item in infrastructure_clean
            ],
            "strict_error_free_raw_files": [item["source_raw_file"] for item in strict],
            "source_raw_files": [item["source_raw_file"] for item in strict],
            "excluded_raw_files": [
                item["source_raw_file"] for item in items if item not in strict
            ],
            "target_n": target_n,
            "infrastructure_clean_additional_trials_needed": max(0, target_n - infrastructure_n),
            "strict_error_free_additional_trials_needed": max(0, target_n - strict_n),
            "additional_trials_needed": max(0, target_n - strict_n),
        })
    return cells


FORMAT_ONLY_BRANCH_COMMIT = "412ba5f"


def _historical_mechanics_level(mechanics: str) -> str:
    """Resolve a declared mechanics level to what the accepted runs actually used.

    `build_system_prompt` gained its `HARD_FORMAT_ONLY` branch in 412ba5f
    (2026-06-25). Every accepted trial predates that commit, so trials tagged
    HARD_FORMAT_ONLY fell through the elif chain and were served the ordinary
    MECHANICS_HARD prompt. The audit describes historical evidence, so it must
    fingerprint what those trials ran, not what the label resolves to in the
    working tree today.
    """
    return "HARD" if mechanics == "HARD_FORMAT_ONLY" else mechanics


def _system_prompt_fingerprint(config: str) -> dict[str, Any]:
    """Fingerprint the treatment a config delivered in the accepted runs.

    The system prompt is a function of (goal, mechanics) only; world and
    feedback act on the per-turn observation. Two configs therefore delivered
    the same treatment when their world/feedback levels match and their system
    prompts were byte-identical at the accepted revision.
    """
    from ka59_game.prompts import build_system_prompt

    levels = CONFIG_LEVELS[config]
    declared = levels["mechanics"]
    effective = _historical_mechanics_level(declared)
    prompt = build_system_prompt(levels["goal"], effective)
    current = build_system_prompt(levels["goal"], declared)
    return {
        "config": config,
        "config_levels": dict(levels),
        "world": levels["world"],
        "feedback": levels["feedback"],
        "declared_mechanics": declared,
        "historical_effective_mechanics": effective,
        "system_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "system_prompt_length": len(prompt),
        "current_revision_sha256": hashlib.sha256(current.encode()).hexdigest(),
        "current_revision_differs": prompt != current,
    }


def prompt_identity_groups() -> dict[str, Any]:
    """Group paper configs that are byte-identical treatments in this revision."""
    fingerprints = [_system_prompt_fingerprint(config) for config in PAPER_CONFIGS]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in fingerprints:
        grouped[(item["world"], item["feedback"], item["system_prompt_sha256"])].append(item)
    collisions = [
        {
            "world": key[0],
            "feedback": key[1],
            "system_prompt_sha256": key[2],
            "configs": [item["config"] for item in items],
            "declared_mechanics_levels": [
                item["config_levels"]["mechanics"] for item in items
            ],
        }
        for key, items in grouped.items()
        if len(items) > 1
    ]
    return {
        "prompt_source": "ka59_game/prompts.py:build_system_prompt at the current revision",
        "rule": (
            "configs sharing world, feedback, and a byte-identical system prompt "
            "deliver the same treatment and may be pooled; the label differs, the "
            "experiment does not"
        ),
        "per_config_fingerprints": fingerprints,
        "identical_treatment_groups": collisions,
        "any_collision": bool(collisions),
    }


def _pooled_cells(
    cells: list[dict[str, Any]], groups: dict[str, Any]
) -> list[dict[str, Any]]:
    """Merge cells whose configs are the same treatment under different labels."""
    pooled: list[dict[str, Any]] = []
    for group in groups["identical_treatment_groups"]:
        members = set(group["configs"])
        by_slice: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for cell in cells:
            if cell["config"] in members:
                by_slice[(cell["model"], cell["reasoning_effort"])].append(cell)
        for (model, effort), items in sorted(by_slice.items()):
            if len(items) < 2:
                continue
            infra_n = sum(item["infrastructure_clean_scored_n"] for item in items)
            infra_w = sum(item["infrastructure_clean_scored_wins"] for item in items)
            strict_n = sum(item["strict_error_free_n"] for item in items)
            strict_w = sum(item["strict_error_free_wins"] for item in items)
            pooled.append({
                "model": model,
                "reasoning_effort": effort,
                "pooled_configs": sorted(item["config"] for item in items),
                "system_prompt_sha256": group["system_prompt_sha256"],
                "component_cells": [
                    {
                        "config": item["config"],
                        "infrastructure_clean_scored_value":
                            item["infrastructure_clean_scored_value"],
                        "strict_error_free_value": item["strict_error_free_value"],
                    }
                    for item in sorted(items, key=lambda c: c["config"])
                ],
                "infrastructure_clean_scored_n": infra_n,
                "infrastructure_clean_scored_wins": infra_w,
                "infrastructure_clean_scored_losses": infra_n - infra_w,
                "infrastructure_clean_scored_value": (
                    f"{infra_w}/{infra_n} ({infra_w / infra_n:.0%})"
                    if infra_n else "NO SCORED TRIALS"
                ),
                "strict_error_free_n": strict_n,
                "strict_error_free_wins": strict_w,
                "strict_error_free_losses": strict_n - strict_w,
                "strict_error_free_value": (
                    f"{strict_w}/{strict_n} ({strict_w / strict_n:.0%})"
                    if strict_n else "NO STRICT TRIALS"
                ),
            })
    return pooled


def build_manifest(target_n: int = 20) -> dict[str, Any]:
    records = _gpt_records() + _deepseek_records()
    _mark_duplicates(records)
    cells = _summarise_cells(records, target_n)
    identity_groups = prompt_identity_groups()
    pooled = _pooled_cells(cells, identity_groups)
    env_source = REPO_ROOT / "environment_files" / "ka59simple" / "20260430" / "ka59simple.py"
    prior_parse_records = [
        record for record in records if record["prior_audit_status"] == "parse_error"
    ]
    return {
        "schema_version": 3,
        "generated_by": "scripts/audit_ka59_camera_ready.py",
        "generation_is_deterministic": True,
        "external_calls": False,
        "accepted_branch_base": "996004f0ed1f28cdd24fe0a3722fa57c082c1df4",
        "environment": {
            "id": "ka59simple",
            "revision_path_recovered": "environment_files/ka59simple/20260430",
            "revision_logged_in_raw_trials": False,
            "source_sha256": _source_digest(env_source) if env_source.exists() else None,
        },
        "denominator_policies": {
            "infrastructure_clean_scored": {
                "included": "completed trials with no transport/provider/environment/harness failure; genuine model protocol failures remain scored by final outcome",
                "excluded": "transport, authentication, insufficient balance, timeout, provider empty response, environment, harness parse, indeterminate parse, incomplete, or duplicate trial",
                "paper_status": "candidate paper-performance denominator",
            },
            "strict_error_free": {
                "included": "infrastructure-clean scored trials bearing no parse error",
                "excluded": "every parse-error-bearing trial in addition to infrastructure-clean exclusions",
                "paper_status": "ultra-conservative sensitivity view",
            },
            "runner_compatibility": "legacy valid_n/wins/losses aliases and plan/resume targets remain strict_error_free in this focused correction",
        },
        "validity_rule": {
            "valid": "backward-compatible alias for strict_error_free",
            "loss": "strict-error-free completed trial with won=false",
            "excluded": "strict-error-free exclusions; see denominator_policies for the candidate view",
            "model_invalid_action": "kept as model behavior when the API response was received and the trial otherwise completed",
            "post_run_understanding_error": "does not invalidate the already completed game outcome",
        },
        "paper_cells": cells,
        "cross_transport_pooling": {
            "deepseek-v4-pro": {
                "decision": "author-approved: pool accepted direct-DeepSeek-API trials "
                            "with new OpenRouter trials pinned to a single upstream",
                "accepted_transport": "api.deepseek.com direct, provider field 'deepseek'",
                "new_transport": "openrouter, upstream pinned via provider.order",
                "model_slug_alias": "deepseek/deepseek-v4-pro -> deepseek-v4-pro",
                "known_differences": [
                    "serving stack differs between the accepted and new trials",
                    "accepted trials ran under max_tokens=4096; the cap never bound at "
                    "effort none (mean 279 output tokens/turn, worst trial 1179)",
                    "accepted upstream was unpinned; new trials pin one provider",
                ],
                "disclosure_required_in_paper": True,
            },
            "openai/gpt-5.2": {
                "decision": "not pooled across transports",
            },
        },
        "prompt_identity_pooling": {
            **identity_groups,
            "pooled_cells": pooled,
        },
        "parse_error_review": {
            "prior_parse_error_trial_count": len(prior_parse_records),
            "disposition_counts": dict(sorted(Counter(
                record["parse_review_disposition"] for record in prior_parse_records
            ).items())),
            "trials": [
                {
                    "source_raw_file": record["source_raw_file"],
                    "model": record["model"],
                    "reasoning_effort": record["reasoning_effort"],
                    "config": record["config"],
                    "classification": record["parse_review_disposition"],
                    "parse_failure_classification": record["parse_failure_classification"],
                    "raw_response_evidence": record["raw_response_evidence"],
                    "raw_response_excerpts": record["raw_response_excerpts"],
                    "parse_turns": record["parse_turns"],
                    "actions_after_last_parse": record["actions_after_last_parse"],
                    "runner_recovered_after_parse": record["runner_recovered_after_parse"],
                    "final_outcome": "win" if record["raw_won"] else "loss",
                }
                for record in prior_parse_records
            ],
        },
        "candidate_trials": sorted(records, key=lambda item: item["source_raw_file"]),
        "findings": {
            "gpt_main_cells":
                "The complete-universe audit recovers N=5 same-protocol candidates per effort. All none trials are infrastructure-clean; all medium trials are provider-failed. In the accepted figure, the none baseline/feedback values retain the "
                "none+medium pool while the medium values scale N=5 rates to N=10; neither row is an "
                "independent N=10 effort-specific sample.",
            "gpt_norules":
                "The complete historical none universe is N=31: 19 clean losses, 10 credit failures, and two provenance-insufficient parse records. The final eligible pool is 0/19 across five batches. Medium is N=10, all provider-empty failures.",
            "gpt_provenance_correction":
                "PR20 effort filenames interleaved May13 world/mechanics trials. The aggregate's effort-major execution order restores none world/mechanics to five clean trials each and medium world/mechanics to five provider-failed trials each.",
            "deepseek_medium":
                "The merged PR18 summary includes 402 Insufficient Balance and provider-empty trials as losses; both denominator policies exclude them as infrastructure failures.",
            "parse_review":
                "Of 49 parse-bearing records now in scope, 46 are explicit DeepSeek empty-content infrastructure failures, one is a recoverable DeepSeek model protocol failure, and two recovered GPT no-rules records are indeterminate because raw response content is absent. No definite harness parse failure is evidenced.",
            "format_only_protocol":
                "Historical prompt code before 412ba5f did not branch on HARD_FORMAT_ONLY, so accepted raw runs received MECHANICS_HARD. The paper-intended action-protocol-retaining prompt was added later.",
        },
    }


def _render_pooling(manifest: dict[str, Any]) -> list[str]:
    """Render the prompt-identity pooling evidence and pooled cells."""
    pooling = manifest["prompt_identity_pooling"]
    lines = [
        "## Prompt-identity pooling",
        "",
        "Config labels are not evidence of distinct treatments. This section fingerprints "
        "what each config actually sent in the accepted runs: the world and feedback levels "
        "plus the SHA-256 of the system prompt built by "
        "`ka59_game/prompts.py:build_system_prompt` at the accepted revision. Configs "
        "agreeing on all three delivered an identical treatment and are pooled. "
        f"`HARD_FORMAT_ONLY` gained a real prompt branch in {FORMAT_ONLY_BRANCH_COMMIT}; "
        "every accepted trial predates it and fell through to `MECHANICS_HARD`.",
        "",
        "| Config | World | Mechanics (declared) | Mechanics (as run) | Feedback | Accepted prompt SHA-256 | Working tree today |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in pooling["per_config_fingerprints"]:
        today = (
            f"`{item['current_revision_sha256'][:16]}` (differs)"
            if item["current_revision_differs"] else "unchanged"
        )
        lines.append(
            f"| {item['config']} | {item['world']} | "
            f"{item['declared_mechanics']} | {item['historical_effective_mechanics']} | "
            f"{item['feedback']} | `{item['system_prompt_sha256'][:16]}` | {today} |"
        )
    lines.append("")
    if not pooling["any_collision"]:
        lines += [
            "No two configs share a treatment at this revision. Every paper cell is "
            "reported on its own label.",
            "",
        ]
        return lines
    for group in pooling["identical_treatment_groups"]:
        lines += [
            f"**Identical treatment:** {', '.join(f'`{c}`' for c in group['configs'])} — "
            f"declared mechanics levels "
            f"{', '.join(f'`{m}`' for m in group['declared_mechanics_levels'])} "
            f"resolved to the same prompt `{group['system_prompt_sha256'][:16]}` in "
            "the accepted runs, which predate the real control branch. The labels "
            "differ; the experiment did not.",
            "",
        ]
    lines += [
        "Pooled cells follow. These are the defensible denominators for the pooled "
        "condition; the component per-label cells above remain as provenance.",
        "",
        "| Model | Effort | Pooled configs | Components (infra-clean) | Pooled infra-clean | Pooled strict |",
        "|---|---|---|---|---|---|",
    ]
    for cell in manifest["prompt_identity_pooling"]["pooled_cells"]:
        components = "; ".join(
            f"{c['config'].replace('mechanics_hard', 'mech')}={c['infrastructure_clean_scored_value']}"
            for c in cell["component_cells"]
        )
        lines.append(
            f"| {cell['model']} | {cell['reasoning_effort']} | "
            f"{', '.join(cell['pooled_configs'])} | {components} | "
            f"**{cell['infrastructure_clean_scored_value']}** | "
            f"{cell['strict_error_free_value']} |"
        )
    lines += [
        "",
        "A pooled cell already at or above the target N needs no further trials. The "
        "runner cannot infer this: `protocol_identity` hashes the config *label*, so it "
        "plans the component cells separately. Pooling is recorded here, not derived at "
        "run time.",
        "",
        "Where the working tree column says *differs*, that config no longer reproduces "
        "the accepted treatment. New trials under that label are a different protocol and "
        "must not be added to the pooled cell above; they need their own identity and "
        "their own denominator.",
        "",
    ]
    return lines


def render_truth(manifest: dict[str, Any]) -> str:
    lines = [
        "# KA59-Simple Camera-Ready Source of Truth",
        "",
        "This document is generated by `scripts/audit_ka59_camera_ready.py` from committed raw JSON. "
        "It does not use the historical Google Sheet as evidence and makes no model calls.",
        "Start with `camera_ready/README.md` for the owner-facing state and decision model.",
        "",
        "## Authority and denominator policies",
        "",
        "GPT-5.2 authority is the complete historical universe in "
        "`camera_ready/ka59_gpt_complete_trial_universe.json`; PR #20's restored directory is an "
        "accepted-evidence subset and duplicate copy layer, not the universe boundary. DeepSeek-V4-Pro "
        "authority remains the raw files named by merged PR #18 summaries. Transport/provider, "
        "environment, and harness failures are never model losses. Model-produced invalid actions and "
        "genuine protocol failures remain model behavior when the request succeeded.",
        "",
        "Two views are intentionally preserved. `infrastructure_clean_scored` is the candidate paper "
        "denominator: it excludes infrastructure/environment/harness/indeterminate failures and exact "
        "duplicates, but scores completed trials containing genuine model protocol failures by final outcome. "
        "`strict_error_free` is the ultra-conservative sensitivity view and additionally excludes every "
        "parse-error-bearing trial. The runner's legacy `valid_n` target remains the strict view in this commit.",
        "",
        "Raw trials do not log an environment Git SHA. The KA59-Simple environment path "
        "`environment_files/ka59simple/20260430` and historical runner revisions are therefore marked "
        "as recovered metadata, never direct observations.",
        "",
        "## Canonical paper cells under both views",
        "",
        "| Model | Effort | Config | Nominal N | Infra-clean N | Infra-clean W/L | Strict N | Strict W/L | Infra excl. | Model protocol | Harness excl. | Indeterminate | Duplicates | Decision |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cell in manifest["paper_cells"]:
        lines.append(
            "| {model} | {reasoning_effort} | {config} | {nominal_historical_n} | "
            "{infrastructure_clean_scored_n} | {infrastructure_clean_scored_wins}/{infrastructure_clean_scored_losses} | "
            "{strict_error_free_n} | {strict_error_free_wins}/{strict_error_free_losses} | "
            "{infrastructure_exclusions} | {model_protocol_failures} | {harness_exclusions} | "
            "{indeterminate_parse_exclusions} | {duplicates_excluded} | {denominator_decision} |".format(**cell)
        )
    lines.extend([
        "",
        "Only DeepSeek none / mechanics-hard differs: the candidate view is 4 wins, 16 losses, N=20 "
        "(20%); the strict view is 4 wins, 15 losses, N=19 (21%). This numeric reporting choice is "
        "**NEEDS HUMAN DECISION**. Every other cell is identical under the two policies, so the "
        "substantive conclusions for those cells do not change.",
        "",
        *_render_pooling(manifest),
        "## Parse-error review",
        "",
        "All 49 parse-bearing records now in scope were inspected. Forty-six DeepSeek-medium "
        "records explicitly say `DeepSeek returned empty content`; these are provider/infrastructure "
        "failures, even where the historical runner later produced actions or a win. The remaining record, "
        "`ka59_game/results/deepseek-v4-pro/run_fE_gE_mH_wE_20260613T050257_120144.json`, persisted the "
        "nonempty fragment `'{\"'` as the failed response at turn 100. It is invalid under the documented "
        "JSON-object protocol, so it is a `model_protocol_failure`, not a harness failure. The runner then "
        "completed 27 meaningful action events through turn 128 and recorded a normal loss. Two newly recovered "
        "GPT no-rules-none trials persist only a `NoneType` parse exception without raw response content; they "
        "are `indeterminate` and excluded rather than guessed to be model or harness behavior. No raw evidence "
        "supports a definite `harness_parse_failure` classification.",
        "",
        "## Why GPT N=5 and the accepted figure's N=10 rows diverged",
        "",
        "The 2026-05-13 GPT-5.2 ablation ran five trials at `none` and five at `medium` for each main "
        "condition. Complete-history recovery found no additional compatible main-condition batch. An early "
        "paper-facing figure pooled those efforts into baseline 9/10 and feedback-hard "
        "8/10, then displayed that row under both effort labels. The accepted figure later changed the "
        "medium percentages to the raw medium rates but kept N=10, displaying 8/10 from 4/5 and 10/10 "
        "from 5/5. The none row retained the pooled 9/10 and 8/10 values. The accepted figure therefore "
        "mixes two reporting operations; neither row is an independent effort-specific N=10 sample. The "
        "effort-specific candidate count remains N=5, but infrastructure-clean evidence is N=5 for every "
        "`none` main cell and N=0 for every `medium` main cell.",
        "",
        "## Newly verified discrepancies",
        "",
        "1. The DeepSeek medium PR #18 summary labels N=20 cells but includes trials whose every turn failed "
        "with HTTP 402 `Insufficient Balance`. Those records are infrastructure failures, not model losses; "
        "the table above reports the remaining valid denominator.",
        "2. PR #20 omitted 21 additional compatible GPT-5.2 no-rules `none` trials. Nineteen are clean losses; "
        "two have indeterminate parse provenance. Together with ten credit failures, the complete candidate "
        "universe is N=31 and the final eligible pool is 0/19.",
        "3. PR #20 interleaved May 13 effort labels for world/mechanics. Aggregate order restores `none` to "
        "five clean trials per cell and `medium` to five provider-failed trials per cell.",
        "4. Before commit `412ba5f`, `HARD_FORMAT_ONLY` fell through to the ordinary `MECHANICS_HARD` prompt. "
        "Thus the accepted raw no-rules runs did not actually retain the detailed action protocol described "
        "in the paper. Reproducing raw behavior and implementing the paper-intended control are different "
        "protocols and must not be pooled.",
        "",
        "## Provenance",
        "",
        "Each manifest cell contains the exact valid and excluded raw filenames, batch/summary source, "
        "reasoning-effort metadata mode, source SHA-256, and semantic duplicate fingerprint. See "
        "`camera_ready/ka59_camera_ready_manifest.json` for the final eligible-protocol record and "
        "`camera_ready/ka59_gpt_complete_trial_universe.json` for every historical GPT candidate and duplicate.",
        "",
    ])
    return "\n".join(lines)


def _encoded_manifest(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def _check(path: Path, expected: str) -> bool:
    return path.exists() and path.read_text() == expected


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-n", type=int, default=20)
    parser.add_argument("--check", action="store_true", help="verify generated files are current")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.target_n < 1:
        parser.error("--target-n must be positive")
    manifest = build_manifest(args.target_n)
    manifest_text = _encoded_manifest(manifest)
    truth_text = render_truth(manifest)
    if args.check:
        stale = [
            str(path.relative_to(REPO_ROOT))
            for path, expected in ((MANIFEST_PATH, manifest_text), (TRUTH_PATH, truth_text))
            if not _check(path, expected)
        ]
        if stale:
            print("Stale generated files: " + ", ".join(stale))
            return 1
        print("KA59 camera-ready manifest and truth document are current.")
        return 0
    CAMERA_READY_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(manifest_text)
    TRUTH_PATH.write_text(truth_text)
    print(f"Wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote {TRUTH_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
