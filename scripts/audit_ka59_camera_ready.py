"""Deterministic KA59-Simple camera-ready evidence audit.

This script reads only committed raw artifacts. It never imports an LLM client,
loads credentials, or makes a network request. The JSON manifest and Markdown
truth table are generated from the same in-memory records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "ka59_game" / "results"
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

INFRASTRUCTURE_MARKERS = (
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


def _fatal_errors(payload: dict[str, Any]) -> tuple[str | None, list[str]]:
    errors = [str(error) for error in payload.get("errors") or []]
    relevant = [
        error for error in errors
        if not error.lower().startswith("understanding prompt failed")
    ]
    lowered = "\n".join(relevant).lower()
    if any(marker in lowered for marker in ENVIRONMENT_MARKERS):
        return "environment_error", relevant
    if any(marker in lowered for marker in INFRASTRUCTURE_MARKERS):
        return "infrastructure_error", relevant
    if any(marker in lowered for marker in PARSE_MARKERS):
        return "parse_error", relevant
    return None, relevant


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
    actions = len(_history_events(payload, "action"))
    invalid_actions = len(_history_events(payload, "invalid_action"))
    completed = fatal_kind is None and (
        bool(payload.get("won")) or actions > 0 or invalid_actions > 0
    )
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
        "status": status,
        "won": bool(payload.get("won")) if completed else None,
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
    records: list[dict[str, Any]] = []
    for path in sorted((RESULTS_ROOT / "gpt-5.2").glob("run_*.json")):
        config = _config_from_filename(path)
        effort = _effort_from_filename(path)
        if config not in PAPER_CONFIGS or effort not in {"none", "medium"}:
            continue
        records.append(
            _record(
                path,
                config_hint=config,
                effort_hint=effort,
                batch=_batch_for_gpt(path, config, effort),
                canonical_membership="restored_by_merged_pr20",
            )
        )
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
            record["status"] = "duplicate"
            record["completed"] = False
            record["won"] = None
        else:
            seen[identity] = record["source_raw_file"]


def _cell_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record["model"]),
        str(record["reasoning_effort"]),
        str(record["config"]),
    )


def _historical_display(model: str, effort: str, config: str) -> str | None:
    if model == "openai/gpt-5.2" and config != "mechanics_hard_format_only":
        pooled = {"baseline": "9/10 (90%)", "feedback_hard": "8/10 (80%)"}
        return pooled.get(config, "0/10 (0%)") + " pooled none+medium, displayed per effort"
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
        valid = [item for item in items if item["status"] in {"win", "loss"}]
        wins = sum(1 for item in valid if item["status"] == "win")
        losses = sum(1 for item in valid if item["status"] == "loss")
        errors = sum(
            1 for item in items
            if item["status"] in {
                "infrastructure_error", "parse_error", "environment_error", "incomplete"
            }
        )
        duplicates = sum(1 for item in items if item["status"] == "duplicate")
        metadata_modes = sorted({item["reasoning_effort_provenance"] for item in items})
        valid_n = len(valid)
        cells.append({
            "model": model,
            "reasoning_effort": effort,
            "config": config,
            "valid_n": valid_n,
            "wins": wins,
            "losses": losses,
            "errors_excluded": errors,
            "duplicates_excluded": duplicates,
            "authoritative_rate": wins / valid_n if valid_n else None,
            "authoritative_value": (
                f"{wins}/{valid_n} ({wins / valid_n:.0%})" if valid_n else "NO VALID TRIALS"
            ),
            "historical_display_value": _historical_display(model, effort, config),
            "metadata_provenance": metadata_modes,
            "provenance": sorted({item["source_batch_or_summary"] for item in items}),
            "source_raw_files": [item["source_raw_file"] for item in valid],
            "excluded_raw_files": [
                item["source_raw_file"] for item in items if item not in valid
            ],
            "target_n": target_n,
            "additional_trials_needed": max(0, target_n - valid_n),
        })
    return cells


def build_manifest(target_n: int = 20) -> dict[str, Any]:
    records = _gpt_records() + _deepseek_records()
    _mark_duplicates(records)
    cells = _summarise_cells(records, target_n)
    env_source = REPO_ROOT / "environment_files" / "ka59simple" / "20260430" / "ka59simple.py"
    return {
        "schema_version": 1,
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
        "validity_rule": {
            "valid": "completed model trial with no LLM/API/parse/environment failure",
            "loss": "valid completed trial with won=false",
            "excluded": "infrastructure, authentication, timeout, parse, environment, incomplete, or duplicate trial",
            "model_invalid_action": "kept as model behavior when the API response was received and the trial otherwise completed",
            "post_run_understanding_error": "does not invalidate the already completed game outcome",
        },
        "paper_cells": cells,
        "candidate_trials": sorted(records, key=lambda item: item["source_raw_file"]),
        "findings": {
            "gpt_main_cells":
                "N=5 per effort; the paper-facing N=10 was a none+medium pool, not N=10 per effort.",
            "gpt_norules":
                "Raw validity must be read from candidate trials; all-credit/API failures are excluded.",
            "deepseek_medium":
                "The merged PR18 summary includes 402 Insufficient Balance trials as losses; they are excluded here.",
            "format_only_protocol":
                "Historical prompt code before 412ba5f did not branch on HARD_FORMAT_ONLY, so accepted raw runs received MECHANICS_HARD. The paper-intended action-protocol-retaining prompt was added later.",
        },
    }


def render_truth(manifest: dict[str, Any]) -> str:
    lines = [
        "# KA59-Simple Camera-Ready Source of Truth",
        "",
        "This document is generated by `scripts/audit_ka59_camera_ready.py` from committed raw JSON. "
        "It does not use the historical Google Sheet as evidence and makes no model calls.",
        "",
        "## Authority and validity rule",
        "",
        "The authoritative evidence is the per-trial raw JSON restored by merged PR #20 for GPT-5.2 "
        "and the raw files named by merged PR #18 summaries for DeepSeek-V4-Pro. A trial is counted "
        "only when it completed without an LLM/API/parse/environment failure. API failures are not losses. "
        "Model-produced invalid actions remain model behavior when the request itself succeeded.",
        "",
        "Raw trials do not log an environment Git SHA. The KA59-Simple environment path "
        "`environment_files/ka59simple/20260430` and historical runner revisions are therefore marked "
        "as recovered metadata, never direct observations.",
        "",
        "## Canonical paper cells",
        "",
        "| Model | Effort | Config | Valid N | Wins | Losses | Errors excluded | Duplicates | Authoritative | Historical display | Need for N=20 |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for cell in manifest["paper_cells"]:
        lines.append(
            "| {model} | {reasoning_effort} | {config} | {valid_n} | {wins} | {losses} | "
            "{errors_excluded} | {duplicates_excluded} | {authoritative_value} | {historical} | "
            "{additional_trials_needed} |".format(
                **cell,
                historical=cell["historical_display_value"] or "-",
            )
        )
    lines.extend([
        "",
        "## Why GPT N=5, pooled N=10, and Sheet N=10-per-effort diverged",
        "",
        "The 2026-05-13 GPT-5.2 ablation ran five trials at `none` and five at `medium` for each main "
        "condition. Paper-facing reporting pooled those distinct efforts, yielding one N=10 row: baseline "
        "9/10 and feedback-hard 8/10. The working Sheet later displayed that pooled row under both effort "
        "labels, which visually implied two independent N=10 cells. No second five-trial batch exists for "
        "those effort-specific cells. The raw effort-specific values are therefore N=5 each.",
        "",
        "## Newly verified discrepancies",
        "",
        "1. The DeepSeek medium PR #18 summary labels N=20 cells but includes trials whose every turn failed "
        "with HTTP 402 `Insufficient Balance`. Those records are infrastructure failures, not model losses; "
        "the table above reports the remaining valid denominator.",
        "2. GPT-5.2 no-rules `none` includes credit-failure records. They are excluded rather than reported as "
        "0% losses.",
        "3. Before commit `412ba5f`, `HARD_FORMAT_ONLY` fell through to the ordinary `MECHANICS_HARD` prompt. "
        "Thus the accepted raw no-rules runs did not actually retain the detailed action protocol described "
        "in the paper. Reproducing raw behavior and implementing the paper-intended control are different "
        "protocols and must not be pooled.",
        "",
        "## Provenance",
        "",
        "Each manifest cell contains the exact valid and excluded raw filenames, batch/summary source, "
        "reasoning-effort metadata mode, source SHA-256, and semantic duplicate fingerprint. See "
        "`camera_ready/ka59_camera_ready_manifest.json` for the complete candidate-level record.",
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
