#!/usr/bin/env python3
"""Recover and classify the complete historical GPT-5.2 KA59-Simple universe.

This is a Git-object audit. It reads preserved repository objects only and
makes no network, provider, model, or paid-compute calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REV = "c9063f293f93d743a648b969d5dfbffe93b9532d"
SOURCE_BRANCH = "levi/ka59simple-level-attempts (local preserved commit)"
OUTPUT_JSON = REPO_ROOT / "camera_ready" / "ka59_gpt_complete_trial_universe.json"
OUTPUT_MD = REPO_ROOT / "camera_ready" / "KA59_GPT_COMPLETE_TRIAL_UNIVERSE.md"

INCLUDED = "INCLUDED_SAME_PROTOCOL"
DIFFERENT = "EXCLUDED_DIFFERENT_PROTOCOL"
INFRASTRUCTURE = "EXCLUDED_INFRASTRUCTURE"
DUPLICATE = "EXCLUDED_DUPLICATE"
SMOKE = "EXCLUDED_EXPLICIT_SMOKE_OR_DEBUG"
INSUFFICIENT = "EXCLUDED_PROVENANCE_INSUFFICIENT"
SEPARATE = "SEPARATE_EXPERIMENT"

PAPER_CONFIGS = (
    "baseline",
    "world_hard",
    "mechanics_hard",
    "mechanics_hard_format_only",
    "feedback_hard",
)

CONFIG_LEVELS = {
    "baseline": {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"},
    "world_hard": {"world": "HARD", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"},
    "goal_hard": {"world": "EASY", "goal": "HARD", "mechanics": "EASY", "feedback": "EASY"},
    "mechanics_hard": {"world": "EASY", "goal": "EASY", "mechanics": "HARD", "feedback": "EASY"},
    "mechanics_ooda": {"world": "EASY", "goal": "EASY", "mechanics": "OODA", "feedback": "EASY"},
    "mechanics_ooda_f": {"world": "EASY", "goal": "EASY", "mechanics": "OODA_F", "feedback": "EASY"},
    "mechanics_hard_format_only": {
        "world": "EASY", "goal": "EASY", "mechanics": "HARD_FORMAT_ONLY", "feedback": "EASY"
    },
    "feedback_hard": {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "HARD"},
}

ACCEPTED_PROTOCOL = {
    "environment_id": "ka59simple",
    "environment_path_recovered": "environment_files/ka59simple/20260430",
    "model": "openai/gpt-5.2",
    "provider": "openrouter",
    "turn_budget_per_attempt": 64,
    "attempt_count": 2,
    "failed_attempt_context_policy": "retained",
    "retry_semantics": "one immediate format retry per turn; failed turn consumed; same level reset between attempts",
    "prompt_revision_recovered": "ff0c184",
    "format_only_implementation": "pre-412ba5f HARD_FORMAT_ONLY fallthrough to ordinary HARD prompt",
}


def _first_repository_commit(batch: str) -> str:
    if batch < "20260513":
        return "d6f790284eaddf616ddd32105fea7ef3e0e84ba8"
    if batch.startswith("20260513"):
        return "d548e19215626aa630f72199f719fef7154de557"
    return SOURCE_REV


def _git(*args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=text)


def _git_json(path: str, rev: str = SOURCE_REV) -> dict[str, Any]:
    return json.loads(_git("show", f"{rev}:{path}"))


def _tree_paths(root: str, rev: str = SOURCE_REV) -> list[str]:
    return [
        path for path in str(_git("ls-tree", "-r", "--name-only", rev, "--", root)).splitlines()
        if path.endswith(".json")
    ]


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _config_from_levels(levels: dict[str, Any]) -> str:
    for name, expected in CONFIG_LEVELS.items():
        if levels == expected:
            return name
    return "unknown"


def _raw_catalog() -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for path in _tree_paths("results/ka59simple_game"):
        payload = _git_json(path)
        if payload.get("env_id") != "ka59simple" or "gpt-5.2" not in str(payload.get("model")):
            continue
        catalog[path] = payload
    return catalog


def _raw_timestamp(payload: dict[str, Any]) -> str:
    return str(payload.get("timestamp") or "")


def _raw_errors(payload: dict[str, Any]) -> list[str]:
    return [str(error) for error in payload.get("errors") or []]


def _infrastructure_error(errors: list[str]) -> bool:
    markers = (
        "insufficient credits", "insufficient balance", "error code: 402",
        "returned empty content", "timed out", "timeout", "error code: 429",
        "error code: 502", "error code: 503",
    )
    return any(any(marker in error.lower() for marker in markers) for error in errors)


def _indeterminate_parse(errors: list[str]) -> bool:
    return any("'nonetype' object is not subscriptable" in error.lower() for error in errors)


def _turn_policy(payload: dict[str, Any]) -> tuple[int | None, int | None]:
    budgets = [
        event.get("state_before", {}).get("resources", {}).get("step_budget")
        for event in payload.get("history") or []
    ]
    budgets = [value for value in budgets if isinstance(value, int)]
    attempts = [
        event.get("attempt") for event in payload.get("history") or []
        if isinstance(event.get("attempt"), int)
    ]
    return (Counter(budgets).most_common(1)[0][0] if budgets else None, max(attempts) if attempts else 1)


def _validate_non_outcome_match(summary: dict[str, Any], raw: dict[str, Any], label: str) -> None:
    """Verify recovered membership without consulting win/loss."""
    pairs = {
        "turns": (summary.get("turns"), raw.get("turns")),
        "levels_completed": (summary.get("levels_completed"), raw.get("levels_completed")),
        "passable_walls": (summary.get("passable_walls"), raw.get("click_actions")),
        "blocked": (summary.get("blocked"), raw.get("moves_blocked")),
        "wall_transfers": (summary.get("wall_transfers"), raw.get("wall_transfers")),
    }
    mismatches = {key: values for key, values in pairs.items() if values[0] != values[1]}
    if mismatches:
        raise RuntimeError(f"non-outcome membership mismatch for {label}: {mismatches}")


def _trial_record(
    *,
    batch: str,
    config: str,
    effort: str,
    index: int,
    source_path: str,
    raw_path: str | None,
    raw: dict[str, Any] | None,
    summary: dict[str, Any],
    category: str,
    reason: str,
    protocol_identity: str,
    effort_provenance: str,
    raw_available: bool,
) -> dict[str, Any]:
    errors = _raw_errors(raw or {})
    won = bool((raw or summary).get("won"))
    timestamp = str((raw or {}).get("timestamp") or batch)
    turn_budget, attempts = _turn_policy(raw or {}) if raw else (None, None)
    if turn_budget is None and protocol_identity == "PRE_LEVEL_ATTEMPTS_32T_UNKNOWN_EFFORT":
        turn_budget, attempts = 32, 1
    blob_path = (raw_path or source_path).split("#", 1)[0]
    return {
        "trial_id": f"{batch}:{effort}:{config}:t{index}",
        "batch": batch,
        "source_path": source_path,
        "raw_source_path": raw_path,
        "source_commit": SOURCE_REV,
        "first_repository_commit": _first_repository_commit(batch),
        "source_branch": SOURCE_BRANCH,
        "source_blob": str(_git("rev-parse", f"{SOURCE_REV}:{blob_path}")).strip(),
        "timestamp": timestamp,
        "batch_run_id": batch,
        "model": (raw or {}).get("model") or "openai/gpt-5.2",
        "provider": (raw or {}).get("provider") or "openrouter",
        "reasoning_effort": effort,
        "reasoning_effort_provenance": effort_provenance,
        "config": config,
        "config_levels": (raw or {}).get("config") or CONFIG_LEVELS.get(config),
        "environment_id": (raw or {}).get("env_id") or "ka59simple",
        "environment_identity": "environment_files/ka59simple/20260430 (recovered; raw Git SHA not logged)",
        "prompt_code_revision": "ff0c184/da9964f recovered from run chronology" if timestamp >= "2026-05-13" else "pre-ff0c184 historical runner",
        "turn_budget_per_attempt": turn_budget,
        "attempt_count_observed": attempts,
        "failed_attempt_context_policy": "retained" if attempts == 2 else "not applicable or unrecovered",
        "retry_semantics": ACCEPTED_PROTOCOL["retry_semantics"] if timestamp >= "2026-05-13" else "pre-two-attempt runner",
        "explicit_smoke_or_debug": category == SMOKE,
        "raw_per_trial_available": raw_available,
        "raw_mapping_status": "exact" if raw_path else "raw files preserved but aggregate-row mapping not uniquely recoverable" if raw_available else "aggregate only",
        "won": won,
        "loss": not won,
        "turns": (raw or summary).get("turns"),
        "infrastructure_errors": errors if _infrastructure_error(errors) else [],
        "other_errors": errors if errors and not _infrastructure_error(errors) else [],
        "duplicate_of": None,
        "protocol_identity": protocol_identity,
        "eligibility": category,
        "eligibility_reason": reason,
        "outcome_read_after_eligibility": True,
        "source_semantic_sha256": _sha256_json(raw or summary),
    }


def _summary_cells(path: str) -> list[tuple[str, str, list[dict[str, Any]]]]:
    payload = _git_json(path)
    if "results" in payload:
        return [
            (str(cell["reasoning_effort"]), str(cell["config"]), list(cell.get("trial_data") or []))
            for cell in payload["results"].values()
        ]
    return [(
        str(payload.get("reasoning_effort") or "unknown"),
        str(payload["config"]),
        list(payload.get("trial_data") or []),
    )]


def _pre_protocol_batches(raw: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    sidecars = defaultdict(list)
    for path in _tree_paths("results/ka59simple_real_ablation"):
        if "/sidecar_" not in path or "gpt-5.2" not in path:
            continue
        payload = _git_json(path)
        stamp = str(payload.get("timestamp") or "")
        if stamp >= "20260513":
            continue
        sidecars[stamp].append(path)

    aggregate_by_stamp: dict[str, str] = {}
    for path in _tree_paths("results/ka59simple_real_ablation"):
        if "/ablation_" not in path or "gpt-5.2" not in path:
            continue
        payload = _git_json(path)
        stamp = str(payload.get("timestamp") or "")
        if stamp < "20260513":
            aggregate_by_stamp[stamp] = path

    batches: list[dict[str, Any]] = []
    for stamp in sorted(sidecars):
        paths = sorted(sidecars[stamp])
        cells: list[tuple[str, str, list[dict[str, Any]], str]] = []
        for path in paths:
            effort, config, trials = _summary_cells(path)[0]
            cells.append((effort, config, trials, path))
        trial_records = []
        for effort, config, trials, path in cells:
            category = SEPARATE if config not in PAPER_CONFIGS else DIFFERENT
            reason = (
                "config is outside the accepted paper matrix"
                if category == SEPARATE else
                "pre-May-13 runner used one 32-turn attempt and lacks accepted effort provenance"
            )
            for index, summary in enumerate(trials, 1):
                trial_records.append(_trial_record(
                    batch=stamp, config=config, effort="unknown", index=index,
                    source_path=f"{path}#trial_data[{index - 1}]", raw_path=None, raw=None,
                    summary=summary, category=category, reason=reason,
                    protocol_identity="PRE_LEVEL_ATTEMPTS_32T_UNKNOWN_EFFORT",
                    effort_provenance="unknown", raw_available=True,
                ))
        batches.append({
            "batch": stamp,
            "source_paths": ([aggregate_by_stamp[stamp]] if stamp in aggregate_by_stamp else []) + paths,
            "source_commit": SOURCE_REV,
            "source_branch": SOURCE_BRANCH,
            "timestamp": stamp,
            "batch_run_id": stamp,
            "model": _git_json(paths[0]).get("model"),
            "provider": _git_json(paths[0]).get("provider"),
            "reasoning_effort": "unknown",
            "reasoning_effort_provenance": "unknown",
            "configs": [cell[1] for cell in cells],
            "protocol_identity": "PRE_LEVEL_ATTEMPTS_32T_UNKNOWN_EFFORT",
            "turn_budget_per_attempt": 32,
            "attempt_count": 1,
            "failed_attempt_context_policy": "not applicable",
            "retry_semantics": "pre-two-attempt runner",
            "explicit_smoke_or_debug": False,
            "raw_per_trial_available": True,
            "raw_n": len(trial_records),
            "trial_records": trial_records,
        })
    return batches


def _ordered_canonical_paths(raw: dict[str, dict[str, Any]]) -> dict[tuple[str, str], list[str]]:
    aggregate = _git_json(
        "results/ka59simple_real_ablation/ablation_openrouter_openai_gpt-5.2_20260513T055440_493226.json"
    )
    candidates = [
        (path, payload) for path, payload in raw.items()
        if "2026-05-13T05:54:40" <= _raw_timestamp(payload) < "2026-05-14"
    ]
    candidates.sort(key=lambda item: _raw_timestamp(item[1]))
    cursor = 0
    mapping: dict[tuple[str, str], list[str]] = {}
    for cell in aggregate["results"].values():
        config = str(cell["config"])
        selected: list[str] = []
        while cursor < len(candidates) and len(selected) < int(cell["trials"]):
            path, payload = candidates[cursor]
            cursor += 1
            if _config_from_levels(payload.get("config") or {}) == config:
                selected.append(path)
        if len(selected) != int(cell["trials"]):
            raise RuntimeError(f"could not recover ordered membership for {cell['reasoning_effort']}::{config}")
        mapping[(str(cell["reasoning_effort"]), config)] = selected
    return mapping


def _may13_batches(raw: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    smoke_groups = [
        (
            "20260513T043402_656079",
            ["results/ka59simple_game/run_fE_gE_mE_wE_20260513T043420_602171.json"],
            ["none"],
            "explicit one-cell preflight; 8 turns per attempt",
        ),
        (
            "20260513T043715_RAW_ONLY",
            [
                "results/ka59simple_game/run_fE_gE_mE_wE_20260513T043729_344894.json",
                "results/ka59simple_game/run_fE_gE_mH_wE_20260513T043823_009265.json",
                "results/ka59simple_game/run_fE_gE_mE_wE_20260513T044935_804096.json",
                "results/ka59simple_game/run_fE_gE_mH_wE_20260513T050127_629004.json",
            ],
            ["none", "none", "medium", "medium"],
            "contemporaneous plan explicitly calls this the end-to-end smoke slice; 16 turns per attempt",
        ),
    ]
    batches: list[dict[str, Any]] = []
    for batch, paths, efforts, reason in smoke_groups:
        records = []
        for index, (path, effort) in enumerate(zip(paths, efforts), 1):
            payload = raw[path]
            config = _config_from_levels(payload["config"])
            records.append(_trial_record(
                batch=batch, config=config, effort=effort, index=index,
                source_path=path, raw_path=path, raw=payload, summary=payload,
                category=SMOKE, reason=reason,
                protocol_identity="MAY13_EXPLICIT_PREFLIGHT_REDUCED_TURN_BUDGET",
                effort_provenance="recovered from prescribed smoke command/order",
                raw_available=True,
            ))
        batches.append({
            "batch": batch,
            "source_paths": paths,
            "source_commit": SOURCE_REV,
            "source_branch": SOURCE_BRANCH,
            "timestamp": _raw_timestamp(raw[paths[0]]),
            "batch_run_id": batch,
            "model": "openai/gpt-5.2",
            "provider": "openrouter",
            "reasoning_effort": sorted(set(efforts)),
            "reasoning_effort_provenance": "recovered from prescribed smoke command/order",
            "configs": sorted({_config_from_levels(raw[path]["config"]) for path in paths}),
            "protocol_identity": "MAY13_EXPLICIT_PREFLIGHT_REDUCED_TURN_BUDGET",
            "turn_budget_per_attempt": _turn_policy(raw[paths[0]])[0],
            "attempt_count": 2,
            "failed_attempt_context_policy": "retained",
            "retry_semantics": ACCEPTED_PROTOCOL["retry_semantics"],
            "explicit_smoke_or_debug": True,
            "raw_per_trial_available": True,
            "raw_n": len(records),
            "trial_records": records,
        })

    aggregate_path = "results/ka59simple_real_ablation/ablation_openrouter_openai_gpt-5.2_20260513T055440_493226.json"
    aggregate = _git_json(aggregate_path)
    membership = _ordered_canonical_paths(raw)
    records = []
    for cell in aggregate["results"].values():
        effort, config = str(cell["reasoning_effort"]), str(cell["config"])
        for index, (summary, path) in enumerate(zip(cell["trial_data"], membership[(effort, config)]), 1):
            payload = raw[path]
            _validate_non_outcome_match(summary, payload, f"May13 {effort}/{config}/t{index}")
            errors = _raw_errors(payload)
            if config not in PAPER_CONFIGS:
                category, reason = SEPARATE, "goal_hard is outside the accepted paper matrix"
            elif _infrastructure_error(errors):
                category, reason = INFRASTRUCTURE, "provider empty-content failures occurred during the trial"
            else:
                category, reason = INCLUDED, "matches the locked two-attempt/64-turn protocol; model invalid actions remain model behavior"
            records.append(_trial_record(
                batch="20260513T055440_493226", config=config, effort=effort, index=index,
                source_path=aggregate_path, raw_path=path, raw=payload, summary=summary,
                category=category, reason=reason,
                protocol_identity="ACCEPTED_20260513_TWO_ATTEMPT_64T",
                effort_provenance="recovered from aggregate cell and runner's effort-major execution order",
                raw_available=True,
            ))
    batches.append({
        "batch": "20260513T055440_493226",
        "source_paths": [aggregate_path],
        "source_commit": SOURCE_REV,
        "source_branch": SOURCE_BRANCH,
        "timestamp": aggregate["timestamp"],
        "batch_run_id": aggregate["timestamp"],
        "model": aggregate["model"],
        "provider": aggregate["provider"],
        "reasoning_effort": ["none", "medium"],
        "reasoning_effort_provenance": "observed in aggregate; mapped to raw by effort-major runner order",
        "configs": [str(cell["config"]) for cell in aggregate["results"].values()],
        "protocol_identity": "ACCEPTED_20260513_TWO_ATTEMPT_64T",
        "turn_budget_per_attempt": 64,
        "attempt_count": 2,
        "failed_attempt_context_policy": "retained",
        "retry_semantics": ACCEPTED_PROTOCOL["retry_semantics"],
        "explicit_smoke_or_debug": False,
        "raw_per_trial_available": True,
        "raw_n": len(records),
        "trial_records": records,
    })
    return batches


JUNE_NONE_MEMBERSHIP = {
    "20260601T020500_243047": ["020506_569494", "020512_890828", "020517_419842", "020521_507524", "020526_685977"],
    "20260601T021239_568866": ["021245_074439", "021249_441290", "021254_293528", "021259_027692", "021304_805982"],
    "20260601T021314_763237": ["021653_050033", "022031_151302", "022355_535115", "022731_389629", "023111_341002"],
    "20260601T021411_105948": ["021729_582282", "022100_733843", "022420_749355", "022750_670418", "023134_029312"],
    "20260601T021458_230073": ["021816_390387"],
    "20260601T021821_835569": ["022144_603161", "022531_005283", "022858_878866", "023252_267856", "023648_629927"],
    "20260601T021844_403409": ["022215_936921", "022553_562767", "022925_078391", "023327_044158", "023710_294447"],
}

JUNE_MEDIUM_MEMBERSHIP = {
    "20260620T213830_707671_p35675": ["20260620T224723_509035", "20260621T000217_095303"],
    "20260620T213830_707671_p35677": ["20260620T225003_562103", "20260621T000425_963814"],
    "20260620T213830_707672_p35674": ["20260620T225127_765579", "20260621T000444_835099"],
    "20260620T213830_707676_p35673": ["20260620T224846_113339", "20260620T235331_776190"],
    "20260620T213830_707882_p35676": ["20260620T225537_223785", "20260621T000657_592382"],
}


def _raw_path_by_suffix(raw: dict[str, dict[str, Any]], suffix: str) -> str:
    matches = [path for path in raw if suffix in path and "mH" in path]
    if len(matches) != 1:
        raise RuntimeError(f"expected one raw path for {suffix}, found {matches}")
    return matches[0]


def _june_batches(raw: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for batch, suffixes in JUNE_NONE_MEMBERSHIP.items():
        aggregate_path = f"results/ka59simple_real_ablation/ablation_openrouter_openai_gpt-5.2_{batch}.json"
        aggregate = _git_json(aggregate_path)
        cell = next(iter(aggregate["results"].values()))
        paths = [_raw_path_by_suffix(raw, suffix) for suffix in suffixes]
        records = []
        for index, (summary, path) in enumerate(zip(cell["trial_data"], paths), 1):
            payload, errors = raw[path], _raw_errors(raw[path])
            _validate_non_outcome_match(summary, payload, f"June none {batch}/t{index}")
            if _infrastructure_error(errors):
                category, reason = INFRASTRUCTURE, "HTTP 402 insufficient-credit failures consumed the entire trial"
            elif _indeterminate_parse(errors):
                category, reason = INSUFFICIENT, "persisted NoneType parse error has no raw response; model-versus-harness cause cannot be recovered"
            else:
                category, reason = INCLUDED, "matches the locked historical fallthrough control protocol"
            records.append(_trial_record(
                batch=batch, config="mechanics_hard_format_only", effort="none", index=index,
                source_path=aggregate_path, raw_path=path, raw=payload, summary=summary,
                category=category, reason=reason,
                protocol_identity="ACCEPTED_HISTORICAL_NORULES_FALLTHROUGH_NONE",
                effort_provenance="observed", raw_available=True,
            ))
        batches.append({
            "batch": batch, "source_paths": [aggregate_path], "source_commit": SOURCE_REV,
            "source_branch": SOURCE_BRANCH, "timestamp": aggregate["timestamp"], "batch_run_id": batch,
            "model": aggregate["model"], "provider": aggregate["provider"], "reasoning_effort": "none",
            "reasoning_effort_provenance": "observed", "configs": ["mechanics_hard_format_only"],
            "protocol_identity": "ACCEPTED_HISTORICAL_NORULES_FALLTHROUGH_NONE",
            "turn_budget_per_attempt": 64, "attempt_count": 2, "failed_attempt_context_policy": "retained",
            "retry_semantics": ACCEPTED_PROTOCOL["retry_semantics"], "explicit_smoke_or_debug": False,
            "raw_per_trial_available": True, "raw_n": len(records), "trial_records": records,
        })

    june9_path = "results/ka59simple_real_ablation/sidecar_openrouter_openai_gpt-5.2_20260609T195446_323788_baseline_none.json"
    june9 = _git_json(june9_path)
    june9_record = _trial_record(
        batch="20260609T195446_323788", config="baseline", effort="none", index=1,
        source_path=f"{june9_path}#trial_data[0]", raw_path=None, raw=None,
        summary=june9["trial_data"][0], category=INSUFFICIENT,
        reason="aggregate-only all-zero record; raw trial and error string are absent, so protocol completion and failure cause cannot be proven",
        protocol_identity="PROTOCOL_LABEL_MATCHES_RAW_PROVENANCE_MISSING",
        effort_provenance="observed in sidecar", raw_available=False,
    )
    batches.append({
        "batch": "20260609T195446_323788", "source_paths": [june9_path], "source_commit": SOURCE_REV,
        "source_branch": SOURCE_BRANCH, "timestamp": june9["timestamp"], "batch_run_id": june9["timestamp"],
        "model": june9["model"], "provider": june9["provider"], "reasoning_effort": "none",
        "reasoning_effort_provenance": "observed in sidecar", "configs": ["baseline"],
        "protocol_identity": "PROTOCOL_LABEL_MATCHES_RAW_PROVENANCE_MISSING", "turn_budget_per_attempt": None,
        "attempt_count": None, "failed_attempt_context_policy": "unknown", "retry_semantics": "unknown without raw",
        "explicit_smoke_or_debug": False, "raw_per_trial_available": False, "raw_n": 1,
        "trial_records": [june9_record],
    })

    for batch, suffixes in JUNE_MEDIUM_MEMBERSHIP.items():
        aggregate_path = f"results/ka59simple_real_ablation/ablation_openrouter_openai_gpt-5.2_{batch}.json"
        aggregate = _git_json(aggregate_path)
        cell = next(iter(aggregate["results"].values()))
        paths = [_raw_path_by_suffix(raw, suffix) for suffix in suffixes]
        records = []
        for index, (summary, path) in enumerate(zip(cell["trial_data"], paths), 1):
            payload = raw[path]
            _validate_non_outcome_match(summary, payload, f"June medium {batch}/t{index}")
            records.append(_trial_record(
                batch=batch, config="mechanics_hard_format_only", effort="medium", index=index,
                source_path=aggregate_path, raw_path=path, raw=payload, summary=summary,
                category=INFRASTRUCTURE,
                reason="OpenRouter empty-content failures occurred during the trial",
                protocol_identity="ACCEPTED_HISTORICAL_NORULES_FALLTHROUGH_MEDIUM",
                effort_provenance="observed", raw_available=True,
            ))
        batches.append({
            "batch": batch, "source_paths": [aggregate_path], "source_commit": SOURCE_REV,
            "source_branch": SOURCE_BRANCH, "timestamp": aggregate["timestamp"], "batch_run_id": batch,
            "model": aggregate["model"], "provider": aggregate["provider"], "reasoning_effort": "medium",
            "reasoning_effort_provenance": "observed", "configs": ["mechanics_hard_format_only"],
            "protocol_identity": "ACCEPTED_HISTORICAL_NORULES_FALLTHROUGH_MEDIUM",
            "turn_budget_per_attempt": 64, "attempt_count": 2, "failed_attempt_context_policy": "retained",
            "retry_semantics": ACCEPTED_PROTOCOL["retry_semantics"], "explicit_smoke_or_debug": False,
            "raw_per_trial_available": True, "raw_n": len(records), "trial_records": records,
        })
    return batches


def _duplicate_artifacts(raw: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    duplicates = []
    head_paths = [
        path for path in _tree_paths("ka59_game/results/gpt-5.2", "HEAD") if "/run_" in path
    ]
    by_identity = defaultdict(list)
    for path, payload in raw.items():
        by_identity[(payload.get("timestamp"), json.dumps(payload.get("config"), sort_keys=True))].append(path)
    for path in head_paths:
        payload = _git_json(path, "HEAD")
        key = (payload.get("timestamp"), json.dumps(payload.get("config"), sort_keys=True))
        originals = by_identity.get(key, [])
        if len(originals) == 1:
            duplicates.append({
                "source_path": path,
                "source_commit": "d1d01d43d5becf27003b37bd6bbd9e3698f33ad7 (merged PR #20 lineage)",
                "eligibility": DUPLICATE,
                "duplicate_of": originals[0],
                "reason": "restored copy of the same historical trial; not an independent draw",
            })
    return duplicates


def _batch_summary(batch: dict[str, Any]) -> dict[str, Any]:
    records = batch["trial_records"]
    included = [record for record in records if record["eligibility"] == INCLUDED]
    counts = Counter(record["eligibility"] for record in records)
    batch["valid_n"] = len(included)
    batch["first_repository_commit"] = _first_repository_commit(str(batch["batch"]))
    batch["wins"] = sum(record["won"] for record in included)
    batch["losses"] = len(included) - batch["wins"]
    batch["eligibility_counts"] = dict(sorted(counts.items()))
    batch["eligibility"] = INCLUDED if included else next(iter(counts))
    batch["exact_exclusion_reason"] = "; ".join(sorted({
        record["eligibility_reason"] for record in records if record["eligibility"] != INCLUDED
    })) or "none"
    return batch


def _cell_table(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for effort in ("none", "medium"):
        for config in PAPER_CONFIGS:
            candidates = [record for record in trials if record["reasoning_effort"] == effort and record["config"] == config]
            same_protocol = [
                record for record in candidates
                if record["eligibility"] in {INCLUDED, INFRASTRUCTURE}
                or (
                    record["eligibility"] == INSUFFICIENT
                    and str(record["protocol_identity"]).startswith("ACCEPTED_")
                )
            ]
            included = [record for record in candidates if record["eligibility"] == INCLUDED]
            exclusions = Counter(record["eligibility"] for record in candidates if record["eligibility"] != INCLUDED)
            wins = sum(record["won"] for record in included)
            rows.append({
                "reasoning_effort": effort,
                "config": config,
                "historical_candidate_n": len(candidates),
                "same_protocol_candidate_n": len(same_protocol),
                "infrastructure_clean_included_n": len(included),
                "wins": wins,
                "losses": len(included) - wins,
                "excluded_by_reason": dict(sorted(exclusions.items())),
                "final_pooled_estimate": f"{wins}/{len(included)} ({wins / len(included):.0%})" if included else "NO ELIGIBLE TRIALS",
                "included_batches": sorted({record["batch"] for record in included}),
            })
    return rows


def build_universe() -> dict[str, Any]:
    raw = _raw_catalog()
    batches = _pre_protocol_batches(raw) + _may13_batches(raw) + _june_batches(raw)
    batches = [_batch_summary(batch) for batch in sorted(batches, key=lambda item: item["batch"])]
    trials = [record for batch in batches for record in batch["trial_records"]]
    category_counts = Counter(record["eligibility"] for record in trials)
    return {
        "schema_version": 1,
        "generated_by": "scripts/audit_ka59_gpt_complete_universe.py",
        "generation_is_deterministic": True,
        "external_calls": False,
        "outcome_blind_rule": "eligibility is assigned from protocol/provenance/error class before won/loss is read",
        "accepted_protocol": ACCEPTED_PROTOCOL,
        "history_scope": {
            "refs_examined": [
                "main", "anon-submission", "levi/ka59simple-level-attempts",
                "camera-ready/ka59-truth", "all local and remote refs via git rev-list --all",
            ],
            "paths_examined": [
                "results/ka59simple_real_ablation", "results/ka59simple_game", "ka59_game/results",
            ],
            "preserved_source_commit": SOURCE_REV,
            "reachable_gpt_json_artifact_count": 346,
            "reachable_gpt_unique_blob_count": 346,
            "local_worktrees_examined": [
                "/Users/edward/Projects/BreakingAgentsIntoPOMDP",
                "/Users/edward/Projects/BreakingAgentsIntoPOMDP-camera-ready",
                "/Users/edward/Projects/BreakingAgentsIntoPOMDP-poster-20260822",
            ],
            "untracked_gpt_result_artifacts_found": 0,
        },
        "logical_trial_count": len(trials),
        "category_counts": dict(sorted(category_counts.items())),
        "batches": batches,
        "trials": trials,
        "duplicate_artifacts": _duplicate_artifacts(raw),
        "final_cells": _cell_table(trials),
        "conclusions": {
            "accepted_subset_complete": False,
            "accepted_subset_issue": "PR #20 omitted 21 June no-rules-none trials and assigned May 13 world/mechanics efforts inconsistently with the aggregate execution order",
            "may13_n1_counts": False,
            "may13_n1_reason": "explicit reduced-turn preflight/smoke, established before outcome",
            "further_compute_required_for_reconciliation": False,
            "further_compute_note": "historical provenance is now exhausted; new compute is needed only if authors choose a larger target N or the paper-intended no-rules control",
        },
    }


def _batch_result(batch: dict[str, Any]) -> str:
    return f"{batch['wins']}/{batch['valid_n']}" if batch["valid_n"] else "—"


def render_markdown(universe: dict[str, Any]) -> str:
    lines = [
        "# KA59-Simple GPT-5.2 Complete Historical Trial Universe",
        "",
        "## Answer first",
        "",
        "PR #20's restored per-trial directory was an accepted-evidence subset, not the complete historical universe. "
        "The exhaustive Git-object audit recovered 21 additional protocol-compatible June 1 no-rules-none trials: "
        "19 are infrastructure-clean losses and two have insufficient persisted parse provenance. It also corrected the "
        "May 13 world/mechanics effort mapping from the effort-major aggregate order. Main `none` world/mechanics are "
        "therefore 0/5, the corresponding `medium` cells have no clean trials, and historical no-rules-none pools to 0/19.",
        "",
        "The May 13 N=1 baseline does **not** count: contemporaneous planning identifies it as reduced-turn preflight/smoke "
        "before the N=5 sweep. No outcome was used to make that decision.",
        "",
        "## Outcome-blind eligibility rule",
        "",
        "Protocol identity comes from `camera_ready/KA59_PROTOCOL_LOCK.md`: KA59-Simple dated environment, OpenRouter "
        "`openai/gpt-5.2`, effort-specific cell, 64 turns per attempt, two same-level attempts, retained failed-attempt "
        "context, historical per-turn retry semantics, and the contemporaneous prompt/config implementation. Classification "
        "is assigned before reading win/loss. Infrastructure errors are then removed; independent compatible batches are pooled.",
        "",
        "The historical no-rules tag remains the pre-`412ba5f` fallthrough to ordinary mechanics-hard. These trials estimate "
        "that historical implementation only; they do not estimate the action-list-retaining control described in the paper.",
        "",
        "## Batch inventory",
        "",
        "A row is one recovered invocation/batch. Mixed batches report the exact trial-level category counts; every logical "
        "trial has exactly one category in the JSON inventory.",
        "",
        "| Batch | Protocol identity | Effort | Config(s) | Raw N | Valid N | Wins | Eligibility | Exact exclusion reason |",
        "|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for batch in universe["batches"]:
        counts = ", ".join(f"{key}={value}" for key, value in batch["eligibility_counts"].items())
        lines.append(
            f"| `{batch['batch']}` | `{batch['protocol_identity']}` | "
            f"{', '.join(batch['reasoning_effort']) if isinstance(batch['reasoning_effort'], list) else batch['reasoning_effort']} | "
            f"{', '.join(batch['configs'])} | {batch['raw_n']} | {batch['valid_n']} | {batch['wins']} | {counts} | "
            f"{batch['exact_exclusion_reason']} |"
        )
    lines.extend([
        "",
        "## Final eligible cells",
        "",
        "`Historical candidate N` includes known-effort candidates, including smoke and later excluded records; the older "
        "May 1 default/unknown-effort runs remain visible in the batch table but are not silently assigned to `none` or `medium`. "
        "`Same-protocol candidate N` is the locked-protocol universe before failure/provenance exclusions.",
        "",
        "| Effort | Config | Historical candidate N | Same-protocol candidate N | Infrastructure-clean included N | Wins | Losses | Excluded by reason | Final pooled estimate |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in universe["final_cells"]:
        excluded = ", ".join(f"{key}={value}" for key, value in row["excluded_by_reason"].items()) or "none"
        lines.append(
            f"| {row['reasoning_effort']} | {row['config']} | {row['historical_candidate_n']} | "
            f"{row['same_protocol_candidate_n']} | {row['infrastructure_clean_included_n']} | {row['wins']} | "
            f"{row['losses']} | {excluded} | {row['final_pooled_estimate']} |"
        )
    lines.extend([
        "",
        "## Per-batch heterogeneity for pooled cells",
        "",
        "| Effort/config | Included batch | Included result |",
        "|---|---|---:|",
    ])
    included_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for trial in universe["trials"]:
        if trial["eligibility"] == INCLUDED:
            included_groups[(trial["reasoning_effort"], trial["config"], trial["batch"])].append(trial)
    for (effort, config, batch), trials in sorted(included_groups.items()):
        wins = sum(trial["won"] for trial in trials)
        lines.append(f"| {effort}/{config} | `{batch}` | {wins}/{len(trials)} |")
    lines.extend([
        "",
        "All five included no-rules-none batches are 0% individually (0/5, 0/4, 0/1, 0/4, 0/5), so the pooled 0/19 "
        "does not conceal directional batch heterogeneity. The two 4/5 batches each exclude one indeterminate parse-provenance trial.",
        "The June 1 N=1 batch (`20260601T021458_230073`) is included: it has the same observed 64-turn/two-attempt "
        "identity as the surrounding parallel batches, and no preserved plan, name, or log marks it as smoke/debug. N=1 alone "
        "is not an exclusion criterion, and its loss was not used in the decision.",
        "",
        "## Provenance correction to PR #20",
        "",
        "The May 13 aggregate preserves effort-major runner order: all `none` configurations precede all `medium` configurations. "
        "PR #20's filenames instead interleaved effort labels within world-hard and mechanics-hard. Using aggregate order plus raw "
        "timestamps restores the correct membership: `none` world-hard and mechanics-hard each have five clean trials; `medium` "
        "world-hard and mechanics-hard each have five provider-empty trials and no clean denominator. The raw outcomes did not "
        "determine this remapping.",
        "",
        "## Exhaustiveness proof and limits",
        "",
        "The audit scanned all objects reachable from all local/remote refs under the three result roots, then checked all local "
        "worktrees. It found 346 GPT-5.2 KA59-Simple JSON artifacts with 346 unique blobs and no untracked GPT result artifact. "
        "Every logical trial represented by an aggregate/sidecar or raw-only smoke file is accounted for here; restored PR #20 "
        "copies are linked as duplicates rather than counted again. The June 9 N=1 sidecar has no raw trial and is therefore "
        "provenance-insufficient, not converted into a model loss.",
        "",
        "This closes historical reconciliation without compute. Additional runs are a separate author decision: they are necessary "
        "only if the camera-ready estimand requires larger per-effort N, a pinned fresh matrix, or the paper-intended no-rules control.",
        "",
    ])
    return "\n".join(lines)


def _encoded_json(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write JSON and Markdown outputs")
    parser.add_argument("--check", action="store_true", help="fail if checked-in outputs differ")
    args = parser.parse_args(argv)
    universe = build_universe()
    encoded_json = _encoded_json(universe)
    markdown = render_markdown(universe)
    if args.check:
        return 0 if OUTPUT_JSON.exists() and OUTPUT_MD.exists() and OUTPUT_JSON.read_text() == encoded_json and OUTPUT_MD.read_text() == markdown else 1
    if args.write:
        OUTPUT_JSON.write_text(encoded_json)
        OUTPUT_MD.write_text(markdown)
    else:
        print(encoded_json, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
