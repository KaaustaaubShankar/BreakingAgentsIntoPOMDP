"""Resume-safe KA59-Simple camera-ready runner.

`--plan` and `--smoke` never instantiate an LLM client. Actual model calls are
made only when neither flag is present and the requested target has a deficit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ka59_game.llm_client import OPENROUTER_MAX_TOKENS

from scripts.audit_ka59_camera_ready import (
    PAPER_CONFIGS,
    REPO_ROOT,
    _fatal_errors,
    _load_json,
    _system_prompt_fingerprint,
    build_manifest,
)


ACCEPTED_CONFIGS: dict[str, dict[str, str]] = {
    "baseline": {
        "world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"
    },
    "world_hard": {
        "world": "HARD", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"
    },
    "mechanics_hard": {
        "world": "EASY", "goal": "EASY", "mechanics": "HARD", "feedback": "EASY"
    },
    # This deliberately matches the accepted raw implementation. Before
    # 412ba5f, HARD_FORMAT_ONLY fell through to MECHANICS_HARD. The
    # paper-intended action-list-retaining control is a different protocol.
    "mechanics_hard_format_only": {
        "world": "EASY", "goal": "EASY", "mechanics": "HARD_FORMAT_ONLY", "feedback": "EASY"
    },
    "feedback_hard": {
        "world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "HARD"
    },
}
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
PROTOCOL_BASE = {
    "protocol_name": "ka59-camera-ready-accepted-raw-v1",
    "environment": "ka59simple",
    "environment_revision": "20260430",
    "environment_source_path": "environment_files/ka59simple/20260430/ka59simple.py",
    "turns_per_attempt": 64,
    "level_attempts": 2,
    "max_levels": 1,
    "failed_attempt_context_retained": True,
}

# Accepted-data model slugs, keyed by the slug a transport requires. DeepSeek
# trials are pooled across transports by explicit author decision; GPT-5.2 is
# not (see _accepted_count).
MODEL_ALIASES = {
    "deepseek/deepseek-v4-pro": "deepseek-v4-pro",
}
CROSS_TRANSPORT_POOLED_MODELS = {"deepseek-v4-pro"}
RUNTIME_ROOT = REPO_ROOT / "camera_ready" / "results"


def protocol_identity(
    provider: str,
    model: str,
    effort: str,
    config: str,
    upstream_provider: str | None = None,
) -> dict[str, Any]:
    # The prompt itself is the identity, not a hardcoded claim about it. A cell
    # whose prompt bytes change gets a new protocol_id automatically.
    fingerprint = _system_prompt_fingerprint(config)
    identity = {
        **PROTOCOL_BASE,
        "provider": provider,
        "upstream_provider": upstream_provider,
        "model": model,
        "reasoning_effort": effort,
        "config": config,
        "config_levels": ACCEPTED_CONFIGS[config],
        "system_prompt_sha256": fingerprint["current_revision_sha256"],
        "reproduces_accepted_prompt": not fingerprint["current_revision_differs"],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return {**identity, "protocol_id": hashlib.sha256(encoded).hexdigest()[:20]}


def _accepted_count(manifest: dict[str, Any], identity: dict[str, Any]) -> tuple[int, int, int]:
    # A cell whose prompt no longer reproduces the accepted one has no accepted
    # evidence, whatever its label says. This is what keeps the ported
    # format-only control from silently inheriting the historical fallthrough
    # trials that ran the ordinary mechanics-hard prompt.
    if not identity["reproduces_accepted_prompt"]:
        return 0, 0, 0
    accepted_model = MODEL_ALIASES.get(identity["model"], identity["model"])
    # Prefer a recorded pooled cell: configs proven to have delivered the same
    # treatment share one denominator, so the planner must not re-run a
    # condition that is already satisfied under its pooled count.
    for pooled in manifest.get("prompt_identity_pooling", {}).get("pooled_cells", []):
        if (
            pooled["model"] == accepted_model
            and pooled["reasoning_effort"] == identity["reasoning_effort"]
            and identity["config"] in pooled["pooled_configs"]
        ):
            return (
                int(pooled["strict_error_free_n"]),
                int(pooled["strict_error_free_wins"]),
                int(pooled["strict_error_free_losses"]),
            )
    for cell in manifest["paper_cells"]:
        if (
            cell["model"] == accepted_model
            and cell["reasoning_effort"] == identity["reasoning_effort"]
            and cell["config"] == identity["config"]
        ):
            # GPT-5.2 accepted artifacts are OpenRouter-only and are not pooled
            # into a different transport. DeepSeek is pooled across transports
            # by explicit author decision, recorded in the manifest.
            if accepted_model not in CROSS_TRANSPORT_POOLED_MODELS:
                accepted_provider = (
                    "openrouter" if accepted_model == "openai/gpt-5.2" else None
                )
                if accepted_provider is not None and identity["provider"] != accepted_provider:
                    return 0, 0, 0
            return int(cell["valid_n"]), int(cell["wins"]), int(cell["losses"])
    return 0, 0, 0


def _runtime_dir(identity: dict[str, Any]) -> Path:
    return RUNTIME_ROOT / str(identity["protocol_id"])


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _runtime_counts(identity: dict[str, Any]) -> dict[str, Any]:
    output_dir = _runtime_dir(identity)
    valid = wins = losses = errors = 0
    incompatible: list[str] = []
    files: list[str] = []
    if not output_dir.exists():
        return {
            "valid_n": 0, "wins": 0, "losses": 0, "errors": 0,
            "files": [], "incompatible": [],
        }
    for path in sorted(output_dir.glob("run_*.json")):
        payload = _load_json(path)
        files.append(_display_path(path))
        if payload.get("protocol_identity") != identity:
            incompatible.append(_display_path(path))
            continue
        fatal, _ = _fatal_errors(payload)
        if fatal:
            errors += 1
            continue
        valid += 1
        if payload.get("won"):
            wins += 1
        else:
            losses += 1
    return {
        "valid_n": valid, "wins": wins, "losses": losses, "errors": errors,
        "files": files, "incompatible": incompatible,
    }


def build_plan(
    *, provider: str, model: str, effort: str, configs: Iterable[str], target_n: int,
    upstream_provider: str | None = None,
) -> dict[str, Any]:
    manifest = build_manifest(target_n)
    cells: list[dict[str, Any]] = []
    for config in configs:
        identity = protocol_identity(provider, model, effort, config, upstream_provider)
        accepted_n, accepted_wins, accepted_losses = _accepted_count(manifest, identity)
        runtime = _runtime_counts(identity)
        if runtime["incompatible"]:
            raise ValueError(
                "Incompatible experimental identities found in the protocol output directory: "
                + ", ".join(runtime["incompatible"])
            )
        current_n = accepted_n + runtime["valid_n"]
        cells.append({
            "config": config,
            "protocol_id": identity["protocol_id"],
            "accepted_valid_n": accepted_n,
            "accepted_wins": accepted_wins,
            "accepted_losses": accepted_losses,
            "resumed_valid_n": runtime["valid_n"],
            "resumed_wins": runtime["wins"],
            "resumed_losses": runtime["losses"],
            "recorded_errors_excluded": runtime["errors"],
            "current_valid_n": current_n,
            "target_n": target_n,
            "deficit": max(0, target_n - current_n),
            "output_directory": str(_runtime_dir(identity).relative_to(REPO_ROOT)),
            "protocol_identity": identity,
        })
    return {
        "external_calls": 0,
        "provider": provider,
        "upstream_provider": upstream_provider,
        "model": model,
        "reasoning_effort": effort,
        "target_n": target_n,
        "cells": cells,
        "total_trials_to_run": sum(cell["deficit"] for cell in cells),
    }


def smoke_report() -> dict[str, Any]:
    from ka59_game.prompts import MECHANICS_HARD, build_system_prompt

    required = [
        REPO_ROOT / "environment_files" / "ka59" / "38d34dbb" / "ka59.py",
        REPO_ROOT / "environment_files" / "ka59simple" / "20260430" / "ka59simple.py",
    ]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.exists()]
    format_only_prompt = build_system_prompt("EASY", "HARD_FORMAT_ONLY")
    mechanics_hard_prompt = build_system_prompt("EASY", "HARD")
    runtime_error = None
    environment_initialised = False
    if not missing:
        try:
            from ka59_game.experiment import _make_env

            env = _make_env("ka59simple")
            environment_initialised = env.observation_space is not None
        except Exception as exc:  # structural report; still no model call
            runtime_error = f"{type(exc).__name__}: {exc}"
    return {
        "external_model_calls": 0,
        "required_environment_files_present": not missing,
        "missing_files": missing,
        "format_only_prompt_regime": (
            "historical_fallthrough_to_MECHANICS_HARD"
            if format_only_prompt == mechanics_hard_prompt
            else "paper_intended_action_protocol_retained"
        ),
        "format_only_is_a_distinct_control": format_only_prompt != mechanics_hard_prompt,
        "environment_initialised": environment_initialised,
        "runtime_error": runtime_error,
    }


def _run(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    from ka59_game.experiment import run_agent, save_result

    infrastructure_errors = 0
    for cell in plan["cells"]:
        identity = cell["protocol_identity"]
        output_dir = REPO_ROOT / cell["output_directory"]
        output_dir.mkdir(parents=True, exist_ok=True)
        current_valid_n = int(cell["current_valid_n"])
        sequence = 0
        while current_valid_n < args.target_n:
            sequence += 1
            result = run_agent(
                world_level=identity["config_levels"]["world"],
                goal_level=identity["config_levels"]["goal"],
                mechanics_level=identity["config_levels"]["mechanics"],
                feedback_level=identity["config_levels"]["feedback"],
                provider=args.provider,
                model=args.model,
                max_levels=1,
                turns_per_level=64,
                verbose=args.verbose,
                reasoning_effort=args.reasoning_effort,
                upstream_provider=args.upstream_provider,
                env_id="ka59simple",
                save=False,
            )
            run_id = (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
                + f"_{cell['config']}_{args.reasoning_effort}_seq{sequence}"
            )
            fatal, fatal_errors = _fatal_errors({"errors": result.errors})
            save_result(
                result,
                run_id=run_id,
                output_dir=output_dir,
                extra_metadata={
                    "protocol_identity": identity,
                    # Provenance, deliberately outside protocol_id: the cap is
                    # not a treatment where it never binds, and hashing it would
                    # orphan completed cells. Recorded so a binding cap is
                    # always visible in the raw file.
                    "generation_max_tokens": OPENROUTER_MAX_TOKENS,
                    "camera_ready_status": fatal or ("win" if result.won else "loss"),
                },
            )
            if fatal:
                infrastructure_errors += 1
                print(
                    f"ERROR (excluded, not a loss): {cell['config']} — "
                    + (fatal_errors[0] if fatal_errors else fatal)
                )
                if infrastructure_errors >= args.max_infrastructure_errors:
                    print(
                        "Stopped after repeated infrastructure failures; valid target was not reached.",
                        file=sys.stderr,
                    )
                    return 2
                continue
            current_valid_n += 1
            print(
                f"VALID {'WIN' if result.won else 'LOSS'}: {cell['config']} "
                f"({current_valid_n}/{args.target_n})"
            )
    return 0


def build_index() -> str:
    """Human-readable index of the opaque protocol_id directories.

    The runner resolves output directories by hash, so they cannot be renamed
    without breaking resume. This maps each hash back to the cell it holds.
    """
    lines = [
        "# Camera-ready run index",
        "",
        "Generated by `python -m scripts.run_ka59_camera_ready --index`. Makes no "
        "model calls. Directories are named by `protocol_id`; the runner resolves "
        "them by hash, so do not rename them.",
        "",
    ]
    dirs = sorted(d for d in RUNTIME_ROOT.glob("*") if d.is_dir()) if RUNTIME_ROOT.exists() else []
    if not dirs:
        lines += ["No runs recorded yet.", ""]
        return "\n".join(lines)
    lines += [
        "| protocol_id | model | effort | config | upstream | trials | W/L | excluded |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    details: list[str] = []
    for directory in dirs:
        runs = sorted(directory.glob("run_*.json"))
        if not runs:
            continue
        identity = _load_json(runs[0]).get("protocol_identity", {})
        wins = losses = excluded = 0
        for path in runs:
            payload = _load_json(path)
            fatal, _ = _fatal_errors(payload)
            if fatal:
                excluded += 1
            elif payload.get("won"):
                wins += 1
            else:
                losses += 1
        lines.append(
            f"| `{directory.name}` | {identity.get('model','?')} | "
            f"{identity.get('reasoning_effort','?')} | {identity.get('config','?')} | "
            f"{identity.get('upstream_provider') or 'unpinned'} | {wins + losses} | "
            f"{wins}/{losses} | {excluded} |"
        )
        details += [
            f"### `{directory.name}`",
            "",
            f"- cell: **{identity.get('model')} / {identity.get('reasoning_effort')} / "
            f"{identity.get('config')}**",
            f"- upstream pin: {identity.get('upstream_provider') or 'unpinned'}",
            f"- system prompt sha256: `{str(identity.get('system_prompt_sha256'))[:32]}`",
            f"- reproduces accepted prompt: {identity.get('reproduces_accepted_prompt')}",
            f"- turn budget: {identity.get('turns_per_attempt')} x "
            f"{identity.get('level_attempts')} attempts",
            f"- valid trials: {wins + losses} ({wins} win / {losses} loss); "
            f"excluded on infrastructure: {excluded}",
            "",
        ]
    lines += ["", "## Cell detail", ""] + details
    return "\n".join(lines)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort", choices=REASONING_EFFORTS)
    parser.add_argument(
        "--upstream-provider",
        help="pin the OpenRouter upstream (e.g. DigitalOcean); recorded in protocol_id",
    )
    parser.add_argument("--target-n", type=int, default=20)
    parser.add_argument("--config", action="append", choices=PAPER_CONFIGS)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--index", action="store_true",
                        help="write camera_ready/results/README.md; no model calls")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-infrastructure-errors", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.target_n < 1:
        parser.error("--target-n must be positive")
    if args.max_infrastructure_errors < 1:
        parser.error("--max-infrastructure-errors must be positive")
    if not (args.smoke or args.index) and not all(
        (args.provider, args.model, args.reasoning_effort)
    ):
        parser.error("--provider, --model, and --reasoning-effort are required unless --smoke is used")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.index:
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        target = RUNTIME_ROOT / "README.md"
        target.write_text(build_index() + "\n")
        print(f"Wrote {target.relative_to(REPO_ROOT)}")
        return 0
    if args.smoke:
        report = smoke_report()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if all((
            report["required_environment_files_present"],
    
            report["environment_initialised"],
        )) else 1
    configs = args.config or list(PAPER_CONFIGS)
    plan = build_plan(
        provider=args.provider,
        model=args.model,
        effort=args.reasoning_effort,
        configs=configs,
        target_n=args.target_n,
        upstream_provider=args.upstream_provider,
    )
    if args.plan:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if plan["total_trials_to_run"] == 0:
        print("All requested cells already meet the target; zero model calls made.")
        return 0
    if not args.resume and any(cell["resumed_valid_n"] for cell in plan["cells"]):
        print(
            "Existing camera-ready trials found. Re-run with --resume to continue without duplication.",
            file=sys.stderr,
        )
        return 2
    return _run(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
