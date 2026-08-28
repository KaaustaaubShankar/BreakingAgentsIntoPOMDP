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

from scripts.audit_ka59_camera_ready import (
    PAPER_CONFIGS,
    REPO_ROOT,
    _fatal_errors,
    _load_json,
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
    "format_only_behavior": "historical_fallthrough_to_MECHANICS_HARD",
    "prompt_revision_recovered": "ff0c184",
}
RUNTIME_ROOT = REPO_ROOT / "camera_ready" / "results"


def protocol_identity(provider: str, model: str, effort: str, config: str) -> dict[str, Any]:
    identity = {
        **PROTOCOL_BASE,
        "provider": provider,
        "model": model,
        "reasoning_effort": effort,
        "config": config,
        "config_levels": ACCEPTED_CONFIGS[config],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return {**identity, "protocol_id": hashlib.sha256(encoded).hexdigest()[:20]}


def _accepted_count(manifest: dict[str, Any], identity: dict[str, Any]) -> tuple[int, int, int]:
    for cell in manifest["paper_cells"]:
        if (
            cell["model"] == identity["model"]
            and cell["reasoning_effort"] == identity["reasoning_effort"]
            and cell["config"] == identity["config"]
        ):
            # Accepted GPT artifacts used OpenRouter; DeepSeek used a mix that
            # is not pooled into a differently requested provider identity.
            accepted_provider = "openrouter" if identity["model"] == "openai/gpt-5.2" else None
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
    *, provider: str, model: str, effort: str, configs: Iterable[str], target_n: int
) -> dict[str, Any]:
    manifest = build_manifest(target_n)
    cells: list[dict[str, Any]] = []
    for config in configs:
        identity = protocol_identity(provider, model, effort, config)
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
        "historical_format_only_fallthrough_verified": format_only_prompt == mechanics_hard_prompt,
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


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort", choices=REASONING_EFFORTS)
    parser.add_argument("--target-n", type=int, default=20)
    parser.add_argument("--config", action="append", choices=PAPER_CONFIGS)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-infrastructure-errors", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.target_n < 1:
        parser.error("--target-n must be positive")
    if args.max_infrastructure_errors < 1:
        parser.error("--max-infrastructure-errors must be positive")
    if not args.smoke and not all((args.provider, args.model, args.reasoning_effort)):
        parser.error("--provider, --model, and --reasoning-effort are required unless --smoke is used")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.smoke:
        report = smoke_report()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if all((
            report["required_environment_files_present"],
            report["historical_format_only_fallthrough_verified"],
            report["environment_initialised"],
        )) else 1
    configs = args.config or list(PAPER_CONFIGS)
    plan = build_plan(
        provider=args.provider,
        model=args.model,
        effort=args.reasoning_effort,
        configs=configs,
        target_n=args.target_n,
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
