"""Per-model ablation summaries for the camera-ready cells.

Mirrors the shape of ka59_game/results/ablation_summary_*.json, but reports the
audited denominator rather than a nominal trial count, and lists the exact files
behind every number: accepted historical trials, new camera-ready trials, and
the infrastructure failures excluded from the denominator.

Makes no model calls.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.audit_ka59_camera_ready import (
    CONFIG_LEVELS,
    GPT_UNIVERSE_PATH,
    PAPER_CONFIGS,
    REPO_ROOT,
    build_manifest,
)

OUT_DIR = REPO_ROOT / "ka59_game" / "results" / "camera_ready"
RUNTIME_ROOT = REPO_ROOT / "camera_ready" / "results"
# direct-API / OpenRouter slug -> the accepted-data model name
SLUG_TO_MODEL = {
    "gpt-5.2": "openai/gpt-5.2",
    "openai/gpt-5.2": "openai/gpt-5.2",
    "deepseek/deepseek-v4-pro": "deepseek-v4-pro",
    "deepseek-v4-pro": "deepseek-v4-pro",
}


def _restored_copies() -> dict[str, str]:
    """Historical trial path -> the restored copy that actually exists on disk.

    The manifest cites GPT trials by the path they had when they were run. PR
    #20 restored those trials under different names, so the cited paths are
    provenance, not locations. This maps one to the other.
    """
    universe = json.loads(GPT_UNIVERSE_PATH.read_text())
    return {
        artifact["duplicate_of"]: artifact["source_path"]
        for artifact in universe.get("duplicate_artifacts", [])
        if artifact.get("duplicate_of") and artifact.get("source_path")
    }


RECOVERED_DIR = REPO_ROOT / "ka59_game" / "results" / "gpt-5.2-recovered"


def _resolve(relative: str, restored: dict[str, str]) -> str | None:
    """An openable repo-relative path for a cited trial, or None if it is gone."""
    if (REPO_ROOT / relative).exists():
        return relative
    candidate = restored.get(relative)
    if candidate and (REPO_ROOT / candidate).exists():
        return candidate
    # trials rebuilt from surviving git blobs by scripts/recover_gpt_raw_trials.py
    recovered = RECOVERED_DIR / Path(relative).name
    if recovered.exists():
        return str(recovered.relative_to(REPO_ROOT))
    return None


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _camera_ready_trials() -> dict[tuple[str, str, str], dict[str, Any]]:
    """New trials grouped by (model, effort, config)."""
    grouped: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"wins": [], "losses": [], "excluded": [], "turns": [],
                 "protocol_ids": set(), "upstream": set()}
    )
    for path in sorted(RUNTIME_ROOT.glob("*/run_*.json")):
        payload = json.loads(path.read_text())
        identity = payload.get("protocol_identity", {})
        model = SLUG_TO_MODEL.get(str(identity.get("model")), str(identity.get("model")))
        key = (model, str(identity.get("reasoning_effort")), str(identity.get("config")))
        bucket = grouped[key]
        bucket["protocol_ids"].add(identity.get("protocol_id"))
        bucket["upstream"].add(identity.get("upstream_provider") or identity.get("provider"))
        status = payload.get("camera_ready_status")
        if status == "win":
            bucket["wins"].append(_display(path)); bucket["turns"].append(payload.get("turns"))
        elif status == "loss":
            bucket["losses"].append(_display(path)); bucket["turns"].append(payload.get("turns"))
        else:
            bucket["excluded"].append(_display(path))
    return grouped


# bp35/results/ablation_summary_*.json field order, with KA59-Simple's metric
# set (the bp35 gravity/undo counters do not exist in this environment).
METRICS = (
    "turns", "levels_completed", "invalid_actions",
    "click_actions", "wall_transfers", "object_pushes", "max_goals_occupied",
)


def _metrics(paths: list[str]) -> dict[str, float]:
    values: dict[str, list[float]] = {name: [] for name in METRICS}
    for relative in paths:
        payload = json.loads((REPO_ROOT / relative).read_text())
        for name in METRICS:
            value = payload.get(name)
            if isinstance(value, (int, float)):
                values[name].append(float(value))
    return {
        f"avg_{name}": round(sum(v) / len(v), 4) if v else None
        for name, v in values.items()
    }


def build_summary(model: str) -> list[dict[str, Any]]:
    manifest = build_manifest(20)
    cells = {
        (c["reasoning_effort"], c["config"]): c
        for c in manifest["paper_cells"] if c["model"] == model
    }
    pooled = {
        p["reasoning_effort"]: p
        for p in manifest["prompt_identity_pooling"]["pooled_cells"]
        if p["model"] == model
    }
    new_trials = _camera_ready_trials()
    rows: list[dict[str, Any]] = []
    for effort in ("none", "medium"):
        for config in PAPER_CONFIGS:
            cell = cells.get((effort, config))
            if cell is None:
                continue
            accepted = list(cell["infrastructure_clean_scored_raw_files"])
            accepted_wins = cell["infrastructure_clean_scored_wins"]
            pooled_from: list[str] = []
            if config == "mechanics_hard" and effort in pooled:
                partner = cells.get((effort, "mechanics_hard_format_only"))
                if partner is not None:
                    pooled_from = list(partner["infrastructure_clean_scored_raw_files"])
                    accepted += pooled_from
                    accepted_wins = pooled[effort]["infrastructure_clean_scored_wins"]
            if config == "mechanics_hard_format_only":
                # the ported control shares no history with the fallthrough trials
                accepted, accepted_wins, pooled_from = [], 0, []
            bucket = new_trials.get((model, effort, config))
            fresh = (bucket["wins"] + bucket["losses"]) if bucket else []
            fresh_wins = len(bucket["wins"]) if bucket else 0
            excluded = bucket["excluded"] if bucket else []
            run_files = accepted + fresh
            wins = accepted_wins + fresh_wins
            n = len(run_files)
            restored = _restored_copies()
            resolved = [r for r in (_resolve(f, restored) for f in run_files) if r]
            missing = len(run_files) - len(resolved)
            providers = sorted({
                str(json.loads((REPO_ROOT / f).read_text()).get("provider"))
                for f in resolved
            }) if resolved else []
            rows.append({
                "config_name": config,
                "config": CONFIG_LEVELS[config],
                "provider": providers[0] if len(providers) == 1 else "pooled",
                "model": model,
                "reasoning_effort": effort,
                "n_trials": n,
                "wins": wins,
                "win_rate": round(wins / n, 4) if n else None,
                **_metrics(resolved),
                # avg_* are computed only from files that exist on disk. Where
                # that is fewer than n_trials, the audited denominator still
                # counts trials whose raw files were never restored.
                "metrics_from_n_files": len(resolved),
                "run_files": resolved,
                # KA59-Simple audit fields with no bp35 equivalent
                "denominator": "infrastructure_clean_scored",
                "target_n": 20,
                "additional_trials_needed": max(0, 20 - n),
                "run_files_unresolvable_on_disk": missing,
                "run_files_as_cited_by_manifest": run_files,
                "accepted_run_files": accepted,
                "camera_ready_run_files": fresh,
                "pooled_from_format_only_fallthrough": pooled_from,
                "excluded_infrastructure_files": excluded,
            })
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for model, stem in (("openai/gpt-5.2", "gpt_5_2"),
                        ("deepseek-v4-pro", "deepseek_v4_pro")):
        rows = build_summary(model)
        path = OUT_DIR / f"ablation_summary_{stem}_camera_ready.json"
        # insertion order matches bp35/results/ablation_summary_*.json; do not sort
        path.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"Wrote {_display(path)}")
        for row in rows:
            if row["n_trials"]:
                print(f"    {row['reasoning_effort']:7s} {row['config_name']:28s} "
                      f"{row['wins']}/{row['n_trials']} ({row['win_rate']:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
