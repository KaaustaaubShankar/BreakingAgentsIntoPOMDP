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
    new = _camera_ready_trials()
    rows: list[dict[str, Any]] = []
    for effort in ("none", "medium"):
        for config in PAPER_CONFIGS:
            cell = cells.get((effort, config))
            if cell is None:
                continue
            accepted_files = list(cell["infrastructure_clean_scored_raw_files"])
            accepted_n = cell["infrastructure_clean_scored_n"]
            accepted_wins = cell["infrastructure_clean_scored_wins"]
            pooled_from: list[str] = []
            if config == "mechanics_hard" and effort in pooled:
                group = pooled[effort]
                partner = cells.get((effort, "mechanics_hard_format_only"))
                if partner is not None:
                    extra = list(partner["infrastructure_clean_scored_raw_files"])
                    accepted_files += extra
                    pooled_from = extra
                    accepted_n = group["infrastructure_clean_scored_n"]
                    accepted_wins = group["infrastructure_clean_scored_wins"]
            if config == "mechanics_hard_format_only":
                # the ported control shares no history with the fallthrough trials
                accepted_files, accepted_n, accepted_wins, pooled_from = [], 0, 0, []
            bucket = new.get((model, effort, config))
            new_wins = bucket["wins"] if bucket else []
            new_losses = bucket["losses"] if bucket else []
            excluded = bucket["excluded"] if bucket else []
            valid_n = accepted_n + len(new_wins) + len(new_losses)
            wins = accepted_wins + len(new_wins)
            rows.append({
                "config": CONFIG_LEVELS[config],
                "config_name": config,
                "model": model,
                "reasoning_effort": effort,
                "valid_n": valid_n,
                "wins": wins,
                "losses": valid_n - wins,
                "win_rate": round(wins / valid_n, 4) if valid_n else None,
                "win_rate_display": f"{wins}/{valid_n}" + (
                    f" ({wins / valid_n:.0%})" if valid_n else " (no scored trials)"
                ),
                "target_n": 20,
                "additional_trials_needed": max(0, 20 - valid_n),
                "denominator": "infrastructure_clean_scored",
                "accepted_trials": {
                    "n": accepted_n,
                    "wins": accepted_wins,
                    "run_files": accepted_files,
                    "pooled_from_format_only_fallthrough": pooled_from,
                },
                "camera_ready_trials": {
                    "n": len(new_wins) + len(new_losses),
                    "wins": len(new_wins),
                    "protocol_ids": sorted(x for x in (bucket["protocol_ids"] if bucket else set()) if x),
                    "routing": sorted(x for x in (bucket["upstream"] if bucket else set()) if x),
                    "win_files": new_wins,
                    "loss_files": new_losses,
                },
                "excluded_infrastructure_files": excluded,
            })
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for model, stem in (("openai/gpt-5.2", "gpt_5_2"),
                        ("deepseek-v4-pro", "deepseek_v4_pro")):
        rows = build_summary(model)
        path = OUT_DIR / f"ablation_summary_{stem}_camera_ready.json"
        path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {_display(path)}")
        for row in rows:
            if row["valid_n"]:
                print(f"    {row['reasoning_effort']:7s} {row['config_name']:28s} "
                      f"{row['win_rate_display']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
