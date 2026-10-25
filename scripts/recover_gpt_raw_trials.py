"""Recover GPT-5.2 raw trial files that survive only as git blobs.

The manifest cites these trials by the path they had on
`levi/ka59simple-level-attempts`, a local commit that is no longer reachable
(c9063f29). The commit object is gone but the blobs it referenced are still in
this repository's object store, so the contents are recoverable exactly.

Each recovered file is verified against the semantic digest the audit recorded
when the trial was still readable. Makes no model calls.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.audit_ka59_camera_ready import (
    GPT_UNIVERSE_PATH,
    MANIFEST_PATH,
    REPO_ROOT,
)
from scripts.audit_ka59_gpt_complete_universe import _sha256_json

RECOVERED_DIR = REPO_ROOT / "ka59_game" / "results" / "gpt-5.2-recovered"


def _needed_paths() -> set[str]:
    """Paths the manifest counts in a scored cell but that are not on disk.

    Scoped deliberately: the universe holds blobs for hundreds of candidates,
    including different-protocol runs and sidecar sub-documents that are not
    files at all. Only trials backing a reported number are worth restoring.
    """
    manifest = json.loads(MANIFEST_PATH.read_text())
    universe = json.loads(GPT_UNIVERSE_PATH.read_text())
    restored = {
        a["duplicate_of"]: a["source_path"]
        for a in universe.get("duplicate_artifacts", [])
        if a.get("duplicate_of") and a.get("source_path")
    }
    needed: set[str] = set()
    for cell in manifest["paper_cells"]:
        if cell["model"] != "openai/gpt-5.2":
            continue
        for cited in cell["infrastructure_clean_scored_raw_files"]:
            if (REPO_ROOT / cited).exists():
                continue
            copy = restored.get(cited)
            if copy and (REPO_ROOT / copy).exists():
                continue
            if "#" in cited:          # sidecar sub-document, not a file
                continue
            needed.add(cited)
    return needed


def _blob_records() -> dict[str, dict[str, Any]]:
    """Cited path -> trial record, for every GPT trial with a recoverable blob."""
    universe = json.loads(GPT_UNIVERSE_PATH.read_text())
    records: dict[str, dict[str, Any]] = {}
    for batch in universe.get("batches", []):
        for trial in batch.get("trial_records", []):
            path = trial.get("raw_source_path") or trial.get("source_path")
            if path and trial.get("source_blob"):
                records.setdefault(str(path), trial)
    return records


def _read_blob(sha: str) -> bytes | None:
    probe = subprocess.run(["git", "cat-file", "-t", sha],
                           capture_output=True, text=True, cwd=REPO_ROOT)
    if probe.stdout.strip() != "blob":
        return None
    return subprocess.run(["git", "cat-file", "-p", sha],
                          capture_output=True, cwd=REPO_ROOT).stdout


def recover(write: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {
        "recovered": [], "verified": [], "digest_mismatch": [],
        "unreadable_blob": [], "already_on_disk": [],
    }
    wanted = _needed_paths()
    for cited, trial in sorted(_blob_records().items()):
        if cited not in wanted:
            continue
        if (REPO_ROOT / cited).exists():
            report["already_on_disk"].append(cited)
            continue
        raw = _read_blob(str(trial["source_blob"]))
        if raw is None:
            report["unreadable_blob"].append(cited)
            continue
        payload = json.loads(raw)
        expected = trial.get("source_semantic_sha256")
        actual = _sha256_json(payload)
        target = RECOVERED_DIR / Path(cited).name
        if write:
            RECOVERED_DIR.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        report["recovered"].append(str(target.relative_to(REPO_ROOT)))
        if expected and actual == expected:
            report["verified"].append(str(target.relative_to(REPO_ROOT)))
        elif expected:
            report["digest_mismatch"].append(
                {"file": cited, "expected": expected, "actual": actual}
            )
    return report


def main() -> int:
    report = recover(write=True)
    for key in ("recovered", "verified", "already_on_disk",
                "unreadable_blob", "digest_mismatch"):
        print(f"{key:20s} {len(report[key])}")
    for bad in report["digest_mismatch"]:
        print(f"  MISMATCH {bad['file']}")
    return 1 if report["digest_mismatch"] or report["unreadable_blob"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
