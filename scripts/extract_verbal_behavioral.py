"""
Post-process per-trial JSONs to extract verbal-vs-behavioral discovery metrics.

For every trial JSON under results/ka59_game/ and results/ka59simple_game/, computes:
  verbal_discovery_turn    - first turn an `orient` text matched check_discovery() patterns
  behavioral_discovery_turn - first turn a wall_transfer or object_push event was recorded
                              (reconstructed from action history + post-state inspection)
  verbal_minus_behavioral  - delta in turns. Negative = said it before doing it.
                              Positive = did it before saying it. None = at least one never happened.
  total_orient_words       - rough verbosity proxy
  total_orient_turns       - turns where the agent emitted any orient text

Usage:
    python3 -m scripts.extract_verbal_behavioral
    python3 -m scripts.extract_verbal_behavioral --csv > out.csv
    python3 -m scripts.extract_verbal_behavioral --dir results/ka59simple_game
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

repo_root = Path(__file__).parents[1]
sys.path.insert(0, str(repo_root))

from ka59_game.prompts import check_discovery


def _verbal_discovery_turn(history: list[dict]) -> int | None:
    """First turn the discovery regex fires on ANY verbal channel (reasoning or
    orient). Candidate-level only: matches include speculative hypotheses
    ("test if the wall is passable"), not just confirmed rule statements —
    downstream LLM-judge classification distinguishes the two."""
    for h in history:
        if h.get("type") != "action":
            continue
        text = " ".join(t for t in (h.get("reasoning"), h.get("orient")) if t)
        if text and check_discovery(text):
            return h.get("turn")
    return None


def _behavioral_discovery_turn(history: list[dict]) -> int | None:
    """First turn a wall-transfer event was recorded, from the per-turn
    `outcome` flags experiment.py attaches to each action record. Older trial
    JSONs (before 2026-06-12) lack `outcome` and return None — for those, only
    the aggregate wall_transfers count is available."""
    for h in history:
        if h.get("type") != "action":
            continue
        if (h.get("outcome") or {}).get("wall_transfers"):
            return h.get("turn")
    return None


def _orient_stats(history: list[dict]) -> tuple[int, int]:
    turns_with_orient = 0
    word_count = 0
    for h in history:
        if h.get("type") != "action":
            continue
        orient = h.get("orient") or ""
        if orient.strip():
            turns_with_orient += 1
            word_count += len(orient.split())
    return turns_with_orient, word_count


def analyze(json_path: Path) -> dict:
    d = json.loads(json_path.read_text())
    history = d.get("history", []) or []
    verbal_turn = _verbal_discovery_turn(history)
    behavioral_turn = _behavioral_discovery_turn(history)
    orient_turns, orient_words = _orient_stats(history)

    delta = None
    if verbal_turn is not None and behavioral_turn is not None:
        delta = verbal_turn - behavioral_turn

    config = d.get("config", {})
    return {
        "file": json_path.name,
        "env_id": d.get("env_id", "ka59"),
        "model": d.get("model", ""),
        "world": config.get("world", ""),
        "goal": config.get("goal", ""),
        "mechanics": config.get("mechanics", ""),
        "feedback": config.get("feedback", ""),
        "won": d.get("won"),
        "turns": d.get("turns"),
        "click_actions": d.get("click_actions", 0),
        "object_pushes": d.get("object_pushes", 0),
        "wall_transfers": d.get("wall_transfers", 0),
        "max_goals_occupied": d.get("max_goals_occupied", 0),
        "forced_reframes": d.get("forced_reframes", 0),
        "verbal_discovery_turn": verbal_turn,
        "behavioral_discovery_aggregate":
            (d.get("wall_transfers", 0) + d.get("object_pushes", 0)) > 0,
        "behavioral_discovery_turn": behavioral_turn,
        "verbal_minus_behavioral": delta,
        "verbal_correct_behavioral_wrong":
            verbal_turn is not None and (d.get("wall_transfers", 0) + d.get("object_pushes", 0)) == 0,
        "verbal_wrong_behavioral_correct":
            verbal_turn is None and (d.get("wall_transfers", 0) + d.get("object_pushes", 0)) > 0,
        "orient_turns": orient_turns,
        "orient_words": orient_words,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", action="append", default=None,
                   help="Trial-JSON directory (repeatable). Default: results/ka59_game and results/ka59simple_game")
    p.add_argument("--csv", action="store_true", help="Emit CSV instead of pretty table")
    args = p.parse_args()

    dirs = [repo_root / d for d in args.dir] if args.dir else [
        repo_root / "results" / "ka59_game",
        repo_root / "results" / "ka59simple_game",
    ]

    rows = []
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("run_*.json")):
            try:
                rows.append(analyze(f))
            except Exception as exc:
                print(f"[skip] {f.name}: {exc}", file=sys.stderr)

    if not rows:
        print("No trial JSONs found.", file=sys.stderr)
        sys.exit(1)

    if args.csv:
        w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        return

    n = len(rows)
    won = sum(1 for r in rows if r["won"])
    v_correct_b_wrong = sum(1 for r in rows if r["verbal_correct_behavioral_wrong"])
    v_wrong_b_correct = sum(1 for r in rows if r["verbal_wrong_behavioral_correct"])
    both_correct = sum(1 for r in rows if r["verbal_discovery_turn"] is not None
                       and r["behavioral_discovery_aggregate"])
    both_wrong = sum(1 for r in rows if r["verbal_discovery_turn"] is None
                     and not r["behavioral_discovery_aggregate"])

    print(f"\n=== Verbal-vs-Behavioral Discovery Audit ===")
    print(f"Trials analyzed: {n}  ({won} wins)")
    print(f"  Both verbal AND behavioral discovery: {both_correct}")
    print(f"  Verbal yes / Behavioral no  (says it, doesn't do it):   {v_correct_b_wrong}")
    print(f"  Verbal no  / Behavioral yes (does it, doesn't say it):  {v_wrong_b_correct}")
    print(f"  Neither (no discovery either way):    {both_wrong}")

    # Group by env_id + config tag
    print(f"\n=== Per-condition breakdown ===")
    print(f"{'env':<12} {'config':<22} {'model':<26} {'n':>3} {'wins':>5} {'verb':>5} {'behav':>6} {'V>B':>4} {'V<B':>4}")
    print("-" * 110)
    by_key: dict[tuple, list[dict]] = {}
    for r in rows:
        cfg = f"f{r['feedback'][:1]}_g{r['goal'][:1]}_m{r['mechanics'][:1]}_w{r['world'][:1]}"
        k = (r["env_id"], cfg, r["model"])
        by_key.setdefault(k, []).append(r)
    for (env, cfg, model), bucket in sorted(by_key.items()):
        n_b = len(bucket)
        wins_b = sum(1 for r in bucket if r["won"])
        verb_b = sum(1 for r in bucket if r["verbal_discovery_turn"] is not None)
        behav_b = sum(1 for r in bucket if r["behavioral_discovery_aggregate"])
        v_yes_b_no = sum(1 for r in bucket if r["verbal_correct_behavioral_wrong"])
        v_no_b_yes = sum(1 for r in bucket if r["verbal_wrong_behavioral_correct"])
        print(f"{env:<12} {cfg:<22} {model:<26} {n_b:>3} {wins_b:>5} {verb_b:>5} {behav_b:>6} {v_yes_b_no:>4} {v_no_b_yes:>4}")


if __name__ == "__main__":
    main()
