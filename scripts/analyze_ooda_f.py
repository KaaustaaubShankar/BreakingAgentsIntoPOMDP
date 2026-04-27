"""
analyze_ooda_f.py — Reframe-efficacy analyzer for OODA-F trial JSONs.

Answers the Tuesday-meeting question: when [FORCED REFRAME] fires, does the
model actually switch action types?

For each trial JSON in results/ka59_game/, this:
  1. Counts forced_reframe events
  2. For each reframe at turn T, locates the action event at the same turn
     (the model's response seen the reframe text in that turn's prompt)
  3. Compares action type at T against action type at T-1
  4. Classifies the reframe as:
       'switched'  — action type changed (CLICK <-> MOVE_*)
       'persisted' — same action type as the turn before the reframe
       'no_action' — action event missing (LLM error / invalid)

Action types collapse to {CLICK, MOVE} since the reframe targets that axis.

Usage:
    python3 -m scripts.analyze_ooda_f                  # all results/ka59_game files
    python3 -m scripts.analyze_ooda_f --since 20260426 # filter by run-id timestamp
    python3 -m scripts.analyze_ooda_f path/to/file.json [more...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).parents[1]
DEFAULT_DIR = REPO / "results" / "ka59_game"


def _action_type(action_name: str) -> str:
    if not action_name:
        return ""
    return "CLICK" if "CLICK" in action_name.upper() else "MOVE"


def analyze_trial(path: Path) -> dict:
    payload = json.loads(path.read_text())
    history = payload.get("history", [])
    config = payload.get("config", {})

    # Index events by turn
    actions_by_turn: dict[int, dict] = {}
    reframes_at_turns: list[dict] = []
    for ev in history:
        et = ev.get("type", "")
        turn = ev.get("turn")
        if et == "action" and turn is not None:
            actions_by_turn[int(turn)] = ev
        elif et == "forced_reframe" and turn is not None:
            reframes_at_turns.append(ev)

    classifications = {"switched": 0, "persisted": 0, "no_action": 0}
    reframe_details: list[dict] = []
    persistence_durations: list[int] = []
    for ev in reframes_at_turns:
        t = int(ev["turn"])
        post = actions_by_turn.get(t)
        prior = actions_by_turn.get(t - 1)
        post_action = post.get("action", "") if post else ""
        prior_action = prior.get("action", "") if prior else ""
        post_type = _action_type(post_action)
        prior_type = _action_type(prior_action)

        if not post_type:
            verdict = "no_action"
        elif prior_type and post_type != prior_type:
            verdict = "switched"
        else:
            verdict = "persisted"
        classifications[verdict] += 1

        # Post-reframe persistence: how many consecutive turns starting at T
        # held the post_type before reverting to a different type?
        persistence = 0
        if post_type:
            persistence = 1
            t2 = t + 1
            while t2 in actions_by_turn:
                next_type = _action_type(actions_by_turn[t2].get("action", ""))
                if next_type == post_type:
                    persistence += 1
                    t2 += 1
                else:
                    break
            persistence_durations.append(persistence)

        # Capture the orient text on the reframe turn (model's hypothesis after seeing reframe)
        post_orient = post.get("orient", "") if post else ""

        reframe_details.append({
            "turn": t,
            "reason": ev.get("summary", ""),
            "prior_action": prior_action,
            "post_action": post_action,
            "post_action_persistence_turns": persistence,
            "verdict": verdict,
            "post_orient": post_orient,
        })

    n_reframes = len(reframes_at_turns)
    switch_rate = (
        classifications["switched"] / n_reframes if n_reframes else 0.0
    )

    # Action histogram across the whole trial
    action_hist: dict[str, int] = {}
    for ev in actions_by_turn.values():
        a = ev.get("action", "?")
        action_hist[a] = action_hist.get(a, 0) + 1

    avg_persist = (
        sum(persistence_durations) / len(persistence_durations)
        if persistence_durations else 0.0
    )
    max_persist = max(persistence_durations) if persistence_durations else 0

    # Extract orient text at discovery_turn (if any) for narrative
    discovery_orient = ""
    d_turn = payload.get("discovery_turn")
    if d_turn is not None:
        d_action = actions_by_turn.get(int(d_turn))
        if d_action:
            discovery_orient = d_action.get("orient", "")

    return {
        "file": path.name,
        "config": config,
        "won": payload.get("won"),
        "turns": payload.get("turns"),
        "discovery_turn": d_turn,
        "discovery_orient": discovery_orient,
        "forced_reframes": payload.get("forced_reframes"),
        "click_actions": payload.get("click_actions"),
        "moves_blocked": payload.get("moves_blocked"),
        "reframe_count_in_history": n_reframes,
        "classifications": classifications,
        "switch_rate": round(switch_rate, 3),
        "avg_post_reframe_persistence_turns": round(avg_persist, 2),
        "max_post_reframe_persistence_turns": max_persist,
        "action_histogram": action_hist,
        "reframe_details": reframe_details,
    }


def render(results: list[dict]) -> str:
    lines: list[str] = []
    lines.append("=" * 96)
    lines.append("OODA-F REFRAME EFFICACY ANALYSIS")
    lines.append("=" * 96)
    lines.append(f"{'File':<46} {'Won':>4} {'Refr':>5} {'Sw%':>5} {'AvgHold':>8} {'MaxHold':>8} {'DTurn':>6}")
    lines.append("-" * 96)
    for r in results:
        lines.append(
            f"{r['file'][:46]:<46} "
            f"{str(r['won'])[:4]:>4} "
            f"{r['reframe_count_in_history']:>5} "
            f"{r['switch_rate']:>5.0%} "
            f"{r['avg_post_reframe_persistence_turns']:>8.2f} "
            f"{r['max_post_reframe_persistence_turns']:>8} "
            f"{str(r['discovery_turn']):>6}"
        )
    lines.append("-" * 96)
    if results:
        total_reframes = sum(r["reframe_count_in_history"] for r in results)
        total_switched = sum(r["classifications"]["switched"] for r in results)
        agg_rate = total_switched / total_reframes if total_reframes else 0.0
        avg_hold = (
            sum(r["avg_post_reframe_persistence_turns"] * r["reframe_count_in_history"] for r in results)
            / max(total_reframes, 1)
        )
        max_hold = max((r["max_post_reframe_persistence_turns"] for r in results), default=0)
        lines.append(
            f"{'AGGREGATE':<46} "
            f"{'':>4} "
            f"{total_reframes:>5} "
            f"{agg_rate:>5.0%} "
            f"{avg_hold:>8.2f} "
            f"{max_hold:>8} "
            f"{'':>6}"
        )
    lines.append("=" * 96)
    lines.append("")
    lines.append("Sw%     = % of reframes that produced a same-turn action-type switch")
    lines.append("AvgHold = avg # of consecutive turns the post-reframe action type was held before reverting")
    lines.append("MaxHold = max # of consecutive turns held across all reframes in trial")
    lines.append("DTurn   = discovery_turn (first turn whose orient text matched a discovery keyword)")
    lines.append("")
    lines.append("Per-trial action histograms:")
    for r in results:
        lines.append(f"  {r['file']}: {r['action_histogram']}")
    lines.append("")
    lines.append("Discovery orient text:")
    for r in results:
        if r.get("discovery_orient"):
            lines.append(f"  {r['file']} t={r['discovery_turn']}:")
            lines.append(f"    {r['discovery_orient'][:200]}")
    lines.append("")
    lines.append("All reframes: prior -> post [hold] :: post-reframe orient (truncated)")
    for r in results:
        lines.append(f"  {r['file']}:")
        for d in r["reframe_details"]:
            orient_snip = (d.get("post_orient") or "")[:120].replace("\n", " ")
            lines.append(
                f"    t={d['turn']:>3} {d['prior_action'] or '(none)':<14} -> "
                f"{d['post_action'] or '(none)':<14} hold={d['post_action_persistence_turns']:>2} :: {orient_snip}"
            )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*", help="Specific trial JSON paths; default = scan results/ka59_game/")
    p.add_argument("--since", default=None, help="Substring filter on filename (e.g. 20260427)")
    p.add_argument("--config", default=None, help="Filter to trials whose config.mechanics matches (OODA_F, OODA, EASY, HARD)")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of formatted table")
    args = p.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        paths = sorted(DEFAULT_DIR.glob("run_*.json"))

    if args.since:
        paths = [p for p in paths if args.since in p.name]

    results: list[dict] = []
    for path in paths:
        if not path.exists():
            print(f"  [skip] missing: {path}", file=sys.stderr)
            continue
        try:
            r = analyze_trial(path)
        except Exception as exc:
            print(f"  [error] {path}: {exc}", file=sys.stderr)
            continue
        if args.config and r["config"].get("mechanics", "").upper() != args.config.upper():
            continue
        results.append(r)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(render(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
