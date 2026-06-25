"""
LLM-judge classification of verbal discovery in KA59/KA59-Simple trials.

The regex in ka59_game.prompts.check_discovery is a cheap candidate filter: it
fires on speculative probing ("test if the wall is passable") as readily as on
confirmed rule statements. This script applies a three-level judgment to every
candidate so the paper can report hypothesis rate and confirmation rate
separately:

  none        - text does not actually claim pieces can pass through walls
  hypothesis  - speculates about / proposes testing wall pass-through, unsure
  confirmed   - asserts pass-through (or the push-vs-move asymmetry) as an
                observed or known rule

Items judged per trial:
  - every action turn whose reasoning/orient text matches check_discovery()
  - the end-of-run `understanding` reflection (ALWAYS judged, no prefilter:
    rule consolidation is often paraphrased and would slip past the regex)

Judgments are cached in results/verbal_judge/judgments.jsonl keyed by a hash
of the text, so re-runs only pay for new items.

Usage:
    python3 -m scripts.judge_verbal_discovery --limit 20          # sample run
    python3 -m scripts.judge_verbal_discovery                     # full run
    python3 -m scripts.judge_verbal_discovery --table-only        # summarize cache
    python3 -m scripts.judge_verbal_discovery --provider anthropic \
        --model claude-haiku-4-5-20251001
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

repo_root = Path(__file__).parents[1]
sys.path.insert(0, str(repo_root))

from ka59_game.llm_client import LLMClient
from ka59_game.prompts import check_discovery

CACHE_PATH = repo_root / "results" / "verbal_judge" / "judgments.jsonl"


def set_cache_path(path: Path) -> None:
    global CACHE_PATH
    CACHE_PATH = path

JUDGE_SYSTEM = """\
You are labeling text written by an AI agent while it played a puzzle game.

Hidden game rule (the agent was never told this): a piece moved directly by
the agent is BLOCKED by walls, but a piece PUSHED by another piece passes
THROUGH walls (a push/move asymmetry, sometimes seen as a teleport-like jump).

Classify whether the given text claims this rule. Labels:
- "none": does not actually claim pieces can pass through walls. Includes:
  mentions of gaps/openings in walls, ordinary pushing, CLICK/selection
  switching, or text where wall pass-through is only mentioned to rule it out.
- "hypothesis": speculates that walls might be passable or proposes an action
  to TEST pass-through, without asserting it happened or is true.
- "confirmed": asserts pass-through as something observed or known ("the block
  passed through the wall", "pushed pieces ignore walls", "walls are passable
  when pushing"), or states the push-vs-move asymmetry as a rule.

Respond with a JSON object only:
{"label": "none" | "hypothesis" | "confirmed", "evidence": "<shortest quote that justifies the label>"}\
"""


def _hash(channel: str, text: str) -> str:
    return hashlib.sha1(f"{channel}\n{text}".encode()).hexdigest()


def collect_items(dirs: list[Path]) -> list[dict]:
    items = []
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("run_*.json")):
            try:
                j = json.loads(f.read_text())
            except Exception:
                continue
            config = j.get("config", {})
            base = {
                "file": f.name,
                "env_id": j.get("env_id", d.name.replace("_game", "")),
                "model": j.get("model", ""),
                "config": "f{}_g{}_m{}_w{}".format(*(
                    (config.get(k) or "?")[:1] for k in ("feedback", "goal", "mechanics", "world"))),
                "won": j.get("won"),
                "wall_transfers": j.get("wall_transfers", 0),
            }
            for h in j.get("history", []) or []:
                if h.get("type") != "action":
                    continue
                text = " ".join(t for t in (h.get("reasoning"), h.get("orient")) if t)
                if text and check_discovery(text):
                    items.append({**base, "channel": "turn", "turn": h.get("turn"),
                                  "text": text, "key": _hash("turn", text)})
            und = j.get("understanding") or {}
            und_text = " ".join(str(v) for v in und.values()).strip()
            if und_text:
                items.append({**base, "channel": "understanding", "turn": None,
                              "text": und_text, "key": _hash("understanding", und_text)})
    return items


def load_cache() -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if CACHE_PATH.is_file():
        for line in CACHE_PATH.read_text().splitlines():
            try:
                rec = json.loads(line)
                cache[rec["key"]] = rec
            except Exception:
                continue
    return cache


def judge(items: list[dict], client: LLMClient, cache: dict[str, dict],
          limit: int | None) -> dict[str, dict]:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pending = [it for it in items if it["key"] not in cache]
    # Dedup identical texts within the pending batch.
    seen: set[str] = set()
    pending = [it for it in pending if not (it["key"] in seen or seen.add(it["key"]))]
    if limit is not None:
        pending = pending[:limit]
    print(f"items: {len(items)} total, {len(pending)} to judge (rest cached)", file=sys.stderr)
    with CACHE_PATH.open("a") as fh:
        for i, it in enumerate(pending, 1):
            user_prompt = (
                "Agent text to classify:\n```\n" + it["text"] + "\n```\n"
                "Respond with the JSON object only."
            )
            try:
                reply = client.generate(JUDGE_SYSTEM, user_prompt)
                parsed = client.parse_json(reply)
                label = str(parsed.get("label", "")).lower().strip()
                if label not in ("none", "hypothesis", "confirmed"):
                    raise ValueError(f"bad label {label!r}")
                rec = {"key": it["key"], "label": label,
                       "evidence": str(parsed.get("evidence", ""))[:300],
                       "judge_model": client.model}
            except Exception as exc:
                print(f"  [err] {it['file']} ({it['channel']}): {exc}", file=sys.stderr)
                continue
            cache[rec["key"]] = rec
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"  judged {i}/{len(pending)}", file=sys.stderr)
    return cache


def summarize(items: list[dict], cache: dict[str, dict]) -> None:
    # Per-trial rollup: best label across the trial's judged items
    # (confirmed > hypothesis > none), split by channel.
    rank = {"none": 0, "hypothesis": 1, "confirmed": 2}
    trials: dict[str, dict] = {}
    judged = unjudged = 0
    for it in items:
        rec = cache.get(it["key"])
        if rec is None:
            unjudged += 1
            continue
        judged += 1
        t = trials.setdefault(it["file"], {
            "env_id": it["env_id"], "model": it["model"], "config": it["config"],
            "won": it["won"], "wall_transfers": it["wall_transfers"],
            "turn_label": "none", "und_label": "none"})
        slot = "turn_label" if it["channel"] == "turn" else "und_label"
        if rank[rec["label"]] > rank[t[slot]]:
            t[slot] = rec["label"]
    print(f"\njudged items: {judged}  | not yet judged: {unjudged}")
    print(f"trials with >=1 judged item: {len(trials)}\n")

    by_cell: dict[tuple, list[dict]] = {}
    for t in trials.values():
        by_cell.setdefault((t["env_id"], t["config"], t["model"]), []).append(t)
    hdr = (f"{'env':<12} {'config':<14} {'model':<28} {'n':>3} {'wins':>4} "
           f"{'hyp':>4} {'conf':>4} {'undC':>4} {'conf&win':>8}")
    print(hdr)
    print("-" * len(hdr))
    for (env, cfg, model), bucket in sorted(by_cell.items()):
        n = len(bucket)
        wins = sum(1 for t in bucket if t["won"])
        hyp = sum(1 for t in bucket if rank[t["turn_label"]] >= 1 or rank[t["und_label"]] >= 1)
        conf = sum(1 for t in bucket if t["turn_label"] == "confirmed" or t["und_label"] == "confirmed")
        und_c = sum(1 for t in bucket if t["und_label"] == "confirmed")
        conf_win = sum(1 for t in bucket
                       if (t["turn_label"] == "confirmed" or t["und_label"] == "confirmed") and t["won"])
        print(f"{env:<12} {cfg:<14} {model:<28} {n:>3} {wins:>4} {hyp:>4} {conf:>4} {und_c:>4} {conf_win:>8}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", action="append", default=None,
                   help="Trial-JSON directory (repeatable). Default: results/ka59_game + results/ka59simple_game")
    p.add_argument("--provider", default="anthropic")
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--limit", type=int, default=None, help="Judge at most N new items (sample run)")
    p.add_argument("--table-only", action="store_true", help="Summarize existing cache; no API calls")
    p.add_argument("--cache-file", default=None,
                   help="Alternate judgments cache (e.g. for a second judge model)")
    args = p.parse_args()
    if args.cache_file:
        set_cache_path(repo_root / args.cache_file)

    dirs = [repo_root / d for d in args.dir] if args.dir else [
        repo_root / "results" / "ka59_game",
        repo_root / "results" / "ka59simple_game",
    ]
    items = collect_items(dirs)
    cache = load_cache()
    if not args.table_only:
        client = LLMClient(provider=args.provider, model=args.model)
        cache = judge(items, client, cache, args.limit)
    summarize(items, cache)


if __name__ == "__main__":
    main()
