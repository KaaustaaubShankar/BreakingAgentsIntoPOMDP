"""
Cross-environment verbal-channel audit for BP35 (env4).

Companion to the KA59 verbal-discovery analysis. Establishes that the KA59
wall-transfer verbal-discovery metric is structurally non-transferable to BP35,
and quantifies the fair analogue: rule RE-INFERENCE under mechanics_hard
(where BP35 withholds rules that MECHANICS_EASY otherwise states outright).

BP35 result files are NOT in the working tree; they live on the
`origin/rerunning-bp35` branch. Extract them first:

    git archive origin/rerunning-bp35 env4/results | tar -x -C /tmp/bp35_audit

then:

    python3 -m scripts.audit_bp35_verbal --dir /tmp/bp35_audit/env4/results

Reports, per config:
  - n runs, total action turns, % turns with non-empty `reasoning`
    (exposes that mechanics_hard has no per-turn verbal channel)
  - KA59 wall-transfer regex hit count on BP35 reasoning (expected 0)
  - for mechanics_hard understanding reflections: recovery rate of the
    withheld dynamics (gravity/fall, breakable/click, gravity-flip) and win.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

repo_root = Path(__file__).parents[1]
sys.path.insert(0, str(repo_root))

from ka59_game.prompts import check_discovery

_CFG = re.compile(r"run_(f._g._m._w.)")
_GRAV = re.compile(r"gravit|fall|fell|drop", re.I)
_FLIP = re.compile(r"flip|revers|change.{0,15}(direction|gravit)|switch.{0,15}gravit|up.?ward|upside", re.I)
_BRK = re.compile(r"break|clickable|remove.{0,15}block|destroy", re.I)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="/tmp/bp35_audit/env4/results",
                   help="Directory of extracted BP35 run_*.json files")
    args = p.parse_args()

    files = sorted(glob.glob(f"{args.dir}/run_*.json"))
    if not files:
        print(f"No run_*.json under {args.dir}. Extract from origin/rerunning-bp35 first "
              f"(see module docstring).", file=sys.stderr)
        sys.exit(1)

    stats = defaultdict(lambda: {"n": 0, "act": 0, "reason": 0, "models": set()})
    ka_hits = ka_texts = 0
    mh_und = []
    for fn in files:
        try:
            j = json.loads(Path(fn).read_text())
        except Exception:
            continue
        m = _CFG.search(fn)
        cfg = m.group(1) if m else "?"
        s = stats[cfg]
        s["n"] += 1
        s["models"].add(j.get("model", ""))
        acts = [h for h in j.get("history", []) if h.get("type") == "action"]
        s["act"] += len(acts)
        for h in acts:
            t = (h.get("reasoning") or "").strip()
            if t:
                s["reason"] += 1
                ka_texts += 1
                if check_discovery(t):
                    ka_hits += 1
        if cfg == "fE_gE_mH_wE":
            u = (j.get("understanding") or {}).get("mechanics_understanding", "")
            if u.strip():
                mh_und.append((bool(_GRAV.search(u)), bool(_FLIP.search(u)),
                               bool(_BRK.search(u)), bool(j.get("won"))))

    print(f"{'config':<14}{'n':>4}{'act_turns':>10}{'%reason':>9}  models")
    for cfg in sorted(stats):
        s = stats[cfg]
        pct = 100 * s["reason"] / s["act"] if s["act"] else 0
        print(f"{cfg:<14}{s['n']:>4}{s['act']:>10}{pct:>8.0f}%  {sorted(x for x in s['models'] if x)}")

    print(f"\nKA59 wall-transfer regex on BP35 reasoning: {ka_hits} hits / {ka_texts} non-empty texts "
          f"(expected ~0 — metric is wall-transfer-specific)")

    n = len(mh_und)
    if n:
        print(f"\nBP35 mechanics_hard understanding reflections (n={n}, rules WITHHELD):")
        print(f"  gravity/fall recovered:   {sum(x[0] for x in mh_und)}/{n} ({100*sum(x[0] for x in mh_und)/n:.0f}%)")
        print(f"  breakable/click recovered:{sum(x[2] for x in mh_und)}/{n} ({100*sum(x[2] for x in mh_und)/n:.0f}%)")
        print(f"  gravity-flip recovered:   {sum(x[1] for x in mh_und)}/{n} ({100*sum(x[1] for x in mh_und)/n:.0f}%)")
        print(f"  won:                      {sum(x[3] for x in mh_und)}/{n} ({100*sum(x[3] for x in mh_und)/n:.0f}%)")


if __name__ == "__main__":
    main()
