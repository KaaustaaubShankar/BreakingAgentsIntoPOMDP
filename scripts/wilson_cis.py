"""Wilson 95% CIs + Fisher's-exact primary comparisons for the knockout grid.

Reads the canonical results CSV (the one the dashboard serves). Outputs:
  1. A paper-ready Wilson 95% CI table for every cell (model x env x reasoning x config).
  2. Primary comparisons: for each (model, env, config) present at both
     reasoning=none and reasoning=medium, a Fisher's exact two-sided test on the
     2x2 (wins/losses x reasoning) -- the proper small-N test behind claims like
     "medium reasoning collapses mechanics_hard (45%->0%)".

Pure stdlib (no scipy) so it runs anywhere after a git pull.

Usage:
  python scripts/wilson_cis.py                 # canonical CSV
  python scripts/wilson_cis.py path/to.csv     # override
"""
from __future__ import annotations

import csv
import math
import sys
from math import comb
from pathlib import Path

REPO = Path(__file__).parents[1]
# Canonical source: the CSV the breaking-agents dashboard serves (tracked in dashboard/).
DEFAULT_CSV = REPO / "dashboard" / "jkj results - Detailed_Results.csv"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact p for [[a,b],[c,d]] (stdlib hypergeometric)."""
    n = a + b + c + d
    if n == 0:
        return 1.0
    r1, c1 = a + b, a + c
    def pmf(x: int) -> float:
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
    p0 = pmf(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    tot = sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p0 * (1 + 1e-9))
    return min(1.0, tot)


def parse_pct(s: str) -> float | None:
    s = s.strip().rstrip("%").strip()
    if not s or s in {"-", "--"}:
        return None
    try:
        return float(s) / 100.0
    except ValueError:
        return None


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")
    print(f"# Source: {csv_path}\n")

    cells = {}  # (model, env, reasoning, config) -> (k, n)
    for r in csv.DictReader(open(csv_path)):
        n_str, win_str = r.get("N", "").strip(), r.get("Win%", "").strip()
        if not n_str:
            continue
        try:
            n = int(n_str)
        except ValueError:
            continue
        p = parse_pct(win_str)
        if p is None or n == 0:
            continue
        # Normalize ka59simple's "-R"-suffixed reasoning labels so none/medium
        # pair correctly with every other env in the Fisher comparisons.
        reason = r["Reasoning"].strip()
        reason = {"no-R": "none", "medium-R": "medium", "default-R": "default"}.get(reason, reason)
        key = (r["Model"].strip(), r["Game"].strip(), reason, r["Config"].strip())
        cells[key] = (round(p * n), n)

    # 1. Wilson CI table
    print("## Wilson 95% CIs (every cell)\n")
    print(f"{'Model':<16}{'Env':<11}{'Reason':<8}{'Config':<18}{'k/n':>8}{'win%':>6}{'  Wilson 95% CI':>18}")
    print("-" * 86)
    for (model, env, reason, config), (k, n) in sorted(cells.items()):
        lo, hi = wilson(k, n)
        print(f"{model:<16}{env:<11}{reason:<8}{config:<18}{f'{k}/{n}':>8}{k/n*100:>5.0f}%"
              f"   [{lo*100:>4.1f}, {hi*100:>5.1f}]")

    # 2. Primary comparisons: none vs medium (Fisher's exact)
    print("\n\n## Primary comparisons -- reasoning none vs medium (Fisher's exact, 2x2)\n")
    print(f"{'Model':<16}{'Env':<11}{'Config':<18}{'none':>9}{'medium':>9}{'  Fisher p':>11}  sig")
    print("-" * 86)
    for (m, e, c) in sorted({(m, e, c) for (m, e, _, c) in cells}):
        none_, med_ = cells.get((m, e, "none", c)), cells.get((m, e, "medium", c))
        if not (none_ and med_):
            continue
        kn, nn = none_; km, nm = med_
        p = fisher_exact(kn, nn - kn, km, nm - km)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{m:<16}{e:<11}{c:<18}{f'{kn}/{nn}':>9}{f'{km}/{nm}':>9}{p:>11.4f}  {sig}")
    print("\n*** p<.001  ** p<.01  * p<.05  ns=not significant (treat as exploratory)")


if __name__ == "__main__":
    main()
