"""Compute Wilson 95% CIs for every cell in the JKJ Detailed_Results CSV.

Output is paper-ready: model, env, config, k/n, win%, CI low, CI high.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

repo_root = Path(__file__).parents[1]
CSV = repo_root / "jkj results - Detailed_Results(1).csv"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return (lo, hi)


def parse_pct(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def main() -> None:
    rows = list(csv.DictReader(open(CSV)))
    print(f"{'Model':<14} {'Env':<11} {'Owner':<6} {'R':<7} {'Config':<16} {'k/n':>7} {'win%':>6} {'CI low':>7} {'CI hi':>7}")
    print("-" * 90)
    for r in rows:
        model = r["Model"].strip()
        env = r["Game"].strip()
        owner = r["Owner"].strip()
        reasoning = r["Reasoning"].strip()
        config = r["Config"].strip()
        n_str = r["N"].strip()
        win_str = r["Win%"].strip()
        if not n_str or not win_str:
            continue
        try:
            n = int(n_str)
        except ValueError:
            continue
        win_p = parse_pct(win_str)
        if win_p is None or n == 0:
            continue
        k = round(win_p * n)
        lo, hi = wilson(k, n)
        print(f"{model:<14} {env:<11} {owner:<6} {reasoning:<7} {config:<16} {k:>3}/{n:<3} {win_p*100:>5.0f}% [{lo*100:>4.0f}%, {hi*100:>4.0f}%]")


if __name__ == "__main__":
    main()
