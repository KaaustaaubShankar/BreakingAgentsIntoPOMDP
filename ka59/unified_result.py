"""
unified_result.py — Shared result schema and Josh table for LS20 / KA59 / BP35.

Column mapping per environment:
  Win%        — episode win rate (all envs)
  Turns       — avg turns / steps used (all envs)
  Levels      — avg levels completed (all envs)
  InvalidClicks — wasted / failed actions:
                  BP35: invalid_actions (unknown action + bad click coords)
                  LS20: wall_collisions
                  KA59: blocked_count (moves that didn't advance the agent)
  Flips       — secondary env metric:
                  BP35: gravity_flips (gravity direction reversals)
                  LS20: goals_ever_activated
                  KA59: passable_walls_found (wall-transfer discoveries)
  RelDiff     — avg_turns / baseline_avg_turns (None if no baseline in set)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Unified per-run result
# ---------------------------------------------------------------------------

@dataclass
class UnifiedRunResult:
    """
    Normalised result for one episode across any supported environment.
    All env-specific metrics are mapped to the shared Josh table columns.
    """
    env: str                      # "bp35" | "ls20" | "ka59"
    config_name: str
    config: dict[str, str]
    provider: str
    model: str
    won: bool
    turns: int
    levels_completed: int
    invalid_clicks: int           # see module docstring for per-env mapping
    flips: int                    # see module docstring for per-env mapping
    run_file: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    raw: Optional[Any] = None     # original env RunResult, kept for traces

    # Optional: hypo-ruling trace for KA59 epistemic signal
    hypo_trace: Optional[dict] = None


# ---------------------------------------------------------------------------
# Aggregated ablation summary (one row per config)
# ---------------------------------------------------------------------------

@dataclass
class AblationRow:
    env: str
    config_name: str
    config: dict[str, str]
    provider: str
    model: str
    n_trials: int
    wins: int
    win_rate: float
    avg_turns: float
    avg_levels: float
    avg_invalid_clicks: float
    avg_flips: float
    relative_difficulty: Optional[float]
    run_files: list[str] = field(default_factory=list)

    @classmethod
    def from_runs(
        cls,
        runs: list[UnifiedRunResult],
        *,
        baseline_avg_turns: Optional[float] = None,
    ) -> "AblationRow":
        if not runs:
            raise ValueError("runs must be non-empty")
        r0 = runs[0]
        n = len(runs)
        wins = sum(1 for r in runs if r.won)
        avg_turns = sum(r.turns for r in runs) / n
        rel: Optional[float] = None
        if baseline_avg_turns and baseline_avg_turns > 0 and avg_turns > 0:
            rel = round(avg_turns / baseline_avg_turns, 3)
        return cls(
            env=r0.env,
            config_name=r0.config_name,
            config=r0.config,
            provider=r0.provider,
            model=r0.model,
            n_trials=n,
            wins=wins,
            win_rate=wins / n,
            avg_turns=avg_turns,
            avg_levels=sum(r.levels_completed for r in runs) / n,
            avg_invalid_clicks=sum(r.invalid_clicks for r in runs) / n,
            avg_flips=sum(r.flips for r in runs) / n,
            relative_difficulty=rel,
            run_files=[r.run_file for r in runs if r.run_file],
        )


# ---------------------------------------------------------------------------
# Josh table renderer
# ---------------------------------------------------------------------------

# Column-width constants keep the table readable at 120 chars
_W_ENV    = 6
_W_CFG    = 20
_W_MODEL  = 32
_W_WIN    = 5
_W_TURNS  = 9
_W_LEVELS = 10
_W_INV    = 7
_W_FLIPS  = 7
_W_REL    = 9

_SEP_WIDTHS = [_W_ENV, _W_CFG, _W_MODEL, _W_WIN, _W_TURNS, _W_LEVELS, _W_INV, _W_FLIPS, _W_REL]
_TOTAL_W    = sum(_SEP_WIDTHS) + 2 * len(_SEP_WIDTHS)  # padding


def _hr(char: str = "─") -> str:
    return char * _TOTAL_W


def _header() -> str:
    cols = [
        f"{'Env':<{_W_ENV}}",
        f"{'Config':<{_W_CFG}}",
        f"{'Model':<{_W_MODEL}}",
        f"{'Win%':>{_W_WIN}}",
        f"{'AvgTurns':>{_W_TURNS}}",
        f"{'AvgLevels':>{_W_LEVELS}}",
        f"{'InvClk':>{_W_INV}}",
        f"{'Flips':>{_W_FLIPS}}",
        f"{'RelDiff':>{_W_REL}}",
    ]
    return "  ".join(cols)


def _row(r: AblationRow) -> str:
    rel = f"{r.relative_difficulty:.2f}x" if r.relative_difficulty is not None else "n/a"
    model_short = r.model.split("/")[-1][:_W_MODEL]
    cols = [
        f"{r.env:<{_W_ENV}}",
        f"{r.config_name:<{_W_CFG}}",
        f"{model_short:<{_W_MODEL}}",
        f"{r.win_rate:>{_W_WIN}.0%}",
        f"{r.avg_turns:>{_W_TURNS}.1f}",
        f"{r.avg_levels:>{_W_LEVELS}.1f}",
        f"{r.avg_invalid_clicks:>{_W_INV}.1f}",
        f"{r.avg_flips:>{_W_FLIPS}.1f}",
        f"{rel:>{_W_REL}}",
    ]
    return "  ".join(cols)


def print_josh_table(rows: list[AblationRow], title: str = "Multi-Env Ablation") -> None:
    """Print the unified Josh table to stdout."""
    print(f"\n{'═' * _TOTAL_W}")
    print(f"  {title}")
    print(f"{'─' * _TOTAL_W}")
    print(_header())
    print(_hr())

    prev_env = None
    for r in rows:
        if prev_env is not None and r.env != prev_env:
            print(_hr("·"))
        print(_row(r))
        prev_env = r.env

    print(f"{'═' * _TOTAL_W}")

    # Legend
    print("\n  Column legend:")
    print("    InvClk  — BP35: invalid_actions | LS20: wall_collisions | KA59: blocked_count")
    print("    Flips   — BP35: gravity_flips   | LS20: goals_activated  | KA59: passable_walls")
    print("    RelDiff — avg_turns / baseline_avg_turns within each (env, model) group\n")


def build_rows_with_rel_diff(
    all_rows: list[AblationRow],
) -> list[AblationRow]:
    """
    Compute relative_difficulty within each (env, model) group, anchored on
    the 'baseline' config. Returns a new list with RelDiff populated.
    """
    from dataclasses import replace

    # Build baseline lookup: (env, model) → avg_turns
    baseline: dict[tuple[str, str], float] = {}
    for r in all_rows:
        if r.config_name == "baseline":
            baseline[(r.env, r.model)] = r.avg_turns

    result: list[AblationRow] = []
    for r in all_rows:
        key = (r.env, r.model)
        bt = baseline.get(key)
        if bt and bt > 0 and r.avg_turns > 0:
            rel = round(r.avg_turns / bt, 3)
        else:
            rel = None
        result.append(replace(r, relative_difficulty=rel))
    return result
