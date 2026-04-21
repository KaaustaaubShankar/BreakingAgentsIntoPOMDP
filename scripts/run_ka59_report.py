"""
KA59 Benchmark Report
======================
Human-readable comparison table across the three baseline agents and all
canonical KA59 scenarios.  No external dependencies, no dashboards.

Usage:
    python3 scripts/run_ka59_report.py

Outputs four sections:
  1. AXIS COVERAGE  — honest declaration of what KA59 currently probes
  2. SCENARIOS      — name / primary_axis / key tags for each scenario
  3. AGENT RESULTS  — moved / blocked / passable_walls per agent per scenario
  4. EPISTEMIC HIGHLIGHTS — belief-state summary for MinimalHypothesisAgent

Everything is engine-internal-free: no obfuscated tag strings appear.
"""

from __future__ import annotations
import sys, os

# Make ka59_ref importable when run directly from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ka59_ref.scenarios  import SCENARIOS, SCENARIO_META, KA59_AXIS_COVERAGE
from ka59_ref.benchmark  import evaluate_agent, BenchmarkResult
from ka59_ref.discovery  import NaiveRightAgent, RotateOnBlockAgent, MinimalHypothesisAgent


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENTS = [NaiveRightAgent, RotateOnBlockAgent, MinimalHypothesisAgent]
MAX_STEPS = None   # run each episode to its natural end (step-budget exhaustion)

# Scenarios to include in the report (ordered for readability)
REPORT_SCENARIOS = [
    "open_move_right",
    "transfer_wall_direct_block",
    "transfer_wall_push",
    "solid_wall_push_blocked",
    "push_chain",
    "transfer_wall_push_world_blind",
]

# Short display names for agents (keep table columns narrow)
AGENT_SHORT = {
    "NaiveRightAgent":       "NaiveRight",
    "RotateOnBlockAgent":    "RotateOnBlock",
    "MinimalHypothesisAgent":"MinimalHypothesis",
}

# Short display names for scenarios
SCENARIO_SHORT = {
    "transfer_wall_push_world_blind": "tw_push_world_blind",
    "transfer_wall_direct_block":     "tw_direct_block",
    "transfer_wall_push":             "tw_push",
    "solid_wall_push_blocked":        "solid_push_blocked",
    "open_move_right":                "open_move_right",
    "push_chain":                     "push_chain",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hr(char: str = "─", width: int = 72) -> str:
    return char * width


def _section(title: str) -> str:
    return f"\n{_hr('═')}\n  {title}\n{_hr('─')}"


def _cell(value, width: int, align: str = "<") -> str:
    return f"{str(value):{align}{width}}"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _axis_coverage_section() -> str:
    lines = [_section("AXIS COVERAGE")]
    cov_level = {"strong": "strong ✓", "partial": "partial ~", "none": "none  ✗"}
    for axis in ("Mechanics", "World", "Goal", "Feedback"):
        cov = KA59_AXIS_COVERAGE[axis]
        badge = cov_level.get(cov.level, cov.level)
        # Truncate notes to fit one line
        note = cov.notes[:70].rstrip()
        if len(cov.notes) > 70:
            note += "…"
        lines.append(f"  {axis:<10}  {badge:<12}  {note}")
    lines.append("")
    lines.append(
        "  KA59 is a Mechanics benchmark.  World/Goal/Feedback coverage is limited."
    )
    return "\n".join(lines)


def _agents_section() -> str:
    lines = [_section("AGENTS")]
    for cls in AGENTS:
        short = AGENT_SHORT.get(cls.__name__, cls.__name__)
        lines.append(f"  {cls.__name__:<28}  (shown as {short!r} in results table)")
    return "\n".join(lines)


def _scenarios_section() -> str:
    lines = [_section("SCENARIOS")]
    header = (
        f"  {'name':<34}  {'axis':<10}  tags"
    )
    lines.append(header)
    lines.append("  " + _hr("·", 68))
    for name in REPORT_SCENARIOS:
        meta = SCENARIO_META.get(name)
        if meta is None:
            continue
        # Use full canonical name in the scenarios table for traceability
        tag_str = ", ".join(meta.tags[:3])
        if len(meta.tags) > 3:
            tag_str += ", …"
        lines.append(
            f"  {name:<38}  {meta.primary_axis:<10}  {tag_str}"
        )
    return "\n".join(lines)


def _results_section(results: dict[str, BenchmarkResult]) -> str:
    lines = [_section("AGENT RESULTS")]
    header = (
        f"  {'scenario':<22}  {'agent':<18}  "
        f"{'moved':>5}  {'blocked':>7}  {'p_walls':>7}"
    )
    lines.append(header)
    lines.append("  " + _hr("·", 68))

    for name in REPORT_SCENARIOS:
        short = SCENARIO_SHORT.get(name, name)
        for cls in AGENTS:
            agent_name = cls.__name__
            short_agent = AGENT_SHORT.get(agent_name, agent_name)
            r = results[agent_name].by_name(name)
            ep = results[agent_name].epistemic_for(name)
            # Mark MinimalHypothesisAgent passable-wall discoveries with ✓
            pw_str = str(r.passable_walls_found)
            if r.passable_walls_found > 0:
                pw_str += " ✓"
            lines.append(
                f"  {short:<22}  {short_agent:<18}  "
                f"{r.moved_count:>5}  {r.blocked_count:>7}  {pw_str:>7}"
            )
        lines.append("")   # blank row between scenarios

    return "\n".join(lines)


def _discovery_gap_section(results: dict[str, BenchmarkResult]) -> str:
    lines = [_section("DISCOVERY GAP SNAPSHOT")]
    lines.append(
        "  scenario                outcome signal                 discovery signal  read"
    )
    lines.append("  " + _hr("·", 68))

    baseline_names = ["NaiveRightAgent", "RotateOnBlockAgent"]
    hyp = results["MinimalHypothesisAgent"]

    tw_push = hyp.by_name("transfer_wall_push")
    tw_push_baseline_best = max(
        results[name].by_name("transfer_wall_push").moved_count
        for name in baseline_names
    )
    lines.append(
        "  "
        f"{'tw_push':<22}  "
        f"{f'hypothesis moved {tw_push.moved_count} vs baseline {tw_push_baseline_best}':<30}  "
        f"{f'p_walls={tw_push.passable_walls_found}':<16}  "
        "worse outcome, stronger discovery"
    )

    tw_blind = hyp.by_name("transfer_wall_push_world_blind")
    tw_blind_baseline_best = max(
        results[name].by_name("transfer_wall_push_world_blind").moved_count
        for name in baseline_names
    )
    lines.append(
        "  "
        f"{'tw_push_world_blind':<22}  "
        f"{f'hypothesis moved {tw_blind.moved_count} vs baseline {tw_blind_baseline_best}':<30}  "
        f"{f'p_walls={tw_blind.passable_walls_found}':<16}  "
        "discovery disappears under degraded observation"
    )

    lines.append("")
    lines.append(
        "  The key paper-facing point: score alone misses the mechanic discovery story."
    )
    return "\n".join(lines)


def _epistemic_section(results: dict[str, BenchmarkResult]) -> str:
    lines = [_section("EPISTEMIC HIGHLIGHTS  (MinimalHypothesisAgent)")]
    lines.append(
        "  Belief state at episode end.  "
        "Only MinimalHypothesisAgent exposes hypothesis tracking."
    )
    lines.append("")

    hyp_results = results["MinimalHypothesisAgent"]

    focus = ["transfer_wall_push", "solid_wall_push_blocked",
             "transfer_wall_push_world_blind"]
    for name in focus:
        short = SCENARIO_SHORT.get(name, name)
        meta  = SCENARIO_META.get(name)
        ep    = hyp_results.epistemic_for(name)

        lines.append(f"  {short}  [{meta.primary_axis}]")

        if ep.blocked_dirs:
            bd = ", ".join(f"{d}={n}" for d, n in sorted(ep.blocked_dirs.items()))
            lines.append(f"    blocked_dirs:      {bd}")
        else:
            lines.append(f"    blocked_dirs:      (none)")

        if ep.push_success_dirs:
            ps = ", ".join(f"{d}={n}" for d, n in sorted(ep.push_success_dirs.items()))
            lines.append(f"    push_success:      {ps}")
        else:
            lines.append(f"    push_success:      (none)")

        pw = ep.passable_walls_found
        if pw > 0:
            lines.append(f"    passable_walls:    {pw}  ← wall-transfer asymmetry discovered")
        else:
            lines.append(f"    passable_walls:    0   ← not discovered")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main report function
# ---------------------------------------------------------------------------

def generate_report(max_steps: int | None = MAX_STEPS) -> str:
    """
    Run all three agents over all canonical scenarios and return the full
    report as a plain-text string.  Deterministic for deterministic agents.
    """
    spec_subset = {name: SCENARIOS[name] for name in REPORT_SCENARIOS}

    results: dict[str, BenchmarkResult] = {}
    for cls in AGENTS:
        br = evaluate_agent(cls, spec_subset, max_steps=max_steps)
        results[cls.__name__] = br

    parts = [
        f"\n{'═' * 72}",
        f"  KA59 Benchmark Report",
        f"{'═' * 72}",
        _axis_coverage_section(),
        _agents_section(),
        _scenarios_section(),
        _results_section(results),
        _discovery_gap_section(results),
        _epistemic_section(results),
        _hr("═"),
        "",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(generate_report())


if __name__ == "__main__":
    main()
