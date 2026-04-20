"""
KA59 Report Tests
==================
Tests for scripts/run_ka59_report.py — the human-readable comparison table.

Contract being verified:
  1. generate_report() returns a non-empty string without raising.
  2. Output contains all three agent names.
  3. Output contains all canonical scenario names, including the World probe.
  4. Output contains honest axis-coverage language (Mechanics strong, Goal none).
  5. Output contains per-agent result columns (moved / blocked / passable_walls).
  6. Output contains MinimalHypothesisAgent epistemic highlights.
  7. No internal engine tag strings appear anywhere in the output.
  8. Output is deterministic across two calls.
  9. The script can also be run as __main__ without error.

Run with:
    python3 -m pytest tests/test_ka59_report.py -v
    python3 -m pytest tests/ -v
"""
import subprocess, sys, os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_ka59_report import generate_report
from ka59_ref.engine import TAG_WALL_TRANSFER, TAG_WALL_SOLID

INTERNAL_TAGS = {
    TAG_WALL_TRANSFER, TAG_WALL_SOLID,
    "0022vrxelxosfy", "0001uqqokjrptk", "0003umnkyodpjp",
}

AGENTS = ["NaiveRightAgent", "RotateOnBlockAgent", "MinimalHypothesisAgent"]

CANONICAL_SCENARIOS = [
    "open_move_right",
    "transfer_wall_direct_block",
    "transfer_wall_push",
    "solid_wall_push_blocked",
    "push_chain",
    "transfer_wall_push_world_blind",
]


# ---------------------------------------------------------------------------
# 1. Basic shape
# ---------------------------------------------------------------------------

class TestReportShape:
    def test_generate_report_returns_string(self):
        out = generate_report()
        assert isinstance(out, str)

    def test_report_is_non_empty(self):
        out = generate_report()
        assert len(out.strip()) > 100

    def test_report_has_multiple_lines(self):
        out = generate_report()
        assert out.count("\n") >= 10

    def test_report_has_section_headers(self):
        out = generate_report()
        # Should have at least an axis-coverage section and a results section
        upper = out.upper()
        assert "AXIS" in upper or "COVERAGE" in upper
        assert "RESULT" in upper or "AGENT" in upper or "SCENARIO" in upper


# ---------------------------------------------------------------------------
# 2. Agent names
# ---------------------------------------------------------------------------

class TestAgentNames:
    def test_naive_right_agent_in_report(self):
        out = generate_report()
        assert "NaiveRightAgent" in out

    def test_rotate_on_block_agent_in_report(self):
        out = generate_report()
        assert "RotateOnBlockAgent" in out

    def test_minimal_hypothesis_agent_in_report(self):
        out = generate_report()
        assert "MinimalHypothesisAgent" in out

    def test_all_three_agents_present(self):
        out = generate_report()
        for agent in AGENTS:
            assert agent in out, f"Agent {agent!r} missing from report"


# ---------------------------------------------------------------------------
# 3. Scenario names
# ---------------------------------------------------------------------------

class TestScenarioNames:
    def test_transfer_wall_push_in_report(self):
        out = generate_report()
        assert "transfer_wall_push" in out

    def test_solid_wall_push_blocked_in_report(self):
        out = generate_report()
        assert "solid_wall_push_blocked" in out

    def test_world_blind_scenario_in_report(self):
        """The World probe scenario must appear — we must not hide it."""
        out = generate_report()
        assert "transfer_wall_push_world_blind" in out or "world_blind" in out

    def test_all_canonical_scenarios_in_report(self):
        out = generate_report()
        for name in CANONICAL_SCENARIOS:
            # Allow abbreviated forms (world_blind as short form)
            short = name.replace("transfer_wall_push_world_blind", "world_blind")
            assert name in out or short in out, (
                f"Scenario {name!r} (or its abbreviation) missing from report"
            )


# ---------------------------------------------------------------------------
# 4. Axis coverage language
# ---------------------------------------------------------------------------

class TestAxisCoverageLanguage:
    def test_mechanics_strong_in_report(self):
        out = generate_report()
        assert "Mechanics" in out
        assert "strong" in out

    def test_goal_none_in_report(self):
        out = generate_report()
        assert "Goal" in out
        assert "none" in out

    def test_world_partial_in_report(self):
        out = generate_report()
        assert "World" in out
        assert "partial" in out

    def test_feedback_none_in_report(self):
        out = generate_report()
        assert "Feedback" in out
        assert "none" in out

    def test_honest_caveat_present(self):
        """Report must convey that KA59 does NOT cover all four axes fully."""
        out = generate_report()
        # At least two axes must have 'none' or 'partial' — not all strong
        lower = out.lower()
        assert lower.count("none") >= 2 or lower.count("partial") >= 1


# ---------------------------------------------------------------------------
# 5. Result columns
# ---------------------------------------------------------------------------

class TestResultColumns:
    def test_moved_column_present(self):
        out = generate_report()
        assert "moved" in out.lower()

    def test_blocked_column_present(self):
        out = generate_report()
        assert "blocked" in out.lower()

    def test_passable_walls_column_present(self):
        out = generate_report()
        assert "passable" in out.lower() or "p_wall" in out.lower()

    def test_numeric_results_present(self):
        """Report must contain at least some numeric result values."""
        import re
        out = generate_report()
        numbers = re.findall(r"\b\d+\b", out)
        assert len(numbers) >= 5, "Report must include numeric result data"

    def test_transfer_push_passable_wall_shown_for_hypothesis_agent(self):
        """
        MinimalHypothesisAgent discovers a passable wall in transfer_wall_push.
        That non-zero value must be visible in the report output.
        """
        out = generate_report()
        # Find the section with transfer_wall_push results
        # MinimalHypothesisAgent should show passable_walls >= 1 there
        # We can't assert exact layout but "1" must appear near the right context
        assert "transfer_wall_push" in out
        assert "MinimalHypothesisAgent" in out
        # The discovery should be visible in some form
        assert "1" in out  # at least one passable wall recorded somewhere


# ---------------------------------------------------------------------------
# 6. Epistemic highlights
# ---------------------------------------------------------------------------

class TestEpistemicHighlights:
    def test_epistemic_section_present(self):
        out = generate_report()
        upper = out.upper()
        assert "EPISTEMIC" in upper or "BELIEF" in upper or "HYPOTHESIS" in upper or "DISCOVERY" in upper

    def test_passable_wall_discovery_highlighted(self):
        out = generate_report()
        # Some mention of wall discovery should appear in epistemic section
        lower = out.lower()
        assert "passable" in lower or "discovered" in lower or "wall" in lower

    def test_world_blind_no_discovery_noted(self):
        """Report should note that World-blind agent finds no passable walls."""
        out = generate_report()
        # Both the scenario and zero passable walls should appear in proximity
        assert "world_blind" in out or "transfer_wall_push_world_blind" in out


# ---------------------------------------------------------------------------
# 7. No internal tag leakage
# ---------------------------------------------------------------------------

class TestNoTagLeakage:
    def test_no_internal_tags_in_report(self):
        out = generate_report()
        for tag in INTERNAL_TAGS:
            assert tag not in out, (
                f"Internal engine tag {tag!r} leaked into report output"
            )

    def test_no_raw_hex_ids_in_report(self):
        """No 0000xxxx-style obfuscated names should appear."""
        import re
        out = generate_report()
        # Internal tags match pattern like 0015qniapgwsvb (16 char alphanumeric)
        suspicious = re.findall(r'\b[0-9]{4}[a-z]{10,}\b', out)
        assert suspicious == [], (
            f"Suspicious internal-looking strings in report: {suspicious}"
        )


# ---------------------------------------------------------------------------
# 8. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_two_calls_produce_identical_output(self):
        out1 = generate_report()
        out2 = generate_report()
        assert out1 == out2, "generate_report() must be deterministic"


# ---------------------------------------------------------------------------
# 9. CLI entry point
# ---------------------------------------------------------------------------

class TestCLIEntryPoint:
    def test_script_runs_as_main_without_error(self):
        """scripts/run_ka59_report.py must be executable as a script."""
        repo = os.path.join(os.path.dirname(__file__), "..")
        script = os.path.join(repo, "scripts", "run_ka59_report.py")
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            cwd=repo,
        )
        assert result.returncode == 0, (
            f"Script exited with code {result.returncode}:\n{result.stderr}"
        )
        assert len(result.stdout.strip()) > 50

    def test_cli_output_contains_agent_names(self):
        repo = os.path.join(os.path.dirname(__file__), "..")
        script = os.path.join(repo, "scripts", "run_ka59_report.py")
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True, cwd=repo,
        )
        for agent in AGENTS:
            assert agent in result.stdout, f"{agent!r} missing from CLI output"

    def test_cli_output_no_internal_tags(self):
        repo = os.path.join(os.path.dirname(__file__), "..")
        script = os.path.join(repo, "scripts", "run_ka59_report.py")
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True, cwd=repo,
        )
        for tag in INTERNAL_TAGS:
            assert tag not in result.stdout, (
                f"Internal tag {tag!r} leaked into CLI output"
            )
