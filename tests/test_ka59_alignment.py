"""
KA59 Paper-Alignment Tests
============================
Tests for the alignment layer that maps KA59 scenarios to the JKJ paper's
World / Goal / Mechanics / Feedback knockout-matrix framing, and that verifies
the epistemic-trace surface of benchmark results.

Research direction (from PROJECT_ARCHAEOLOGY.md + MONDAY_EXPERIMENT_MATRIX.md):
  - Primary contribution is *epistemic trace*, not win/loss
  - 4 separable axes: World / Goal / Mechanics / Feedback
  - KA59's hidden wall-transfer asymmetry is a Mechanics-axis probe
  - Scenario pairs (transfer_wall_push / solid_wall_push_blocked) form the
    contrastive probe that isolates the Mechanics axis
  - MinimalHypothesisAgent's belief state IS the epistemic trace

Contract being verified:
  1. Every canonical scenario has paper-facing metadata (primary_axis, tags, rationale).
  2. Metadata contains no engine-internal tag strings.
  3. transfer_wall_push maps to Mechanics and carries the hidden-transition tag.
  4. benchmark results expose an epistemic summary per scenario.
  5. MinimalHypothesisAgent produces an informative epistemic summary (has_beliefs=True).
  6. NaiveRightAgent + RotateOnBlockAgent produce reduced/empty summaries.
  7. No internal wall tags appear anywhere in the alignment surface.

Run with:
    python3 -m pytest tests/test_ka59_alignment.py -v
    python3 -m pytest tests/ -v
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ka59_ref.scenarios  import SCENARIOS, SCENARIO_META, ScenarioMeta
from ka59_ref.benchmark  import evaluate_agent, BenchmarkResult, ScenarioResult, EpistemicSummary
from ka59_ref.discovery  import MinimalHypothesisAgent, NaiveRightAgent, RotateOnBlockAgent
from ka59_ref.engine     import TAG_WALL_TRANSFER, TAG_WALL_SOLID

INTERNAL_TAGS = {
    TAG_WALL_TRANSFER, TAG_WALL_SOLID,
    "0022vrxelxosfy", "0001uqqokjrptk", "0003umnkyodpjp",
}
VALID_AXES = {"baseline", "World", "Goal", "Mechanics", "Feedback"}
CANONICAL = set(SCENARIOS.keys())


# ---------------------------------------------------------------------------
# 1. Scenario metadata structure
# ---------------------------------------------------------------------------

class TestScenarioMetadata:
    def test_scenario_meta_exists(self):
        assert isinstance(SCENARIO_META, dict)

    def test_all_canonical_scenarios_have_metadata(self):
        missing = CANONICAL - set(SCENARIO_META)
        assert not missing, f"Scenarios missing metadata: {missing}"

    def test_metadata_instances_are_scenario_meta(self):
        for name, meta in SCENARIO_META.items():
            assert isinstance(meta, ScenarioMeta), (
                f"{name!r}: expected ScenarioMeta, got {type(meta)}"
            )

    def test_each_meta_has_primary_axis(self):
        for name, meta in SCENARIO_META.items():
            assert hasattr(meta, "primary_axis")
            assert meta.primary_axis, f"{name!r}: primary_axis must be non-empty"

    def test_each_meta_primary_axis_is_valid(self):
        for name, meta in SCENARIO_META.items():
            assert meta.primary_axis in VALID_AXES, (
                f"{name!r}: primary_axis {meta.primary_axis!r} not in {VALID_AXES}"
            )

    def test_each_meta_has_tags(self):
        for name, meta in SCENARIO_META.items():
            assert hasattr(meta, "tags")
            assert isinstance(meta.tags, tuple), (
                f"{name!r}: tags must be a tuple for hashability"
            )

    def test_each_meta_has_rationale(self):
        for name, meta in SCENARIO_META.items():
            assert hasattr(meta, "rationale")
            assert isinstance(meta.rationale, str)
            assert len(meta.rationale) > 10, (
                f"{name!r}: rationale is too short to be meaningful"
            )

    def test_no_internal_tags_in_primary_axis(self):
        for name, meta in SCENARIO_META.items():
            assert meta.primary_axis not in INTERNAL_TAGS, (
                f"{name!r}: primary_axis {meta.primary_axis!r} is an internal tag"
            )

    def test_no_internal_tags_in_scenario_tags(self):
        for name, meta in SCENARIO_META.items():
            for tag in meta.tags:
                assert tag not in INTERNAL_TAGS, (
                    f"{name!r}: tag {tag!r} is an internal engine tag"
                )

    def test_no_internal_tags_in_rationale(self):
        for name, meta in SCENARIO_META.items():
            for internal in INTERNAL_TAGS:
                assert internal not in meta.rationale, (
                    f"{name!r}: rationale contains internal tag {internal!r}"
                )


# ---------------------------------------------------------------------------
# 2. Paper-axis classification correctness
# ---------------------------------------------------------------------------

class TestPaperAxisClassification:
    def test_transfer_wall_push_is_mechanics(self):
        """
        transfer_wall_push is the core Mechanics probe: hidden transition
        asymmetry that is only observable through interaction.
        """
        meta = SCENARIO_META["transfer_wall_push"]
        assert meta.primary_axis == "Mechanics", (
            f"transfer_wall_push must be Mechanics axis, got {meta.primary_axis!r}"
        )

    def test_transfer_wall_push_has_hidden_transition_tag(self):
        meta = SCENARIO_META["transfer_wall_push"]
        assert "hidden_transition" in meta.tags, (
            f"transfer_wall_push must carry 'hidden_transition' tag; tags={meta.tags}"
        )

    def test_solid_wall_push_blocked_is_mechanics(self):
        """
        solid_wall_push_blocked is the Mechanics contrast condition:
        visually identical to transfer_wall_push but mechanically different.
        """
        meta = SCENARIO_META["solid_wall_push_blocked"]
        assert meta.primary_axis == "Mechanics"

    def test_open_move_right_is_baseline(self):
        meta = SCENARIO_META["open_move_right"]
        assert meta.primary_axis == "baseline"

    def test_transfer_and_solid_form_contrastive_pair(self):
        """
        Both scenarios must share a common tag marking them as a pair.
        This supports the knockout-matrix design: same visual layout,
        isolated Mechanics axis difference.
        """
        meta_t = SCENARIO_META["transfer_wall_push"]
        meta_s = SCENARIO_META["solid_wall_push_blocked"]
        # They must share at least one common tag (e.g. "wall_probe")
        shared = set(meta_t.tags) & set(meta_s.tags)
        assert shared, (
            f"transfer_wall_push and solid_wall_push_blocked must share at least "
            f"one tag to mark them as a contrastive pair; "
            f"transfer={meta_t.tags}, solid={meta_s.tags}"
        )

    def test_mechanics_scenarios_outnumber_other_axes(self):
        """
        Most KA59 scenarios probe the Mechanics axis (hidden transition rules).
        This reflects KA59's focus on that dimension.
        """
        mechanics_count = sum(
            1 for m in SCENARIO_META.values() if m.primary_axis == "Mechanics"
        )
        assert mechanics_count >= 3, (
            f"Expected ≥3 Mechanics scenarios, got {mechanics_count}"
        )


# ---------------------------------------------------------------------------
# 3. EpistemicSummary type
# ---------------------------------------------------------------------------

class TestEpistemicSummaryType:
    def _run(self, agent_cls, scenario_name="transfer_wall_push", max_steps=5):
        result = evaluate_agent(
            agent_cls,
            {scenario_name: SCENARIOS[scenario_name]},
            max_steps=max_steps,
        )
        return result.epistemic_for(scenario_name)

    def test_returns_epistemic_summary(self):
        ep = self._run(MinimalHypothesisAgent)
        assert isinstance(ep, EpistemicSummary)

    def test_has_has_beliefs_field(self):
        ep = self._run(MinimalHypothesisAgent)
        assert hasattr(ep, "has_beliefs")
        assert isinstance(ep.has_beliefs, bool)

    def test_has_blocked_dirs_field(self):
        ep = self._run(MinimalHypothesisAgent)
        assert hasattr(ep, "blocked_dirs")
        assert isinstance(ep.blocked_dirs, dict)

    def test_has_push_success_dirs_field(self):
        ep = self._run(MinimalHypothesisAgent)
        assert hasattr(ep, "push_success_dirs")
        assert isinstance(ep.push_success_dirs, dict)

    def test_has_passable_walls_found_field(self):
        ep = self._run(MinimalHypothesisAgent)
        assert hasattr(ep, "passable_walls_found")
        assert isinstance(ep.passable_walls_found, int)

    def test_epistemic_for_unknown_scenario_raises(self):
        result = evaluate_agent(MinimalHypothesisAgent, SCENARIOS, max_steps=3)
        with pytest.raises(KeyError):
            result.epistemic_for("does_not_exist")


# ---------------------------------------------------------------------------
# 4. MinimalHypothesisAgent epistemic content
# ---------------------------------------------------------------------------

class TestMinimalHypothesisEpistemic:
    def test_has_beliefs_is_true(self):
        result = evaluate_agent(
            MinimalHypothesisAgent,
            {"transfer_wall_push": SCENARIOS["transfer_wall_push"]},
            max_steps=5,
        )
        ep = result.epistemic_for("transfer_wall_push")
        assert ep.has_beliefs is True

    def test_transfer_wall_push_records_passable_wall(self):
        result = evaluate_agent(
            MinimalHypothesisAgent,
            {"transfer_wall_push": SCENARIOS["transfer_wall_push"]},
            max_steps=5,
        )
        ep = result.epistemic_for("transfer_wall_push")
        assert ep.passable_walls_found > 0, (
            "MinimalHypothesisAgent must discover a passable wall in transfer_wall_push"
        )

    def test_transfer_wall_push_records_push_success(self):
        result = evaluate_agent(
            MinimalHypothesisAgent,
            {"transfer_wall_push": SCENARIOS["transfer_wall_push"]},
            max_steps=5,
        )
        ep = result.epistemic_for("transfer_wall_push")
        total_successes = sum(ep.push_success_dirs.values())
        assert total_successes > 0, (
            "MinimalHypothesisAgent must record at least one push success"
        )

    def test_solid_push_blocked_records_no_passable_wall(self):
        result = evaluate_agent(
            MinimalHypothesisAgent,
            {"solid_wall_push_blocked": SCENARIOS["solid_wall_push_blocked"]},
            max_steps=5,
        )
        ep = result.epistemic_for("solid_wall_push_blocked")
        assert ep.passable_walls_found == 0, (
            "MinimalHypothesisAgent must not discover a passable wall in solid_wall_push_blocked"
        )

    def test_solid_push_records_blocks(self):
        result = evaluate_agent(
            MinimalHypothesisAgent,
            {"solid_wall_push_blocked": SCENARIOS["solid_wall_push_blocked"]},
            max_steps=5,
        )
        ep = result.epistemic_for("solid_wall_push_blocked")
        total_blocks = sum(ep.blocked_dirs.values())
        assert total_blocks > 0

    def test_epistemic_keys_are_directions_not_tags(self):
        """blocked_dirs and push_success_dirs must use direction strings, not internal tags."""
        result = evaluate_agent(
            MinimalHypothesisAgent,
            {"transfer_wall_push": SCENARIOS["transfer_wall_push"]},
            max_steps=5,
        )
        ep = result.epistemic_for("transfer_wall_push")
        valid_dirs = {"right", "left", "up", "down"}
        for k in ep.blocked_dirs:
            assert k in valid_dirs, f"blocked_dirs key {k!r} not a direction"
            assert k not in INTERNAL_TAGS
        for k in ep.push_success_dirs:
            assert k in valid_dirs, f"push_success_dirs key {k!r} not a direction"
            assert k not in INTERNAL_TAGS


# ---------------------------------------------------------------------------
# 5. Baseline agents: reduced/empty epistemic summaries
# ---------------------------------------------------------------------------

class TestBaselineEpistemic:
    def test_naive_right_has_beliefs_false(self):
        result = evaluate_agent(
            NaiveRightAgent,
            {"transfer_wall_push": SCENARIOS["transfer_wall_push"]},
            max_steps=5,
        )
        ep = result.epistemic_for("transfer_wall_push")
        assert ep.has_beliefs is False

    def test_naive_right_empty_blocked_dirs(self):
        result = evaluate_agent(
            NaiveRightAgent,
            {"transfer_wall_push": SCENARIOS["transfer_wall_push"]},
            max_steps=5,
        )
        ep = result.epistemic_for("transfer_wall_push")
        assert ep.blocked_dirs == {}

    def test_naive_right_empty_push_success_dirs(self):
        result = evaluate_agent(
            NaiveRightAgent,
            {"transfer_wall_push": SCENARIOS["transfer_wall_push"]},
            max_steps=5,
        )
        ep = result.epistemic_for("transfer_wall_push")
        assert ep.push_success_dirs == {}

    def test_rotate_has_beliefs_false(self):
        result = evaluate_agent(
            RotateOnBlockAgent,
            {"transfer_wall_push": SCENARIOS["transfer_wall_push"]},
            max_steps=5,
        )
        ep = result.epistemic_for("transfer_wall_push")
        assert ep.has_beliefs is False

    def test_rotate_empty_push_success_dirs(self):
        result = evaluate_agent(
            RotateOnBlockAgent,
            {"transfer_wall_push": SCENARIOS["transfer_wall_push"]},
            max_steps=5,
        )
        ep = result.epistemic_for("transfer_wall_push")
        assert ep.push_success_dirs == {}

    def test_epistemic_depth_ordering(self):
        """
        MinimalHypothesisAgent should have strictly more epistemic content
        than the baselines in transfer_wall_push.
        Operationalised as: its total signal count (blocked + push_success)
        exceeds that of agents without beliefs.
        """
        def total_signal(ep: EpistemicSummary) -> int:
            return sum(ep.blocked_dirs.values()) + sum(ep.push_success_dirs.values())

        spec = {"transfer_wall_push": SCENARIOS["transfer_wall_push"]}

        ep_hyp   = evaluate_agent(MinimalHypothesisAgent, spec, max_steps=5).epistemic_for("transfer_wall_push")
        ep_naive = evaluate_agent(NaiveRightAgent,        spec, max_steps=5).epistemic_for("transfer_wall_push")
        ep_rot   = evaluate_agent(RotateOnBlockAgent,     spec, max_steps=5).epistemic_for("transfer_wall_push")

        assert total_signal(ep_hyp) > total_signal(ep_naive), (
            "MinimalHypothesisAgent should carry more epistemic signal than NaiveRightAgent"
        )
        assert total_signal(ep_hyp) > total_signal(ep_rot), (
            "MinimalHypothesisAgent should carry more epistemic signal than RotateOnBlockAgent"
        )


# ---------------------------------------------------------------------------
# 6. No internal tag leakage in alignment surface
# ---------------------------------------------------------------------------

class TestAlignmentNoTagLeakage:
    def test_primary_axes_contain_no_internal_tags(self):
        for name, meta in SCENARIO_META.items():
            assert meta.primary_axis not in INTERNAL_TAGS

    def test_scenario_tags_contain_no_internal_tags(self):
        for name, meta in SCENARIO_META.items():
            for tag in meta.tags:
                assert tag not in INTERNAL_TAGS, (
                    f"Scenario {name!r} tag {tag!r} is an internal engine constant"
                )

    def test_epistemic_summary_contains_no_internal_tags(self):
        result = evaluate_agent(MinimalHypothesisAgent, SCENARIOS, max_steps=5)
        for name in SCENARIOS:
            ep = result.epistemic_for(name)
            for k in ep.blocked_dirs:
                assert k not in INTERNAL_TAGS
            for k in ep.push_success_dirs:
                assert k not in INTERNAL_TAGS

    def test_benchmark_result_agent_name_no_internal_tags(self):
        result = evaluate_agent(MinimalHypothesisAgent, SCENARIOS, max_steps=5)
        for tag in INTERNAL_TAGS:
            assert tag not in result.agent_name


# ---------------------------------------------------------------------------
# 7. BenchmarkResult.epistemic_for surface
# ---------------------------------------------------------------------------

class TestBenchmarkResultEpistemicFor:
    def test_epistemic_for_all_scenarios(self):
        result = evaluate_agent(MinimalHypothesisAgent, SCENARIOS, max_steps=5)
        for name in SCENARIOS:
            ep = result.epistemic_for(name)
            assert isinstance(ep, EpistemicSummary), f"Missing EpistemicSummary for {name!r}"

    def test_epistemic_for_is_deterministic(self):
        spec = {"transfer_wall_push": SCENARIOS["transfer_wall_push"]}
        r1 = evaluate_agent(MinimalHypothesisAgent, spec, max_steps=5)
        r2 = evaluate_agent(MinimalHypothesisAgent, spec, max_steps=5)
        ep1 = r1.epistemic_for("transfer_wall_push")
        ep2 = r2.epistemic_for("transfer_wall_push")
        assert ep1.has_beliefs           == ep2.has_beliefs
        assert ep1.blocked_dirs          == ep2.blocked_dirs
        assert ep1.push_success_dirs     == ep2.push_success_dirs
        assert ep1.passable_walls_found  == ep2.passable_walls_found
