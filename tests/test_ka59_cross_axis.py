"""
KA59 Cross-Axis Tests
=======================
Tests for the axis-coverage declaration and the World-probe extension.

Honest framing (see PROJECT_ARCHAEOLOGY.md / MONDAY_EXPERIMENT_MATRIX.md):
  KA59 is a strong Mechanics probe.
  World, Goal, and Feedback are NOT fully covered by this benchmark version.
  This module verifies we SAY THAT EXPLICITLY and do not oversell coverage.

Additionally verifies ONE minimal honest World-axis probe:
  transfer_wall_push_world_blind — same mechanics as transfer_wall_push but
  non-selected object positions are hidden from the agent (observe_positions=False).
  This is testable through the blind env surface and genuinely degrades World
  observability: the agent can no longer detect block displacement, so it cannot
  discover passable walls even though the same push mechanic is present.

World probe behavioral implication:
  transfer_wall_push     (full obs):  passable_walls_found > 0
  transfer_wall_push_world_blind (degraded obs): passable_walls_found = 0
  — Same mechanics, different observability, different epistemic outcome.

Run with:
    python3 -m pytest tests/test_ka59_cross_axis.py -v
    python3 -m pytest tests/ -v
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ka59_ref.scenarios  import SCENARIOS, SCENARIO_META, ScenarioMeta, KA59_AXIS_COVERAGE, AxisCoverage
from ka59_ref.benchmark  import evaluate_agent
from ka59_ref.discovery  import MinimalHypothesisAgent, NaiveRightAgent
from ka59_ref.env        import KA59BlindEnv, MOVE_RIGHT, MOVE_DOWN
from ka59_ref.engine     import TAG_WALL_TRANSFER, TAG_WALL_SOLID, STEP

INTERNAL_TAGS = {TAG_WALL_TRANSFER, TAG_WALL_SOLID,
                 "0022vrxelxosfy", "0001uqqokjrptk", "0003umnkyodpjp"}


# ---------------------------------------------------------------------------
# 1. Axis-coverage declaration
# ---------------------------------------------------------------------------

class TestAxisCoverageDeclaration:
    def test_axis_coverage_dict_exists(self):
        assert isinstance(KA59_AXIS_COVERAGE, dict)

    def test_all_four_axes_declared(self):
        required = {"World", "Goal", "Mechanics", "Feedback"}
        missing  = required - set(KA59_AXIS_COVERAGE)
        assert not missing, f"Missing axis declarations: {missing}"

    def test_each_entry_is_axis_coverage(self):
        for axis, cov in KA59_AXIS_COVERAGE.items():
            assert isinstance(cov, AxisCoverage), (
                f"KA59_AXIS_COVERAGE[{axis!r}] must be AxisCoverage, got {type(cov)}"
            )

    def test_each_coverage_has_level(self):
        valid_levels = {"strong", "partial", "none"}
        for axis, cov in KA59_AXIS_COVERAGE.items():
            assert hasattr(cov, "level")
            assert cov.level in valid_levels, (
                f"{axis}: coverage level {cov.level!r} not in {valid_levels}"
            )

    def test_each_coverage_has_notes(self):
        for axis, cov in KA59_AXIS_COVERAGE.items():
            assert hasattr(cov, "notes")
            assert len(cov.notes) > 20, f"{axis}: notes too short to be useful"

    def test_each_coverage_has_is_implemented(self):
        for axis, cov in KA59_AXIS_COVERAGE.items():
            assert hasattr(cov, "is_implemented")
            assert isinstance(cov.is_implemented, bool)

    def test_mechanics_is_strong(self):
        cov = KA59_AXIS_COVERAGE["Mechanics"]
        assert cov.level == "strong", (
            f"Mechanics must be 'strong', got {cov.level!r}"
        )
        assert cov.is_implemented is True

    def test_goal_is_not_strong(self):
        """
        KA59 has no win condition. Goal axis is not varied.
        This must be honestly declared — we must NOT claim strong Goal coverage.
        """
        cov = KA59_AXIS_COVERAGE["Goal"]
        assert cov.level != "strong", (
            "Goal axis must NOT be declared 'strong' — KA59 has no varied win condition"
        )

    def test_feedback_is_not_strong(self):
        """
        KA59 provides fixed binary feedback. Feedback richness is not varied.
        Must not be declared 'strong'.
        """
        cov = KA59_AXIS_COVERAGE["Feedback"]
        assert cov.level != "strong", (
            "Feedback axis must NOT be declared 'strong' — not independently varied"
        )

    def test_world_has_partial_or_none_coverage_unless_implemented(self):
        """
        World axis: we have one partial probe (position-blind variant).
        Must be declared 'partial' (not 'strong') given the limited coverage.
        """
        cov = KA59_AXIS_COVERAGE["World"]
        assert cov.level in ("partial", "none"), (
            f"World axis must be 'partial' or 'none', got {cov.level!r}"
        )

    def test_no_internal_tags_in_coverage_notes(self):
        for axis, cov in KA59_AXIS_COVERAGE.items():
            for tag in INTERNAL_TAGS:
                assert tag not in cov.notes, (
                    f"Axis {axis!r} notes contain internal tag {tag!r}"
                )

    def test_no_internal_tags_in_coverage_axes(self):
        for axis in KA59_AXIS_COVERAGE:
            assert axis not in INTERNAL_TAGS


# ---------------------------------------------------------------------------
# 2. World probe scenario
# ---------------------------------------------------------------------------

class TestWorldProbeScenario:
    def test_world_blind_scenario_exists(self):
        assert "transfer_wall_push_world_blind" in SCENARIOS

    def test_world_blind_has_observe_positions_false(self):
        spec = SCENARIOS["transfer_wall_push_world_blind"]
        assert spec.get("observe_positions") is False, (
            "World-blind scenario must set observe_positions=False"
        )

    def test_world_blind_has_same_objects_as_transfer_push(self):
        """Same object layout as transfer_wall_push — only observability differs."""
        blind  = SCENARIOS["transfer_wall_push_world_blind"]["objects"]
        normal = SCENARIOS["transfer_wall_push"]["objects"]
        blind_ids  = {o["id"]: o["kind"] for o in blind}
        normal_ids = {o["id"]: o["kind"] for o in normal}
        assert blind_ids == normal_ids, (
            "World-blind scenario must have same object ids/kinds as transfer_wall_push"
        )

    def test_world_blind_metadata_is_world_axis(self):
        meta = SCENARIO_META["transfer_wall_push_world_blind"]
        assert meta.primary_axis == "World", (
            f"World-blind scenario must be primary_axis='World', got {meta.primary_axis!r}"
        )

    def test_world_blind_metadata_has_world_probe_tag(self):
        meta = SCENARIO_META["transfer_wall_push_world_blind"]
        assert "world_probe" in meta.tags

    def test_world_blind_metadata_no_internal_tags(self):
        meta = SCENARIO_META["transfer_wall_push_world_blind"]
        for tag in meta.tags:
            assert tag not in INTERNAL_TAGS
        assert meta.primary_axis not in INTERNAL_TAGS


# ---------------------------------------------------------------------------
# 3. World probe env behaviour
# ---------------------------------------------------------------------------

class TestWorldProbeEnvBehaviour:
    def _reset_blind(self):
        env = KA59BlindEnv()
        spec = SCENARIOS["transfer_wall_push_world_blind"]
        return env, env.reset(spec)

    def test_selected_piece_has_real_position_in_blind_mode(self):
        """Even in world-blind mode, the selected piece must have real position."""
        env, obs = self._reset_blind()
        sel = next(v for v in obs if v.is_selected)
        assert sel.x == 0 and sel.y == 0, (
            "Selected piece should be at its true starting position"
        )

    def test_non_selected_objects_have_zeroed_positions_in_blind_mode(self):
        """
        Non-selected objects must have x=y=w=h=0 when observe_positions=False.
        This is the World-probe degradation: agent cannot locate walls or blocks.
        """
        env, obs = self._reset_blind()
        non_sel = [v for v in obs if not v.is_selected]
        assert non_sel, "Should have non-selected objects in world-blind scenario"
        for v in non_sel:
            assert v.x == 0 and v.y == 0, (
                f"Non-selected object {v.id!r} must have x=y=0 in world-blind mode, "
                f"got ({v.x}, {v.y})"
            )

    def test_full_obs_scenario_shows_real_positions(self):
        """Control: standard transfer_wall_push shows real positions."""
        env = KA59BlindEnv()
        obs = env.reset(SCENARIOS["transfer_wall_push"])
        pb  = next(v for v in obs if v.id == "pb")
        assert pb.x == STEP, f"Block should be at x={STEP}, got {pb.x}"

    def test_obs_still_includes_all_object_ids_in_blind_mode(self):
        """Agent can still see that objects exist (by id/kind); just not where."""
        env, obs = self._reset_blind()
        ids_blind = {v.id for v in obs}
        env2 = KA59BlindEnv()
        obs2 = env2.reset(SCENARIOS["transfer_wall_push"])
        ids_full = {v.id for v in obs2}
        assert ids_blind == ids_full, (
            "World-blind mode must not hide object existence, only positions"
        )

    def test_blind_mode_kinds_are_still_visible(self):
        """Agent can still see what kind each object is."""
        env, obs = self._reset_blind()
        kinds = {v.kind for v in obs}
        assert "wall" in kinds and "block" in kinds and "controllable" in kinds

    def test_blind_obs_no_internal_tags(self):
        env, obs = self._reset_blind()
        for v in obs:
            assert v.kind not in INTERNAL_TAGS


# ---------------------------------------------------------------------------
# 4. World probe epistemic impact
# ---------------------------------------------------------------------------

class TestWorldProbeEpistemicImpact:
    """
    The core behavioral test for the World probe:
    degraded observability prevents the agent from discovering passable walls,
    even though the same push mechanic is present.
    """

    def test_full_obs_agent_discovers_passable_wall(self):
        result = evaluate_agent(
            MinimalHypothesisAgent,
            {"transfer_wall_push": SCENARIOS["transfer_wall_push"]},
            max_steps=5,
        )
        ep = result.epistemic_for("transfer_wall_push")
        assert ep.passable_walls_found > 0, (
            "With full observation, agent must discover the passable wall"
        )

    def test_world_blind_agent_cannot_discover_passable_wall(self):
        """
        With observe_positions=False, agent cannot detect block displacement,
        so passable_walls_found must remain 0 — even though the push succeeded.
        This is the World-probe behavioral signature.
        """
        result = evaluate_agent(
            MinimalHypothesisAgent,
            {"transfer_wall_push_world_blind": SCENARIOS["transfer_wall_push_world_blind"]},
            max_steps=5,
        )
        ep = result.epistemic_for("transfer_wall_push_world_blind")
        assert ep.passable_walls_found == 0, (
            "With degraded world observation, agent must NOT discover passable wall"
        )

    def test_full_vs_blind_passable_wall_difference(self):
        """
        Combined: same agent, same mechanics, different World observability
        → different epistemic outcome. This is the paper-useful comparison.
        """
        specs = {
            "transfer_wall_push":            SCENARIOS["transfer_wall_push"],
            "transfer_wall_push_world_blind": SCENARIOS["transfer_wall_push_world_blind"],
        }
        result = evaluate_agent(MinimalHypothesisAgent, specs, max_steps=5)

        ep_full  = result.epistemic_for("transfer_wall_push")
        ep_blind = result.epistemic_for("transfer_wall_push_world_blind")

        assert ep_full.passable_walls_found > ep_blind.passable_walls_found, (
            f"Full obs ({ep_full.passable_walls_found}) must exceed blind obs "
            f"({ep_blind.passable_walls_found}) for passable wall discovery"
        )

    def test_world_probe_results_are_deterministic(self):
        spec = {"transfer_wall_push_world_blind": SCENARIOS["transfer_wall_push_world_blind"]}
        r1 = evaluate_agent(MinimalHypothesisAgent, spec, max_steps=5)
        r2 = evaluate_agent(MinimalHypothesisAgent, spec, max_steps=5)
        ep1 = r1.epistemic_for("transfer_wall_push_world_blind")
        ep2 = r2.epistemic_for("transfer_wall_push_world_blind")
        assert ep1.passable_walls_found == ep2.passable_walls_found


# ---------------------------------------------------------------------------
# 5. Honesty guards
# ---------------------------------------------------------------------------

class TestHonestyGuards:
    """
    Verify we are not overselling KA59's axis coverage to the paper.
    """

    def test_goal_declared_not_implemented_or_none(self):
        """
        KA59 has no varied win condition.
        Goal axis must be declared 'none' or 'partial' with is_implemented=False.
        """
        cov = KA59_AXIS_COVERAGE["Goal"]
        assert cov.level in ("none", "partial")
        if cov.level == "none":
            assert cov.is_implemented is False

    def test_feedback_declared_not_implemented_or_none(self):
        """KA59 does not vary feedback richness. Must not claim it does."""
        cov = KA59_AXIS_COVERAGE["Feedback"]
        assert cov.level in ("none", "partial")

    def test_mechanics_is_the_dominant_axis(self):
        """
        Most scenarios should be primary_axis='Mechanics'.
        KA59's identity as a Mechanics benchmark must be clear.
        """
        mechanics = sum(
            1 for m in SCENARIO_META.values() if m.primary_axis == "Mechanics"
        )
        world = sum(1 for m in SCENARIO_META.values() if m.primary_axis == "World")
        goal  = sum(1 for m in SCENARIO_META.values() if m.primary_axis == "Goal")
        feedback = sum(1 for m in SCENARIO_META.values() if m.primary_axis == "Feedback")

        assert mechanics > world + goal + feedback, (
            f"Mechanics ({mechanics}) must dominate other axes "
            f"(world={world}, goal={goal}, feedback={feedback})"
        )

    def test_at_least_one_world_scenario_exists(self):
        """
        We do have ONE World probe. This must be acknowledged.
        """
        world_scenarios = [
            name for name, m in SCENARIO_META.items() if m.primary_axis == "World"
        ]
        assert len(world_scenarios) >= 1, (
            "Must have at least one World-axis scenario to justify 'partial' coverage"
        )

    def test_no_goal_scenarios_present(self):
        """
        No Goal scenarios are implemented. Coverage is declared 'none'.
        This test exists so someone NOTICES if Goal scenarios are added
        without updating the coverage declaration.
        """
        goal_scenarios = [
            name for name, m in SCENARIO_META.items() if m.primary_axis == "Goal"
        ]
        goal_cov = KA59_AXIS_COVERAGE["Goal"]
        if goal_cov.is_implemented is False:
            assert len(goal_scenarios) == 0, (
                "Goal coverage is declared not-implemented but Goal scenarios exist"
            )

    def test_benchmark_results_still_work_after_cross_axis_extension(self):
        """Regression: adding World-probe scenario must not break existing benchmark."""
        result = evaluate_agent(MinimalHypothesisAgent, SCENARIOS, max_steps=5)
        for name in SCENARIOS:
            r = result.by_name(name)
            assert r.steps_taken >= 0
            assert r.termination in ("done", "max_steps")
