"""
KA59 Benchmark Tests
======================
Tests for ka59_ref/scenarios.py and ka59_ref/benchmark.py.

Contract being verified:
  1. SCENARIOS dict loads correctly; each entry is a valid level spec.
  2. evaluate_agent() runs an agent factory across named scenarios and returns
     a structured BenchmarkResult without exposing engine internals.
  3. Results are deterministic for deterministic agents.
  4. Each ScenarioResult has interpretable outcome fields.
  5. No internal wall tags / engine constants appear in result payloads.
  6. Transfer-wall and solid-wall scenarios produce meaningfully different
     numeric outcomes for the same baseline agent.

Run with:
    python3 -m pytest tests/test_ka59_benchmark.py -v
    python3 -m pytest tests/ -v        # full suite
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ka59_ref.scenarios import SCENARIOS
from ka59_ref.benchmark  import evaluate_agent, BenchmarkResult, ScenarioResult
from ka59_ref.discovery  import MinimalHypothesisAgent
from ka59_ref.engine     import TAG_WALL_TRANSFER, TAG_WALL_SOLID, STEP

INTERNAL_TAGS = {
    TAG_WALL_TRANSFER, TAG_WALL_SOLID,
    "0022vrxelxosfy", "0001uqqokjrptk", "0003umnkyodpjp",
}
VALID_KINDS = {"wall", "block", "controllable"}

# Expected canonical scenario names
EXPECTED_SCENARIOS = {
    "open_move_right",
    "transfer_wall_direct_block",
    "transfer_wall_push",
    "solid_wall_push_blocked",
    "push_chain",
}


# ---------------------------------------------------------------------------
# 1. Scenario corpus
# ---------------------------------------------------------------------------

class TestScenarios:
    def test_scenarios_is_a_dict(self):
        assert isinstance(SCENARIOS, dict)

    def test_all_expected_scenarios_present(self):
        missing = EXPECTED_SCENARIOS - set(SCENARIOS)
        assert not missing, f"Missing scenarios: {missing}"

    def test_each_scenario_has_steps_field(self):
        for name, spec in SCENARIOS.items():
            assert "steps" in spec, f"{name!r} missing 'steps'"
            assert isinstance(spec["steps"], int)
            assert spec["steps"] > 0

    def test_each_scenario_has_objects_list(self):
        for name, spec in SCENARIOS.items():
            assert "objects" in spec, f"{name!r} missing 'objects'"
            assert isinstance(spec["objects"], list)
            assert len(spec["objects"]) >= 1

    def test_each_object_has_required_fields(self):
        required = {"id", "x", "y", "w", "h", "kind"}
        for name, spec in SCENARIOS.items():
            for obj in spec["objects"]:
                missing = required - set(obj)
                assert not missing, (
                    f"Scenario {name!r}: object {obj.get('id', '?')!r} "
                    f"missing fields {missing}"
                )

    def test_each_scenario_has_exactly_one_selected(self):
        for name, spec in SCENARIOS.items():
            sel_count = sum(1 for o in spec["objects"] if o["kind"] == "selected")
            assert sel_count == 1, (
                f"Scenario {name!r} should have exactly 1 'selected' object, "
                f"got {sel_count}"
            )

    def test_spec_kinds_are_valid_engine_kinds(self):
        valid = {"selected", "block", "cross", "wall_transfer", "wall_solid"}
        for name, spec in SCENARIOS.items():
            for obj in spec["objects"]:
                assert obj["kind"] in valid, (
                    f"Scenario {name!r}: unknown kind {obj['kind']!r}"
                )

    def test_scenarios_can_be_passed_to_env_reset(self):
        """Each scenario spec must be parseable by KA59BlindEnv.reset()."""
        from ka59_ref.env import KA59BlindEnv
        env = KA59BlindEnv()
        for name, spec in SCENARIOS.items():
            try:
                obs = env.reset(spec)
            except Exception as e:
                pytest.fail(f"Scenario {name!r} failed to reset: {e}")
            assert len(obs) > 0

    def test_scenarios_have_no_internal_tags_as_object_ids(self):
        for name, spec in SCENARIOS.items():
            for obj in spec["objects"]:
                assert obj["id"] not in INTERNAL_TAGS, (
                    f"Scenario {name!r}: object ID {obj['id']!r} is an internal tag"
                )


# ---------------------------------------------------------------------------
# 2. Evaluator basics
# ---------------------------------------------------------------------------

class TestEvaluatorBasics:
    def test_returns_benchmark_result(self):
        result = evaluate_agent(MinimalHypothesisAgent, SCENARIOS, max_steps=5)
        assert isinstance(result, BenchmarkResult)

    def test_result_has_one_scenario_result_per_scenario(self):
        result = evaluate_agent(MinimalHypothesisAgent, SCENARIOS, max_steps=5)
        assert len(result.results) == len(SCENARIOS)

    def test_each_result_is_scenario_result(self):
        result = evaluate_agent(MinimalHypothesisAgent, SCENARIOS, max_steps=5)
        for r in result.results:
            assert isinstance(r, ScenarioResult)

    def test_result_has_agent_name(self):
        result = evaluate_agent(MinimalHypothesisAgent, SCENARIOS, max_steps=5)
        assert hasattr(result, "agent_name")
        assert isinstance(result.agent_name, str)
        assert result.agent_name  # non-empty

    def test_agent_name_reflects_class(self):
        result = evaluate_agent(MinimalHypothesisAgent, SCENARIOS, max_steps=5)
        assert "MinimalHypothesisAgent" in result.agent_name

    def test_by_name_returns_correct_result(self):
        result = evaluate_agent(MinimalHypothesisAgent, SCENARIOS, max_steps=5)
        for name in SCENARIOS:
            r = result.by_name(name)
            assert isinstance(r, ScenarioResult)
            assert r.scenario_name == name

    def test_by_name_raises_on_unknown(self):
        result = evaluate_agent(MinimalHypothesisAgent, SCENARIOS, max_steps=5)
        with pytest.raises(KeyError):
            result.by_name("does_not_exist")

    def test_subset_of_scenarios(self):
        """Evaluator should work on a hand-picked subset, not only SCENARIOS."""
        subset = {k: SCENARIOS[k] for k in ("open_move_right", "push_chain")}
        result = evaluate_agent(MinimalHypothesisAgent, subset, max_steps=5)
        assert len(result.results) == 2


# ---------------------------------------------------------------------------
# 3. ScenarioResult fields
# ---------------------------------------------------------------------------

class TestScenarioResultFields:
    def _get_result(self, scenario_name, max_steps=5):
        result = evaluate_agent(
            MinimalHypothesisAgent,
            {scenario_name: SCENARIOS[scenario_name]},
            max_steps=max_steps,
        )
        return result.by_name(scenario_name)

    def test_scenario_name_field(self):
        r = self._get_result("open_move_right")
        assert r.scenario_name == "open_move_right"

    def test_steps_taken_is_non_negative_int(self):
        r = self._get_result("open_move_right")
        assert isinstance(r.steps_taken, int)
        assert r.steps_taken >= 0

    def test_termination_is_valid_string(self):
        r = self._get_result("open_move_right")
        assert r.termination in ("done", "max_steps")

    def test_final_selected_pos_is_int_pair(self):
        r = self._get_result("open_move_right")
        assert isinstance(r.final_selected_pos, tuple)
        assert len(r.final_selected_pos) == 2
        x, y = r.final_selected_pos
        assert isinstance(x, int)
        assert isinstance(y, int)

    def test_moved_count_is_non_negative(self):
        r = self._get_result("open_move_right")
        assert isinstance(r.moved_count, int)
        assert r.moved_count >= 0

    def test_blocked_count_is_non_negative(self):
        r = self._get_result("open_move_right")
        assert isinstance(r.blocked_count, int)
        assert r.blocked_count >= 0

    def test_moved_plus_blocked_equals_steps_taken(self):
        for name in SCENARIOS:
            r = self._get_result(name)
            assert r.moved_count + r.blocked_count == r.steps_taken, (
                f"{name}: moved={r.moved_count} + blocked={r.blocked_count} "
                f"!= steps_taken={r.steps_taken}"
            )

    def test_passable_walls_found_is_non_negative(self):
        r = self._get_result("transfer_wall_push", max_steps=5)
        assert isinstance(r.passable_walls_found, int)
        assert r.passable_walls_found >= 0

    def test_steps_taken_respects_max_steps(self):
        r = self._get_result("open_move_right", max_steps=3)
        # Level has steps=20; max_steps=3 should cap it
        assert r.steps_taken <= 3

    def test_open_scenario_selected_moved_forward(self):
        """In open_move_right, selected piece should advance rightward."""
        r = self._get_result("open_move_right", max_steps=3)
        x, _ = r.final_selected_pos
        assert x > 0, "Selected should have moved right at least once"


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_agent_same_scenarios_same_results(self):
        run1 = evaluate_agent(MinimalHypothesisAgent, SCENARIOS, max_steps=5)
        run2 = evaluate_agent(MinimalHypothesisAgent, SCENARIOS, max_steps=5)

        for name in SCENARIOS:
            r1 = run1.by_name(name)
            r2 = run2.by_name(name)
            assert r1.steps_taken        == r2.steps_taken,        name
            assert r1.termination        == r2.termination,         name
            assert r1.final_selected_pos == r2.final_selected_pos,  name
            assert r1.moved_count        == r2.moved_count,         name
            assert r1.blocked_count      == r2.blocked_count,       name


# ---------------------------------------------------------------------------
# 5. No internal tag leakage
# ---------------------------------------------------------------------------

class TestNoTagLeakage:
    def test_scenario_names_contain_no_internal_tags(self):
        result = evaluate_agent(MinimalHypothesisAgent, SCENARIOS, max_steps=5)
        for r in result.results:
            assert r.scenario_name not in INTERNAL_TAGS

    def test_termination_contains_no_internal_tags(self):
        result = evaluate_agent(MinimalHypothesisAgent, SCENARIOS, max_steps=5)
        for r in result.results:
            assert r.termination not in INTERNAL_TAGS
            assert r.termination in ("done", "max_steps")

    def test_agent_name_contains_no_internal_tags(self):
        result = evaluate_agent(MinimalHypothesisAgent, SCENARIOS, max_steps=5)
        assert result.agent_name not in INTERNAL_TAGS
        for tag in INTERNAL_TAGS:
            assert tag not in result.agent_name

    def test_final_pos_is_pure_int_pair(self):
        """final_selected_pos must be (int, int), not leaking any object repr."""
        result = evaluate_agent(MinimalHypothesisAgent, SCENARIOS, max_steps=5)
        for r in result.results:
            assert isinstance(r.final_selected_pos, tuple)
            for v in r.final_selected_pos:
                assert isinstance(v, int)
                assert not isinstance(v, bool)  # bool is a subclass of int


# ---------------------------------------------------------------------------
# 6. Meaningful differences between scenarios
# ---------------------------------------------------------------------------

class TestMeaningfulDifferences:
    """
    KEY REQUIREMENT: same baseline agent, different canonical scenarios
    → verifiably different outcomes.  This is the 'paper slice' core.
    """

    def _run(self, *names, max_steps=8):
        specs = {n: SCENARIOS[n] for n in names}
        return evaluate_agent(MinimalHypothesisAgent, specs, max_steps=max_steps)

    def test_transfer_push_discovers_passable_wall(self):
        """
        In transfer_wall_push scenario the agent should empirically discover
        a passable wall (block observed at wall position after successful push).
        """
        result = self._run("transfer_wall_push")
        r = result.by_name("transfer_wall_push")
        assert r.passable_walls_found > 0, (
            "Agent must discover a passable wall in transfer_wall_push scenario"
        )

    def test_solid_wall_push_blocked_discovers_no_passable_wall(self):
        """
        In solid_wall_push_blocked scenario no block ever crosses a wall,
        so passable_walls should remain empty.
        """
        result = self._run("solid_wall_push_blocked")
        r = result.by_name("solid_wall_push_blocked")
        assert r.passable_walls_found == 0, (
            "Agent must not discover a passable wall in solid_wall_push_blocked"
        )

    def test_passable_walls_found_differs_between_wall_types(self):
        """
        Core assertion: transfer scenario finds passable walls, solid does not.
        Captures the mechanistic difference from the agent's perspective.
        """
        result = self._run("transfer_wall_push", "solid_wall_push_blocked")
        t = result.by_name("transfer_wall_push")
        s = result.by_name("solid_wall_push_blocked")
        assert t.passable_walls_found > s.passable_walls_found, (
            f"transfer={t.passable_walls_found}, solid={s.passable_walls_found}"
        )

    def test_open_move_right_has_no_blocked_steps_initially(self):
        """
        In a completely open level, every step moves the selected piece.
        moved_count should equal steps_taken, blocked_count should be 0.
        """
        result = self._run("open_move_right", max_steps=3)
        r = result.by_name("open_move_right")
        assert r.moved_count == r.steps_taken
        assert r.blocked_count == 0

    def test_direct_block_scenario_starts_with_a_block(self):
        """
        In transfer_wall_direct_block, the very first move (right) is blocked
        by the wall (no push candidate either), so the agent's first step
        should be blocked.  It may succeed in other directions after rotating.
        """
        result = self._run("transfer_wall_direct_block", max_steps=2)
        r = result.by_name("transfer_wall_direct_block")
        # At least one blocked step must have occurred
        assert r.blocked_count >= 1

    def test_push_chain_selected_advances(self):
        """
        push_chain has no walls, just two blocks in a chain.
        First move right: sel pushes pa, pa pushes pb → sel moves.
        """
        result = self._run("push_chain", max_steps=3)
        r = result.by_name("push_chain")
        assert r.moved_count >= 1
        x, _ = r.final_selected_pos
        assert x > 0, "Selected should have advanced right in push_chain"

    def test_transfer_push_selected_moves_first_step(self):
        """
        In transfer_wall_push, the first move succeeds (push through wall).
        moved_count should be >= 1.
        """
        result = self._run("transfer_wall_push", max_steps=3)
        r = result.by_name("transfer_wall_push")
        assert r.moved_count >= 1

    def test_solid_push_blocked_first_step_blocked(self):
        """
        In solid_wall_push_blocked, the first move right is blocked.
        The agent will rotate to other directions so moved_count > 0 eventually,
        but blocked_count >= 1.
        """
        result = self._run("solid_wall_push_blocked", max_steps=3)
        r = result.by_name("solid_wall_push_blocked")
        assert r.blocked_count >= 1


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_max_steps_zero_gives_zero_steps_taken(self):
        """max_steps=0 means the runner is never called; results have 0 steps."""
        result = evaluate_agent(
            MinimalHypothesisAgent,
            {"open_move_right": SCENARIOS["open_move_right"]},
            max_steps=0,
        )
        r = result.by_name("open_move_right")
        assert r.steps_taken == 0

    def test_empty_scenarios_dict_returns_empty_results(self):
        result = evaluate_agent(MinimalHypothesisAgent, {}, max_steps=5)
        assert len(result.results) == 0

    def test_custom_lambda_factory_works(self):
        """agent_factory can be a lambda, not just a class."""
        result = evaluate_agent(
            lambda: MinimalHypothesisAgent(),
            {"open_move_right": SCENARIOS["open_move_right"]},
            max_steps=3,
        )
        assert len(result.results) == 1
