"""
KA59 Episode Runner Tests
===========================
Tests for run_episode() / EpisodeTrace / EpisodeStep.

Contract being verified:
  1. run_episode(level_spec, policy_fn) returns a full EpisodeTrace.
  2. policy_fn receives only the blinded observation — no engine internals.
  3. Trajectory is captured deterministically (step records, initial obs).
  4. Termination reasons: "done" (stamina exhausted) and "max_steps" (cap hit).
  5. Invalid actions from the policy raise InvalidActionError, tested explicitly.
  6. No internal wall-tag strings appear anywhere in the trace payload.

Run with:
    python3 -m pytest tests/test_ka59_runner.py -v
    python3 -m pytest tests/ -v            # all KA59 tests together
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ka59_ref.runner import (
    run_episode,
    EpisodeTrace,
    EpisodeStep,
    InvalidActionError,
)
from ka59_ref.env import (
    MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT,
    SELECT,
    ObjectView,
)
from ka59_ref.engine import TAG_WALL_TRANSFER, TAG_WALL_SOLID, STEP

INTERNAL_TAGS = {
    TAG_WALL_TRANSFER,
    TAG_WALL_SOLID,
    "0022vrxelxosfy",
    "0001uqqokjrptk",
    "0003umnkyodpjp",
}
VALID_KINDS = {"wall", "block", "controllable"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def open_level(steps=5):
    """Single selected piece, no walls."""
    return {
        "steps": steps,
        "objects": [
            {"id": "sel", "x": 0, "y": 0, "w": STEP, "h": STEP, "kind": "selected"},
        ],
    }


def two_piece_level(steps=10):
    return {
        "steps": steps,
        "objects": [
            {"id": "a", "x": 0,         "y": 0, "w": STEP, "h": STEP, "kind": "selected"},
            {"id": "b", "x": STEP * 4,  "y": 0, "w": STEP, "h": STEP, "kind": "selected"},
        ],
    }


def walled_level(steps=5):
    return {
        "steps": steps,
        "objects": [
            {"id": "sel", "x": 0,        "y": 0, "w": STEP, "h": STEP, "kind": "selected"},
            {"id": "wt",  "x": STEP * 3, "y": 0, "w": STEP, "h": STEP, "kind": "wall_transfer"},
            {"id": "ws",  "x": STEP * 6, "y": 0, "w": STEP, "h": STEP, "kind": "wall_solid"},
        ],
    }


def obs_by_id(obs, obj_id):
    for v in obs:
        if v.id == obj_id:
            return v
    raise KeyError(obj_id)


# ---------------------------------------------------------------------------
# 1. Return types / structure
# ---------------------------------------------------------------------------

class TestReturnTypes:
    def test_returns_episode_trace(self):
        trace = run_episode(open_level(steps=1), lambda obs: MOVE_RIGHT)
        assert isinstance(trace, EpisodeTrace)

    def test_trace_has_initial_obs(self):
        trace = run_episode(open_level(steps=1), lambda obs: MOVE_RIGHT)
        assert hasattr(trace, "initial_obs")
        assert isinstance(trace.initial_obs, (list, tuple))
        assert len(trace.initial_obs) > 0

    def test_trace_has_steps(self):
        trace = run_episode(open_level(steps=2), lambda obs: MOVE_RIGHT)
        assert hasattr(trace, "steps")
        assert isinstance(trace.steps, (list, tuple))

    def test_trace_has_termination(self):
        trace = run_episode(open_level(steps=1), lambda obs: MOVE_RIGHT)
        assert hasattr(trace, "termination")
        assert isinstance(trace.termination, str)

    def test_steps_are_episode_step_instances(self):
        trace = run_episode(open_level(steps=2), lambda obs: MOVE_RIGHT)
        for s in trace.steps:
            assert isinstance(s, EpisodeStep)

    def test_episode_step_fields(self):
        trace = run_episode(open_level(steps=1), lambda obs: MOVE_RIGHT)
        s = trace.steps[0]
        assert hasattr(s, "step_num")
        assert hasattr(s, "action")
        assert hasattr(s, "obs")
        assert hasattr(s, "moved")
        assert hasattr(s, "steps_remaining")
        assert hasattr(s, "done")


# ---------------------------------------------------------------------------
# 2. Scripted policies — trajectory content
# ---------------------------------------------------------------------------

class TestScriptedPolicies:
    def test_move_right_until_done(self):
        """
        Simple 'always move right' policy runs to stamina exhaustion.
        All step records must be captured in order.
        """
        steps_budget = 4
        trace = run_episode(open_level(steps=steps_budget), lambda obs: MOVE_RIGHT)

        assert trace.termination == "done"
        assert len(trace.steps) == steps_budget

        # step_num should be 1-indexed and strictly increasing
        for i, s in enumerate(trace.steps):
            assert s.step_num == i + 1

        # final step should be marked done
        assert trace.steps[-1].done is True
        assert trace.steps[-1].steps_remaining == 0

        # selected piece moved right each step
        sel = obs_by_id(trace.initial_obs, "sel")
        assert sel.x == 0  # initial obs is before any action
        sel_final = obs_by_id(trace.steps[-1].obs, "sel")
        assert sel_final.x == steps_budget * STEP

    def test_selection_then_move_policy(self):
        """
        Policy: first action selects piece 'b', subsequent actions move left.
        Verifies runner handles SELECT then directional actions correctly.
        """
        call_count = [0]

        def policy(obs):
            call_count[0] += 1
            if call_count[0] == 1:
                return SELECT("b")
            return MOVE_LEFT

        trace = run_episode(two_piece_level(steps=10), policy, max_steps=4)

        # Step 1: select b (no positional change for b, but is_selected flips)
        b_after_select = obs_by_id(trace.steps[0].obs, "b")
        assert b_after_select.is_selected is True
        a_after_select = obs_by_id(trace.steps[0].obs, "a")
        assert a_after_select.is_selected is False

        # Steps 2–4: b moves left 3 times
        b_final = obs_by_id(trace.steps[-1].obs, "b")
        assert b_final.x == STEP * 4 - 3 * STEP   # started at STEP*4, moved left 3×

    def test_stationary_policy_terminates_on_max_steps(self):
        """
        A policy that always moves into a wall (stays put) should still
        terminate once max_steps is reached.
        """
        spec = {
            "steps": 20,
            "objects": [
                {"id": "sel", "x": 0,    "y": 0, "w": STEP, "h": STEP, "kind": "selected"},
                {"id": "ws",  "x": STEP, "y": 0, "w": STEP, "h": STEP, "kind": "wall_solid"},
            ],
        }
        trace = run_episode(spec, lambda obs: MOVE_RIGHT, max_steps=3)

        assert trace.termination == "max_steps"
        assert len(trace.steps) == 3

        # Selected never moved (wall blocked every attempt)
        sel_final = obs_by_id(trace.steps[-1].obs, "sel")
        assert sel_final.x == 0

    def test_initial_obs_matches_reset_state(self):
        """initial_obs must reflect the state before any action is taken."""
        trace = run_episode(open_level(steps=3), lambda obs: MOVE_RIGHT)
        sel = obs_by_id(trace.initial_obs, "sel")
        assert sel.x == 0   # untouched start position

    def test_obs_in_each_step_reflects_state_after_action(self):
        """Each step's obs must reflect the state resulting from that step's action."""
        trace = run_episode(open_level(steps=3), lambda obs: MOVE_RIGHT)
        for i, s in enumerate(trace.steps, start=1):
            sel = obs_by_id(s.obs, "sel")
            assert sel.x == i * STEP  # moved right i times

    def test_policy_receives_previous_step_obs(self):
        """
        The observation handed to policy_fn at step N must match
        the obs recorded in step N-1 (i.e. the env state after the last action).
        """
        received_obs = []

        def recording_policy(obs):
            received_obs.append(obs)
            return MOVE_RIGHT

        trace = run_episode(open_level(steps=3), recording_policy)

        # First call to policy gets initial_obs
        assert list(trace.initial_obs) == list(received_obs[0])

        # Second call gets step[0].obs (state after first action)
        assert list(trace.steps[0].obs) == list(received_obs[1])


# ---------------------------------------------------------------------------
# 3. Termination conditions
# ---------------------------------------------------------------------------

class TestTermination:
    def test_terminates_done_when_stamina_exhausted(self):
        trace = run_episode(open_level(steps=2), lambda obs: MOVE_RIGHT)
        assert trace.termination == "done"

    def test_terminates_max_steps_before_done(self):
        trace = run_episode(open_level(steps=10), lambda obs: MOVE_RIGHT, max_steps=3)
        assert trace.termination == "max_steps"
        assert len(trace.steps) == 3

    def test_done_takes_precedence_over_max_steps_same_tick(self):
        """If done and max_steps arrive on the same step, report 'done'."""
        # steps=3, max_steps=3: they coincide on step 3
        trace = run_episode(open_level(steps=3), lambda obs: MOVE_RIGHT, max_steps=3)
        assert trace.termination == "done"

    def test_no_steps_taken_when_done_immediately(self):
        """steps=0 in level spec means the env is done before any action."""
        trace = run_episode(open_level(steps=0), lambda obs: MOVE_RIGHT)
        # env already done at reset, runner should not call policy at all
        assert len(trace.steps) == 0
        assert trace.termination == "done"

    def test_max_steps_none_runs_until_done(self):
        trace = run_episode(open_level(steps=2), lambda obs: MOVE_RIGHT, max_steps=None)
        assert trace.termination == "done"

    def test_step_count_correct_for_max_steps(self):
        trace = run_episode(open_level(steps=100), lambda obs: MOVE_RIGHT, max_steps=7)
        assert len(trace.steps) == 7


# ---------------------------------------------------------------------------
# 4. Invalid action handling
# ---------------------------------------------------------------------------

class TestInvalidAction:
    def test_none_action_raises_invalid_action_error(self):
        with pytest.raises(InvalidActionError):
            run_episode(open_level(), lambda obs: None, max_steps=5)

    def test_string_action_raises_invalid_action_error(self):
        with pytest.raises(InvalidActionError):
            run_episode(open_level(), lambda obs: "move_right", max_steps=5)

    def test_integer_action_raises_invalid_action_error(self):
        with pytest.raises(InvalidActionError):
            run_episode(open_level(), lambda obs: 42, max_steps=5)

    def test_invalid_action_error_is_value_error(self):
        """InvalidActionError should subclass ValueError for easy catching."""
        with pytest.raises(ValueError):
            run_episode(open_level(), lambda obs: None, max_steps=5)

    def test_error_includes_step_number(self):
        """The exception message should mention which step failed."""
        call_count = [0]

        def bad_after_two(obs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return MOVE_RIGHT
            return None  # goes bad on step 3

        with pytest.raises(InvalidActionError, match="3"):
            run_episode(open_level(steps=10), bad_after_two, max_steps=10)


# ---------------------------------------------------------------------------
# 5. No internal tag leakage in trajectory
# ---------------------------------------------------------------------------

class TestNoTagLeakageInTrace:
    def _check_obs_clean(self, obs):
        for view in obs:
            assert isinstance(view, ObjectView)
            assert view.kind in VALID_KINDS, (
                f"Internal/unknown kind {view.kind!r} leaked into trace obs"
            )
            assert view.kind not in INTERNAL_TAGS

    def test_initial_obs_has_no_internal_tags(self):
        trace = run_episode(walled_level(), lambda obs: MOVE_RIGHT, max_steps=3)
        self._check_obs_clean(trace.initial_obs)

    def test_step_obs_has_no_internal_tags(self):
        trace = run_episode(walled_level(), lambda obs: MOVE_RIGHT, max_steps=3)
        for step in trace.steps:
            self._check_obs_clean(step.obs)

    def test_policy_input_has_no_internal_tags(self):
        """
        The observation the policy function *receives* must also be clean.
        Verify by checking what arrives at policy_fn itself.
        """
        leaked = []

        def checking_policy(obs):
            for view in obs:
                if view.kind in INTERNAL_TAGS or view.kind not in VALID_KINDS:
                    leaked.append(view.kind)
            return MOVE_RIGHT

        run_episode(walled_level(), checking_policy, max_steps=3)
        assert leaked == [], f"Internal tags arrived at policy_fn: {leaked}"

    def test_action_stored_in_step_is_action_object(self):
        """step.action must be an Action, not a raw string or tag."""
        from ka59_ref.env import Action
        trace = run_episode(open_level(steps=2), lambda obs: MOVE_RIGHT)
        for step in trace.steps:
            assert isinstance(step.action, Action)


# ---------------------------------------------------------------------------
# 6. Determinism / reproducibility
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_policy_same_level_produces_identical_traces(self):
        spec = open_level(steps=4)
        policy = lambda obs: MOVE_RIGHT

        trace1 = run_episode(spec, policy)
        trace2 = run_episode(spec, policy)

        assert trace1.termination == trace2.termination
        assert len(trace1.steps) == len(trace2.steps)
        for s1, s2 in zip(trace1.steps, trace2.steps):
            assert s1.step_num == s2.step_num
            assert s1.moved == s2.moved
            assert s1.steps_remaining == s2.steps_remaining
            # obs should match position-wise
            for v1, v2 in zip(s1.obs, s2.obs):
                assert v1.id == v2.id
                assert v1.x == v2.x
                assert v1.y == v2.y
