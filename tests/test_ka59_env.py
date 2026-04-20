"""
KA59 Blinded Agent Environment Tests
======================================
Tests for the agent-facing wrapper (KA59BlindEnv) over KA59State.

Design contract being verified:
  1. Observation exposes positions/sizes/opaque-kinds, NOT internal rule names.
  2. Both wall types (WALL_TRANSFER and WALL_SOLID) appear as kind="wall" —
     the agent cannot distinguish them from the observation alone.
  3. The wall-transfer asymmetry IS discoverable purely from transitions:
       - direct move into a "wall" → blocked
       - pushing a block into the same "wall" space → succeeds
  4. Step counter depletes correctly through the env API.
  5. SELECT action switches the controlled piece.
  6. done=True when steps are exhausted.

Run with:
    python3 -m pytest tests/test_ka59_env.py -v
    python3 -m pytest tests/ -v           # runs both test files together
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ka59_ref.env import (
    KA59BlindEnv,
    MOVE_UP,
    MOVE_DOWN,
    MOVE_LEFT,
    MOVE_RIGHT,
    SELECT,
    ObjectView,
    StepResult,
)
# Internal constants imported ONLY to verify they do NOT appear in observations.
from ka59_ref.engine import TAG_WALL_TRANSFER, TAG_WALL_SOLID, STEP

INTERNAL_TAGS = {TAG_WALL_TRANSFER, TAG_WALL_SOLID, "0022vrxelxosfy", "0001uqqokjrptk", "0003umnkyodpjp"}
VALID_KINDS   = {"wall", "block", "controllable"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def simple_level(extra_objects=(), steps=10):
    """Level with one selected piece and optional extras."""
    objects = [
        {"id": "sel", "x": 0, "y": 0, "w": 3, "h": 3, "kind": "selected"},
    ] + list(extra_objects)
    return {"steps": steps, "objects": objects}


def obs_by_id(obs, obj_id):
    """Find an ObjectView by its id from an observation list."""
    for view in obs:
        if view.id == obj_id:
            return view
    raise KeyError(f"No object with id={obj_id!r} in observation")


def all_kinds(obs):
    return {v.kind for v in obs}


# ---------------------------------------------------------------------------
# 1. Basic reset / initial observation shape
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_list_of_object_views(self):
        env = KA59BlindEnv()
        obs = env.reset(simple_level())
        assert isinstance(obs, list)
        assert len(obs) == 1
        assert isinstance(obs[0], ObjectView)

    def test_object_view_has_required_fields(self):
        env = KA59BlindEnv()
        obs = env.reset(simple_level())
        v = obs[0]
        assert hasattr(v, "id")
        assert hasattr(v, "x")
        assert hasattr(v, "y")
        assert hasattr(v, "w")
        assert hasattr(v, "h")
        assert hasattr(v, "kind")
        assert hasattr(v, "is_selected")

    def test_selected_piece_is_marked(self):
        env = KA59BlindEnv()
        obs = env.reset(simple_level())
        sel = obs_by_id(obs, "sel")
        assert sel.is_selected is True

    def test_initial_position_correct(self):
        env = KA59BlindEnv()
        obs = env.reset(simple_level())
        sel = obs_by_id(obs, "sel")
        assert sel.x == 0
        assert sel.y == 0
        assert sel.w == 3
        assert sel.h == 3

    def test_reset_is_idempotent(self):
        """Calling reset twice re-initialises to the same starting state."""
        env = KA59BlindEnv()
        spec = simple_level([{"id": "pb", "x": 6, "y": 0, "w": 3, "h": 3, "kind": "block"}])
        obs1 = env.reset(spec)
        env.step(MOVE_RIGHT)  # mutate state
        obs2 = env.reset(spec)
        pb1 = obs_by_id(obs1, "pb")
        pb2 = obs_by_id(obs2, "pb")
        assert pb1.x == pb2.x == 6


# ---------------------------------------------------------------------------
# 2. No internal tags / rule names leaked in observations
# ---------------------------------------------------------------------------

class TestNoTagLeakage:
    def test_wall_transfer_not_labeled_as_internal_tag(self):
        """
        A WALL_TRANSFER object must NOT appear as kind=<internal tag string>.
        Both wall types must appear as kind='wall' only.
        """
        env = KA59BlindEnv()
        spec = simple_level([
            {"id": "wt", "x": 3, "y": 0, "w": 3, "h": 3, "kind": "wall_transfer"},
        ])
        obs = env.reset(spec)
        wt = obs_by_id(obs, "wt")
        assert wt.kind not in INTERNAL_TAGS, (
            f"Internal tag {wt.kind!r} was leaked into the observation"
        )
        assert wt.kind == "wall"

    def test_wall_solid_not_labeled_as_internal_tag(self):
        env = KA59BlindEnv()
        spec = simple_level([
            {"id": "ws", "x": 3, "y": 0, "w": 3, "h": 3, "kind": "wall_solid"},
        ])
        obs = env.reset(spec)
        ws = obs_by_id(obs, "ws")
        assert ws.kind not in INTERNAL_TAGS
        assert ws.kind == "wall"

    def test_both_wall_types_appear_identical_in_observation(self):
        """
        Agent cannot distinguish WALL_TRANSFER from WALL_SOLID by observation alone.
        Both map to kind='wall'.
        """
        env = KA59BlindEnv()
        spec = simple_level([
            {"id": "wt", "x": 3, "y": 0, "w": 3, "h": 3, "kind": "wall_transfer"},
            {"id": "ws", "x": 6, "y": 0, "w": 3, "h": 3, "kind": "wall_solid"},
        ])
        obs = env.reset(spec)
        wt_kind = obs_by_id(obs, "wt").kind
        ws_kind = obs_by_id(obs, "ws").kind
        assert wt_kind == ws_kind == "wall", (
            "Both wall types must look identical in the observation"
        )

    def test_observation_kinds_are_only_valid_kinds(self):
        """No unknown or internal kind strings appear in any observation."""
        env = KA59BlindEnv()
        spec = {
            "steps": 5,
            "objects": [
                {"id": "sel", "x": 0,  "y": 0, "w": 3, "h": 3, "kind": "selected"},
                {"id": "pb",  "x": 3,  "y": 0, "w": 3, "h": 3, "kind": "block"},
                {"id": "wt",  "x": 9,  "y": 0, "w": 3, "h": 3, "kind": "wall_transfer"},
                {"id": "ws",  "x": 12, "y": 0, "w": 3, "h": 3, "kind": "wall_solid"},
            ],
        }
        obs = env.reset(spec)
        for view in obs:
            assert view.kind in VALID_KINDS, (
                f"Object {view.id!r} has unknown/leaked kind={view.kind!r}"
            )

    def test_object_view_has_no_tags_attribute(self):
        """ObjectView must not expose a 'tags' attribute."""
        env = KA59BlindEnv()
        obs = env.reset(simple_level([
            {"id": "wt", "x": 3, "y": 0, "w": 3, "h": 3, "kind": "wall_transfer"},
        ]))
        for view in obs:
            assert not hasattr(view, "tags"), (
                f"ObjectView for {view.id!r} exposes internal 'tags' attribute"
            )

    def test_step_result_obs_has_no_internal_tags(self):
        """StepResult.obs must also be clean of internal tag strings."""
        env = KA59BlindEnv()
        env.reset(simple_level())
        result = env.step(MOVE_RIGHT)
        for view in result.obs:
            assert view.kind not in INTERNAL_TAGS


# ---------------------------------------------------------------------------
# 3. Transfer-wall asymmetry discoverable through transitions alone
# ---------------------------------------------------------------------------

class TestAsymmetryViaTransitions:
    """
    THE CRITICAL TEST SUITE.

    Both scenarios use identical observations for the wall (kind='wall').
    The agent can only distinguish the behaviours from transition outcomes.
    """

    def _direct_blocked_scenario(self):
        """Selected is directly adjacent to a WALL_TRANSFER tile."""
        return {
            "steps": 5,
            "objects": [
                # selected sits immediately left of the transfer wall
                {"id": "sel", "x": 0, "y": 0, "w": STEP, "h": STEP, "kind": "selected"},
                {"id": "wt",  "x": STEP, "y": 0, "w": STEP, "h": STEP, "kind": "wall_transfer"},
            ],
        }

    def _push_through_scenario(self):
        """Pushable block sits between selected and a WALL_TRANSFER tile."""
        return {
            "steps": 5,
            "objects": [
                {"id": "sel", "x": 0,        "y": 0, "w": STEP, "h": STEP, "kind": "selected"},
                {"id": "pb",  "x": STEP,     "y": 0, "w": STEP, "h": STEP, "kind": "block"},
                {"id": "wt",  "x": STEP * 2, "y": 0, "w": STEP, "h": STEP, "kind": "wall_transfer"},
            ],
        }

    def test_direct_move_into_transfer_wall_is_blocked(self):
        """
        Direct move of selected into a WALL_TRANSFER must be blocked.
        The agent sees kind='wall' — no rule explanation is given.
        """
        env = KA59BlindEnv()
        obs = env.reset(self._direct_blocked_scenario())

        # Confirm the wall is opaque (no internal kind exposed)
        wt_obs = obs_by_id(obs, "wt")
        assert wt_obs.kind == "wall"

        result = env.step(MOVE_RIGHT)
        sel_after = obs_by_id(result.obs, "sel")

        assert result.moved is False, (
            "Selected piece must not move directly into a wall"
        )
        assert sel_after.x == 0, "Selected must be at original x after blocked move"

    def test_push_block_through_transfer_wall_succeeds(self):
        """
        Pushing a block through a WALL_TRANSFER must succeed.
        The agent sees kind='wall' for the transfer wall — identical to
        test_direct_move_into_transfer_wall_is_blocked above.
        Discovery requires observing that this transition DOES commit.
        """
        env = KA59BlindEnv()
        obs = env.reset(self._push_through_scenario())

        # Confirm the wall looks identical to the blocked-case wall
        wt_obs = obs_by_id(obs, "wt")
        assert wt_obs.kind == "wall"

        result = env.step(MOVE_RIGHT)
        sel_after = obs_by_id(result.obs, "sel")
        pb_after  = obs_by_id(result.obs, "pb")

        assert result.moved is True, (
            "Selected must move when pushable clears the path through WALL_TRANSFER"
        )
        assert sel_after.x == STEP,     "Selected must have advanced one STEP"
        assert pb_after.x  == STEP * 2, "Pushable must have been pushed into WALL_TRANSFER space"

    def test_wall_kind_is_identical_in_both_scenarios(self):
        """
        Definitively proves the agent cannot distinguish wall types from
        observation — only from transition outcomes.
        """
        env = KA59BlindEnv()

        obs_a = env.reset(self._direct_blocked_scenario())
        wt_kind_a = obs_by_id(obs_a, "wt").kind

        obs_b = env.reset(self._push_through_scenario())
        wt_kind_b = obs_by_id(obs_b, "wt").kind

        assert wt_kind_a == wt_kind_b == "wall", (
            "Wall type must be indistinguishable from observation in both scenarios"
        )


# ---------------------------------------------------------------------------
# 4. Solid wall: push IS blocked (confirms asymmetry is WALL_TRANSFER-specific)
# ---------------------------------------------------------------------------

class TestSolidWallStillBlocks:
    def test_push_into_solid_wall_is_blocked(self):
        """
        A WALL_SOLID still stops pushable blocks.
        This confirms the pass-through effect is specific to WALL_TRANSFER.
        NOTE: from the agent's perspective, both walls look identical (kind='wall').
        """
        env = KA59BlindEnv()
        spec = {
            "steps": 5,
            "objects": [
                {"id": "sel", "x": 0,        "y": 0, "w": STEP, "h": STEP, "kind": "selected"},
                {"id": "pb",  "x": STEP,     "y": 0, "w": STEP, "h": STEP, "kind": "block"},
                {"id": "ws",  "x": STEP * 2, "y": 0, "w": STEP, "h": STEP, "kind": "wall_solid"},
            ],
        }
        obs = env.reset(spec)
        # Both walls look the same to the agent
        assert obs_by_id(obs, "ws").kind == "wall"

        result = env.step(MOVE_RIGHT)
        assert result.moved is False

        pb_after = obs_by_id(result.obs, "pb")
        assert pb_after.x == STEP, "Block must not move when WALL_SOLID is behind it"


# ---------------------------------------------------------------------------
# 5. Step counter via env
# ---------------------------------------------------------------------------

class TestEnvStepCounter:
    def test_step_decrements_stamina(self):
        env = KA59BlindEnv()
        env.reset(simple_level(steps=5))
        result = env.step(MOVE_RIGHT)
        assert result.steps_remaining == 4

    def test_steps_on_blocked_move_still_decrement(self):
        env = KA59BlindEnv()
        env.reset(simple_level([
            {"id": "ws", "x": STEP, "y": 0, "w": STEP, "h": STEP, "kind": "wall_solid"},
        ], steps=5))
        result = env.step(MOVE_RIGHT)
        assert result.steps_remaining == 4

    def test_done_when_steps_exhausted(self):
        env = KA59BlindEnv()
        env.reset(simple_level(steps=1))
        result = env.step(MOVE_RIGHT)
        assert result.done is True

    def test_not_done_while_steps_remain(self):
        env = KA59BlindEnv()
        env.reset(simple_level(steps=3))
        result = env.step(MOVE_RIGHT)
        assert result.done is False

    def test_steps_do_not_go_negative(self):
        env = KA59BlindEnv()
        env.reset(simple_level(steps=1))
        env.step(MOVE_RIGHT)
        result = env.step(MOVE_RIGHT)
        assert result.steps_remaining == 0


# ---------------------------------------------------------------------------
# 6. SELECT action
# ---------------------------------------------------------------------------

class TestEnvSelectAction:
    def _two_controllable_level(self):
        return {
            "steps": 10,
            "objects": [
                {"id": "a", "x": 0,  "y": 0, "w": STEP, "h": STEP, "kind": "selected"},
                {"id": "b", "x": 12, "y": 0, "w": STEP, "h": STEP, "kind": "selected"},
            ],
        }

    def test_select_switches_is_selected_flag(self):
        env = KA59BlindEnv()
        env.reset(self._two_controllable_level())
        result = env.step(SELECT("b"))
        a = obs_by_id(result.obs, "a")
        b = obs_by_id(result.obs, "b")
        assert b.is_selected is True
        assert a.is_selected is False

    def test_move_after_select_moves_new_piece(self):
        env = KA59BlindEnv()
        env.reset(self._two_controllable_level())
        env.step(SELECT("b"))
        result = env.step(MOVE_LEFT)
        b = obs_by_id(result.obs, "b")
        assert b.x == 12 - STEP

    def test_select_consumes_a_step(self):
        env = KA59BlindEnv()
        env.reset(self._two_controllable_level())
        result = env.step(SELECT("b"))
        assert result.steps_remaining == 9

    def test_select_unknown_id_raises(self):
        env = KA59BlindEnv()
        env.reset(self._two_controllable_level())
        with pytest.raises((KeyError, ValueError)):
            env.step(SELECT("does_not_exist"))


# ---------------------------------------------------------------------------
# 7. All directional moves
# ---------------------------------------------------------------------------

class TestEnvDirections:
    def test_move_up(self):
        env = KA59BlindEnv()
        env.reset({"steps": 5, "objects": [
            {"id": "sel", "x": 0, "y": STEP, "w": STEP, "h": STEP, "kind": "selected"},
        ]})
        result = env.step(MOVE_UP)
        sel = obs_by_id(result.obs, "sel")
        assert sel.y == 0
        assert result.moved is True

    def test_move_down(self):
        env = KA59BlindEnv()
        env.reset({"steps": 5, "objects": [
            {"id": "sel", "x": 0, "y": 0, "w": STEP, "h": STEP, "kind": "selected"},
        ]})
        result = env.step(MOVE_DOWN)
        sel = obs_by_id(result.obs, "sel")
        assert sel.y == STEP

    def test_move_left(self):
        env = KA59BlindEnv()
        env.reset({"steps": 5, "objects": [
            {"id": "sel", "x": STEP, "y": 0, "w": STEP, "h": STEP, "kind": "selected"},
        ]})
        result = env.step(MOVE_LEFT)
        sel = obs_by_id(result.obs, "sel")
        assert sel.x == 0

    def test_move_right(self):
        env = KA59BlindEnv()
        env.reset({"steps": 5, "objects": [
            {"id": "sel", "x": 0, "y": 0, "w": STEP, "h": STEP, "kind": "selected"},
        ]})
        result = env.step(MOVE_RIGHT)
        sel = obs_by_id(result.obs, "sel")
        assert sel.x == STEP


# ---------------------------------------------------------------------------
# 8. StepResult structure
# ---------------------------------------------------------------------------

class TestStepResultStructure:
    def test_step_returns_step_result(self):
        env = KA59BlindEnv()
        env.reset(simple_level())
        result = env.step(MOVE_RIGHT)
        assert isinstance(result, StepResult)

    def test_step_result_has_obs(self):
        env = KA59BlindEnv()
        env.reset(simple_level())
        result = env.step(MOVE_RIGHT)
        assert hasattr(result, "obs")
        assert isinstance(result.obs, (list, tuple))

    def test_step_result_has_moved(self):
        env = KA59BlindEnv()
        env.reset(simple_level())
        result = env.step(MOVE_RIGHT)
        assert hasattr(result, "moved")
        assert isinstance(result.moved, bool)

    def test_step_result_has_steps_remaining(self):
        env = KA59BlindEnv()
        env.reset(simple_level())
        result = env.step(MOVE_RIGHT)
        assert hasattr(result, "steps_remaining")

    def test_step_result_has_done(self):
        env = KA59BlindEnv()
        env.reset(simple_level())
        result = env.step(MOVE_RIGHT)
        assert hasattr(result, "done")
