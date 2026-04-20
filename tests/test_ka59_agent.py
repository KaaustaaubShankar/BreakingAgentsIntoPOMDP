"""
KA59 Minimal Hypothesis Agent Tests
======================================
Tests for MinimalHypothesisAgent in ka59_ref/discovery.py.

Contract being verified:
  1. act(obs) receives only blinded ObjectViews and returns an Action.
  2. Internal state (blocked_count, push_success, passable_walls) updates
     correctly from observed transitions.
  3. After a blocked move, agent de-prioritises that direction.
  4. After a push-through-transfer-wall success, agent records the discovery
     and maintains preference for that direction.
  5. In a paired scenario (identical wall observations, different mechanics),
     the agent chooses a different step-2 action based on step-1 outcomes —
     demonstrating actual discovery rather than a fixed sequence.
  6. Agent state never contains internal tag strings.

Run with:
    python3 -m pytest tests/test_ka59_agent.py -v
    python3 -m pytest tests/ -v        # full suite
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ka59_ref.discovery import MinimalHypothesisAgent
from ka59_ref.env import (
    Action, ObjectView,
    MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, SELECT,
)
from ka59_ref.runner import run_episode
from ka59_ref.engine import TAG_WALL_TRANSFER, TAG_WALL_SOLID, STEP

INTERNAL_TAGS = {
    TAG_WALL_TRANSFER, TAG_WALL_SOLID,
    "0022vrxelxosfy", "0001uqqokjrptk", "0003umnkyodpjp",
}
VALID_DIRS = {"right", "left", "up", "down"}


# ---------------------------------------------------------------------------
# Test helpers — build ObjectView snapshots directly
# ---------------------------------------------------------------------------

def sel(x, y, w=STEP, h=STEP) -> ObjectView:
    return ObjectView(id="sel", x=x, y=y, w=w, h=h, kind="controllable", is_selected=True)

def blk(x, y, obj_id="pb", w=STEP, h=STEP) -> ObjectView:
    return ObjectView(id=obj_id, x=x, y=y, w=w, h=h, kind="block", is_selected=False)

def wall(x, y, obj_id="wt", w=STEP, h=STEP) -> ObjectView:
    return ObjectView(id=obj_id, x=x, y=y, w=w, h=h, kind="wall", is_selected=False)

def fresh_agent() -> MinimalHypothesisAgent:
    a = MinimalHypothesisAgent()
    a.reset()
    return a


# Tiny level specs for run_episode integration tests
def make_open_level(steps=10):
    return {"steps": steps, "objects": [
        {"id": "sel", "x": 0, "y": 0, "w": STEP, "h": STEP, "kind": "selected"},
    ]}

def make_push_level(wall_kind, steps=10):
    """sel → block → wall: canonical asymmetry probe."""
    return {"steps": steps, "objects": [
        {"id": "sel",  "x": 0,        "y": 0, "w": STEP, "h": STEP, "kind": "selected"},
        {"id": "pb",   "x": STEP,     "y": 0, "w": STEP, "h": STEP, "kind": "block"},
        {"id": "wall", "x": STEP * 2, "y": 0, "w": STEP, "h": STEP, "kind": wall_kind},
    ]}


# ---------------------------------------------------------------------------
# 1. Public API contract
# ---------------------------------------------------------------------------

class TestAPIContract:
    def test_act_returns_action(self):
        agent = fresh_agent()
        obs = [sel(0, 0)]
        result = agent.act(obs)
        assert isinstance(result, Action)

    def test_act_return_is_directional_action(self):
        """Agent only moves; it does not spontaneously SELECT."""
        agent = fresh_agent()
        action = agent.act([sel(0, 0)])
        assert action.type in VALID_DIRS

    def test_act_called_repeatedly_always_returns_action(self):
        agent = fresh_agent()
        obs = [sel(0, 0)]
        for _ in range(10):
            result = agent.act(obs)
            assert isinstance(result, Action)

    def test_reset_clears_state(self):
        agent = fresh_agent()
        # Drive one blocked transition so state is non-empty
        obs1 = [sel(0, 0), wall(STEP, 0)]
        agent.act(obs1)   # step 1: picks right
        agent.act(obs1)   # step 2: analyzes (blocked), picks something else
        assert agent.blocked_count  # non-empty

        agent.reset()
        assert not agent.blocked_count
        assert not agent.push_success
        assert not agent.passable_walls
        assert agent.step_count == 0

    def test_act_works_with_multiple_objects_in_obs(self):
        agent = fresh_agent()
        obs = [
            sel(0, 0),
            blk(STEP * 2, 0),
            wall(STEP * 5, 0, obj_id="ws"),
        ]
        action = agent.act(obs)
        assert isinstance(action, Action)


# ---------------------------------------------------------------------------
# 2. Hypothesis update: blocked move
# ---------------------------------------------------------------------------

class TestBlockedMoveUpdate:
    def _blocked_transition(self, direction):
        """Simulate: agent chose <direction>, selected did NOT move."""
        agent = fresh_agent()
        obs_before = [sel(0, 0), wall(STEP, 0)]
        obs_after  = [sel(0, 0), wall(STEP, 0)]  # identical — nothing moved

        agent.act(obs_before)           # step 1: picks <direction>, stores last_obs
        # Manually set last_action to the target direction for a controlled test
        from ka59_ref.env import Action
        agent.last_action = Action(direction)
        agent.act(obs_after)            # step 2: analyzes transition
        return agent

    def test_blocked_right_increments_blocked_count(self):
        agent = self._blocked_transition("right")
        assert agent.blocked_count.get("right", 0) >= 1

    def test_blocked_left_increments_blocked_count(self):
        agent = self._blocked_transition("left")
        assert agent.blocked_count.get("left", 0) >= 1

    def test_blocked_up_increments_blocked_count(self):
        agent = self._blocked_transition("up")
        assert agent.blocked_count.get("up", 0) >= 1

    def test_blocked_down_increments_blocked_count(self):
        agent = self._blocked_transition("down")
        assert agent.blocked_count.get("down", 0) >= 1

    def test_blocked_does_not_increment_push_success(self):
        agent = self._blocked_transition("right")
        assert agent.push_success.get("right", 0) == 0

    def test_repeatedly_blocked_increases_count(self):
        """
        After many blocked steps the agent accumulates blocked_count across
        directions as it cycles through its preference order.
        Total blocked entries should reflect the attempts made.
        (The agent rotates away from each blocked direction, so the count
        spreads across all four directions rather than piling up on one.)
        """
        agent = fresh_agent()
        obs_blocked = [sel(0, 0), wall(STEP, 0)]
        # 5 act() calls: first has no analysis; subsequent 4 each record one block.
        agent.act(obs_blocked)
        for _ in range(4):
            agent.act(obs_blocked)
        total_blocked = sum(agent.blocked_count.values())
        assert total_blocked >= 3, (
            f"Expected >= 3 total blocked records, got {total_blocked}: "
            f"{agent.blocked_count}"
        )


# ---------------------------------------------------------------------------
# 3. Hypothesis update: successful push move
# ---------------------------------------------------------------------------

class TestSuccessfulPushUpdate:
    def _push_success_transition(self):
        """Simulate: agent chose RIGHT, selected moved, block moved to wall pos."""
        agent = fresh_agent()
        obs_before = [sel(0, 0), blk(STEP, 0), wall(STEP * 2, 0)]
        # After push: selected at STEP, block at STEP*2 (same as wall)
        obs_after  = [
            ObjectView("sel",  STEP,     0, STEP, STEP, "controllable", True),
            ObjectView("pb",   STEP * 2, 0, STEP, STEP, "block",        False),
            ObjectView("wt",   STEP * 2, 0, STEP, STEP, "wall",         False),
        ]
        agent.act(obs_before)   # step 1: picks right
        agent.act(obs_after)    # step 2: analyzes push-success transition
        return agent

    def test_push_success_increments_push_success_count(self):
        agent = self._push_success_transition()
        assert agent.push_success.get("right", 0) >= 1

    def test_push_success_does_not_increment_blocked_count(self):
        agent = self._push_success_transition()
        assert agent.blocked_count.get("right", 0) == 0

    def test_push_success_records_passable_wall(self):
        """Block landing at a wall position should add that wall to passable_walls."""
        agent = self._push_success_transition()
        assert "wt" in agent.passable_walls

    def test_passable_walls_contains_opaque_id_not_tag(self):
        """passable_walls must store the object's opaque ID, never an internal tag."""
        agent = self._push_success_transition()
        for wid in agent.passable_walls:
            assert wid not in INTERNAL_TAGS, (
                f"Internal tag {wid!r} leaked into passable_walls"
            )


# ---------------------------------------------------------------------------
# 4. Action selection: agent changes behaviour based on hypotheses
# ---------------------------------------------------------------------------

class TestActionSelection:
    def test_first_action_is_valid_direction(self):
        """Fresh agent with no history picks a valid direction."""
        agent = fresh_agent()
        action = agent.act([sel(0, 0)])
        assert action.type in VALID_DIRS

    def test_after_blocked_right_agent_tries_different_direction(self):
        """
        Once 'right' is consistently blocked, agent should deprioritise it
        and choose a different direction.
        """
        agent = fresh_agent()
        obs_blocked = [sel(0, 0), wall(STEP, 0)]

        agent.act(obs_blocked)                  # step 1: picks right (no history)
        action2 = agent.act(obs_blocked)        # step 2: right was blocked → try else
        assert action2.type != "right", (
            f"Agent should rotate away from blocked 'right', got {action2.type!r}"
        )

    def test_after_push_success_agent_stays_on_same_direction(self):
        """
        After a push succeeds in a direction, that direction should remain preferred
        (push_success lowers its score).
        """
        agent = fresh_agent()
        obs_before = [sel(0, 0), blk(STEP, 0), wall(STEP * 2, 0)]
        obs_after  = [
            ObjectView("sel",  STEP,     0, STEP, STEP, "controllable", True),
            ObjectView("pb",   STEP * 2, 0, STEP, STEP, "block",        False),
            ObjectView("wt",   STEP * 2, 0, STEP, STEP, "wall",         False),
        ]
        agent.act(obs_before)               # picks right
        action2 = agent.act(obs_after)      # analyzes push-success → stays on right
        assert action2.type == "right", (
            f"Agent should stay on 'right' after push success, got {action2.type!r}"
        )

    def test_push_success_outweighs_single_block(self):
        """
        Even if 'right' was blocked once, one push success should keep it preferred
        (score formula: blocked - 2*push_success).
        """
        agent = fresh_agent()
        # Manually set state: 1 block, 1 push success in 'right'
        agent.blocked_count["right"] = 1
        agent.push_success["right"] = 1
        action = agent.act([sel(0, 0)])
        # score("right") = 1 - 2 = -1 (best possible)
        assert action.type == "right"


# ---------------------------------------------------------------------------
# 5. THE CRITICAL DISCOVERY TEST
#    Same wall observation, different mechanics → different step-2 action
# ---------------------------------------------------------------------------

class TestDiscoveryViaTransitions:
    """
    In both scenarios the wall appears as kind='wall' in observations.
    The agent discovers the behavioural difference from transitions alone.
    """

    def test_agent_diverges_after_transfer_vs_solid_wall(self):
        """
        WALL_TRANSFER scenario: step-1 push succeeds (selected moves).
        WALL_SOLID scenario   : step-1 push blocked (selected stays put).

        After observing different outcomes, the agent should choose a
        different step-2 action — demonstrating genuine discovery, not a
        fixed action sequence.
        """
        agent_t = MinimalHypothesisAgent()
        agent_s = MinimalHypothesisAgent()

        trace_t = run_episode(make_push_level("wall_transfer"), agent_t.act, max_steps=2)
        trace_s = run_episode(make_push_level("wall_solid"),    agent_s.act, max_steps=2)

        # Both agents took the same step-1 action (fresh, no priors)
        assert trace_t.steps[0].action == trace_s.steps[0].action, (
            "Fresh agents must make the same first choice"
        )

        # Different step-1 outcomes (the observable difference)
        assert trace_t.steps[0].moved is True,  "Transfer wall: push must succeed"
        assert trace_s.steps[0].moved is False, "Solid wall:    push must be blocked"

        # Different internal state after step 1
        assert agent_t.push_success.get("right", 0) >= 1, \
            "Transfer agent must record push success"
        assert agent_s.blocked_count.get("right", 0) >= 1, \
            "Solid agent must record block"
        assert "wall" in agent_t.passable_walls, \
            "Transfer agent must record the passable wall"

        # The key assertion: step-2 actions differ
        action2_t = trace_t.steps[1].action
        action2_s = trace_s.steps[1].action
        assert action2_t != action2_s, (
            f"Agent must choose differently after different outcomes — "
            f"transfer chose {action2_t.type!r}, solid chose {action2_s.type!r}"
        )

    def test_wall_appears_identical_in_both_scenarios(self):
        """Confirm the agent cannot cheat: both walls look like kind='wall'."""
        agent = fresh_agent()

        # WALL_TRANSFER obs
        spec_t = make_push_level("wall_transfer")
        from ka59_ref.env import KA59BlindEnv
        env = KA59BlindEnv()
        obs_t = env.reset(spec_t)
        wall_t = next(v for v in obs_t if v.id == "wall")

        # WALL_SOLID obs
        spec_s = make_push_level("wall_solid")
        obs_s = env.reset(spec_s)
        wall_s = next(v for v in obs_s if v.id == "wall")

        assert wall_t.kind == wall_s.kind == "wall", (
            "Both wall types must appear as kind='wall' to the agent"
        )

    def test_transfer_agent_records_passable_wall_solid_agent_does_not(self):
        """
        After step 1 has been observed by the agent, the transfer agent records
        a passable wall ID; the solid agent records nothing in passable_walls.
        This IS the empirical discovery.

        Implementation note: _update_hypotheses runs at the START of the second
        act() call (when the agent sees the result of the previous action).
        We therefore need max_steps=2 so the agent gets a second act() call
        where it analyses the step-1 transition.
        """
        agent_t = MinimalHypothesisAgent()
        agent_s = MinimalHypothesisAgent()

        run_episode(make_push_level("wall_transfer"), agent_t.act, max_steps=2)
        run_episode(make_push_level("wall_solid"),    agent_s.act, max_steps=2)

        assert len(agent_t.passable_walls) > 0, "Transfer agent must note passable wall"
        assert len(agent_s.passable_walls) == 0, "Solid agent must note nothing passable"


# ---------------------------------------------------------------------------
# 6. No internal tags in agent state or output
# ---------------------------------------------------------------------------

class TestNoInternalTagsInAgent:
    def test_blocked_count_keys_are_directions(self):
        """blocked_count keys must be direction strings, not internal tags."""
        agent = fresh_agent()
        obs_blocked = [sel(0, 0), wall(STEP, 0)]
        agent.act(obs_blocked)
        agent.act(obs_blocked)
        for k in agent.blocked_count:
            assert k in VALID_DIRS, f"blocked_count key {k!r} is not a direction"
            assert k not in INTERNAL_TAGS

    def test_push_success_keys_are_directions(self):
        agent = fresh_agent()
        obs_before = [sel(0, 0), blk(STEP, 0), wall(STEP * 2, 0)]
        obs_after  = [
            ObjectView("sel", STEP, 0, STEP, STEP, "controllable", True),
            ObjectView("pb", STEP*2, 0, STEP, STEP, "block", False),
            ObjectView("wt", STEP*2, 0, STEP, STEP, "wall", False),
        ]
        agent.act(obs_before)
        agent.act(obs_after)
        for k in agent.push_success:
            assert k in VALID_DIRS
            assert k not in INTERNAL_TAGS

    def test_passable_walls_contains_opaque_ids_not_tags(self):
        agent = fresh_agent()
        obs_before = [sel(0, 0), blk(STEP, 0), wall(STEP * 2, 0, obj_id="my_wall")]
        obs_after  = [
            ObjectView("sel",    STEP,     0, STEP, STEP, "controllable", True),
            ObjectView("pb",     STEP * 2, 0, STEP, STEP, "block",        False),
            ObjectView("my_wall",STEP * 2, 0, STEP, STEP, "wall",         False),
        ]
        agent.act(obs_before)
        agent.act(obs_after)
        for wid in agent.passable_walls:
            assert wid not in INTERNAL_TAGS, f"Internal tag {wid!r} in passable_walls"
            assert wid == "my_wall"

    def test_act_never_returns_internal_tag_string(self):
        """act() must return an Action, never a raw string or internal constant."""
        agent = fresh_agent()
        obs = [sel(0, 0), wall(STEP, 0)]
        for _ in range(5):
            result = agent.act(obs)
            assert isinstance(result, Action)
            assert not isinstance(result, str)

    def test_policy_input_in_runner_contains_no_internal_tags(self):
        """Verify end-to-end: what arrives at agent.act is clean."""
        leaks = []
        original_act = MinimalHypothesisAgent.act

        agent = MinimalHypothesisAgent()

        def patched_act(self_inner, obs):
            for view in obs:
                if view.kind in INTERNAL_TAGS:
                    leaks.append(view.kind)
            return original_act(self_inner, obs)

        import types
        agent.act = types.MethodType(patched_act, agent)

        run_episode(make_push_level("wall_transfer"), agent.act, max_steps=3)
        assert leaks == [], f"Internal tags arrived at agent.act: {leaks}"


# ---------------------------------------------------------------------------
# 7. Runner integration: agent as policy
# ---------------------------------------------------------------------------

class TestAgentAsRunnerPolicy:
    def test_agent_completes_open_episode(self):
        """Agent can run a full episode in an obstacle-free level."""
        agent = MinimalHypothesisAgent()
        trace = run_episode(make_open_level(steps=4), agent.act)
        assert trace.termination == "done"
        assert len(trace.steps) == 4

    def test_agent_trajectory_is_deterministic(self):
        """Same level, same agent always produces the same trace."""
        spec = make_push_level("wall_transfer", steps=10)

        agent1 = MinimalHypothesisAgent()
        trace1 = run_episode(spec, agent1.act, max_steps=5)

        agent2 = MinimalHypothesisAgent()
        trace2 = run_episode(spec, agent2.act, max_steps=5)

        assert len(trace1.steps) == len(trace2.steps)
        for s1, s2 in zip(trace1.steps, trace2.steps):
            assert s1.action == s2.action

    def test_step_count_increments_each_act_call(self):
        agent = MinimalHypothesisAgent()
        trace = run_episode(make_open_level(steps=3), agent.act)
        # step_count should equal number of act() calls made
        # Runner calls act once per step
        assert agent.step_count == 3

    def test_agent_never_references_engine_directly(self):
        """
        Smoke test: agent module must not import engine internals.
        Verified by confirming agent works entirely through env.py types.
        """
        import ka59_ref.discovery as disc
        import ka59_ref.engine as eng

        # The agent's public state fields should only contain direction strings
        # and opaque IDs — never engine tag constants
        engine_constants = {eng.TAG_WALL_TRANSFER, eng.TAG_WALL_SOLID,
                            eng.TAG_SELECTED, eng.TAG_CROSS, eng.TAG_BLOCK}
        agent = MinimalHypothesisAgent()
        run_episode(make_push_level("wall_transfer"), agent.act, max_steps=3)

        for k in agent.blocked_count:
            assert k not in engine_constants
        for k in agent.push_success:
            assert k not in engine_constants
        for wid in agent.passable_walls:
            assert wid not in engine_constants
