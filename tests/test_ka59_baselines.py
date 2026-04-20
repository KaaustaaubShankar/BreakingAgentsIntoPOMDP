"""
KA59 Baseline Agent Tests + Multi-Agent Spread
================================================
Tests for NaiveRightAgent and RotateOnBlockAgent, and for the benchmark spread
across all three agents.

Behavioral spread being verified
---------------------------------
In transfer_wall_push (sel → block → WALL_TRANSFER → open):
  NaiveRightAgent:         step-1 push succeeds, then stuck on right forever
                           moved_count=1 / blocked_count=4  passable_walls=0
  RotateOnBlockAgent:      step-1 push succeeds, step-2 blocked → rotates DOWN
                           moved_count=4 / blocked_count=1  passable_walls=0
  MinimalHypothesisAgent:  push success in step-1 lifts right's score; persists
                           on right longer before eventually rotating
                           moved_count=2 / blocked_count=3  passable_walls≥1

In solid_wall_push_blocked (sel → block → WALL_SOLID):
  NaiveRightAgent:         always right → always blocked → moved_count=0
  RotateOnBlockAgent:      step-1 blocked → rotates DOWN, moves freely after
  MinimalHypothesisAgent:  step-1 blocked → DOWN wins next, moves freely after
  → NaiveRightAgent is clearly separated from both others here.

Run with:
    python3 -m pytest tests/test_ka59_baselines.py -v
    python3 -m pytest tests/ -v
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ka59_ref.discovery import (
    NaiveRightAgent,
    RotateOnBlockAgent,
    MinimalHypothesisAgent,
)
from ka59_ref.env      import Action, ObjectView, MOVE_RIGHT, MOVE_DOWN, MOVE_LEFT, MOVE_UP
from ka59_ref.runner   import run_episode
from ka59_ref.scenarios import SCENARIOS
from ka59_ref.benchmark import evaluate_agent
from ka59_ref.engine    import TAG_WALL_TRANSFER, TAG_WALL_SOLID, STEP

INTERNAL_TAGS = {
    TAG_WALL_TRANSFER, TAG_WALL_SOLID,
    "0022vrxelxosfy", "0001uqqokjrptk", "0003umnkyodpjp",
}
VALID_DIRS = {"right", "left", "up", "down"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sel_view(x=0, y=0):
    return ObjectView("sel", x, y, STEP, STEP, "controllable", True)

def wall_view(x, y, obj_id="w"):
    return ObjectView(obj_id, x, y, STEP, STEP, "wall", False)


# ---------------------------------------------------------------------------
# 1. NaiveRightAgent
# ---------------------------------------------------------------------------

class TestNaiveRightAgent:
    def test_act_returns_action(self):
        agent = NaiveRightAgent()
        assert isinstance(agent.act([sel_view()]), Action)

    def test_always_returns_move_right(self):
        agent = NaiveRightAgent()
        obs = [sel_view()]
        for _ in range(10):
            assert agent.act(obs) == MOVE_RIGHT

    def test_ignores_observation_content(self):
        """NaiveRightAgent must return MOVE_RIGHT regardless of what is in obs."""
        agent = NaiveRightAgent()
        obs_simple  = [sel_view()]
        obs_complex = [sel_view(), wall_view(STEP, 0), wall_view(0, STEP)]
        assert agent.act(obs_simple)  == MOVE_RIGHT
        assert agent.act(obs_complex) == MOVE_RIGHT

    def test_reset_does_not_change_behaviour(self):
        """Reset is a no-op for NaiveRightAgent; still returns MOVE_RIGHT."""
        agent = NaiveRightAgent()
        agent.act([sel_view()])
        agent.reset()
        assert agent.act([sel_view()]) == MOVE_RIGHT

    def test_deterministic(self):
        spec = SCENARIOS["open_move_right"]
        t1 = run_episode(spec, NaiveRightAgent().act, max_steps=5)
        t2 = run_episode(spec, NaiveRightAgent().act, max_steps=5)
        for s1, s2 in zip(t1.steps, t2.steps):
            assert s1.action == s2.action

    def test_no_internal_tags_in_state(self):
        """NaiveRightAgent has no hypothesis state so nothing to leak."""
        agent = NaiveRightAgent()
        run_episode(SCENARIOS["transfer_wall_push"], agent.act, max_steps=5)
        # No hidden state to inspect; just confirm act() still works
        assert isinstance(agent.act([sel_view()]), Action)

    def test_in_solid_push_scenario_never_moves(self):
        """
        NaiveRightAgent is stuck forever in solid_wall_push_blocked.
        Every move is blocked (always chooses right).
        """
        trace = run_episode(
            SCENARIOS["solid_wall_push_blocked"],
            NaiveRightAgent().act,
            max_steps=5,
        )
        assert trace.steps[0].moved is False
        moved_total = sum(s.moved for s in trace.steps)
        assert moved_total <= 1   # at most step-1 push succeeds (it doesn't here)
        assert moved_total == 0   # solid wall → push fails immediately

    def test_in_transfer_push_only_first_step_moves(self):
        """
        NaiveRightAgent gets one successful push (step 1), then the selected
        piece is adjacent to the wall and all subsequent MOVE_RIGHT are blocked.
        """
        trace = run_episode(
            SCENARIOS["transfer_wall_push"],
            NaiveRightAgent().act,
            max_steps=5,
        )
        assert trace.steps[0].moved is True      # push succeeded
        for step in trace.steps[1:]:
            assert step.moved is False            # stuck against transfer wall


# ---------------------------------------------------------------------------
# 2. RotateOnBlockAgent
# ---------------------------------------------------------------------------

class TestRotateOnBlockAgent:
    def test_act_returns_action(self):
        agent = RotateOnBlockAgent()
        assert isinstance(agent.act([sel_view()]), Action)

    def test_first_action_is_right(self):
        agent = RotateOnBlockAgent()
        assert agent.act([sel_view()]).type == "right"

    def test_rotates_to_next_direction_after_blocked(self):
        """
        Simulate one blocked step: sel doesn't move → agent should rotate.
        """
        agent = RotateOnBlockAgent()
        obs_before = [sel_view(0, 0), wall_view(STEP, 0)]
        obs_after  = [sel_view(0, 0), wall_view(STEP, 0)]  # nothing moved

        agent.act(obs_before)              # step 1: picks right
        action2 = agent.act(obs_after)     # step 2: analyzes blocked → rotates
        assert action2.type != "right", (
            f"RotateOnBlockAgent must rotate after a blocked step, got {action2.type!r}"
        )

    def test_stays_on_direction_when_not_blocked(self):
        """
        When last move succeeded (sel changed position), agent keeps direction.
        """
        agent = RotateOnBlockAgent()
        obs1 = [sel_view(0, 0)]
        obs2 = [sel_view(STEP, 0)]   # sel moved right

        agent.act(obs1)              # step 1: right
        action2 = agent.act(obs2)   # step 2: success → no rotation
        assert action2.type == "right"

    def test_rotation_cycles_through_all_directions(self):
        """Four consecutive blocks should cycle through all four directions."""
        agent = RotateOnBlockAgent()
        obs_stuck = [sel_view(0, 0)]   # same obs each time (blocked in every dir)

        actions_seen = set()
        agent.act(obs_stuck)            # call 1: no analysis yet
        for _ in range(4):
            a = agent.act(obs_stuck)   # each call blocks and rotates
            actions_seen.add(a.type)
        assert actions_seen == VALID_DIRS

    def test_reset_restores_initial_direction(self):
        agent = RotateOnBlockAgent()
        obs_blocked = [sel_view(0, 0), wall_view(STEP, 0)]
        agent.act(obs_blocked)
        agent.act(obs_blocked)   # rotated to "down"
        agent.reset()
        assert agent.act([sel_view()]).type == "right"  # back to start

    def test_deterministic(self):
        spec = SCENARIOS["solid_wall_push_blocked"]
        t1 = run_episode(spec, RotateOnBlockAgent().act, max_steps=5)
        t2 = run_episode(spec, RotateOnBlockAgent().act, max_steps=5)
        for s1, s2 in zip(t1.steps, t2.steps):
            assert s1.action == s2.action

    def test_no_internal_tags_in_output(self):
        agent = RotateOnBlockAgent()
        run_episode(SCENARIOS["transfer_wall_push"], agent.act, max_steps=5)
        result = agent.act([sel_view()])
        assert isinstance(result, Action)
        assert result.type in VALID_DIRS

    def test_rotates_away_in_solid_push_scenario(self):
        """
        In solid_wall_push_blocked, step-1 is blocked → agent rotates to DOWN.
        Subsequent steps succeed (nothing blocks downward movement).
        """
        trace = run_episode(
            SCENARIOS["solid_wall_push_blocked"],
            RotateOnBlockAgent().act,
            max_steps=5,
        )
        assert trace.steps[0].moved is False   # first step blocked
        # After rotating, the agent should find free movement
        moved_after_first = sum(s.moved for s in trace.steps[1:])
        assert moved_after_first > 0, "RotateOnBlockAgent should move freely after rotating"

    def test_step_count_increments(self):
        agent = RotateOnBlockAgent()
        run_episode(SCENARIOS["open_move_right"], agent.act, max_steps=4)
        assert agent.step_count == 4


# ---------------------------------------------------------------------------
# 3. Behavioral diffs vs MinimalHypothesisAgent
# ---------------------------------------------------------------------------

class TestBehavioralDiffs:
    """
    Each test confirms at least one agent pair behaves differently on a
    canonical scenario.  These are the observable differences that
    justify having three distinct baselines.
    """

    def _run(self, agent_cls, scenario_name, max_steps=5):
        return evaluate_agent(
            agent_cls,
            {scenario_name: SCENARIOS[scenario_name]},
            max_steps=max_steps,
        ).by_name(scenario_name)

    def test_naive_vs_rotate_in_solid_push_blocked(self):
        """
        NaiveRightAgent stays stuck; RotateOnBlockAgent escapes.
        moved_count: NaiveRightAgent=0, RotateOnBlockAgent>0.
        """
        r_naive  = self._run(NaiveRightAgent,     "solid_wall_push_blocked")
        r_rotate = self._run(RotateOnBlockAgent,   "solid_wall_push_blocked")

        assert r_naive.moved_count == 0, (
            f"NaiveRightAgent must stay stuck, got moved={r_naive.moved_count}"
        )
        assert r_rotate.moved_count > 0, (
            f"RotateOnBlockAgent must escape, got moved={r_rotate.moved_count}"
        )

    def test_naive_vs_hypothesis_in_solid_push_blocked(self):
        """
        MinimalHypothesisAgent also escapes; NaiveRightAgent does not.
        """
        r_naive = self._run(NaiveRightAgent,          "solid_wall_push_blocked")
        r_hyp   = self._run(MinimalHypothesisAgent,   "solid_wall_push_blocked")

        assert r_naive.moved_count == 0
        assert r_hyp.moved_count > 0

    def test_hypothesis_discovers_passable_wall_naive_does_not(self):
        """
        MinimalHypothesisAgent records a passable wall in transfer_wall_push.
        NaiveRightAgent and RotateOnBlockAgent record none (no hypothesis state).
        """
        r_naive  = self._run(NaiveRightAgent,          "transfer_wall_push")
        r_rotate = self._run(RotateOnBlockAgent,        "transfer_wall_push")
        r_hyp    = self._run(MinimalHypothesisAgent,    "transfer_wall_push")

        assert r_hyp.passable_walls_found > 0,    "Discovery agent must find passable wall"
        assert r_naive.passable_walls_found == 0,  "NaiveRightAgent has no hypothesis"
        assert r_rotate.passable_walls_found == 0, "RotateOnBlockAgent has no hypothesis"

    def test_all_three_differ_in_transfer_push_moved_count(self):
        """
        transfer_wall_push gives a three-way split in moved_count:
          NaiveRightAgent:        =1  (one push, then stuck)
          MinimalHypothesisAgent: =2  (push then persists on right; rotates late)
          RotateOnBlockAgent:     =4  (push then immediately rotates to free dir)
        """
        r_naive  = self._run(NaiveRightAgent,        "transfer_wall_push")
        r_hyp    = self._run(MinimalHypothesisAgent,  "transfer_wall_push")
        r_rotate = self._run(RotateOnBlockAgent,      "transfer_wall_push")

        moved = {r_naive.moved_count, r_hyp.moved_count, r_rotate.moved_count}
        assert len(moved) == 3, (
            f"All three agents should have distinct moved_counts in transfer_wall_push; "
            f"got naive={r_naive.moved_count}, hyp={r_hyp.moved_count}, "
            f"rotate={r_rotate.moved_count}"
        )

    def test_transfer_push_exact_counts(self):
        """Pin the exact moved/blocked counts for the three-way spread."""
        r_naive  = self._run(NaiveRightAgent,        "transfer_wall_push")
        r_hyp    = self._run(MinimalHypothesisAgent,  "transfer_wall_push")
        r_rotate = self._run(RotateOnBlockAgent,      "transfer_wall_push")

        assert r_naive.moved_count  == 1,  f"NaiveRightAgent: expected 1, got {r_naive.moved_count}"
        assert r_hyp.moved_count    == 2,  f"MinimalHypothesisAgent: expected 2, got {r_hyp.moved_count}"
        assert r_rotate.moved_count == 4,  f"RotateOnBlockAgent: expected 4, got {r_rotate.moved_count}"


# ---------------------------------------------------------------------------
# 4. Multi-agent benchmark spread
# ---------------------------------------------------------------------------

class TestMultiAgentBenchmarkSpread:
    """
    Run evaluate_agent for each factory and confirm the results diverge
    in at least one canonical scenario.  This is the 'meaningful spread'
    for paper purposes.
    """

    ALL_AGENTS = [NaiveRightAgent, RotateOnBlockAgent, MinimalHypothesisAgent]

    def _eval_all(self, max_steps=5):
        return {
            cls.__name__: evaluate_agent(cls, SCENARIOS, max_steps=max_steps)
            for cls in self.ALL_AGENTS
        }

    def test_all_agents_return_benchmark_results(self):
        results = self._eval_all()
        for name, br in results.items():
            assert br is not None, f"{name} returned None"
            assert len(br.results) == len(SCENARIOS)

    def test_agent_names_differ(self):
        results = self._eval_all()
        names = [br.agent_name for br in results.values()]
        assert len(set(names)) == 3, f"Expected 3 distinct agent names, got {names}"

    def test_benchmark_results_deterministic_per_agent(self):
        for cls in self.ALL_AGENTS:
            r1 = evaluate_agent(cls, SCENARIOS, max_steps=5)
            r2 = evaluate_agent(cls, SCENARIOS, max_steps=5)
            for name in SCENARIOS:
                s1 = r1.by_name(name)
                s2 = r2.by_name(name)
                assert s1.moved_count == s2.moved_count, (
                    f"{cls.__name__}/{name}: moved_count not deterministic"
                )

    def test_spread_in_solid_push_blocked(self):
        """
        solid_wall_push_blocked: NaiveRightAgent=0 moved, others>0.
        This is a clean 1 vs rest split.
        """
        results = self._eval_all()
        naive_moved  = results["NaiveRightAgent"].by_name("solid_wall_push_blocked").moved_count
        rotate_moved = results["RotateOnBlockAgent"].by_name("solid_wall_push_blocked").moved_count
        hyp_moved    = results["MinimalHypothesisAgent"].by_name("solid_wall_push_blocked").moved_count

        assert naive_moved  == 0, f"NaiveRightAgent should be stuck: {naive_moved}"
        assert rotate_moved  > 0, f"RotateOnBlockAgent should escape: {rotate_moved}"
        assert hyp_moved     > 0, f"MinimalHypothesisAgent should escape: {hyp_moved}"

    def test_spread_in_transfer_push_passable_walls(self):
        """
        transfer_wall_push: only MinimalHypothesisAgent records passable_walls.
        """
        results = self._eval_all()
        naive_pw  = results["NaiveRightAgent"].by_name("transfer_wall_push").passable_walls_found
        rotate_pw = results["RotateOnBlockAgent"].by_name("transfer_wall_push").passable_walls_found
        hyp_pw    = results["MinimalHypothesisAgent"].by_name("transfer_wall_push").passable_walls_found

        assert hyp_pw    > 0,  "MinimalHypothesisAgent must discover passable wall"
        assert naive_pw  == 0, "NaiveRightAgent must not (no hypothesis)"
        assert rotate_pw == 0, "RotateOnBlockAgent must not (no hypothesis)"

    def test_no_internal_tags_in_any_result(self):
        """No internal tag string must appear in any result payload."""
        results = self._eval_all()
        for agent_name, br in results.items():
            for r in br.results:
                assert r.termination in ("done", "max_steps"), (
                    f"{agent_name}/{r.scenario_name}: bad termination {r.termination!r}"
                )
                assert r.scenario_name not in INTERNAL_TAGS, (
                    f"Scenario name {r.scenario_name!r} is an internal tag"
                )
                assert br.agent_name not in INTERNAL_TAGS, (
                    f"Agent name {br.agent_name!r} is an internal tag"
                )

    def test_open_level_same_for_all_agents(self):
        """
        In open_move_right (no obstacles) all three agents always move right
        and get maximum moved_count.  Confirms the ceiling baseline.
        """
        results = self._eval_all(max_steps=3)
        for cls in self.ALL_AGENTS:
            r = results[cls.__name__].by_name("open_move_right")
            assert r.moved_count == 3, (
                f"{cls.__name__} in open_move_right: expected moved=3, got {r.moved_count}"
            )
            assert r.blocked_count == 0
