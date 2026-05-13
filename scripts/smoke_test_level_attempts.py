"""Smoke-test LEVEL_ATTEMPTS=2 retry path in ka59_game.experiment.run_agent.

Scenario 1: stub LLMClient that always returns MOVE_LEFT (which on ka59simple
either bumps the wall or wastes turns), forces the agent to exhaust its
turn budget on attempt 1, and verifies:
  - history contains a `level_retry` event
  - action log entries have `attempt` and `attempt_turn` fields
  - result.turns counts cumulative across attempts
  - env.step(RESET) was called between attempts (level_retry event proves it)

Scenario 2: scripted stub that plays the canonical 6-action ka59simple winning
sequence and verifies:
  - level wins on attempt 1
  - no `level_retry` event fired (retry loop did NOT trigger spuriously)
  - both `level_complete` and `win` events present
  - all action events have attempt == 1

Run: python3 -m scripts.smoke_test_level_attempts
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

repo_root = Path(__file__).parents[1]
sys.path.insert(0, str(repo_root))

from ka59_game.experiment import LEVEL_ATTEMPTS, run_agent
from ka59_game.llm_client import LLMClient


class StubLLMClient(LLMClient):
    """Always responds with MOVE_LEFT. Inherits parse_json + usage tracking."""
    def __init__(self):
        super().__init__(provider="openrouter", model="stub", reasoning_effort=None)

    def generate(self, system_prompt, user_prompt):
        return '{"reasoning": "stub: always left", "action": "MOVE_LEFT"}'


class WinningStubLLMClient(LLMClient):
    """Returns the canonical ka59simple winning sequence in order, one per call."""
    SCRIPT = [
        '{"reasoning": "stub: right 1", "action": "MOVE_RIGHT"}',
        '{"reasoning": "stub: right 2", "action": "MOVE_RIGHT"}',
        '{"reasoning": "stub: right 3", "action": "MOVE_RIGHT"}',
        '{"reasoning": "stub: click pushee", "action": "CLICK", "target_position": [33, 21]}',
        '{"reasoning": "stub: right 4", "action": "MOVE_RIGHT"}',
        '{"reasoning": "stub: up to goal", "action": "MOVE_UP"}',
    ]

    def __init__(self):
        super().__init__(provider="openrouter", model="stub", reasoning_effort=None)
        self.step = 0

    def generate(self, system_prompt, user_prompt):
        if self.step >= len(self.SCRIPT):
            # Defensive: if called beyond the script, just no-op to MOVE_LEFT
            return '{"reasoning": "stub: out of script", "action": "MOVE_LEFT"}'
        response = self.SCRIPT[self.step]
        self.step += 1
        return response


assert LEVEL_ATTEMPTS == 2, f"Expected LEVEL_ATTEMPTS=2, got {LEVEL_ATTEMPTS}"

# ============================================================================
# Scenario 1: always-MOVE_LEFT, exhaust budget, retry should fire
# ============================================================================
stub = StubLLMClient()
result = run_agent(
    world_level="EASY",
    goal_level="EASY",
    mechanics_level="EASY",
    feedback_level="EASY",
    max_levels=1,
    turns_per_level=4,  # tiny budget — guarantees attempt 1 fails
    verbose=False,
    llm_client=stub,
    env_id="ka59simple",
)

retry_events = [e for e in result.history if e.get("type") == "level_retry"]
action_events = [e for e in result.history if e.get("type") == "action"]

print(f"won={result.won}  turns={result.turns}  levels_completed={result.levels_completed}")
print(f"  retry events    : {len(retry_events)}")
print(f"  action events   : {len(action_events)}")
print(f"  first action ev : {json.dumps(action_events[0], indent=2)[:400] if action_events else '(none)'}")

assert len(retry_events) >= 1, "Expected at least one level_retry event (attempt 1 should fail with 4-turn budget)"
assert all("attempt" in e and "attempt_turn" in e for e in action_events), \
    "Every action event must include `attempt` and `attempt_turn` fields"
assert result.turns > 4, f"Expected global_turn > turns_per_level=4 (multiple attempts); got {result.turns}"
print("\nSMOKE TEST PASSED")

# ============================================================================
# Scenario 2: winning stub on attempt 1, retry loop must NOT fire
# ============================================================================
win_stub = WinningStubLLMClient()
result2 = run_agent(
    world_level="EASY",
    goal_level="EASY",
    mechanics_level="EASY",
    feedback_level="EASY",
    max_levels=1,
    turns_per_level=16,  # plenty of room — winning sequence is 6 actions
    verbose=False,
    llm_client=win_stub,
    env_id="ka59simple",
)

retry_events2 = [e for e in result2.history if e.get("type") == "level_retry"]
action_events2 = [e for e in result2.history if e.get("type") == "action"]
complete_events2 = [e for e in result2.history if e.get("type") == "level_complete"]
win_events2 = [e for e in result2.history if e.get("type") == "win"]

print(f"\nwon={result2.won}  turns={result2.turns}  levels_completed={result2.levels_completed}")
print(f"  retry events    : {len(retry_events2)}")
print(f"  action events   : {len(action_events2)}")
print(f"  complete events : {len(complete_events2)}")
print(f"  win events      : {len(win_events2)}")

assert result2.won is True, f"Expected result.won=True (winning sequence should clear level 1); got {result2.won}"
assert result2.levels_completed == 1, f"Expected levels_completed=1; got {result2.levels_completed}"
assert len(retry_events2) == 0, \
    f"Expected NO level_retry events (attempt 1 should win); got {len(retry_events2)}"
assert len(complete_events2) >= 1, "Expected at least one level_complete event"
assert len(win_events2) >= 1, "Expected at least one win event"
assert all(e.get("attempt") == 1 for e in action_events2), \
    f"All action events must have attempt==1; got attempts={[e.get('attempt') for e in action_events2]}"
print("\nSMOKE TEST 2 PASSED")
