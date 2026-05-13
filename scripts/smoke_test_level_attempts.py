"""Smoke-test LEVEL_ATTEMPTS=2 retry path in ka59_game.experiment.run_agent.

Injects a stub LLMClient that always returns MOVE_LEFT (which on ka59simple
either bumps the wall or wastes turns), forces the agent to exhaust its
turn budget on attempt 1, and verifies:
  - history contains a `level_retry` event
  - action log entries have `attempt` and `attempt_turn` fields
  - result.turns counts cumulative across attempts
  - env.step(RESET) was called between attempts (level_retry event proves it)

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


assert LEVEL_ATTEMPTS == 2, f"Expected LEVEL_ATTEMPTS=2, got {LEVEL_ATTEMPTS}"

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
