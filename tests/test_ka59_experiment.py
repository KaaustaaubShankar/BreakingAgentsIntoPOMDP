"""
End-to-end smoke test for ka59_ref.experiment.run_agent using a mock
LLMClient. No network calls, no API key required.
"""

from __future__ import annotations

import json
from typing import Any

from ka59_ref.experiment import RunResult, run_agent
from ka59_ref.llm_client import LLMClient


class _ScriptedMockClient(LLMClient):
    """LLMClient whose generate() replays a pre-recorded script of replies."""

    def __init__(self, script: list[str]) -> None:
        self.provider = "mock"
        self.model = "mock-test"
        self._script = list(script)
        self._default = self._script[-1] if self._script else '{"action": "MOVE_RIGHT"}'
        self.reset_usage()

    def generate(self, system_prompt: str, user_prompt: str) -> str:  # noqa: ARG002
        if self._script:
            return self._script.pop(0)
        return self._default

    def parse_json(self, text: str) -> dict[str, Any]:
        return json.loads(text)


def test_run_agent_wins_transfer_wall_push_in_one_move_with_scripted_client() -> None:
    """
    A scripted agent that plays MOVE_RIGHT once wins transfer_wall_push — the
    single MOVE_RIGHT pushes the block through the wall_transfer. This proves
    the end-to-end harness (prompts → client → env → feedback → win check)
    works.
    """
    client = _ScriptedMockClient([
        '{"reasoning": "push right to test the wall", "action": "MOVE_RIGHT"}',
        # Understanding reflection at end:
        '{"goal_understanding": "push a block past target_x", "mechanics_understanding": "some walls let pushed blocks pass"}',
    ])

    result = run_agent(
        scenario="transfer_wall_push",
        world_level="EASY",
        goal_level="EASY",
        mechanics_level="EASY",
        feedback_level="EASY",
        provider="mock",
        model="mock-test",
        verbose=False,
        llm_client=client,
    )

    assert isinstance(result, RunResult)
    assert result.won is True
    assert result.turns == 1
    assert result.levels_completed == 1
    assert result.invalid_actions == 0
    assert result.push_events >= 1
    assert result.config["scenario"] == "transfer_wall_push"
    assert result.config["world"] == "EASY"


def test_run_agent_counts_invalid_and_select_actions() -> None:
    client = _ScriptedMockClient([
        '{"action": "NONSENSE"}',                          # invalid
        '{"action": "SELECT"}',                            # SELECT missing target_id → invalid
        '{"action": "SELECT", "target_id": "pb"}',         # valid SELECT
        '{"action": "MOVE_RIGHT"}',                        # pb now selected
        '{"goal_understanding": "test", "mechanics_understanding": "test"}',
    ])
    result = run_agent(
        scenario="transfer_wall_push",
        provider="mock",
        model="mock-test",
        max_turns=4,
        verbose=False,
        llm_client=client,
    )
    assert result.invalid_actions == 2
    assert result.select_actions == 1


def test_run_agent_world_hard_preserves_win_detection() -> None:
    """WORLD_HARD hides non-selected positions in the observation dict, but
    the harness still detects the win via true engine positions."""
    client = _ScriptedMockClient([
        '{"action": "MOVE_RIGHT"}',
        '{"goal_understanding": "test", "mechanics_understanding": "test"}',
    ])
    result = run_agent(
        scenario="transfer_wall_push",
        world_level="HARD",
        provider="mock",
        model="mock-test",
        verbose=False,
        llm_client=client,
    )
    assert result.won is True
    assert result.config["world"] == "HARD"


def test_run_agent_losses_on_solid_wall_push_blocked() -> None:
    """Contrastive loss row: no number of MOVE_RIGHT wins this scenario."""
    script = ['{"action": "MOVE_RIGHT"}'] * 20 + [
        '{"goal_understanding": "blocked", "mechanics_understanding": "wall stopped push"}'
    ]
    client = _ScriptedMockClient(script)
    result = run_agent(
        scenario="solid_wall_push_blocked",
        provider="mock",
        model="mock-test",
        verbose=False,
        llm_client=client,
    )
    assert result.won is False
