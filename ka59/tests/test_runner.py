"""
test_runner.py — pytest suite for the multi-env runner.

Tests are fully mocked — no arc_agi, no LLM calls, no network.
Coverage:
  1. Dispatcher routes to correct adapter for each env
  2. UnifiedRunResult fields are correctly populated from each env's RunResult
  3. AblationRow aggregation (win_rate, avg_turns, relative_difficulty)
  4. Josh table renders without error (spot-check columns)
  5. BP35 hypo trace extracts gravity-flip turns
  6. LS20 hypo trace extracts goal-activation turns
  7. KA59 hypo trace extracts passable-wall discoveries
  8. Sample table (--sample flag) is consistent
"""

from __future__ import annotations

import sys
import os
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Any

# Make ka59/ importable when tests run from repo root or from ka59/
_KA59_DIR = Path(__file__).parents[1]
if str(_KA59_DIR) not in sys.path:
    sys.path.insert(0, str(_KA59_DIR))

import pytest

from unified_result import UnifiedRunResult, AblationRow, build_rows_with_rel_diff, print_josh_table


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — canned per-env RunResult objects
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FakeBP35Result:
    config: dict = field(default_factory=lambda: {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"})
    won: bool = True
    turns: int = 42
    levels_completed: int = 1
    provider: str = "openrouter"
    model: str = "test/model"
    vision: bool = False
    errors: list = field(default_factory=list)
    history: list = field(default_factory=list)
    understanding: dict = field(default_factory=dict)
    timestamp: str = "2026-04-22T00:00:00Z"
    invalid_actions: int = 3
    click_actions: int = 2
    gravity_flips: int = 1
    undos: int = 0


@dataclass
class FakeLS20Result:
    config: dict = field(default_factory=lambda: {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"})
    won: bool = False
    turns: int = 100
    levels_completed: int = 0
    provider: str = "openrouter"
    model: str = "test/model"
    vision: bool = False
    errors: list = field(default_factory=list)
    history: list = field(default_factory=list)
    understanding: dict = field(default_factory=dict)
    timestamp: str = "2026-04-22T00:00:00Z"
    wall_collisions: int = 7
    goals_ever_activated: int = 2


@dataclass
class FakeKA59Result:
    config: dict = field(default_factory=lambda: {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"})
    scenario: str = "transfer_wall_push"
    won: bool = False
    turns: int = 64
    levels_completed: int = 0
    provider: str = "openrouter"
    model: str = "test/model"
    errors: list = field(default_factory=list)
    history: list = field(default_factory=list)
    understanding: dict = field(default_factory=dict)
    timestamp: str = "2026-04-22T00:00:00Z"
    blocked_count: int = 38
    passable_walls_found: int = 0
    select_actions: int = 2
    moved_count: int = 26


# ─────────────────────────────────────────────────────────────────────────────
# Helper — patch runner's lazy imports
# ─────────────────────────────────────────────────────────────────────────────

_CONFIG = {"world": "EASY", "goal": "EASY", "mechanics": "EASY", "feedback": "EASY"}


def _patch_bp35(fake: FakeBP35Result):
    """Return context manager that patches bp35_runner.run_agent."""
    mock_module = MagicMock()
    mock_module.run_agent.return_value = fake
    mock_module.RunResult = FakeBP35Result
    return patch.dict("sys.modules", {"envs.bp35_runner": mock_module})


def _patch_ls20(fake: FakeLS20Result):
    mock_module = MagicMock()
    mock_module.run_agent.return_value = fake
    mock_module.RunResult = FakeLS20Result
    return patch.dict("sys.modules", {"envs.ls20_runner": mock_module})


def _patch_ka59(fake: FakeKA59Result):
    mock_module = MagicMock()
    mock_module.run_agent.return_value = fake
    mock_module.RunResult = FakeKA59Result
    return patch.dict("sys.modules", {"envs.ka59_llm_runner": mock_module})


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Dispatcher routing
# ─────────────────────────────────────────────────────────────────────────────

class TestDispatcher:
    def test_bp35_route(self):
        fake = FakeBP35Result()
        with _patch_bp35(fake):
            import runner as r
            # Force module reload so new patches are picked up
            import importlib; importlib.reload(r)
            result = r.run_one("bp35", "baseline", _CONFIG, "openrouter", "test/model")
        assert result.env == "bp35"

    def test_ls20_route(self):
        fake = FakeLS20Result()
        with _patch_ls20(fake):
            import runner as r
            import importlib; importlib.reload(r)
            result = r.run_one("ls20", "baseline", _CONFIG, "openrouter", "test/model")
        assert result.env == "ls20"

    def test_ka59_route(self):
        fake = FakeKA59Result()
        with _patch_ka59(fake):
            import runner as r
            import importlib; importlib.reload(r)
            result = r.run_one("ka59", "baseline", _CONFIG, "openrouter", "test/model")
        assert result.env == "ka59"

    def test_unknown_env_raises(self):
        import runner as r
        with pytest.raises(ValueError, match="Unknown env"):
            r.run_one("zendo", "baseline", _CONFIG, "openrouter", "test/model")

    def test_env_case_insensitive(self):
        fake = FakeBP35Result()
        with _patch_bp35(fake):
            import runner as r
            import importlib; importlib.reload(r)
            result = r.run_one("BP35", "baseline", _CONFIG, "openrouter", "test/model")
        assert result.env == "bp35"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — UnifiedRunResult field mapping
# ─────────────────────────────────────────────────────────────────────────────

class TestUnifiedResultMapping:
    def test_bp35_fields(self):
        fake = FakeBP35Result(won=True, turns=42, invalid_actions=3, gravity_flips=1)
        with _patch_bp35(fake):
            import runner as r
            import importlib; importlib.reload(r)
            result = r.run_one("bp35", "baseline", _CONFIG, "openrouter", "test/model")
        assert result.won is True
        assert result.turns == 42
        assert result.invalid_clicks == 3   # invalid_actions → invalid_clicks
        assert result.flips == 1            # gravity_flips → flips
        assert result.env == "bp35"

    def test_ls20_fields(self):
        fake = FakeLS20Result(won=False, turns=100, wall_collisions=7, goals_ever_activated=2)
        with _patch_ls20(fake):
            import runner as r
            import importlib; importlib.reload(r)
            result = r.run_one("ls20", "baseline", _CONFIG, "openrouter", "test/model")
        assert result.won is False
        assert result.turns == 100
        assert result.invalid_clicks == 7   # wall_collisions → invalid_clicks
        assert result.flips == 2            # goals_ever_activated → flips
        assert result.env == "ls20"

    def test_ka59_fields(self):
        fake = FakeKA59Result(blocked_count=38, passable_walls_found=0, turns=64)
        with _patch_ka59(fake):
            import runner as r
            import importlib; importlib.reload(r)
            result = r.run_one("ka59", "baseline", _CONFIG, "openrouter", "test/model")
        assert result.turns == 64
        assert result.invalid_clicks == 38  # blocked_count → invalid_clicks
        assert result.flips == 0            # passable_walls_found → flips
        assert result.env == "ka59"

    def test_model_preserved(self):
        fake = FakeBP35Result()
        fake.model = "anthropic/claude-sonnet-4-6"
        with _patch_bp35(fake):
            import runner as r
            import importlib; importlib.reload(r)
            result = r.run_one("bp35", "baseline", _CONFIG, "openrouter", "anthropic/claude-sonnet-4-6")
        assert result.model == "anthropic/claude-sonnet-4-6"

    def test_hypo_trace_present(self):
        fake = FakeBP35Result()
        with _patch_bp35(fake):
            import runner as r
            import importlib; importlib.reload(r)
            result = r.run_one("bp35", "baseline", _CONFIG, "openrouter", "test/model")
        assert result.hypo_trace is not None
        assert result.hypo_trace["env"] == "bp35"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — AblationRow aggregation
# ─────────────────────────────────────────────────────────────────────────────

class TestAblationRow:
    def _make_runs(self, env: str, won_list: list[bool], turns_list: list[int]) -> list[UnifiedRunResult]:
        return [
            UnifiedRunResult(
                env=env,
                config_name="baseline",
                config=_CONFIG,
                provider="openrouter",
                model="test/model",
                won=won,
                turns=t,
                levels_completed=1 if won else 0,
                invalid_clicks=2,
                flips=1,
            )
            for won, t in zip(won_list, turns_list)
        ]

    def test_win_rate(self):
        runs = self._make_runs("bp35", [True, False, True], [40, 128, 60])
        row = AblationRow.from_runs(runs)
        assert row.wins == 2
        assert abs(row.win_rate - 2/3) < 0.001

    def test_avg_turns(self):
        runs = self._make_runs("ls20", [False, False], [80, 100])
        row = AblationRow.from_runs(runs)
        assert row.avg_turns == 90.0

    def test_relative_difficulty(self):
        runs = self._make_runs("bp35", [True, True], [60, 80])
        row = AblationRow.from_runs(runs, baseline_avg_turns=70.0)
        # avg_turns=70, baseline=70 → rel=1.0
        assert row.relative_difficulty == 1.0

    def test_relative_difficulty_none_when_no_baseline(self):
        runs = self._make_runs("bp35", [True], [50])
        row = AblationRow.from_runs(runs, baseline_avg_turns=None)
        assert row.relative_difficulty is None

    def test_empty_runs_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            AblationRow.from_runs([])

    def test_build_rows_with_rel_diff(self):
        baseline_runs = self._make_runs("bp35", [True, True, True], [60, 60, 60])
        hard_runs = self._make_runs("bp35", [False, False, False], [128, 128, 128])
        hard_runs_r = [UnifiedRunResult(
            env="bp35", config_name="world_hard", config=_CONFIG,
            provider="openrouter", model="test/model",
            won=False, turns=128, levels_completed=0,
            invalid_clicks=5, flips=0,
        ) for _ in range(3)]

        baseline_row = AblationRow.from_runs(baseline_runs)
        baseline_row.config_name = "baseline"
        hard_row = AblationRow.from_runs(hard_runs_r)
        hard_row.config_name = "world_hard"

        rows = build_rows_with_rel_diff([baseline_row, hard_row])
        baseline_out = next(r for r in rows if r.config_name == "baseline")
        hard_out = next(r for r in rows if r.config_name == "world_hard")

        assert baseline_out.relative_difficulty == 1.0
        assert hard_out.relative_difficulty == pytest.approx(128 / 60, rel=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Josh table renders correctly
# ─────────────────────────────────────────────────────────────────────────────

class TestJoshTable:
    def _sample_rows(self) -> list[AblationRow]:
        envs_configs = [
            ("bp35", "baseline", 2, 3, 68.0, 0.67, 3.3, 1.0),
            ("bp35", "world_hard", 0, 3, 128.0, 0.0, 8.7, 0.0),
            ("ls20", "baseline", 1, 3, 52.0, 0.33, 6.7, 0.7),
            ("ka59", "baseline", 0, 3, 64.0, 0.0, 38.7, 0.0),
        ]
        rows = []
        for (env, cfg, wins, n, avg_t, avg_l, avg_i, avg_f) in envs_configs:
            rows.append(AblationRow(
                env=env,
                config_name=cfg,
                config=_CONFIG,
                provider="openrouter",
                model="test/model",
                n_trials=n,
                wins=wins,
                win_rate=wins/n,
                avg_turns=avg_t,
                avg_levels=avg_l,
                avg_invalid_clicks=avg_i,
                avg_flips=avg_f,
                relative_difficulty=None,
            ))
        return build_rows_with_rel_diff(rows)

    def test_table_renders(self, capsys):
        rows = self._sample_rows()
        print_josh_table(rows)
        captured = capsys.readouterr()
        assert "Win%" in captured.out
        assert "Turns" in captured.out
        assert "bp35" in captured.out
        assert "ls20" in captured.out
        assert "ka59" in captured.out

    def test_table_has_header(self, capsys):
        rows = self._sample_rows()
        print_josh_table(rows, title="TEST TABLE")
        captured = capsys.readouterr()
        assert "TEST TABLE" in captured.out

    def test_win_pct_column(self, capsys):
        rows = self._sample_rows()
        print_josh_table(rows)
        captured = capsys.readouterr()
        # 2/3 wins = 67%
        assert "67%" in captured.out

    def test_legend_present(self, capsys):
        rows = self._sample_rows()
        print_josh_table(rows)
        captured = capsys.readouterr()
        assert "InvClk" in captured.out
        assert "Column legend" in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — BP35 hypo trace
# ─────────────────────────────────────────────────────────────────────────────

class TestBP35HypoTrace:
    def test_gravity_flip_extracted(self):
        import runner as r
        fake = FakeBP35Result()
        # Simulate history with gravity changes
        fake.history = [
            {"type": "action", "turn": 5, "action": "CLICK",
             "state_before": {"player": {"gravity": "down"}}},
            {"type": "action", "turn": 6, "action": "MOVE_RIGHT",
             "state_before": {"player": {"gravity": "up"}}},  # flipped
        ]
        trace = r._bp35_hypo_trace(fake)
        assert "gravity_flip_turns" in trace
        assert 6 in trace["gravity_flip_turns"]

    def test_invalid_turns_extracted(self):
        import runner as r
        fake = FakeBP35Result()
        fake.history = [
            {"type": "invalid_action", "turn": 3},
            {"type": "invalid_action", "turn": 7},
        ]
        trace = r._bp35_hypo_trace(fake)
        assert 3 in trace["invalid_click_turns"]
        assert 7 in trace["invalid_click_turns"]

    def test_hypothesis_field_present(self):
        import runner as r
        fake = FakeBP35Result()
        trace = r._bp35_hypo_trace(fake)
        assert "hypothesis" in trace


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — LS20 hypo trace
# ─────────────────────────────────────────────────────────────────────────────

class TestLS20HypoTrace:
    def test_goal_activation_extracted(self):
        import runner as r
        fake = FakeLS20Result()
        fake.history = [
            {"type": "action", "turn": 10,
             "state_before": {"goals": {"activated_list": [False, False]}}},
            {"type": "action", "turn": 15,
             "state_before": {"goals": {"activated_list": [True, False]}}},  # 1 activated
        ]
        trace = r._ls20_hypo_trace(fake)
        assert 15 in trace["goal_activation_turns"]

    def test_hypothesis_modifier_discovery(self):
        import runner as r
        fake = FakeLS20Result()
        fake.history = [
            {"type": "action", "turn": 5,
             "state_before": {"goals": {"activated_list": [True]}}},
        ]
        trace = r._ls20_hypo_trace(fake)
        assert "modifier" in trace["hypothesis"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — KA59 hypo trace
# ─────────────────────────────────────────────────────────────────────────────

class TestKA59HypoTrace:
    def test_passable_walls_0(self):
        import runner as r
        fake = FakeKA59Result(passable_walls_found=0)
        trace = r._ka59_hypo_trace(fake)
        assert trace["passable_walls_found"] == 0
        assert "NOT discovered" in trace["hypothesis"]

    def test_passable_walls_found(self):
        import runner as r
        fake = FakeKA59Result(passable_walls_found=1)
        trace = r._ka59_hypo_trace(fake)
        assert trace["passable_walls_found"] == 1
        assert "discovered" in trace["hypothesis"]
        assert "NOT" not in trace["hypothesis"]

    def test_blocked_count_in_trace(self):
        import runner as r
        fake = FakeKA59Result(blocked_count=38)
        trace = r._ka59_hypo_trace(fake)
        assert trace["blocked_count"] == 38


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 — Sample table (ablation --sample)
# ─────────────────────────────────────────────────────────────────────────────

class TestSampleTable:
    def test_sample_rows_all_envs(self):
        from ablation import _make_sample_rows
        rows = _make_sample_rows()
        envs = {r.env for r in rows}
        assert "bp35" in envs
        assert "ls20" in envs
        assert "ka59" in envs

    def test_sample_has_baseline(self):
        from ablation import _make_sample_rows
        rows = _make_sample_rows()
        baseline_rows = [r for r in rows if r.config_name == "baseline"]
        assert len(baseline_rows) == 3   # one per env

    def test_sample_relative_difficulty_set(self):
        from ablation import _make_sample_rows
        rows = _make_sample_rows()
        # Baseline should have rel_diff = 1.0
        for r in rows:
            if r.config_name == "baseline":
                assert r.relative_difficulty == pytest.approx(1.0, rel=0.01)

    def test_sample_table_renders(self, capsys):
        from ablation import _make_sample_rows
        rows = _make_sample_rows()
        print_josh_table(rows, title="SAMPLE")
        captured = capsys.readouterr()
        assert "SAMPLE" in captured.out
        assert "RelDiff" in captured.out
