from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import audit_ka59_camera_ready as audit
from scripts import run_ka59_camera_ready as runner


class CameraReadyAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = audit.build_manifest(20)
        cls.cells = {
            (cell["model"], cell["reasoning_effort"], cell["config"]): cell
            for cell in cls.manifest["paper_cells"]
        }

    def test_raw_counts_lock_strict_camera_ready_denominators(self) -> None:
        expected = {
            ("openai/gpt-5.2", "none", "baseline"): (5, 5, 0),
            ("openai/gpt-5.2", "none", "feedback_hard"): (5, 3, 0),
            ("openai/gpt-5.2", "medium", "baseline"): (0, 0, 5),
            ("deepseek-v4-pro", "none", "baseline"): (20, 12, 0),
            ("deepseek-v4-pro", "none", "mechanics_hard"): (19, 4, 1),
            ("deepseek-v4-pro", "medium", "baseline"): (0, 0, 11),
        }
        for key, (valid_n, wins, errors) in expected.items():
            with self.subTest(key=key):
                cell = self.cells[key]
                self.assertEqual(cell["valid_n"], valid_n)
                self.assertEqual(cell["wins"], wins)
                self.assertEqual(cell["errors_excluded"], errors)

    def test_errors_and_duplicates_are_never_losses(self) -> None:
        excluded = {
            "infrastructure_error", "parse_error", "environment_error",
            "incomplete", "duplicate",
        }
        for trial in self.manifest["candidate_trials"]:
            if trial["status"] in excluded:
                self.assertIsNone(trial["won"])

    def test_fatal_error_classifier_separates_model_invalid_actions(self) -> None:
        kind, _ = audit._fatal_errors({
            "errors": ["Turn 1: LLM/parse error — Error code: 402 - Insufficient Balance"]
        })
        self.assertEqual(kind, "infrastructure_error")
        kind, _ = audit._fatal_errors({
            "errors": ["Turn 1: LLM/parse error — No JSON object found"]
        })
        self.assertEqual(kind, "parse_error")
        kind, _ = audit._fatal_errors({"errors": ["Turn 1: unknown action 'SPIN'"]})
        self.assertIsNone(kind)

    def test_generated_files_are_current(self) -> None:
        self.assertEqual(
            audit.MANIFEST_PATH.read_text(),
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
        )
        self.assertEqual(audit.TRUTH_PATH.read_text(), audit.render_truth(self.manifest))


class CameraReadyRunnerTests(unittest.TestCase):
    def test_target_n_means_deficit_not_additional_runs(self) -> None:
        plan = runner.build_plan(
            provider="openrouter",
            model="openai/gpt-5.2",
            effort="none",
            configs=["baseline"],
            target_n=20,
        )
        self.assertEqual(plan["cells"][0]["current_valid_n"], 5)
        self.assertEqual(plan["cells"][0]["deficit"], 15)
        self.assertEqual(plan["total_trials_to_run"], 15)
        self.assertEqual(plan["external_calls"], 0)

    def test_plan_does_not_construct_provider_client(self) -> None:
        with mock.patch("openai.OpenAI", side_effect=AssertionError("external client constructed")):
            code = runner.main([
                "--provider", "openrouter",
                "--model", "openai/gpt-5.2",
                "--reasoning-effort", "none",
                "--target-n", "5",
                "--config", "baseline",
                "--plan",
            ])
        self.assertEqual(code, 0)

    def test_incompatible_identity_refuses_pooling(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(runner, "RUNTIME_ROOT", Path(temp)):
                identity = runner.protocol_identity(
                    "openrouter", "openai/gpt-5.2", "none", "baseline"
                )
                output = Path(temp) / identity["protocol_id"]
                output.mkdir()
                (output / "run_bad.json").write_text(json.dumps({
                    "protocol_identity": {**identity, "turns_per_attempt": 32},
                    "won": False,
                    "errors": [],
                }))
                with self.assertRaisesRegex(ValueError, "Incompatible experimental identities"):
                    runner.build_plan(
                        provider="openrouter",
                        model="openai/gpt-5.2",
                        effort="none",
                        configs=["baseline"],
                        target_n=20,
                    )

    def test_historical_format_only_prompt_is_locked(self) -> None:
        from ka59_game.prompts import build_system_prompt

        self.assertEqual(
            build_system_prompt("EASY", "HARD_FORMAT_ONLY"),
            build_system_prompt("EASY", "HARD"),
        )

    def test_provider_failure_aborts_trial_instead_of_scoring_loss(self) -> None:
        from ka59_game.experiment import run_agent

        class FailingClient:
            calls = 0

            def generate(self, _system: str, _user: str) -> str:
                self.calls += 1
                raise RuntimeError("synthetic provider failure")

            def parse_json(self, _text: str) -> dict:
                raise AssertionError("parse_json should not be reached")

            def get_usage_summary(self) -> dict:
                return {}

        client = FailingClient()
        result = run_agent(
            provider="openrouter",
            model="test-model",
            max_levels=1,
            turns_per_level=64,
            llm_client=client,
            reasoning_effort="none",
            env_id="ka59simple",
            verbose=False,
            save=False,
        )
        self.assertFalse(result.won)
        self.assertEqual(result.turns, 1)
        self.assertTrue(any("synthetic provider failure" in error for error in result.errors))
        self.assertEqual(client.calls, 1)


if __name__ == "__main__":
    unittest.main()
