from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import audit_ka59_camera_ready as audit
from scripts import audit_ka59_gpt_complete_universe as gpt_universe
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

    def test_deepseek_empty_content_is_infrastructure(self) -> None:
        kind, _ = audit._fatal_errors({
            "errors": ["Turn 1: LLM/parse error — DeepSeek returned empty content."]
        })
        self.assertEqual(kind, "infrastructure_error")

    def test_dual_denominators_change_only_deepseek_none_mechanics(self) -> None:
        cells = {
            (cell["model"], cell["reasoning_effort"], cell["config"]): cell
            for cell in self.manifest["paper_cells"]
        }
        changed = [
            key for key, cell in cells.items()
            if cell["denominator_policies_differ"]
        ]
        key = ("deepseek-v4-pro", "none", "mechanics_hard")
        self.assertEqual(changed, [key])
        cell = cells[key]
        self.assertEqual(
            (
                cell["infrastructure_clean_scored_n"],
                cell["infrastructure_clean_scored_wins"],
                cell["infrastructure_clean_scored_losses"],
            ),
            (20, 4, 16),
        )
        self.assertEqual(
            (
                cell["strict_error_free_n"],
                cell["strict_error_free_wins"],
                cell["strict_error_free_losses"],
            ),
            (19, 4, 15),
        )
        self.assertEqual(cell["model_protocol_failures"], 1)
        self.assertEqual(cell["denominator_decision"], "NEEDS HUMAN DECISION")

    def test_every_prior_parse_error_has_evidence_based_review(self) -> None:
        review = self.manifest["parse_error_review"]
        self.assertEqual(review["prior_parse_error_trial_count"], 49)
        self.assertEqual(
            review["disposition_counts"],
            {
                "indeterminate": 2,
                "infrastructure_failure": 46,
                "model_protocol_failure": 1,
            },
        )
        model_failures = [
            trial for trial in review["trials"]
            if trial["classification"] == "model_protocol_failure"
        ]
        self.assertEqual(len(model_failures), 1)
        failure = model_failures[0]
        self.assertEqual(failure["parse_turns"], [100])
        self.assertEqual(failure["raw_response_excerpts"], ['{"'])
        self.assertEqual(failure["actions_after_last_parse"], 27)
        self.assertTrue(failure["runner_recovered_after_parse"])
        self.assertEqual(failure["final_outcome"], "loss")

    def test_complete_gpt_universe_corrects_effort_mapping_and_pools_all_batches(self) -> None:
        universe = gpt_universe.build_universe()
        cells = {
            (cell["reasoning_effort"], cell["config"]): cell
            for cell in universe["final_cells"]
        }
        self.assertEqual(universe["logical_trial_count"], 219)
        self.assertEqual(cells[("none", "world_hard")]["final_pooled_estimate"], "0/5 (0%)")
        self.assertEqual(cells[("none", "mechanics_hard")]["final_pooled_estimate"], "0/5 (0%)")
        self.assertEqual(cells[("none", "mechanics_hard_format_only")]["final_pooled_estimate"], "0/19 (0%)")
        self.assertEqual(cells[("medium", "world_hard")]["infrastructure_clean_included_n"], 0)
        self.assertEqual(cells[("medium", "mechanics_hard")]["infrastructure_clean_included_n"], 0)

    def test_may13_n1_is_outcome_blind_smoke_exclusion(self) -> None:
        universe = gpt_universe.build_universe()
        trial = next(
            trial for trial in universe["trials"]
            if trial["batch"] == "20260513T043402_656079"
        )
        self.assertEqual(trial["eligibility"], "EXCLUDED_EXPLICIT_SMOKE_OR_DEBUG")
        self.assertEqual(trial["turn_budget_per_attempt"], 8)
        self.assertTrue(trial["outcome_read_after_eligibility"])

    def test_complete_universe_generated_files_are_current(self) -> None:
        universe = gpt_universe.build_universe()
        self.assertEqual(
            gpt_universe.OUTPUT_JSON.read_text(),
            json.dumps(universe, indent=2, sort_keys=True) + "\n",
        )
        self.assertEqual(gpt_universe.OUTPUT_MD.read_text(), gpt_universe.render_markdown(universe))

    def test_every_cell_exposes_requested_policy_counts(self) -> None:
        required = {
            "nominal_historical_n",
            "infrastructure_clean_scored_n",
            "strict_error_free_n",
            "infrastructure_exclusions",
            "model_protocol_failures",
            "harness_exclusions",
            "indeterminate_parse_exclusions",
            "duplicates_excluded",
            "infrastructure_clean_scored_wins",
            "infrastructure_clean_scored_losses",
            "strict_error_free_wins",
            "strict_error_free_losses",
        }
        for cell in self.manifest["paper_cells"]:
            self.assertTrue(required.issubset(cell))

    def test_accepted_gpt_figure_lineage_is_explicit(self) -> None:
        none_base = self.cells[("openai/gpt-5.2", "none", "baseline")]
        medium_base = self.cells[("openai/gpt-5.2", "medium", "baseline")]
        medium_feedback = self.cells[("openai/gpt-5.2", "medium", "feedback_hard")]
        self.assertEqual(none_base["nominal_historical_n"], 5)
        self.assertIn("pooled none+medium", none_base["historical_display_value"])
        self.assertEqual(medium_base["nominal_historical_n"], 5)
        self.assertIn("8/10 (80%)", medium_base["historical_display_value"])
        self.assertIn("raw 4/5", medium_base["historical_display_value"])
        self.assertIn("10/10 (100%)", medium_feedback["historical_display_value"])
        self.assertIn("raw 5/5", medium_feedback["historical_display_value"])

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

    def test_format_only_control_is_a_distinct_protocol_from_mechanics_hard(self) -> None:
        """The ported control must never share an identity with the fallthrough."""
        from ka59_game.prompts import build_system_prompt

        self.assertNotEqual(
            build_system_prompt("EASY", "HARD_FORMAT_ONLY"),
            build_system_prompt("EASY", "HARD"),
            "the real format-only control is not present; the ported prompt is missing",
        )
        hard = runner.protocol_identity(
            "openrouter", "deepseek/deepseek-v4-pro", "none", "mechanics_hard"
        )
        fmt = runner.protocol_identity(
            "openrouter", "deepseek/deepseek-v4-pro", "none",
            "mechanics_hard_format_only",
        )
        self.assertNotEqual(hard["protocol_id"], fmt["protocol_id"])
        self.assertTrue(hard["reproduces_accepted_prompt"])
        self.assertFalse(fmt["reproduces_accepted_prompt"])

    def test_ported_control_inherits_no_historical_trials(self) -> None:
        """The 20 historical fallthrough trials must not fill the new control."""
        manifest = audit.build_manifest(20)
        identity = runner.protocol_identity(
            "openrouter", "deepseek/deepseek-v4-pro", "none",
            "mechanics_hard_format_only",
        )
        self.assertEqual(runner._accepted_count(manifest, identity), (0, 0, 0))

    def test_upstream_provider_pin_changes_the_protocol_id(self) -> None:
        unpinned = runner.protocol_identity(
            "openrouter", "deepseek/deepseek-v4-pro", "none", "baseline"
        )
        pinned = runner.protocol_identity(
            "openrouter", "deepseek/deepseek-v4-pro", "none", "baseline",
            upstream_provider="DigitalOcean",
        )
        self.assertNotEqual(unpinned["protocol_id"], pinned["protocol_id"])
        self.assertEqual(pinned["upstream_provider"], "DigitalOcean")

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


class PromptIdentityPoolingTests(unittest.TestCase):
    """Pooling must follow the prompt bytes, never the config label."""

    def test_accepted_format_only_trials_ran_the_mechanics_hard_prompt(self):
        """The pooled cell rests on what the accepted runs sent, not on today's code."""
        from ka59_game.prompts import build_system_prompt

        fingerprint = audit._system_prompt_fingerprint("mechanics_hard_format_only")
        self.assertEqual(fingerprint["historical_effective_mechanics"], "HARD")
        self.assertEqual(
            fingerprint["system_prompt_sha256"],
            audit._system_prompt_fingerprint("mechanics_hard")["system_prompt_sha256"],
        )

    def test_working_tree_divergence_is_recorded_not_silently_pooled(self):
        """Once the real control exists, new trials under that label are a new protocol."""
        from ka59_game.prompts import build_system_prompt

        fingerprint = audit._system_prompt_fingerprint("mechanics_hard_format_only")
        real_control_present = (
            build_system_prompt("EASY", "HARD_FORMAT_ONLY")
            != build_system_prompt("EASY", "HARD")
        )
        self.assertEqual(fingerprint["current_revision_differs"], real_control_present)

    def test_configs_differing_only_in_world_or_feedback_are_not_pooled(self):
        groups = audit.prompt_identity_groups()
        for group in groups["identical_treatment_groups"]:
            configs = set(group["configs"])
            self.assertNotIn(
                "baseline", configs,
                "baseline shares a system prompt with world_hard/feedback_hard but "
                "differs in the observation stream; it must never be pooled with them.",
            )

    def test_pooled_gpt_none_mechanics_reaches_target_without_new_trials(self):
        manifest = audit.build_manifest(20)
        pooled = {
            (cell["model"], cell["reasoning_effort"]): cell
            for cell in manifest["prompt_identity_pooling"]["pooled_cells"]
        }
        cell = pooled[("openai/gpt-5.2", "none")]
        self.assertEqual(cell["pooled_configs"],
                         ["mechanics_hard", "mechanics_hard_format_only"])
        self.assertEqual(cell["infrastructure_clean_scored_n"], 24)
        self.assertGreaterEqual(cell["infrastructure_clean_scored_n"], 20)

    def test_pooled_components_sum_to_the_pooled_denominator(self):
        manifest = audit.build_manifest(20)
        by_cell = {
            (c["model"], c["reasoning_effort"], c["config"]): c
            for c in manifest["paper_cells"]
        }
        for cell in manifest["prompt_identity_pooling"]["pooled_cells"]:
            expected = sum(
                by_cell[(cell["model"], cell["reasoning_effort"], config)][
                    "infrastructure_clean_scored_n"
                ]
                for config in cell["pooled_configs"]
            )
            self.assertEqual(cell["infrastructure_clean_scored_n"], expected)


class SmokeAndIndexTests(unittest.TestCase):
    def test_smoke_exits_zero_and_reports_the_active_prompt_regime(self):
        self.assertEqual(runner.main(["--smoke"]), 0)
        report = runner.smoke_report()
        self.assertIn(
            report["format_only_prompt_regime"],
            {"historical_fallthrough_to_MECHANICS_HARD",
             "paper_intended_action_protocol_retained"},
        )

    def test_index_lists_every_recorded_run_directory(self):
        text = runner.build_index()
        for directory in sorted(runner.RUNTIME_ROOT.glob("*")):
            if directory.is_dir() and list(directory.glob("run_*.json")):
                self.assertIn(directory.name, text)


class OpenRouterTransportTests(unittest.TestCase):
    """Truncated bodies must be retried, not allowed to void a whole trial."""

    def _client(self):
        from ka59_game.llm_client import LLMClient
        return LLMClient(provider="openrouter", model="m", reasoning_effort="medium",
                         upstream_provider="DigitalOcean,StreamLake")

    def _response(self, content):
        msg = mock.Mock(content=content)
        choice = mock.Mock(message=msg, finish_reason="length" if content is None else "stop")
        return mock.Mock(choices=[choice], usage=None, provider="StreamLake")

    def test_empty_content_is_retried_and_recovers(self):
        client = self._client()
        api = mock.Mock()
        api.chat.completions.create.side_effect = [
            self._response(None), self._response(None), self._response('{"action":"MOVE_UP"}')
        ]
        with mock.patch.object(type(client), "_openrouter_client", return_value=api), \
             mock.patch("time.sleep"):
            out = client._generate_openrouter("sys", "user")
        self.assertEqual(out, '{"action":"MOVE_UP"}')
        self.assertEqual(api.chat.completions.create.call_count, 3)

    def test_persistent_empty_content_still_fails_and_names_the_upstream(self):
        client = self._client()
        api = mock.Mock()
        api.chat.completions.create.return_value = self._response(None)
        with mock.patch.object(type(client), "_openrouter_client", return_value=api), \
             mock.patch("time.sleep"):
            with self.assertRaises(ValueError) as ctx:
                client._generate_openrouter("sys", "user")
        self.assertIn("StreamLake", str(ctx.exception))
        self.assertIn("finish_reason=length", str(ctx.exception))

    def test_cap_clears_the_measured_per_turn_draw(self):
        from ka59_game.llm_client import OPENROUTER_MAX_TOKENS
        self.assertGreaterEqual(OPENROUTER_MAX_TOKENS, 4 * 10_000)


class UpstreamRoutingTests(unittest.TestCase):
    def test_sort_and_pin_are_both_sent(self):
        from ka59_game.llm_client import LLMClient
        client = LLMClient(provider="openrouter", model="m", reasoning_effort="medium",
                           upstream_provider="A,B", upstream_sort="throughput")
        api = mock.Mock()
        msg = mock.Mock(content='{"action":"MOVE_UP"}')
        api.chat.completions.create.return_value = mock.Mock(
            choices=[mock.Mock(message=msg, finish_reason="stop")], usage=None, provider="A")
        with mock.patch.object(type(client), "_openrouter_client", return_value=api):
            client._generate_openrouter("s", "u")
        sent = api.chat.completions.create.call_args.kwargs["extra_body"]["provider"]
        self.assertEqual(sent, {"only": ["A", "B"], "sort": "throughput"})

    def test_sort_changes_the_protocol_id(self):
        a = runner.protocol_identity("openrouter", "deepseek/deepseek-v4-pro", "medium",
                                     "baseline", "A,B")
        b = runner.protocol_identity("openrouter", "deepseek/deepseek-v4-pro", "medium",
                                     "baseline", "A,B", "throughput")
        self.assertNotEqual(a["protocol_id"], b["protocol_id"])
class GptCrossTransportPoolingTests(unittest.TestCase):
    """GPT-5.2 accepted OpenRouter trials pool into direct-API runs."""

    def test_direct_api_slug_resolves_to_the_accepted_model(self):
        manifest = audit.build_manifest(20)
        identity = runner.protocol_identity("openai", "gpt-5.2", "none", "baseline")
        n, wins, _ = runner._accepted_count(manifest, identity)
        self.assertEqual((n, wins), (5, 5))

    def test_ported_control_still_inherits_nothing(self):
        manifest = audit.build_manifest(20)
        identity = runner.protocol_identity(
            "openai", "gpt-5.2", "none", "mechanics_hard_format_only"
        )
        self.assertEqual(runner._accepted_count(manifest, identity), (0, 0, 0))

    def test_pooled_mechanics_cell_needs_no_new_trials(self):
        plan = runner.build_plan(provider="openai", model="gpt-5.2", effort="none",
                                 configs=["mechanics_hard"], target_n=20)
        self.assertEqual(plan["cells"][0]["current_valid_n"], 24)
        self.assertEqual(plan["cells"][0]["deficit"], 0)

    def test_cross_transport_decision_is_recorded_with_its_risk(self):
        manifest = audit.build_manifest(20)
        entry = manifest["cross_transport_pooling"]["openai/gpt-5.2"]
        self.assertTrue(entry["disclosure_required_in_paper"])
        self.assertIn("weaker_than_deepseek_pooling", entry)


class QuotaVsRateLimitTests(unittest.TestCase):
    """A quota 429 is terminal; a rate-limit 429 is worth retrying."""

    def _client(self):
        from ka59_game.llm_client import LLMClient
        return LLMClient(provider="openrouter", model="m", reasoning_effort="none")

    def _fail(self, message):
        api = mock.Mock()
        api.chat.completions.create.side_effect = Exception(message)
        return api

    def test_quota_exhaustion_is_not_retried(self):
        client, api = self._client(), self._fail(
            "Error code: 429 - {'code': 'credit_balance_exhausted', "
            "'message': 'You have no credits remaining.'}")
        with mock.patch.object(type(client), "_openrouter_client", return_value=api), \
             mock.patch("time.sleep"):
            with self.assertRaises(Exception):
                client._generate_openrouter("s", "u")
        self.assertEqual(api.chat.completions.create.call_count, 1)

    def test_rate_limit_429_is_still_retried(self):
        client, api = self._client(), self._fail("Error code: 429 - rate limit exceeded")
        with mock.patch.object(type(client), "_openrouter_client", return_value=api), \
             mock.patch("time.sleep"):
            with self.assertRaises(Exception):
                client._generate_openrouter("s", "u")
        self.assertEqual(api.chat.completions.create.call_count, 4)
