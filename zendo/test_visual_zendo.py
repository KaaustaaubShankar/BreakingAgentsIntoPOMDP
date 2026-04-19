import argparse
import os
import tempfile
import unittest

from test_agent import parse_axis_value
from visual_zendo import (
    DEFAULT_ARTIFACTS_DIR,
    FeedbackAxis,
    GoalAxis,
    PIL_AVAILABLE,
    LithicArrayEnv,
    MechanicsAxis,
    Arrangement,
    Shape,
    WorldAxis,
)


class LithicArrayEnvTests(unittest.TestCase):
    def test_default_artifacts_dir_is_module_local(self):
        env = LithicArrayEnv()
        self.assertEqual(env.artifacts_dir, os.path.abspath(DEFAULT_ARTIFACTS_DIR))

    def test_high_alias_maps_to_hard_world(self):
        env = LithicArrayEnv(world="high")
        self.assertEqual(env.world, WorldAxis.HARD)

    def test_low_alias_maps_to_easy_world_in_parser(self):
        parsed = parse_axis_value("low", WorldAxis, "world")
        self.assertEqual(parsed, WorldAxis.EASY)

    def test_high_alias_maps_to_hard_world_in_parser(self):
        parsed = parse_axis_value("high", WorldAxis, "world")
        self.assertEqual(parsed, WorldAxis.HARD)

    def test_medium_world_is_rejected(self):
        with self.assertRaises(ValueError):
            LithicArrayEnv(world="medium")

    def test_medium_world_is_rejected_by_parser(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_axis_value("medium", WorldAxis, "world")

    def test_easy_world_returns_json_representation(self):
        arrangement = Arrangement(
            shapes=[
                Shape(color="red", size="large", type_="triangle"),
                Shape(color="blue", size="small", type_="square"),
            ]
        )

        env = LithicArrayEnv(world=WorldAxis.EASY)
        result = env._format_arrangement(arrangement, label=False)

        self.assertEqual(result["representation_type"], "json")
        self.assertEqual(result["representation"], arrangement.to_json_dict())
        self.assertEqual(result["label"], "Shale")

    def test_hard_world_returns_image_metadata_or_json_fallback(self):
        arrangement = Arrangement(
            shapes=[
                Shape(color="red", size="large", type_="triangle"),
                Shape(color="blue", size="small", type_="square"),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            env = LithicArrayEnv(world=WorldAxis.HARD, artifacts_dir=tmpdir)
            result = env._format_arrangement(arrangement, label=True, filename="sample.png")

            expected_path = os.path.join(tmpdir, "sample.png")
            self.assertEqual(result["label"], "Quartz")
            if PIL_AVAILABLE:
                self.assertEqual(result["representation_type"], "image")
                self.assertEqual(result["representation"], expected_path)
                self.assertEqual(result["image_path"], expected_path)
                self.assertTrue(os.path.exists(expected_path))
            else:
                self.assertEqual(result["representation_type"], "json")
                self.assertEqual(result["representation"], arrangement.to_json_dict())
                self.assertEqual(result["fallback_reason"], "PIL not available for image rendering")

    def test_goal_and_mechanics_only_have_easy_and_hard_variants(self):
        easy_env = LithicArrayEnv(goal=GoalAxis.EASY, mechanics=MechanicsAxis.EASY)
        hard_env = LithicArrayEnv(goal=GoalAxis.HARD, mechanics=MechanicsAxis.HARD)

        self.assertIn("discover the hidden rule", easy_env._get_goal_instruction())
        self.assertEqual(hard_env._get_goal_instruction(), "")
        self.assertIn("You have two actions available.", easy_env._get_mechanics_instruction())
        self.assertEqual(
            hard_env._get_mechanics_instruction(),
            "Respond with a JSON object containing your action ('STRATA' or 'PROPOSE').",
        )

    def test_mechanics_prompt_does_not_repeat_goal_language(self):
        mechanics_prompt = LithicArrayEnv(mechanics=MechanicsAxis.EASY)._get_mechanics_instruction().lower()

        self.assertNotIn("discover the hidden rule", mechanics_prompt)
        self.assertNotIn("state the rule explicitly to win", mechanics_prompt)
        self.assertIn("propose costs 1 token", mechanics_prompt)

    def test_feedback_easy_returns_counterexample_on_failed_proposal(self):
        env = LithicArrayEnv(feedback=FeedbackAxis.EASY)
        counterexample = Arrangement(shapes=[Shape(color="red", size="small", type_="circle")])
        env.reset([], "always true", lambda arr: True, lambda _: (counterexample, True))
        env.tokens = 1

        result = env.propose_rule("always false", lambda arr: False)

        self.assertEqual(result["result"], "Rejected")
        self.assertIn("counter_example", result)

    def test_feedback_hard_omits_counterexample_on_failed_proposal(self):
        env = LithicArrayEnv(feedback=FeedbackAxis.HARD)
        counterexample = Arrangement(shapes=[Shape(color="red", size="small", type_="circle")])
        env.reset([], "always true", lambda arr: True, lambda _: (counterexample, True))
        env.tokens = 1

        result = env.propose_rule("always false", lambda arr: False)

        self.assertEqual(result["result"], "Rejected")
        self.assertNotIn("counter_example", result)


if __name__ == "__main__":
    unittest.main()
