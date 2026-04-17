import json
import tempfile
import unittest

from test_agent import run_llm_agent


class _StubUsageClient:
    def __init__(self):
        self.responses = [
            '{"action": "STRATA", "arrangement": [], "prediction": false}',
            '{"goal_understanding": "Find the rule.", "mechanics_understanding": "Query examples and propose a rule."}',
        ]
        self.per_call_usage = [
            {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
            {"input_tokens": 6, "output_tokens": 3, "total_tokens": 9},
        ]
        self.reset_usage()

    def reset_usage(self):
        self.index = 0
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
            "calls_with_usage": 0,
        }

    def generate(self, system_prompt, user_prompt):
        usage = self.per_call_usage[self.index]
        self.usage["calls"] += 1
        self.usage["calls_with_usage"] += 1
        self.usage["input_tokens"] += usage["input_tokens"]
        self.usage["output_tokens"] += usage["output_tokens"]
        self.usage["total_tokens"] += usage["total_tokens"]
        response = self.responses[self.index]
        self.index += 1
        return response

    def get_usage_summary(self):
        return dict(self.usage)


class RunLLMAgentUsageTests(unittest.TestCase):
    def test_run_llm_agent_records_llm_usage_in_result_and_history(self):
        client = _StubUsageClient()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_llm_agent(
                provider="openrouter",
                model="openai/gpt-4o",
                max_turns=1,
                client=client,
                verbose=False,
                save_history=True,
                artifacts_dir=tmpdir,
            )

            self.assertEqual(
                result["llm_usage"],
                {
                    "input_tokens": 16,
                    "output_tokens": 7,
                    "total_tokens": 23,
                    "calls": 2,
                    "calls_with_usage": 2,
                },
            )
            self.assertEqual(result["usage"], result["llm_usage"])

            with open(result["history_file"], "r") as f:
                history = json.load(f)

            events = history["events"] if isinstance(history, dict) else history
            usage_events = [entry for entry in events if entry["event"] == "llm_usage_summary"]
            self.assertEqual(len(usage_events), 1)
            self.assertEqual(
                usage_events[0]["data"],
                {
                    "provider": "openrouter",
                    "model": "openai/gpt-4o",
                    "input_tokens": 16,
                    "output_tokens": 7,
                    "total_tokens": 23,
                    "calls": 2,
                    "calls_with_usage": 2,
                },
            )


if __name__ == "__main__":
    unittest.main()
