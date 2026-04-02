import os
import sys
import types
import unittest
from unittest.mock import patch

from test_agent import LLMClient


class _FakeOpenAIClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )
        self.last_request = None

    def _create(self, **kwargs):
        self.last_request = kwargs
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content='{"action": "STRATA"}')
                )
            ]
        )


class LLMClientTests(unittest.TestCase):
    def test_openrouter_uses_openai_compatible_client(self):
        fake_holder = {}

        def fake_openai_factory(**kwargs):
            client = _FakeOpenAIClient(**kwargs)
            fake_holder["client"] = client
            return client

        fake_openai_module = types.SimpleNamespace(OpenAI=fake_openai_factory)

        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_HTTP_REFERER": "https://example.com",
            "OPENROUTER_APP_TITLE": "Zendo Tester",
        }, clear=False):
            with patch.dict(sys.modules, {"openai": fake_openai_module}):
                client = LLMClient("openrouter", "openai/gpt-4o")
                content = client.generate("system prompt", "user prompt")

        self.assertEqual(content, '{"action": "STRATA"}')
        self.assertEqual(
            fake_holder["client"].kwargs,
            {
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "test-key",
                "default_headers": {
                    "HTTP-Referer": "https://example.com",
                    "X-Title": "Zendo Tester",
                },
            },
        )
        self.assertEqual(
            fake_holder["client"].last_request,
            {
                "model": "openai/gpt-4o",
                "messages": [
                    {"role": "system", "content": "system prompt"},
                    {"role": "user", "content": "user prompt"},
                ],
            },
        )

    def test_openrouter_requires_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            client = LLMClient("openrouter", "openai/gpt-4o")
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                client.generate("system prompt", "user prompt")


if __name__ == "__main__":
    unittest.main()
