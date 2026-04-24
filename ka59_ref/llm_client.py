"""
llm_client.py — OpenRouter client using the OpenAI-compatible SDK.

Copied verbatim from env4/llm_client.py so both environments use the same
provider abstraction, key resolution, and JSON-extraction behaviour.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider.lower()
        self.model = model
        self.reset_usage()

    def reset_usage(self) -> None:
        self.last_usage = self._empty_usage()
        self.usage_totals = self._empty_usage()

    def _empty_usage(self) -> Dict[str, int]:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
            "calls_with_usage": 0,
        }

    def _extract_usage(self, response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage", None)

        def _val(u: Any, *keys: str) -> int | None:
            for key in keys:
                v = u.get(key) if isinstance(u, dict) else getattr(u, key, None)
                if v is not None:
                    try:
                        return int(v)
                    except (TypeError, ValueError):
                        pass
            return None

        input_tokens = _val(usage, "input_tokens", "prompt_tokens") or 0
        output_tokens = _val(usage, "output_tokens", "completion_tokens") or 0
        total_tokens = _val(usage, "total_tokens") or (input_tokens + output_tokens)
        has_usage = usage is not None and any(x > 0 for x in (input_tokens, output_tokens, total_tokens))
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "calls": 1,
            "calls_with_usage": 1 if has_usage else 0,
        }

    def _record_usage(self, response: Any) -> None:
        usage = self._extract_usage(response)
        self.last_usage = usage
        for key, value in usage.items():
            self.usage_totals[key] += value

    def get_usage_summary(self) -> Dict[str, int]:
        return dict(self.usage_totals)

    def _client(self):
        import openai

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required.")
        return openai.OpenAI(base_url=self.OPENROUTER_BASE_URL, api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider != "openrouter":
            raise ValueError(f"Unknown provider: {self.provider}. Use 'openrouter'.")

        import time
        for attempt in range(4):
            try:
                response = self._client().chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                self._record_usage(response)
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenRouter returned empty content.")
                return str(content)
            except Exception as exc:
                msg = str(exc)
                if "429" in msg and attempt < 3:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise

    def parse_json(self, text: str) -> Dict[str, Any]:
        try:
            return dict(json.loads(text.strip()))
        except json.JSONDecodeError:
            pass

        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return dict(json.loads(match.group(1)))

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return dict(json.loads(match.group(0)))

        raise ValueError(f"No JSON object found in response:\n{text!r}")
