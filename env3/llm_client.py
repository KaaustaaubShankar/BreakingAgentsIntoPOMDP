"""
llm_client.py — OpenRouter client using the OpenAI-compatible SDK.

OpenRouter acts as a single endpoint for hundreds of models.
Free models require the ":free" suffix in the model ID.

Recommended free models (April 2026):
  Text:   "meta-llama/llama-3.3-70b-instruct:free"
          "deepseek/deepseek-r1:free"
  Vision: "google/gemma-3-27b-it:free"
          "google/gemma-4-31b-it:free"

Paid models for the paper:
  "openai/gpt-4o"
  "anthropic/claude-sonnet-4-6"
  "google/gemini-2.5-pro-preview"

Set OPENROUTER_API_KEY in your .env file.
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

    # ── Usage tracking ────────────────────────────────────────────────────────

    def reset_usage(self) -> None:
        self.last_usage = self._empty_usage()
        self.usage_totals = self._empty_usage()

    def _empty_usage(self) -> Dict[str, int]:
        return {
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "calls": 0, "calls_with_usage": 0,
        }

    def _extract_usage(self, response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage", None)

        def _val(u: Any, *keys: str) -> int | None:
            for k in keys:
                v = u.get(k) if isinstance(u, dict) else getattr(u, k, None)
                if v is not None:
                    try:
                        return int(v)
                    except (TypeError, ValueError):
                        pass
            return None

        inp = _val(usage, "input_tokens", "prompt_tokens") or 0
        out = _val(usage, "output_tokens", "completion_tokens") or 0
        tot = _val(usage, "total_tokens") or (inp + out)
        has = usage is not None and any(x > 0 for x in (inp, out, tot))
        return {
            "input_tokens": inp, "output_tokens": out, "total_tokens": tot,
            "calls": 1, "calls_with_usage": 1 if has else 0,
        }

    def _record_usage(self, response: Any) -> None:
        u = self._extract_usage(response)
        self.last_usage = u
        for k, v in u.items():
            self.usage_totals[k] += v

    def get_usage_summary(self) -> Dict[str, int]:
        return dict(self.usage_totals)

    # ── Generation ────────────────────────────────────────────────────────────

    def _client(self):
        import openai
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required.")
        return openai.OpenAI(base_url=self.OPENROUTER_BASE_URL, api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Send a text-only request and return the reply."""
        if self.provider != "openrouter":
            raise ValueError(f"Unknown provider: {self.provider}. Use 'openrouter'.")

        resp = self._client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        self._record_usage(resp)
        content = resp.choices[0].message.content
        if content is None:
            raise ValueError("OpenRouter returned empty content.")
        return str(content)

    def generate_with_image(
        self,
        system_prompt: str,
        user_prompt: str,
        base64_png: str,
    ) -> str:
        """
        Send a request with both text and an inline PNG image.
        Used for the visual Hard World axis level.
        Requires a vision-capable model (e.g. google/gemma-3-27b-it:free).
        """
        if self.provider != "openrouter":
            raise ValueError(f"Unknown provider: {self.provider}. Use 'openrouter'.")

        resp = self._client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_png}"},
                        },
                    ],
                },
            ],
        )
        self._record_usage(resp)
        content = resp.choices[0].message.content
        if content is None:
            raise ValueError("OpenRouter returned empty content.")
        return str(content)

    # ── JSON parsing ──────────────────────────────────────────────────────────

    def parse_json(self, text: str) -> Dict[str, Any]:
        """Extract and parse the first JSON object found in text."""
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
