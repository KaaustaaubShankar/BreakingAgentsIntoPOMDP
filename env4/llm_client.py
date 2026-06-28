"""
llm_client.py — OpenRouter client using the OpenAI-compatible SDK.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
# arc_agi/base.py loads `.env.example` at import (override=False). When run from
# env4/, that pulls the placeholder OPENROUTER_API_KEY from env4/.env.example,
# which — depending on import order — can shadow the real key and cause silent
# 401 "Missing Authentication header" on every turn (0-token trials that look
# like normal losses). Force the real key from the repo-root .env.
_ROOT_ENV = Path(__file__).resolve().parents[1] / ".env"
if _ROOT_ENV.exists():
    load_dotenv(_ROOT_ENV, override=True)


class LLMClient:
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, provider: str, model: str, reasoning_effort: str | None = None) -> None:
        self.provider = provider.lower()
        self.model = model
        self.reasoning_effort = reasoning_effort.lower().strip() if reasoning_effort else None
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
            "cost": 0.0,
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

        # try to extract float cost if present
        cost_val = None
        try:
            if isinstance(usage, dict):
                cost_val = usage.get("cost")
            else:
                cost_val = getattr(usage, "cost", None)
            if cost_val is None:
                cost_val = 0.0
            else:
                cost_val = float(cost_val)
        except Exception:
            cost_val = 0.0

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "calls": 1,
            "calls_with_usage": 1 if has_usage else 0,
            "cost": cost_val,
        }

    def _record_usage(self, response: Any) -> None:
        usage = self._extract_usage(response)
        self.last_usage = usage
        for key, value in usage.items():
            # ensure numeric accumulation works for both ints and floats
            prev = self.usage_totals.get(key, 0)
            try:
                self.usage_totals[key] = prev + value
            except Exception:
                self.usage_totals[key] = value

    def get_usage_summary(self) -> Dict[str, int]:
        return dict(self.usage_totals)

    def _client(self):
        import openai

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required.")
        return openai.OpenAI(base_url=self.OPENROUTER_BASE_URL, api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "qwen-local":
            return self._generate_qwen_local(system_prompt, user_prompt)
        if self.provider != "openrouter":
            raise ValueError(f"Unknown provider: {self.provider}. Use 'openrouter' or 'qwen-local'.")

        kwargs = {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.reasoning_effort and self.reasoning_effort != "none":
            kwargs["extra_body"] = {"reasoning": {"enabled": True}}

        response = self._client().chat.completions.create(**kwargs)
        self._record_usage(response)
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenRouter returned empty content.")
        return str(content)

    def _generate_qwen_local(self, system_prompt: str, user_prompt: str) -> str:
        """Local Qwen 3 inference. Delegates to top-level qwen_local module
        so ka59_game / env3 / env4 share one model cache per process."""
        import sys, pathlib
        repo_root = str(pathlib.Path(__file__).resolve().parents[1])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from qwen_local import generate as _qwen_generate
        text, input_tokens, output_tokens = _qwen_generate(
            self.model, system_prompt, user_prompt, self.reasoning_effort
        )

        class _U:
            pass

        _U.input_tokens = input_tokens
        _U.output_tokens = output_tokens
        _U.total_tokens = input_tokens + output_tokens

        class _R:
            usage = _U()

        self._record_usage(_R())
        return text

    def generate_with_image(
        self,
        system_prompt: str,
        user_prompt: str,
        base64_png: str,
    ) -> str:
        if self.provider != "openrouter":
            raise ValueError(f"Unknown provider: {self.provider}. Use 'openrouter'.")

        kwargs = {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "messages": [
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
        }
        if self.reasoning_effort and self.reasoning_effort != "none":
            kwargs["extra_body"] = {"reasoning": {"enabled": True, "effort": self.reasoning_effort}}

        response = self._client().chat.completions.create(**kwargs)


        self._record_usage(response)
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenRouter returned empty content.")
        return str(content)

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
