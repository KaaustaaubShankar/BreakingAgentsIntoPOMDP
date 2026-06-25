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

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
# arc_agi/base.py loads `.env.example` at import (override=False). When this
# pipeline is run from env3/, that pulls the placeholder OPENROUTER_API_KEY from
# env3/.env.example, which — depending on import order — can shadow the real key
# and cause silent 401 "Missing Authentication header" on every turn. Force the
# real key from the repo-root .env so we never run against the placeholder.
_ROOT_ENV = Path(__file__).resolve().parents[1] / ".env"
if _ROOT_ENV.exists():
    load_dotenv(_ROOT_ENV, override=True)


class LLMClient:
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, provider: str, model: str, reasoning_effort: str | None = None) -> None:
        self.provider = provider.lower()
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.reset_usage()

    # ── Usage tracking ────────────────────────────────────────────────────────

    def reset_usage(self) -> None:
        self.last_usage = self._empty_usage()
        self.usage_totals = self._empty_usage()

    def _empty_usage(self) -> Dict[str, int]:
        return {
            "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
            "total_tokens": 0, "calls": 0, "calls_with_usage": 0,
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
        # OpenRouter/OpenAI report reasoning tokens under completion_tokens_details
        details = (usage.get("completion_tokens_details") if isinstance(usage, dict)
                   else getattr(usage, "completion_tokens_details", None))
        reason = (_val(details, "reasoning_tokens") or 0) if details is not None else 0
        has = usage is not None and any(x > 0 for x in (inp, out, tot))
        return {
            "input_tokens": inp, "output_tokens": out, "reasoning_tokens": reason,
            "total_tokens": tot, "calls": 1, "calls_with_usage": 1 if has else 0,
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
        # Per-request timeout so a single stalled OpenRouter call can't hang a
        # whole cell indefinitely (env3 had no timeout). A timed-out turn is
        # caught per-turn in experiment.py and the loop continues.
        return openai.OpenAI(base_url=self.OPENROUTER_BASE_URL, api_key=api_key, timeout=300.0)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Send a text-only request and return the reply."""
        if self.provider == "qwen-local":
            return self._generate_qwen_local(system_prompt, user_prompt)
        if self.provider != "openrouter":
            raise ValueError(f"Unknown provider: {self.provider}. Use 'openrouter' or 'qwen-local'.")

        import time
        # Cap generation so reasoning models don't stall a turn for minutes on a
        # multi-thousand-token chain. high/medium effort gets a larger budget so
        # the chain isn't truncated mid-thought. Mirrors ka59_game/llm_client.py.
        _high_effort = self.reasoning_effort in ("xhigh", "high", "medium")
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=4096 if _high_effort else 1024,
        )
        if self.reasoning_effort:
            # OpenRouter unified reasoning param. effort="none" disables thinking
            # (verified reasoning_tokens=0 for deepseek-v4-pro). Passed via
            # extra_body since the OpenAI SDK has no top-level `reasoning`.
            kwargs["extra_body"] = {"reasoning": {"effort": self.reasoning_effort}}
        for attempt in range(4):
            try:
                resp = self._client().chat.completions.create(**kwargs)
                self._record_usage(resp)
                content = resp.choices[0].message.content
                if content is None:
                    raise ValueError("OpenRouter returned empty content.")
                return str(content)
            except Exception as exc:
                if "429" in str(exc) and attempt < 3:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise
        raise RuntimeError("OpenRouter call exhausted retries without raising.")

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
