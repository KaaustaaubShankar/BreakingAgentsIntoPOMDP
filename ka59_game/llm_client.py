"""
llm_client.py — LLM client for the KA59 real-game harness.

Providers: openrouter, anthropic, claude-cli, openai, xai.
Each provider's API key comes from the environment (loaded via dotenv on import).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()


# DeepSeek-V4-Pro at medium effort draws 8,000-10,000 output tokens per turn on
# real KA59 states (measured: 8,101 and 9,934 per turn across two trials). A cap
# near that mean truncates the tail, and a truncated response returns
# content=None. Every historical medium failure fits this one mechanism: the
# direct API at 4,096 lost 81% of turns; OpenRouter at 16,384 lost the tail.
OPENROUTER_MAX_TOKENS = 65536


class LLMClient:
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        provider: str,
        model: str,
        reasoning_effort: str | None = None,
        upstream_provider: str | None = None,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.reasoning_effort = reasoning_effort
        # OpenRouter routes across many upstreams by default. Pinning one keeps
        # the served backend a recorded part of the experimental identity.
        # Accepts one name or a comma-separated allow-list. OpenRouter picks
        # among the listed upstreams; the served one is recorded per call in
        # last_upstream_provider so the identity stays auditable.
        self.upstream_provider = upstream_provider
        self.last_upstream_provider: str | None = None
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

    def _openrouter_client(self):
        import openai
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set.")
        if not hasattr(self, "_or_client"):
            self._or_client = openai.OpenAI(
                base_url=self.OPENROUTER_BASE_URL,
                api_key=api_key,
                timeout=120.0,
            )
        return self._or_client

    def _anthropic_client(self):
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY") or self._claude_code_token()
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set and no Claude Code credentials found "
                "at ~/.claude/.credentials.json"
            )
        return anthropic.Anthropic(api_key=api_key)

    @staticmethod
    def _claude_code_token() -> str | None:
        """Fall back to the local Claude Code OAuth token if available."""
        from pathlib import Path
        creds_path = Path.home() / ".claude" / ".credentials.json"
        try:
            creds = json.loads(creds_path.read_text())
            return creds.get("claudeAiOauth", {}).get("accessToken")
        except Exception:
            return None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "anthropic":
            return self._generate_anthropic(system_prompt, user_prompt)
        if self.provider == "openrouter":
            return self._generate_openrouter(system_prompt, user_prompt)
        if self.provider == "claude-cli":
            return self._generate_claude_cli(system_prompt, user_prompt)
        if self.provider == "openai":
            return self._generate_openai(system_prompt, user_prompt)
        if self.provider == "xai":
            return self._generate_xai(system_prompt, user_prompt)
        raise ValueError(
            f"Unknown provider: {self.provider!r}. "
            "Use 'openrouter', 'anthropic', 'claude-cli', 'openai', or 'xai'."
        )

    def _generate_xai(self, system_prompt: str, user_prompt: str) -> str:
        import openai as _openai
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY not set.")
        client = _openai.OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=4096,
        )
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        resp = client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        if content is None:
            raise ValueError("xAI returned empty content.")
        return str(content)

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        import openai as _openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set.")
        client = _openai.OpenAI(api_key=api_key)
        # max_completion_tokens covers both reasoning and non-reasoning models;
        # max_tokens is rejected by gpt-5.x and o-series reasoning models. The
        # budget must clear the reasoning draw or the model spends it all on
        # reasoning tokens and returns content=None.
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=16384,
        )
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        resp = client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI returned empty content.")
        return str(content)

    def _generate_claude_cli(self, system_prompt: str, user_prompt: str) -> str:
        """Route through `claude -p` CLI — uses Claude Code OAuth with token refresh.

        Runs from /tmp to avoid loading workspace CLAUDE.md context.
        """
        import subprocess
        result = subprocess.run(
            ["claude", "-p", "--output-format", "json",
             "--model", self.model,
             "--system-prompt", system_prompt or "You are a helpful assistant."],
            input=user_prompt,
            capture_output=True,
            text=True,
            timeout=120,
            cwd="/tmp",
        )
        if result.returncode != 0:
            raise RuntimeError(f"claude-cli error: {result.stderr[:300]}")
        data = json.loads(result.stdout)
        if data.get("is_error") or data.get("subtype") != "success":
            raise RuntimeError(f"claude-cli returned error: {result.stdout[:300]}")
        return str(data["result"])

    def _generate_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        import time
        client = self._anthropic_client()
        response = None
        for attempt in range(4):
            try:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                break
            except Exception as exc:
                msg = str(exc)
                if ("429" in msg or "rate_limit" in msg.lower()) and attempt < 3:
                    time.sleep(15 * (attempt + 1))
                    continue
                raise
        if response is None:
            raise RuntimeError("Anthropic call did not produce a response.")
        content = response.content[0].text if response.content else None
        if content is None:
            raise ValueError("Anthropic returned empty content.")

        class _FakeUsage:
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = response.usage.input_tokens + response.usage.output_tokens

        class _FakeResponse:
            usage = _FakeUsage()

        self._record_usage(_FakeResponse())
        return str(content)

    def _generate_openrouter(self, system_prompt: str, user_prompt: str) -> str:
        import time
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=OPENROUTER_MAX_TOKENS,
        )
        extra_body: Dict[str, Any] = {}
        if self.reasoning_effort:
            extra_body["reasoning"] = {"effort": self.reasoning_effort}
        if self.upstream_provider:
            extra_body["provider"] = {
                "only": [p.strip() for p in self.upstream_provider.split(",") if p.strip()]
            }
        if extra_body:
            kwargs["extra_body"] = extra_body
        for attempt in range(4):
            try:
                response = self._openrouter_client().chat.completions.create(**kwargs)
                self._record_usage(response)
                self.last_upstream_provider = getattr(response, "provider", None)
                content = response.choices[0].message.content
                if content is None:
                    served = self.last_upstream_provider or "unknown upstream"
                    finish = response.choices[0].finish_reason
                    raise ValueError(
                        f"OpenRouter returned empty content (upstream={served}, "
                        f"finish_reason={finish}, cap={OPENROUTER_MAX_TOKENS})."
                    )
                return str(content)
            except Exception as exc:
                msg = str(exc)
                # An empty/truncated body is transport flakiness, not a model
                # answer: retry rather than voiding a trial that has most of its
                # turns left. The retry may also land on a healthier upstream.
                # A quota-exhaustion 429 never succeeds on retry; only rate
                # limiting is worth backing off for.
                quota_exhausted = (
                    "insufficient_quota" in msg or "credit_balance_exhausted" in msg
                    or "no credits remaining" in msg
                )
                retryable = (
                    ("429" in msg and not quota_exhausted) or "empty content" in msg
                )
                if retryable and attempt < 3:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise
        raise RuntimeError("OpenRouter call exhausted retries without raising.")

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
