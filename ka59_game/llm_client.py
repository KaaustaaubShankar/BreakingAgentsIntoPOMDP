"""
llm_client.py — LLM client for the KA59 real-game harness.

Providers: openrouter, anthropic, claude-cli, claude-proxy, openai, xai.
Each provider's API key comes from the environment (loaded via dotenv on import).

`claude-proxy` routes through the local OpenClaw billing proxy on :18801,
which uses the Claude Max OAuth credentials and reshapes requests to pass
Anthropic's classifier. ~10x faster per turn than `claude-cli` (no
subprocess fork, persistent HTTP connection, prompt caching).
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

    def __init__(self, provider: str, model: str, reasoning_effort: str | None = None) -> None:
        self.provider = provider.lower()
        self.model = model
        self.reasoning_effort = reasoning_effort  # None | 'minimal' | 'low' | 'medium' | 'high'
        self.reset_usage()

    def reset_usage(self) -> None:
        self.last_usage = self._empty_usage()
        self.usage_totals = self._empty_usage()

    def _empty_usage(self) -> Dict[str, int]:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
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
        # OpenRouter/OpenAI return reasoning tokens under completion_tokens_details
        details = (usage.get("completion_tokens_details") if isinstance(usage, dict)
                   else getattr(usage, "completion_tokens_details", None))
        reasoning_tokens = (_val(details, "reasoning_tokens") or 0) if details is not None else 0
        # OpenRouter/OpenAI report prompt-cache hits under prompt_tokens_details.
        # cached_tokens is the portion of input_tokens served from cache (billed
        # at a discount), so it lets us verify prompt caching is actually landing.
        prompt_details = (usage.get("prompt_tokens_details") if isinstance(usage, dict)
                          else getattr(usage, "prompt_tokens_details", None))
        cached_tokens = (_val(prompt_details, "cached_tokens") or 0) if prompt_details is not None else 0
        has_usage = usage is not None and any(x > 0 for x in (input_tokens, output_tokens, total_tokens))
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_tokens": cached_tokens,
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
        if self.provider == "claude-proxy":
            return self._generate_claude_proxy(system_prompt, user_prompt)
        if self.provider == "openai":
            return self._generate_openai(system_prompt, user_prompt)
        if self.provider == "deepseek":
            return self._generate_deepseek(system_prompt, user_prompt)
        if self.provider == "xai":
            return self._generate_xai(system_prompt, user_prompt)
        if self.provider == "qwen-mlx":
            return self._generate_qwen_mlx(system_prompt, user_prompt)
        if self.provider == "qwen-local":
            return self._generate_qwen_local(system_prompt, user_prompt)
        raise ValueError(
            f"Unknown provider: {self.provider!r}. "
            "Use 'openrouter', 'anthropic', 'claude-cli', 'claude-proxy', "
            "'openai', 'deepseek', 'xai', 'qwen-local', or 'qwen-mlx'."
        )

    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    def _deepseek_client(self):
        import openai
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not set.")
        if not hasattr(self, "_ds_client"):
            self._ds_client = openai.OpenAI(
                base_url=self.DEEPSEEK_BASE_URL,
                api_key=api_key,
                timeout=120.0,
            )
        return self._ds_client

    def _generate_deepseek(self, system_prompt: str, user_prompt: str) -> str:
        """Direct DeepSeek API (OpenAI-compatible) for deepseek-v4-pro/flash.

        deepseek-v4-pro is a thinking model by default. The native API does NOT
        accept reasoning_effort="none" (valid: low/medium/high/max/xhigh), so the
        no-reasoning condition is expressed by explicitly disabling thinking via
        extra_body {"thinking": {"type": "disabled"}}. Higher efforts pass through
        as reasoning_effort. max_tokens is kept generous so a reasoning chain (when
        enabled) doesn't consume the whole budget and leave empty content.
        """
        import time
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4096,
        )
        effort = (self.reasoning_effort or "none").lower()
        if effort in ("none", "off", "disabled"):
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            kwargs["reasoning_effort"] = effort
        for attempt in range(4):
            try:
                response = self._deepseek_client().chat.completions.create(**kwargs)
                self._record_usage(response)
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("DeepSeek returned empty content.")
                return str(content)
            except Exception as exc:
                msg = str(exc)
                if "429" in msg and attempt < 3:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise
        raise RuntimeError("DeepSeek call exhausted retries without raising.")

    def _generate_xai(self, system_prompt: str, user_prompt: str) -> str:
        import openai as _openai
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY not set.")
        client = _openai.OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
        )
        content = resp.choices[0].message.content
        if content is None:
            raise ValueError("xAI returned empty content.")
        return str(content)

    def _generate_qwen_local(self, system_prompt: str, user_prompt: str) -> str:
        """Local Qwen 3 inference. Delegates to top-level qwen_local module
        so ka59_game / env3 / env4 all share one model cache per process."""
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

    def _generate_qwen_mlx(self, system_prompt: str, user_prompt: str) -> str:
        """MLX-based local Qwen inference (Apple Silicon only).

        Delegates to top-level qwen_mlx module so the model cache is shared
        across ka59_game / env3 / env4 within one process. Mirrors the
        transformers-based qwen-local provider, but uses mlx_lm instead.
        """
        import sys, pathlib
        repo_root = str(pathlib.Path(__file__).resolve().parents[1])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from qwen_mlx import generate as _qwen_mlx_generate
        text, input_tokens, output_tokens = _qwen_mlx_generate(
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

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        import openai as _openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set.")
        client = _openai.OpenAI(api_key=api_key)
        # max_completion_tokens covers both reasoning and non-reasoning models;
        # max_tokens is rejected by gpt-5.x and o-series reasoning models.
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

    PROXY_BASE_URL = "http://127.0.0.1:18801"

    def _generate_claude_proxy(self, system_prompt: str, user_prompt: str) -> str:
        """Route through local OpenClaw billing proxy → Claude Max OAuth.

        Persistent client across turns within a trial — no subprocess fork.
        System prompt carries cache_control so subsequent turns hit the
        prompt cache (5-min TTL, comfortably within turn cadence).
        """
        import time
        if not hasattr(self, "_proxy_client"):
            import anthropic
            # x-openclaw-no-lift=1 disables Plan-7 lift-and-replace at the
            # proxy. Required for experimental purity: our `system` field is
            # the variable under study and must reach the model unmodified.
            self._proxy_client = anthropic.Anthropic(
                base_url=self.PROXY_BASE_URL,
                api_key="dummy-proxy-injects-oauth",
                max_retries=0,
                default_headers={"x-openclaw-no-lift": "1"},
            )
        response = None
        for attempt in range(4):
            try:
                response = self._proxy_client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=[{
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }],
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
            raise RuntimeError("claude-proxy call did not produce a response.")
        content = response.content[0].text if response.content else None
        if content is None:
            raise ValueError("claude-proxy returned empty content.")
        self._record_usage(response)
        return str(content)

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
        # Cap generation to prevent reasoning models from stalling a trial for
        # many minutes on multi-thousand-token chains. xhigh/high effort gets a
        # larger budget so the reasoning chain isn't truncated mid-thought.
        _high_effort = self.reasoning_effort in ("xhigh", "high", "medium")
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4096 if _high_effort else 1024,
        )
        extra_body: Dict[str, Any] = {}
        if self.reasoning_effort:
            # OpenRouter unified reasoning param works for all supported
            # providers (OpenAI gpt-5.x, xAI Grok, Anthropic). Pass via
            # extra_body since OpenAI Python SDK has no top-level `reasoning`.
            extra_body["reasoning"] = {"effort": self.reasoning_effort}
        # Pin OpenAI-hosted models to the OpenAI upstream. A prompt cache only
        # lives on the backend that created it, so consistent routing is what
        # makes caching actually land across turns/trials; it also keeps the
        # serving backend reproducible across runs and avoids the flaky
        # third-party resellers that previously returned zero-token responses.
        if self.model.startswith("openai/"):
            extra_body["provider"] = {"order": ["OpenAI"], "allow_fallbacks": False}
        if extra_body:
            kwargs["extra_body"] = extra_body
        for attempt in range(4):
            try:
                response = self._openrouter_client().chat.completions.create(**kwargs)
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
