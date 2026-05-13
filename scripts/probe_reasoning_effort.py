"""
Probe what `reasoning_effort` values actually do via OpenRouter.

Sends a tiny prompt to gpt-5.2 at multiple effort levels and prints
the request kwargs + the full usage object, so we can see whether
reasoning_tokens differ between settings.

Run:
    python3 -m scripts.probe_reasoning_effort
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

repo_root = Path(__file__).parents[1]
sys.path.insert(0, str(repo_root))

from dotenv import load_dotenv
import openai

load_dotenv(repo_root / ".env")

PROMPT = (
    "What is 17 * 23? Respond with just the number."
)
SYSTEM = "You are a calculator. Answer with just digits."

LEVELS = [None, "minimal", "low", "medium", "high"]
MODEL = "openai/gpt-5.2"


def _usage_to_dict(usage):
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return dict(usage)
    return {k: getattr(usage, k, None) for k in dir(usage) if not k.startswith("_")}


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    rows = []
    for effort in LEVELS:
        kwargs = dict(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT},
            ],
        )
        if effort is not None:
            kwargs["extra_body"] = {"reasoning": {"effort": effort}}

        label = effort if effort is not None else "OMITTED"
        print(f"\n=== reasoning_effort={label} ===")
        print(f"request kwargs: {json.dumps({k: v for k, v in kwargs.items() if k != 'messages'}, indent=2)}")

        try:
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            usage = _usage_to_dict(resp.usage)
            print(f"response: {content!r}")
            print(f"usage: {json.dumps(usage, indent=2, default=str)}")

            ctd = (usage or {}).get("completion_tokens_details") or {}
            r_tokens = (
                ctd.get("reasoning_tokens")
                if isinstance(ctd, dict)
                else None
            )
            rows.append({
                "effort": label,
                "completion_tokens": (usage or {}).get("completion_tokens"),
                "reasoning_tokens": r_tokens,
                "content": content,
            })
        except Exception as exc:
            print(f"ERROR: {exc!r}")
            rows.append({"effort": label, "error": repr(exc)})

    print("\n=== SUMMARY ===")
    print(f"{'effort':<10} {'completion':<12} {'reasoning':<12} content")
    for r in rows:
        if "error" in r:
            print(f"{r['effort']:<10} ERROR: {r['error'][:80]}")
        else:
            print(f"{r['effort']:<10} {str(r['completion_tokens']):<12} {str(r['reasoning_tokens']):<12} {r['content']!r}")


if __name__ == "__main__":
    main()
