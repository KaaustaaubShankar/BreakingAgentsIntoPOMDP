"""Smoke-test the qwen-mlx provider end-to-end.

Loads a small MLX Qwen model and runs one generation per reasoning_effort
level via the LLMClient, confirming the full chain
  LLMClient(provider='qwen-mlx') -> qwen_mlx.generate() -> mlx_lm
works on Apple Silicon and returns non-empty text with usage tokens recorded.

Run:
    source .venv/bin/activate
    python -m scripts.smoke_test_qwen_mlx
    python -m scripts.smoke_test_qwen_mlx --model mlx-community/Qwen2.5-3B-Instruct-4bit
    python -m scripts.smoke_test_qwen_mlx --efforts none medium high
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

repo_root = Path(__file__).parents[1]
sys.path.insert(0, str(repo_root))

from ka59_game.llm_client import LLMClient


DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DEFAULT_EFFORTS = ["none", "minimal", "low", "medium", "high"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"MLX model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--efforts", nargs="+", default=DEFAULT_EFFORTS,
                        help="Reasoning_effort levels to test")
    parser.add_argument("--system", default="You are a helpful assistant. Reply in one short sentence.")
    parser.add_argument("--user", default="What is 2 + 2?")
    args = parser.parse_args()

    print(f"\n=== qwen-mlx smoke test ===")
    print(f"Model:   {args.model}")
    print(f"Efforts: {args.efforts}")
    print()

    failures = 0
    for effort in args.efforts:
        # The ablation runner uses the string "none" for the no-thinking case
        # whereas the rest of the harness uses Python None; LLMClient stores
        # whatever it's given. qwen_mlx normalizes both to a single sentinel.
        client = LLMClient(provider="qwen-mlx", model=args.model,
                           reasoning_effort=None if effort == "none" else effort)
        t0 = time.time()
        try:
            reply = client.generate(args.system, args.user)
        except Exception as exc:
            failures += 1
            print(f"[{effort:>7}] ERROR: {exc}")
            continue
        dt = time.time() - t0
        usage = client.get_usage_summary()
        snippet = reply.strip().replace("\n", " ")[:80]
        print(f"[{effort:>7}] {dt:5.1f}s  "
              f"in={usage['input_tokens']:>4}  out={usage['output_tokens']:>4}  "
              f"reply={snippet!r}")
        if not reply.strip():
            failures += 1
            print(f"           (empty reply — counted as failure)")

    print()
    if failures:
        print(f"FAILED: {failures}/{len(args.efforts)} efforts errored or returned empty")
        return 1
    print(f"OK: {len(args.efforts)}/{len(args.efforts)} efforts produced replies")
    return 0


if __name__ == "__main__":
    sys.exit(main())
