"""MLX-based Qwen local inference. Apple Silicon (M-series) only.

Parallel to qwen_local.py (transformers-based, CPU/CUDA). Loads MLX
weights once per process via module-level cache keyed by model name.
Maps the harness-wide `reasoning_effort` knob to Qwen 3's
`enable_thinking` chat-template toggle plus a proportional max_tokens
budget.

Model IDs typically come from the `mlx-community/` HuggingFace org,
e.g. `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`.

Imported lazily from ka59_game/llm_client.py's qwen-mlx branch so
non-Apple environments don't hit a hard import error at top-of-file.
"""
from __future__ import annotations

import re
from typing import Any

_MLX_CACHE: dict[str, Any] = {}

# Same shape as qwen_local._MAX_NEW_TOKENS — proportional budget per
# reasoning level. The runtime ablation script passes the *string* "none"
# while the rest of the harness uses Python None for the same meaning;
# both must disable thinking and use the small budget.
_MAX_NEW_TOKENS = {
    None: 1024,
    "none": 1024,
    "minimal": 1536,
    "low": 2560,
    "medium": 4608,
    "high": 8704,
    "xhigh": 16896,
}

_NO_THINKING = {None, "none"}


def _load(model_name: str) -> Any:
    if model_name in _MLX_CACHE:
        return _MLX_CACHE[model_name]
    from mlx_lm import load
    model, tokenizer = load(model_name)
    _MLX_CACHE[model_name] = (model, tokenizer)
    return _MLX_CACHE[model_name]


def generate(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str | None,
) -> tuple[str, int, int]:
    """Run one MLX Qwen generation.

    Returns (visible_text, input_tokens, output_tokens). The Qwen
    <think>...</think> block (when emitted) is stripped before return
    so callers see only the visible reply.
    """
    from mlx_lm import generate as mlx_generate

    model, tokenizer = _load(model_name)
    enable_thinking = reasoning_effort not in _NO_THINKING
    max_tokens = _MAX_NEW_TOKENS.get(reasoning_effort, 1024)

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    # mlx_lm.generate returns the generated text directly. To get token
    # counts, encode the prompt ourselves and count the generated chars
    # roughly via re-encoding the response.
    input_tokens = len(tokenizer.encode(prompt))
    full_text = mlx_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    output_tokens = len(tokenizer.encode(full_text))

    match = re.search(r"</think>\s*", full_text)
    content = full_text[match.end():] if match else full_text
    return content.strip(), int(input_tokens), int(output_tokens)
