"""Shared Qwen local-inference helper used by ka59_game / env3 / env4 LLM clients.

Loads weights once per process via the module-level cache. The cache is keyed
by model name, so a single process running multiple ablations against the
same model only pays the ~30-60s load cost once. Maps the harness-wide
`reasoning_effort` knob to Qwen 3's `enable_thinking` toggle plus a
proportional `max_new_tokens` budget.

Only imported lazily from each LLM client's qwen-local branch — local devs
without torch/transformers installed won't hit an import error at top-of-file.
"""
from __future__ import annotations

import re
from typing import Any

_QWEN_CACHE: dict[str, Any] = {}

_MAX_NEW_TOKENS = {
    None: 1024,
    "minimal": 1536,
    "low": 2560,
    "medium": 4608,
    "high": 8704,
}


def _load(model_name: str) -> Any:
    if model_name in _QWEN_CACHE:
        return _QWEN_CACHE[model_name]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.requires_grad_(False)
    _QWEN_CACHE[model_name] = (model, tokenizer)
    return _QWEN_CACHE[model_name]


def generate(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str | None,
) -> tuple[str, int, int]:
    """Run a single Qwen generation.

    Returns (visible_text, input_tokens, output_tokens). The Qwen
    <think>...</think> block (when emitted) is stripped before return so
    callers see only the visible reply, matching API-provider semantics.
    """
    import torch
    model, tokenizer = _load(model_name)
    enable_thinking = reasoning_effort is not None
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    max_new = _MAX_NEW_TOKENS.get(reasoning_effort, 1024)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
        )
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:].tolist()
    full_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    match = re.search(r"</think>\s*", full_text)
    content = full_text[match.end():] if match else full_text
    return content.strip(), int(inputs.input_ids.shape[1]), len(new_tokens)
