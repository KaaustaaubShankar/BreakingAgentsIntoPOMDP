"""
test_chirality_agent.py — LLM agent runner for the Chirality environment (Env4).

Mirrors the structure of test_pt_agent.py so all three environments share the
same harness conventions.

Usage:
    python test_chirality_agent.py --agent llm --provider openrouter --model openai/gpt-4o
    python test_chirality_agent.py --agent mock
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from env4.environment import ChiralityEnv, generate_examples
from env4.prompts import AxisLevel, build_system_prompt, format_classify_result, format_query_result
from env4.rules import get_rule_name

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# LLM Client (reuses OpenRouter pattern from PT)
# ---------------------------------------------------------------------------

class LLMClient:
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, provider: str, model: str):
        self.provider = provider.lower()
        self.model = model
        self.reset_usage()

    def reset_usage(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0

    def _get_client(self):
        import openai
        if self.provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY not set")
            return openai.OpenAI(api_key=api_key, base_url=self.OPENROUTER_BASE_URL)
        elif self.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            return openai.OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def chat(self, messages: List[Dict]) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        usage = response.usage
        if usage:
            self.total_prompt_tokens += usage.prompt_tokens
            self.total_completion_tokens += usage.completion_tokens
            # Estimate cost via OpenRouter usage field if available
            cost_attr = getattr(usage, "cost", None) or getattr(response, "cost", None)
            if cost_attr is not None:
                try:
                    self.total_cost += float(cost_attr)
                except (TypeError, ValueError):
                    pass
        return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def parse_action(text: str) -> Optional[Dict]:
    """Extract JSON action from LLM response."""
    # Try to find JSON block
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            # Normalize: if the agent used a non-standard key but provided an R/L value,
            # treat it as a classify action (needed when mechanics_HARD hides the format)
            if "action" not in parsed:
                for val in parsed.values():
                    if str(val).upper() in ("R", "L"):
                        label = str(val).upper()
                        seq = parsed.get("sequence", None)
                        return {"action": "classify", "sequence": seq, "label": label}
            return parsed
        except json.JSONDecodeError:
            pass
    # Try to find classify/query keywords
    text_lower = text.lower()
    if "query" in text_lower and "classify" not in text_lower:
        return {"action": "query"}
    # Try to extract R or L classification
    label_match = re.search(r'\b([RL])\b', text)
    if label_match:
        # Try to find a sequence (5 uppercase A-E chars)
        seq_match = re.search(r'\b([A-E]{5})\b', text)
        if seq_match:
            return {"action": "classify", "sequence": seq_match.group(1), "label": label_match.group(1)}
    return None


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

@dataclass
class GameHistory:
    rule_index: int
    rule_name: str
    world: str
    goal: str
    mechanics: str
    feedback: str
    turns: List[Dict] = field(default_factory=list)
    won: bool = False
    turns_taken: int = 0
    total_classifies: int = 0
    total_queries: int = 0
    errors: List[str] = field(default_factory=list)


def run_chirality_llm_agent(
    provider: str,
    model: str,
    max_turns: int,
    world: AxisLevel,
    goal: AxisLevel,
    mechanics: AxisLevel,
    feedback: AxisLevel,
    rule_index: int,
    seed: int = 42,
    n_easy_examples: int = 5,
    n_classify_to_win: int = 3,
    verbose: bool = False,
) -> Dict:
    """
    Run one game of ChiralityEnv with an LLM agent.
    The agent must correctly classify n_classify_to_win sequences to win.
    """
    env = ChiralityEnv(rule_index=rule_index, seed=seed)
    client = LLMClient(provider=provider, model=model)

    # Generate initial examples (HARD = zero examples, cold start)
    n_examples = n_easy_examples if world == AxisLevel.EASY else 0
    examples = env.get_initial_examples(n_examples) if n_examples > 0 else []

    system_prompt = build_system_prompt(
        world=world,
        goal=goal,
        mechanics=mechanics,
        feedback=feedback,
        examples=examples,
        n_easy_examples=n_easy_examples,
    )

    import random
    rng = random.Random(seed + 100)
    SYMBOLS = list("ABCDE")

    messages = [{"role": "system", "content": system_prompt}]
    history = GameHistory(
        rule_index=rule_index,
        rule_name=get_rule_name(rule_index),
        world=world.value,
        goal=goal.value,
        mechanics=mechanics.value,
        feedback=feedback.value,
    )

    correct_streak = 0
    turn = 0
    start_time = time.time()

    # Give the agent a sequence to classify
    current_sequence = "".join(rng.choices(SYMBOLS, k=5))
    messages.append({
        "role": "user",
        "content": f"Classify this sequence: {current_sequence}\n\nRespond with JSON."
    })

    while turn < max_turns:
        turn += 1

        try:
            response_text = client.chat(messages)
        except Exception as e:
            history.errors.append(f"Turn {turn}: LLM error: {e}")
            break

        if verbose:
            print(f"\n[Turn {turn}] Agent: {response_text[:200]}")

        action = parse_action(response_text)
        messages.append({"role": "assistant", "content": response_text})

        if action is None:
            if mechanics == AxisLevel.HARD:
                feedback_msg = "Invalid response. Please respond with valid JSON."
            else:
                feedback_msg = "Invalid response. Please respond with JSON: {\"action\": \"classify\", \"sequence\": \"...\", \"label\": \"R\" or \"L\"} or {\"action\": \"query\"}"
            history.errors.append(f"Turn {turn}: parse failure")
        elif action.get("action") == "query":
            if mechanics == AxisLevel.HARD:
                feedback_msg = f"Unknown action. Classify this sequence: {current_sequence}"
                history.errors.append(f"Turn {turn}: query blocked (mechanics_HARD)")
            else:
                history.total_queries += 1
                seq, label = env.query()
                feedback_msg = format_query_result(seq, label)
                feedback_msg += f"\n\nNow classify: {current_sequence}"
        elif action.get("action") == "classify":
            seq = action.get("sequence") or current_sequence
            pred_label = action.get("label", "").upper()
            if pred_label not in ("R", "L"):
                feedback_msg = "Invalid label. Must be 'R' or 'L'."
                history.errors.append(f"Turn {turn}: invalid label {pred_label!r}")
            else:
                result = env.classify(seq, pred_label)
                history.total_classifies += 1
                feedback_msg = format_classify_result(result, feedback)

                if result["correct"]:
                    correct_streak += 1
                    if correct_streak >= n_classify_to_win:
                        history.won = True
                        history.turns_taken = turn
                        if verbose:
                            print(f"\n✓ Won in {turn} turns!")
                        break
                    # Give next sequence to classify
                    current_sequence = "".join(rng.choices(SYMBOLS, k=5))
                    feedback_msg += f"\n\nNext sequence to classify: {current_sequence}"
                else:
                    correct_streak = 0
                    # Give same or new sequence
                    current_sequence = "".join(rng.choices(SYMBOLS, k=5))
                    feedback_msg += f"\n\nNext sequence to classify: {current_sequence}"
        else:
            if mechanics == AxisLevel.HARD:
                feedback_msg = "Invalid response. Please respond with valid JSON."
            else:
                feedback_msg = "Unknown action. Use 'classify' or 'query'."

        messages.append({"role": "user", "content": feedback_msg})

    if not history.won:
        history.turns_taken = turn

    elapsed = time.time() - start_time

    # Build history file
    import tempfile
    history_dir = os.path.join(os.path.dirname(__file__), "game_logs")
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, f"chirality_{rule_index}_{seed}_{int(time.time())}.json")
    with open(history_path, "w") as f:
        json.dump(asdict(history), f, indent=2)

    return {
        "won": history.won,
        "turns_taken": history.turns_taken,
        "total_classifies": history.total_classifies,
        "total_queries": history.total_queries,
        "errors": history.errors,
        "history_file": history_path,
        "elapsed": elapsed,
        "usage": {
            "prompt_tokens": client.total_prompt_tokens,
            "completion_tokens": client.total_completion_tokens,
            "cost": client.total_cost,
        },
        "provider": provider,
        "model": model,
        "rule_index": rule_index,
        "rule_name": history.rule_name,
    }


# ---------------------------------------------------------------------------
# Mock agent (for testing without API calls)
# ---------------------------------------------------------------------------

def run_chirality_mock_agent(rule_index: int, **kwargs) -> Dict:
    """Simple mock agent that randomly classifies."""
    import random
    rng = random.Random(rule_index)
    from env4.rules import get_rule

    env = ChiralityEnv(rule_index=rule_index, seed=kwargs.get("seed", 42))
    rule = get_rule(rule_index)

    SYMBOLS = list("ABCDE")
    n_to_win = kwargs.get("n_classify_to_win", 3)
    max_turns = kwargs.get("max_turns", 40)

    correct = 0
    turns = 0
    while correct < n_to_win and turns < max_turns:
        seq = "".join(rng.choices(SYMBOLS, k=5))
        # Mock agent guesses correctly 60% of the time
        true_label = rule(seq)
        pred = true_label if rng.random() < 0.6 else ("L" if true_label == "R" else "R")
        result = env.classify(seq, pred)
        if result["correct"]:
            correct += 1
        turns += 1

    return {
        "won": correct >= n_to_win,
        "turns_taken": turns,
        "errors": [],
        "history_file": None,
        "elapsed": 0.0,
        "usage": {"cost": 0.0},
        "provider": "mock",
        "model": "mock",
        "rule_index": rule_index,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["llm", "mock"], default="llm")
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--model", default="openai/gpt-4o")
    parser.add_argument("--rule", type=int, default=1)
    parser.add_argument("--world", choices=["EASY", "HARD"], default="EASY")
    parser.add_argument("--goal", choices=["EASY", "HARD"], default="EASY")
    parser.add_argument("--mechanics", choices=["EASY", "HARD"], default="EASY")
    parser.add_argument("--feedback", choices=["EASY", "HARD"], default="EASY")
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.agent == "mock":
        result = run_chirality_mock_agent(rule_index=args.rule, max_turns=args.turns, seed=args.seed)
    else:
        result = run_chirality_llm_agent(
            provider=args.provider,
            model=args.model,
            max_turns=args.turns,
            world=AxisLevel(args.world),
            goal=AxisLevel(args.goal),
            mechanics=AxisLevel(args.mechanics),
            feedback=AxisLevel(args.feedback),
            rule_index=args.rule,
            seed=args.seed,
            verbose=args.verbose,
        )

    print(json.dumps(result, indent=2))
