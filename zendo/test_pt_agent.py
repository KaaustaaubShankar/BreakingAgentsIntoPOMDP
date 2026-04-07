"""
test_pt_agent.py — LLM agent runner for the Parameter Tuning environment.

Mirrors the structure of test_agent.py (Zendo) so both environments share
the same harness conventions and the ablation study can drive either one.

Usage:
    python test_pt_agent.py --agent llm --provider openrouter --model openai/gpt-4o
    python test_pt_agent.py --agent mock
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from parameter_tuning import (
    ParameterTuningEnv,
    create_counter_example_generator,
    generate_initial_examples,
    get_rule_by_index,
    get_target_by_index,
)
from visual_zendo import WorldAxis, GoalAxis, MechanicsAxis, FeedbackAxis

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# LLM client (reuse from test_agent — same OpenRouter wrapper)
# ---------------------------------------------------------------------------

class LLMClient:
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, provider: str, model: str):
        self.provider = provider.lower()
        self.model = model
        self.reset_usage()

    def reset_usage(self):
        self.last_usage = self._empty_usage()
        self.usage_totals = self._empty_usage()

    def _empty_usage(self) -> Dict[str, int]:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "calls": 0, "calls_with_usage": 0}

    def _extract_usage(self, response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage", None)
        def _val(u, *keys):
            for k in keys:
                v = u.get(k) if isinstance(u, dict) else getattr(u, k, None)
                if v is not None:
                    try: return int(v)
                    except (TypeError, ValueError): pass
            return None

        inp = _val(usage, "input_tokens", "prompt_tokens") or 0
        out = _val(usage, "output_tokens", "completion_tokens") or 0
        tot = _val(usage, "total_tokens") or (inp + out)
        has = usage is not None and any(x > 0 for x in (inp, out, tot))
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": tot,
                "calls": 1, "calls_with_usage": 1 if has else 0}

    def _record_usage(self, response: Any):
        u = self._extract_usage(response)
        self.last_usage = u
        for k, v in u.items():
            self.usage_totals[k] += v

    def get_usage_summary(self) -> Dict[str, int]:
        return dict(self.usage_totals)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable is required.")
            import openai
            client = openai.OpenAI(base_url=self.OPENROUTER_BASE_URL, api_key=api_key)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            self._record_usage(resp)
            content = resp.choices[0].message.content
            if content is None:
                raise ValueError("OpenRouter returned empty content.")
            return content
        raise ValueError(f"Unknown provider: {self.provider}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class RunLogger:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.messages: List[str] = []

    def log(self, msg: str):
        self.messages.append(msg)
        if self.verbose:
            print(msg)


@dataclass
class PTRunResult:
    agent: str
    won: bool
    turns_taken: int
    max_turns: int
    tokens_remaining: int
    history_file: Optional[str]
    history_log: str
    logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    provider: Optional[str] = None
    model: Optional[str] = None
    target_index: Optional[int] = None
    true_target_name: Optional[str] = None
    llm_usage: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from LLM output: {text}") from e


def _build_system_prompt(presentation: Dict[str, Any]) -> str:
    param_range = presentation.get("param_range", [0, 10])
    param_names = presentation.get("param_names", ["P1", "P2", "P3"])
    return (
        f"{presentation['instruction']}\n"
        f"{presentation['mechanics']}\n\n"
        f"Parameters: {', '.join(param_names)} (integers, range {param_range[0]}-{param_range[1]})"
    )


def _build_history_log(presentation: Dict[str, Any]) -> str:
    return (
        "Initial examples:\n"
        + json.dumps(presentation["initial_examples"], indent=2)
        + "\n\n"
    )


# ---------------------------------------------------------------------------
# LLM agent runner
# ---------------------------------------------------------------------------

def run_pt_llm_agent(
    provider: str,
    model: str,
    max_turns: int = 40,
    world: WorldAxis = WorldAxis.EASY,
    goal: GoalAxis = GoalAxis.EASY,
    mechanics: MechanicsAxis = MechanicsAxis.EASY,
    feedback: FeedbackAxis = FeedbackAxis.EASY,
    target_index: int = 0,
    *,
    client: Optional[Any] = None,
    verbose: bool = True,
    save_history: bool = True,
    artifacts_dir: Optional[str] = None,
):
    logger = RunLogger(verbose=verbose)
    logger.log(f"--- PT Agent ({provider}/{model}) | Target {target_index} ---")

    env_kwargs: Dict[str, Any] = dict(world=world, goal=goal, mechanics=mechanics, feedback=feedback)
    if artifacts_dir:
        env_kwargs["artifacts_dir"] = artifacts_dir
    env = ParameterTuningEnv(**env_kwargs)

    true_name, true_fn = get_rule_by_index(target_index)
    hidden_target = get_target_by_index(target_index)
    ce_gen = create_counter_example_generator(true_fn)
    examples = generate_initial_examples(true_fn)

    presentation = env.reset(examples, true_name, true_fn, ce_gen, hidden_target=hidden_target)

    client = client or LLMClient(provider, model)
    client.reset_usage()

    system_prompt = _build_system_prompt(presentation)
    history_log = _build_history_log(presentation)
    turns_taken = 0
    won = False
    errors: List[str] = []

    for turn in range(max_turns):
        turns_taken = turn + 1
        logger.log(f"\n--- Turn {turns_taken} | Tokens: {env.tokens} ---")
        prompt = f"Game history so far:\n{history_log}\nChoose your next action (SET or PROPOSE)."

        try:
            llm_response = client.generate(system_prompt, prompt)
            logger.log(f"LLM:\n{llm_response}")
            action_data = _parse_json(llm_response)
        except Exception as e:
            msg = f"Error: {e}"
            errors.append(msg)
            logger.log(msg)
            history_log += f"\nTurn {turns_taken} Error: {msg}\n"
            continue

        action = action_data.get("action", "").upper()

        if action == "SET":
            params = action_data.get("params", {})
            prediction = bool(action_data.get("prediction", False))
            res = env.set_params(params, prediction)
            log = f"SET {params} predict={prediction} → {res}"
            logger.log(log)
            history_log += log + "\n"

        elif action == "PROPOSE":
            # Agent submits a target vector guess, not Python code
            guessed = action_data.get("target", action_data.get("params", {}))
            if not guessed:
                msg = "PROPOSE missing 'target' field"
                errors.append(msg)
                logger.log(msg)
                history_log += msg + "\n"
                continue
            try:
                guessed = {k: int(v) for k, v in guessed.items()}
            except Exception as e:
                msg = f"PROPOSE bad target format: {e}"
                errors.append(msg)
                logger.log(msg)
                history_log += msg + "\n"
                continue

            res = env.propose_target(guessed)
            log = f"PROPOSE target={guessed} → {res}"
            logger.log(log)
            history_log += log + "\n"

            if res.get("result") == "Accepted":
                logger.log("Agent won!")
                won = True
                break

        else:
            msg = f"Unrecognised action '{action}'"
            errors.append(msg)
            logger.log(msg)
            history_log += msg + "\n"

    logger.log(f"\nGame over. Turns: {turns_taken}")

    llm_usage = client.get_usage_summary() if hasattr(client, "get_usage_summary") else {}
    env._log_event("llm_usage_summary", {"provider": provider, "model": model, **llm_usage})

    history_file = None
    if save_history:
        history_file = env.save_history()
        logger.log(f"Saved history → {history_file}")

    return PTRunResult(
        agent="llm",
        provider=provider,
        model=model,
        won=won,
        turns_taken=turns_taken,
        max_turns=max_turns,
        tokens_remaining=env.tokens,
        history_file=history_file,
        history_log=history_log,
        logs=logger.messages,
        errors=errors,
        target_index=target_index,
        true_target_name=true_name,
        llm_usage=llm_usage,
    ).to_dict()


# ---------------------------------------------------------------------------
# Mock agent (for local smoke-testing without an API key)
# ---------------------------------------------------------------------------

def run_pt_mock_agent(
    world: WorldAxis = WorldAxis.EASY,
    goal: GoalAxis = GoalAxis.EASY,
    mechanics: MechanicsAxis = MechanicsAxis.EASY,
    feedback: FeedbackAxis = FeedbackAxis.EASY,
    target_index: int = 0,
    *,
    verbose: bool = True,
    save_history: bool = True,
    artifacts_dir: Optional[str] = None,
):
    logger = RunLogger(verbose=verbose)
    logger.log("--- PT Mock Agent ---")

    env_kwargs: Dict[str, Any] = dict(world=world, goal=goal, mechanics=mechanics, feedback=feedback)
    if artifacts_dir:
        env_kwargs["artifacts_dir"] = artifacts_dir
    env = ParameterTuningEnv(**env_kwargs)

    true_name, true_fn = get_rule_by_index(target_index)
    hidden_target = get_target_by_index(target_index)
    ce_gen = create_counter_example_generator(true_fn)
    examples = generate_initial_examples(true_fn)
    env.reset(examples, true_name, true_fn, ce_gen, hidden_target=hidden_target)

    # Probe with the actual target (should activate) and a random config (likely not)
    env.set_params(hidden_target, prediction=True)
    env.set_params({k: 0 for k in hidden_target}, prediction=False)

    # Propose a wrong target, then the correct one
    wrong = {k: 0 for k in hidden_target}
    env.propose_target(wrong)
    result = env.propose_target(hidden_target)  # exact match = accepted

    llm_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                 "calls": 0, "calls_with_usage": 0}
    env._log_event("llm_usage_summary", {"provider": None, "model": None, **llm_usage})

    history_file = None
    if save_history:
        history_file = env.save_history()
        logger.log(f"Mock run done. History → {history_file}")

    return PTRunResult(
        agent="mock",
        won=result.get("result") == "Accepted",
        turns_taken=4,
        max_turns=4,
        tokens_remaining=env.tokens,
        history_file=history_file,
        history_log="",
        logs=logger.messages,
        target_index=target_index,
        true_target_name=true_name,
        llm_usage=llm_usage,
    ).to_dict()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_axis(value: str, axis_type, name: str):
    aliases = {"LOW": "EASY", "HIGH": "HARD"}
    norm = aliases.get(value.strip().upper(), value.strip().upper())
    try:
        return axis_type[norm]
    except KeyError:
        valid = ", ".join(m.name.lower() for m in axis_type)
        raise argparse.ArgumentTypeError(f"Invalid {name} '{value}'. Expected: {valid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PT Agent for Parameter Tuning environment")
    parser.add_argument("--agent", choices=["mock", "llm"], default="llm")
    parser.add_argument("--provider", choices=["openrouter"], default="openrouter")
    parser.add_argument("--model", default="openai/gpt-4o")
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--world", default="EASY")
    parser.add_argument("--goal", default="EASY")
    parser.add_argument("--mechanics", default="EASY")
    parser.add_argument("--feedback", default="EASY")
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    world = _parse_axis(args.world, WorldAxis, "world")
    goal = _parse_axis(args.goal, GoalAxis, "goal")
    mechanics = _parse_axis(args.mechanics, MechanicsAxis, "mechanics")
    feedback = _parse_axis(args.feedback, FeedbackAxis, "feedback")

    if args.agent == "mock":
        run_pt_mock_agent(world=world, goal=goal, mechanics=mechanics, feedback=feedback,
                          target_index=args.target_index, verbose=not args.quiet)
    else:
        run_pt_llm_agent(args.provider, args.model, args.turns,
                         world=world, goal=goal, mechanics=mechanics, feedback=feedback,
                         target_index=args.target_index, verbose=not args.quiet)
