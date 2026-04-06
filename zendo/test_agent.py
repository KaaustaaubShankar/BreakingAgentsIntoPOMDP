import argparse
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from rules import (
    create_counter_example_generator,
    generate_initial_examples,
    generate_random_arrangement,
    get_rule_by_index,
)
from visual_zendo import (
    Arrangement,
    FeedbackAxis,
    GoalAxis,
    LithicArrayEnv,
    MechanicsAxis,
    Shape,
    WorldAxis,
)

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


class LLMClient:
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, provider: str, model: str):
        self.provider = provider.lower()
        self.model = model
        self.reset_usage()

    def reset_usage(self):
        self.last_usage = self._empty_usage_summary()
        self.usage_totals = self._empty_usage_summary()

    def _empty_usage_summary(self) -> Dict[str, int]:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
            "calls_with_usage": 0,
        }

    def _usage_value(self, usage: Any, *keys: str) -> Optional[int]:
        for key in keys:
            if isinstance(usage, dict):
                value = usage.get(key)
            else:
                value = getattr(usage, key, None)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    def _extract_usage(self, response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage", None)
        input_tokens = self._usage_value(usage, "input_tokens", "prompt_tokens")
        output_tokens = self._usage_value(usage, "output_tokens", "completion_tokens")
        total_tokens = self._usage_value(usage, "total_tokens")
        has_usage = usage is not None and any(
            value is not None for value in (input_tokens, output_tokens, total_tokens)
        )

        input_tokens = input_tokens or 0
        output_tokens = output_tokens or 0
        total_tokens = total_tokens if total_tokens is not None else input_tokens + output_tokens

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "calls": 1,
            "calls_with_usage": 1 if has_usage else 0,
        }

    def _record_usage(self, response: Any):
        usage = self._extract_usage(response)
        self.last_usage = usage
        for key, value in usage.items():
            self.usage_totals[key] += value

    def get_usage_summary(self) -> Dict[str, int]:
        return dict(self.usage_totals)

    def _openrouter_client(self):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required when using OpenRouter.")

        import openai

        client_kwargs = {
            "base_url": self.OPENROUTER_BASE_URL,
            "api_key": api_key,
        }
        default_headers = {}
        referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        app_title = os.environ.get("OPENROUTER_APP_TITLE")
        if referer:
            default_headers["HTTP-Referer"] = referer
        if app_title:
            default_headers["X-Title"] = app_title
        if default_headers:
            client_kwargs["default_headers"] = default_headers

        return openai.OpenAI(**client_kwargs)
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "openrouter":
            client = self._openrouter_client()
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            self._record_usage(resp)
            content = resp.choices[0].message.content
            if content is None:
                raise ValueError("OpenRouter returned an empty message content.")
            return content

        raise ValueError(f"Unknown provider {self.provider}. Choose from openrouter.")


class RunLogger:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.messages: List[str] = []

    def log(self, message: str):
        self.messages.append(message)
        if self.verbose:
            print(message)


@dataclass
class AgentRunResult:
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
    rule_index: Optional[int] = None
    true_rule_name: Optional[str] = None
    understanding: Optional[Dict[str, str]] = None
    llm_usage: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_json_from_text(text: str) -> dict:
    text = text.strip()
    match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to decode JSON from LLM output: {text}") from exc

def parse_axis_value(value: str, axis_type, axis_name: str):
    normalized = value.strip().upper()
    aliases = {
        "LOW": "EASY",
        "HIGH": "HARD",
    }
    normalized = aliases.get(normalized, normalized)
    try:
        return axis_type[normalized]
    except KeyError as exc:
        valid_values = ", ".join(member.name.lower() for member in axis_type)
        raise argparse.ArgumentTypeError(
            f"Invalid {axis_name} value '{value}'. Expected one of: {valid_values}, high, low."
        ) from exc

def _build_environment(
    *,
    world: WorldAxis,
    goal: GoalAxis,
    mechanics: MechanicsAxis,
    feedback: FeedbackAxis,
    rule_index: int,
    artifacts_dir: Optional[str] = None,
):
    env_kwargs = {
        "world": world,
        "goal": goal,
        "mechanics": mechanics,
        "feedback": feedback,
    }
    if artifacts_dir is not None:
        env_kwargs["artifacts_dir"] = artifacts_dir

    env = LithicArrayEnv(**env_kwargs)
    true_rule_name, true_rule_eval_fn = get_rule_by_index(rule_index)
    ce_generator = create_counter_example_generator(true_rule_eval_fn)
    initial_examples = generate_initial_examples(true_rule_eval_fn)
    presentation = env.reset(initial_examples, true_rule_name, true_rule_eval_fn, ce_generator)

    return env, presentation, true_rule_name, true_rule_eval_fn


def _build_system_prompt(presentation: Dict[str, Any]) -> str:
    return f"""{presentation['instruction']}
{presentation['mechanics']}

Valid colors: red, blue, green, yellow, black. Sizes: small, medium, large. Types: triangle, circle, square."""


def _build_history_log(presentation: Dict[str, Any]) -> str:
    return f"Basalt Initialization:\nExamples: {json.dumps(presentation['initial_examples'], indent=2)}\n\n"


def _save_history(env: LithicArrayEnv, save_history: bool) -> Optional[str]:
    if not save_history:
        return None
    return env.save_history()


def _empty_llm_usage_summary() -> Dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
        "calls_with_usage": 0,
    }


def _reset_client_usage(client: Any):
    if hasattr(client, "reset_usage"):
        client.reset_usage()


def _get_client_usage_summary(client: Any) -> Dict[str, int]:
    if hasattr(client, "get_usage_summary"):
        usage = client.get_usage_summary()
        if usage is not None:
            return usage
    return _empty_llm_usage_summary()


def run_llm_agent(
    provider: str,
    model: str,
    max_turns: int = 20,
    world: WorldAxis = WorldAxis.EASY,
    goal: GoalAxis = GoalAxis.EASY,
    mechanics: MechanicsAxis = MechanicsAxis.EASY,
    feedback: FeedbackAxis = FeedbackAxis.EASY,
    rule_index: int = 1,
    *,
    client: Optional[Any] = None,
    verbose: bool = True,
    save_history: bool = True,
    artifacts_dir: Optional[str] = None,
):
    logger = RunLogger(verbose=verbose)
    logger.log(f"--- Starting LLM Agent ({provider} / {model}) ---")

    client = client or LLMClient(provider, model)
    _reset_client_usage(client)
    env, presentation, true_rule_name, _ = _build_environment(
        world=world,
        goal=goal,
        mechanics=mechanics,
        feedback=feedback,
        rule_index=rule_index,
        artifacts_dir=artifacts_dir,
    )

    system_prompt = _build_system_prompt(presentation)
    history_log = _build_history_log(presentation)
    turns_taken = 0
    won = False
    understanding = None
    errors: List[str] = []

    for turn in range(max_turns):
        turns_taken = turn + 1
        logger.log(f"\n--- Turn {turns_taken} | Tokens: {env.tokens} ---")
        prompt = f"Current Game State History:\n{history_log}\nChoose your next action (STRATA or PROPOSE)."

        try:
            llm_response = client.generate(system_prompt, prompt)
            logger.log(f"LLM Response:\n{llm_response}")
            action_data = parse_json_from_text(llm_response)
        except Exception as e:
            msg = f"Error querying LLM or parsing output: {e}"
            errors.append(msg)
            logger.log(msg)
            history_log += f"\nTurn {turn+1} System Error: {msg}\n"
            continue

        action = action_data.get("action")
        if action == "STRATA":
            shapes_data = action_data.get("arrangement", [])
            pred = action_data.get("prediction", False)
            shapes = [Shape(**s) for s in shapes_data]
            arr = Arrangement(shapes=shapes)
            res = env.strata(arr, pred)
            log = f"Action: STRATA | Predicted: {pred} | Arrangement: {shapes_data}\nResult: {res}"
            logger.log(log)
            history_log += log + "\n"
        elif action == "PROPOSE":
            desc = action_data.get("rule_description", "")
            code_str = action_data.get("rule_code", "")

            local_env = {}
            try:
                exec(code_str, {}, local_env)
                agent_eval_fn = local_env["agent_eval_fn"]
            except Exception as e:
                log = f"Action: PROPOSE | Failed to parse agent_eval_fn: {e}"
                errors.append(log)
                logger.log(log)
                history_log += log + "\n"
                continue

            res = env.propose_rule(desc, agent_eval_fn)
            log = f"Action: PROPOSE | Rule: {desc}\nResult: {res}"
            logger.log(log)
            history_log += log + "\n"

            if res.get("result") == "Accepted":
                logger.log("\nAgent won!!")
                won = True
                break
        else:
            log = f"System Error: Unrecognized action '{action}'"
            errors.append(log)
            logger.log(log)
            history_log += log + "\n"

    logger.log(f"\nGame Over. Turns taken: {turns_taken}")

    understanding_prompt = (
        "The game has ended. Based on your experience playing, please explain your inferred understanding of:\n"
        "1. The Goal of the game.\n"
        "2. The Mechanics of the game.\n\n"
        "Output ONLY a JSON block with exactly two keys: 'goal_understanding' (string) and 'mechanics_understanding' (string)."
    )
    logger.log("--- Eliciting Agent Understanding ---")
    try:
        final_history = f"Current Game State History:\n{history_log}\n{understanding_prompt}"
        resp = client.generate(system_prompt, final_history)
        understanding = parse_json_from_text(resp)
        logger.log(f"Goal Understanding: {understanding.get('goal_understanding', '')}")
        logger.log(f"Mechanics Understanding: {understanding.get('mechanics_understanding', '')}")
        env._log_event("agent_understanding", understanding)
    except Exception as e:
        msg = f"Failed to get understanding: {e}"
        errors.append(msg)
        logger.log(msg)

    llm_usage = _get_client_usage_summary(client)
    env._log_event("llm_usage_summary", {
        "provider": provider,
        "model": model,
        **llm_usage,
    })
    logger.log(
        "LLM token usage: "
        f"{llm_usage['input_tokens']} input, "
        f"{llm_usage['output_tokens']} output "
        f"across {llm_usage['calls']} calls."
    )

    history_file = _save_history(env, save_history)
    if history_file is not None:
        logger.log(f"\nSaved history to {history_file}")

    return AgentRunResult(
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
        rule_index=rule_index,
        true_rule_name=true_rule_name,
        understanding=understanding,
        llm_usage=llm_usage,
    ).to_dict()


def run_mock_agent(
    world: WorldAxis = WorldAxis.EASY,
    goal: GoalAxis = GoalAxis.EASY,
    mechanics: MechanicsAxis = MechanicsAxis.EASY,
    feedback: FeedbackAxis = FeedbackAxis.EASY,
    rule_index: int = 1,
    *,
    verbose: bool = True,
    save_history: bool = True,
    artifacts_dir: Optional[str] = None,
):
    logger = RunLogger(verbose=verbose)
    logger.log("--- Starting Mock Test Agent ---")
    env, _, true_rule_name, true_rule_eval_fn = _build_environment(
        world=world,
        goal=goal,
        mechanics=mechanics,
        feedback=feedback,
        rule_index=rule_index,
        artifacts_dir=artifacts_dir,
    )

    arr1 = generate_random_arrangement()
    env.strata(arr1, not true_rule_eval_fn(arr1))

    arr2 = generate_random_arrangement()
    env.strata(arr2, true_rule_eval_fn(arr2))

    bad_rule_name = "Exactly two large shapes"
    _, bad_eval = get_rule_by_index(2)
    env.propose_rule(bad_rule_name, bad_eval)

    arr3 = generate_random_arrangement()
    env.strata(arr3, true_rule_eval_fn(arr3))
    final_response = env.propose_rule(true_rule_name, true_rule_eval_fn)
    llm_usage = _empty_llm_usage_summary()
    env._log_event("llm_usage_summary", {
        "provider": None,
        "model": None,
        **llm_usage,
    })

    history_file = _save_history(env, save_history)
    if history_file is not None:
        logger.log(f"Mock run complete. Saved history to {history_file}")

    return AgentRunResult(
        agent="mock",
        won=final_response.get("result") == "Accepted",
        turns_taken=5,
        max_turns=5,
        tokens_remaining=env.tokens,
        history_file=history_file,
        history_log="",
        logs=logger.messages,
        rule_index=rule_index,
        true_rule_name=true_rule_name,
        llm_usage=llm_usage,
    ).to_dict()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Agent for Lithic Array")
    parser.add_argument("--agent", type=str, choices=["mock", "llm"], default="llm", help="Choose 'mock' for local rules or 'llm' for AI agent.")
    parser.add_argument("--provider", type=str, choices=["openrouter"], default="openrouter", help="LLM Provider")
    parser.add_argument("--model", type=str, default="openai/gpt-4o", help="OpenRouter model name (e.g. openai/gpt-4o)")
    parser.add_argument("--turns", type=int, default=100, help="Max interaction turns for LLM")
    parser.add_argument("--world", type=str, default="EASY", help="Axis level: easy or hard/high.")
    parser.add_argument("--goal", type=str, default="EASY", help="Axis level: easy or hard/high.")
    parser.add_argument("--mechanics", type=str, default="EASY", help="Axis level: easy or hard/high.")
    parser.add_argument("--feedback", type=str, default="EASY", help="Axis level: easy or hard/high.")
    parser.add_argument("--rule-index", type=int, default=1, help="Index of the rule to evaluate from rules.py")
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step terminal logging.")
    
    args = parser.parse_args()
    
    world_axis = parse_axis_value(args.world, WorldAxis, "world")
    goal_axis = parse_axis_value(args.goal, GoalAxis, "goal")
    mechanics_axis = parse_axis_value(args.mechanics, MechanicsAxis, "mechanics")
    feedback_axis = parse_axis_value(args.feedback, FeedbackAxis, "feedback")
    
    if args.agent == "mock":
        run_mock_agent(
            world=world_axis,
            goal=goal_axis,
            mechanics=mechanics_axis,
            feedback=feedback_axis,
            rule_index=args.rule_index,
            verbose=not args.quiet,
        )
    else:
        run_llm_agent(
            args.provider,
            args.model,
            args.turns,
            world=world_axis,
            goal=goal_axis,
            mechanics=mechanics_axis,
            feedback=feedback_axis,
            rule_index=args.rule_index,
            verbose=not args.quiet,
        )
