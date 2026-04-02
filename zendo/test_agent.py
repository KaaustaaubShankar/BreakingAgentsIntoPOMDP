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
    load_dotenv()
except ImportError:
    pass


class LLMClient:
    def __init__(self, provider: str, model: str):
        self.provider = provider.lower()
        self.model = model
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "openai":
            import openai
            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            return resp.choices[0].message.content
            
        elif self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return resp.content[0].text
            
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(self.model, system_instruction=system_prompt)
            resp = model_inst.generate_content(user_prompt)
            return resp.text
            
        else:
            raise ValueError(f"Unknown provider {self.provider}. Choose from openai, anthropic, gemini.")


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
    ).to_dict()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Agent for Lithic Array")
    parser.add_argument("--agent", type=str, choices=["mock", "llm"], default="llm", help="Choose 'mock' for local rules or 'llm' for AI agent.")
    parser.add_argument("--provider", type=str, choices=["openai", "anthropic", "gemini"], default="openai", help="LLM Provider")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name (e.g. gpt-4o, claude-3-5-sonnet-20241022, gemini-2.0-flash)")
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
