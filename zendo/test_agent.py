import json
import os
import argparse
import re
from visual_zendo import LithicArrayEnv, WorldAxis, GoalAxis, MechanicsAxis, FeedbackAxis, Arrangement, Shape
from rules import get_rule_by_index, generate_initial_examples, create_counter_example_generator, generate_random_arrangement

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

def parse_json_from_text(text: str) -> dict:
    text = text.strip()
    # Try to find JSON block
    match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except Exception as e:
        print("Failed to decode JSON from LLM: ", text)
        raise e

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

def run_llm_agent(provider: str, model: str, max_turns: int = 20, world: WorldAxis = WorldAxis.EASY, goal: GoalAxis = GoalAxis.EASY, mechanics: MechanicsAxis = MechanicsAxis.EASY, feedback: FeedbackAxis = FeedbackAxis.EASY, rule_index: int = 1):
    print(f"--- Starting LLM Agent ({provider} / {model}) ---")
    
    client = LLMClient(provider, model)
    env = LithicArrayEnv(world=world, goal=goal, mechanics=mechanics, feedback=feedback)
    
    true_rule_name, true_rule_eval_fn = get_rule_by_index(rule_index)
    ce_generator = create_counter_example_generator(true_rule_eval_fn)
    initial_examples = generate_initial_examples(true_rule_eval_fn)
    
    presentation = env.reset(initial_examples, true_rule_name, true_rule_eval_fn, ce_generator)
    
    system_prompt = f"""{presentation['instruction']}
{presentation['mechanics']}

Valid colors: red, blue, green, yellow, black. Sizes: small, medium, large. Types: triangle, circle, square."""

    history_log = f"Basalt Initialization:\nExamples: {json.dumps(presentation['initial_examples'], indent=2)}\n\n"
    
    turns_taken = 0
    won = False
    for turn in range(max_turns):
        turns_taken = turn + 1
        print(f"\n--- Turn {turns_taken} | Tokens: {env.tokens} ---")        
        prompt = f"Current Game State History:\n{history_log}\nChoose your next action (STRATA or PROPOSE)."
        
        try:
            llm_response = client.generate(system_prompt, prompt)
            print(f"LLM Response:\n{llm_response}")
            action_data = parse_json_from_text(llm_response)
        except Exception as e:
            msg = f"Error querying LLM or parsing output: {e}"
            print(msg)
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
            print(log)
            history_log += log + "\n"
        elif action == "PROPOSE":
            desc = action_data.get("rule_description", "")
            code_str = action_data.get("rule_code", "")
            
            # Execute the function string safely into a local dictionary
            local_env = {}
            try:
                exec(code_str, globals(), local_env)
                agent_eval_fn = local_env['agent_eval_fn']
            except Exception as e:
                log = f"Action: PROPOSE | Failed to parse agent_eval_fn: {e}"
                print(log)
                history_log += log + "\n"
                continue
                
            res = env.propose_rule(desc, agent_eval_fn)
            log = f"Action: PROPOSE | Rule: {desc}\nResult: {res}"
            print(log)
            history_log += log + "\n"
            
            if res.get("result") == "Accepted":
                print("\nAgent won!!")
                won = True
                break
        else:
            log = f"System Error: Unrecognized action '{action}'"
            print(log)
            history_log += log + "\n"
            
    print(f"\nGame Over. Turns taken: {turns_taken}")
    
    understanding_prompt = (
        "The game has ended. Based on your experience playing, please explain your inferred understanding of:\n"
        "1. The Goal of the game.\n"
        "2. The Mechanics of the game.\n\n"
        "Output ONLY a JSON block with exactly two keys: 'goal_understanding' (string) and 'mechanics_understanding' (string)."
    )
    print("--- Eliciting Agent Understanding ---")
    try:
        final_history = f"Current Game State History:\n{history_log}\n{understanding_prompt}"
        resp = client.generate(system_prompt, final_history)
        und_data = parse_json_from_text(resp)
        print(f"Goal Understanding: {und_data.get('goal_understanding', '')}")
        print(f"Mechanics Understanding: {und_data.get('mechanics_understanding', '')}")
        env._log_event("agent_understanding", und_data)
    except Exception as e:
        print(f"Failed to get understanding: {e}")

    history_file = env.save_history()
    print(f"\nSaved history to {history_file}")


def run_mock_agent(world: WorldAxis = WorldAxis.EASY, goal: GoalAxis = GoalAxis.EASY, mechanics: MechanicsAxis = MechanicsAxis.EASY, feedback: FeedbackAxis = FeedbackAxis.EASY):
    print("--- Starting Mock Test Agent ---")
    env = LithicArrayEnv(world=world, goal=goal, mechanics=mechanics, feedback=feedback)
    true_rule_name, true_rule_eval_fn = get_rule_by_index(1)
    ce_generator = create_counter_example_generator(true_rule_eval_fn)
    initial_examples = generate_initial_examples(true_rule_eval_fn)
    
    env.reset(initial_examples, true_rule_name, true_rule_eval_fn, ce_generator)
    
    # 1. Incorrect Strata
    arr1 = generate_random_arrangement()
    env.strata(arr1, not true_rule_eval_fn(arr1))
    
    # 2. Correct Strata
    arr2 = generate_random_arrangement()
    env.strata(arr2, true_rule_eval_fn(arr2))
    
    # 3. Incorrect Proposal
    bad_rule_name = "Exactly two large shapes"
    _, bad_eval = get_rule_by_index(2)
    env.propose_rule(bad_rule_name, bad_eval)
    
    # 4. Another Correct Strata for token
    arr3 = generate_random_arrangement()
    env.strata(arr3, true_rule_eval_fn(arr3))
    
    # 5. Correct Proposal
    env.propose_rule(true_rule_name, true_rule_eval_fn)
    
    history_file = env.save_history()
    print(f"Mock run complete. Saved history to {history_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Agent for Lithic Array")
    parser.add_argument("--agent", type=str, choices=["mock", "llm"], default="mock", help="Choose 'mock' for local rules or 'llm' for AI agent.")
    parser.add_argument("--provider", type=str, choices=["openai", "anthropic", "gemini"], default="openai", help="LLM Provider")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name (e.g. gpt-4o, claude-3-5-sonnet-20241022, gemini-2.0-flash)")
    parser.add_argument("--turns", type=int, default=100, help="Max interaction turns for LLM")
    parser.add_argument("--world", type=str, default="EASY", help="Axis level: easy, medium, hard/high.")
    parser.add_argument("--goal", type=str, default="EASY", help="Axis level: easy, medium, hard/high.")
    parser.add_argument("--mechanics", type=str, default="EASY", help="Axis level: easy, medium, hard/high.")
    parser.add_argument("--feedback", type=str, default="EASY", help="Axis level: easy, medium, hard/high.")
    parser.add_argument("--rule-index", type=int, default=1, help="Index of the rule to evaluate from rules.py")
    
    args = parser.parse_args()
    
    world_axis = parse_axis_value(args.world, WorldAxis, "world")
    goal_axis = parse_axis_value(args.goal, GoalAxis, "goal")
    mechanics_axis = parse_axis_value(args.mechanics, MechanicsAxis, "mechanics")
    feedback_axis = parse_axis_value(args.feedback, FeedbackAxis, "feedback")
    
    if args.agent == "mock":
        run_mock_agent(world=world_axis, goal=goal_axis, mechanics=mechanics_axis, feedback=feedback_axis)
    else:
        run_llm_agent(args.provider, args.model, args.turns, world=world_axis, goal=goal_axis, mechanics=mechanics_axis, feedback=feedback_axis, rule_index=args.rule_index)
