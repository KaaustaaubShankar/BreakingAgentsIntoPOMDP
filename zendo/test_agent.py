import json
import os
import argparse
import re
from visual_zendo import VisualZendoEnv, WorldAxis, GoalAxis, MechanicsAxis, FeedbackAxis, Arrangement, Shape
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

def run_llm_agent(provider: str, model: str, max_turns: int = 20, world: WorldAxis = WorldAxis.EASY, goal: GoalAxis = GoalAxis.EASY, mechanics: MechanicsAxis = MechanicsAxis.EASY, feedback: FeedbackAxis = FeedbackAxis.EASY):
    print(f"--- Starting LLM Agent ({provider} / {model}) ---")
    
    client = LLMClient(provider, model)
    env = VisualZendoEnv(world=world, goal=goal, mechanics=mechanics, feedback=feedback)
    
    true_rule_name, true_rule_eval_fn = get_rule_by_index(1) # Default: "All pieces are the same color"
    ce_generator = create_counter_example_generator(true_rule_eval_fn)
    initial_examples = generate_initial_examples(true_rule_eval_fn)
    
    presentation = env.reset(initial_examples, true_rule_name, true_rule_eval_fn, ce_generator)
    
    system_prompt = f"""You are an inductive reasoning agent playing Visual Zendo.
Your goal: {presentation['instruction']}
Mechanics: {presentation['mechanics']}

Data structures:
Arrangement is a list of Shapes. Each Shape is a dict: {{"color": str, "size": str, "type_": str}}.
Valid colors: red, blue, green, yellow, black. Sizes: small, medium, large. Types: triangle, circle, square.

You must respond with only a JSON block containing your action. Do not include markdown or other text outside the JSON.
Mondo Action Schema:
{{
  "action": "MONDO",
  "arrangement": [{{"color": "red", "size": "small", "type_": "circle"}}],
  "prediction": true  // true for Harmonious, false for Discordant
}}

Propose Action Schema (Requires 1 token):
{{
  "action": "PROPOSE",
  "rule_description": "All pieces are red",
  "rule_code": "def agent_eval_fn(arr):\n    # arr is an Arrangement obj. You can access shapes via arr.shapes. Each shape has .color, .size, .type_\n    return all(s.color == 'red' for s in arr.shapes)"
}}"""

    history_log = f"Master Initialization:\nExamples: {json.dumps(presentation['initial_examples'], indent=2)}\n\n"
    
    for turn in range(max_turns):
        print(f"\n--- Turn {turn+1} | Tokens: {env.tokens} ---")
        prompt = f"Current Game State History:\n{history_log}\nChoose your next action (MONDO or PROPOSE)."
        
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
        if action == "MONDO":
            shapes_data = action_data.get("arrangement", [])
            pred = action_data.get("prediction", False)
            shapes = [Shape(**s) for s in shapes_data]
            arr = Arrangement(shapes=shapes)
            res = env.mondo(arr, pred)
            log = f"Action: MONDO | Predicted: {pred} | Arrangement: {shapes_data}\nResult: {res}"
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
                break
        else:
            log = f"System Error: Unrecognized action '{action}'"
            print(log)
            history_log += log + "\n"
            
    history_file = env.save_history()
    print(f"\nSaved history to {history_file}")


def run_mock_agent(world: WorldAxis = WorldAxis.EASY, goal: GoalAxis = GoalAxis.EASY, mechanics: MechanicsAxis = MechanicsAxis.EASY, feedback: FeedbackAxis = FeedbackAxis.EASY):
    print("--- Starting Mock Test Agent ---")
    env = VisualZendoEnv(world=world, goal=goal, mechanics=mechanics, feedback=feedback)
    true_rule_name, true_rule_eval_fn = get_rule_by_index(1)
    ce_generator = create_counter_example_generator(true_rule_eval_fn)
    initial_examples = generate_initial_examples(true_rule_eval_fn)
    
    env.reset(initial_examples, true_rule_name, true_rule_eval_fn, ce_generator)
    
    # 1. Incorrect Mondo
    arr1 = generate_random_arrangement()
    env.mondo(arr1, not true_rule_eval_fn(arr1))
    
    # 2. Correct Mondo
    arr2 = generate_random_arrangement()
    env.mondo(arr2, true_rule_eval_fn(arr2))
    
    # 3. Incorrect Proposal
    bad_rule_name = "Exactly two large shapes"
    _, bad_eval = get_rule_by_index(2)
    env.propose_rule(bad_rule_name, bad_eval)
    
    # 4. Another Correct Mondo for token
    arr3 = generate_random_arrangement()
    env.mondo(arr3, true_rule_eval_fn(arr3))
    
    # 5. Correct Proposal
    env.propose_rule(true_rule_name, true_rule_eval_fn)
    
    history_file = env.save_history()
    print(f"Mock run complete. Saved history to {history_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Agent for Visual Zendo")
    parser.add_argument("--agent", type=str, choices=["mock", "llm"], default="mock", help="Choose 'mock' for local rules or 'llm' for AI agent.")
    parser.add_argument("--provider", type=str, choices=["openai", "anthropic", "gemini"], default="openai", help="LLM Provider")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name (e.g. gpt-4o, claude-3-5-sonnet-20241022, gemini-2.0-flash)")
    parser.add_argument("--turns", type=int, default=100, help="Max interaction turns for LLM")
    parser.add_argument("--world", type=str, choices=["EASY", "MEDIUM", "HARD"], default="EASY")
    parser.add_argument("--goal", type=str, choices=["EASY", "MEDIUM", "HARD"], default="EASY")
    parser.add_argument("--mechanics", type=str, choices=["EASY", "MEDIUM", "HARD"], default="EASY")
    parser.add_argument("--feedback", type=str, choices=["EASY", "MEDIUM", "HARD"], default="EASY")
    
    args = parser.parse_args()
    
    world_axis = WorldAxis[args.world]
    goal_axis = GoalAxis[args.goal]
    mechanics_axis = MechanicsAxis[args.mechanics]
    feedback_axis = FeedbackAxis[args.feedback]
    
    if args.agent == "mock":
        run_mock_agent(world=world_axis, goal=goal_axis, mechanics=mechanics_axis, feedback=feedback_axis)
    else:
        run_llm_agent(args.provider, args.model, args.turns, world=world_axis, goal=goal_axis, mechanics=mechanics_axis, feedback=feedback_axis)
