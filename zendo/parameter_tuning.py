import json
import os
import random
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Reuse the same axis enums as Zendo for consistency
from visual_zendo import WorldAxis, GoalAxis, MechanicsAxis, FeedbackAxis


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARTIFACTS_DIR = os.path.join(MODULE_DIR, "game_logs")

N_PARAMS = 3          # Number of sliders
PARAM_MIN = 0
PARAM_MAX = 10


# ---------------------------------------------------------------------------
# Rule registry — hidden functions the agent must discover
# ---------------------------------------------------------------------------

def _make_params(n: int = N_PARAMS) -> Dict[str, int]:
    """Generate a random parameter configuration."""
    return {f"P{i+1}": random.randint(PARAM_MIN, PARAM_MAX) for i in range(n)}


# Each rule: (name, eval_fn)
# eval_fn takes a dict like {"P1": 7, "P2": 3, "P3": 9} → bool
RULES_REGISTRY = [
    (
        "P1 > 5",
        lambda p: p["P1"] > 5,
    ),
    (
        "P1 > 5 AND P2 > 5",
        lambda p: p["P1"] > 5 and p["P2"] > 5,
    ),
    (
        "P1 + P2 > 12",
        lambda p: p["P1"] + p["P2"] > 12,
    ),
    (
        "(P1 + P2 > 12) AND (P3 < 4)",
        lambda p: (p["P1"] + p["P2"] > 12) and (p["P3"] < 4),
    ),
]


def get_rule_by_index(index: int) -> Tuple[str, Callable]:
    if 0 <= index < len(RULES_REGISTRY):
        return RULES_REGISTRY[index]
    raise ValueError(f"Rule index {index} out of bounds (max {len(RULES_REGISTRY)-1}).")


def generate_initial_examples(rule_fn: Callable, num_on: int = 3, num_off: int = 3) -> List[Tuple[Dict, bool]]:
    """Generate labeled examples: (params_dict, is_online)."""
    examples = []
    on_count = off_count = 0
    attempts = 0
    while (on_count < num_on or off_count < num_off) and attempts < 10000:
        attempts += 1
        params = _make_params()
        label = rule_fn(params)
        if label and on_count < num_on:
            examples.append((params, True))
            on_count += 1
        elif not label and off_count < num_off:
            examples.append((params, False))
            off_count += 1
    random.shuffle(examples)
    return examples


def create_counter_example_generator(true_fn: Callable, max_attempts: int = 1000) -> Callable:
    """Returns a function that finds a counter-example to an agent's proposed rule."""
    def generator(agent_fn: Callable) -> Optional[Tuple[Dict, bool]]:
        for _ in range(max_attempts):
            params = _make_params()
            try:
                true_label = true_fn(params)
                agent_label = agent_fn(params)
                if true_label != agent_label:
                    return params, true_label
            except Exception:
                return params, true_fn(params)
        return None
    return generator


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ParameterTuningEnv:
    """
    Environment 2: Parameter Tuning
    
    The agent adjusts N integer sliders (P1..PN, range 0-10) and must discover
    the hidden boolean rule that determines whether the system comes "online".
    
    Actions:
      SET   — set parameter values, observe output (+ optional partial feedback)
      PROPOSE — submit the inferred rule as a Python function

    Follows the same pattern as LithicArrayEnv.
    """

    def __init__(
        self,
        world: WorldAxis = WorldAxis.EASY,
        goal: GoalAxis = GoalAxis.EASY,
        mechanics: MechanicsAxis = MechanicsAxis.EASY,
        feedback: FeedbackAxis = FeedbackAxis.EASY,
        artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
        n_params: int = N_PARAMS,
    ):
        self.world = world
        self.goal = goal
        self.mechanics = mechanics
        self.feedback = feedback
        self.artifacts_dir = os.path.abspath(artifacts_dir)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.n_params = n_params

        # State
        self.tokens = 0
        self.true_rule_name = None
        self.true_rule_fn = None
        self.counter_example_generator_fn = None
        self.failed_proposals_count = 0
        self.history = []

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        initial_examples: List[Tuple[Dict, bool]],
        true_rule_name: str,
        true_rule_fn: Callable,
        counter_example_generator_fn: Callable,
    ) -> Dict[str, Any]:
        self.tokens = 0
        self.true_rule_name = true_rule_name
        self.true_rule_fn = true_rule_fn
        self.counter_example_generator_fn = counter_example_generator_fn
        self.failed_proposals_count = 0
        self.history = []

        self._log_event("episode_start", {
            "axes": {
                "world": self.world.name,
                "goal": self.goal.name,
                "mechanics": self.mechanics.name,
                "feedback": self.feedback.name,
            },
            "true_rule": self.true_rule_name,
            "n_params": self.n_params,
        })

        presentation = {
            "instruction": self._get_goal_instruction(),
            "mechanics": self._get_mechanics_instruction(),
            "param_names": [f"P{i+1}" for i in range(self.n_params)],
            "param_range": [PARAM_MIN, PARAM_MAX],
            "initial_examples": [
                self._format_example(params, label)
                for params, label in initial_examples
            ],
        }

        self._log_event("initial_presentation", presentation)
        return presentation

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def set_params(self, params: Dict[str, int], prediction: bool) -> Dict[str, Any]:
        """Agent sets parameter values and predicts whether the system is online."""
        # Validate and coerce params
        params = {k: int(v) for k, v in params.items()}
        true_online = self.true_rule_fn(params)
        match = (prediction == true_online)

        if match:
            self.tokens += 1
            feedback_msg = "Correct! System is {}. You earned 1 token.".format(
                "online" if true_online else "offline"
            )
        else:
            feedback_msg = "Incorrect. System is {}. No token earned.".format(
                "online" if true_online else "offline"
            )

        response: Dict[str, Any] = {
            "result": "online" if true_online else "offline",
            "feedback": feedback_msg,
            "tokens_current": self.tokens,
        }

        # Feedback axis: EASY gives partial signal (which params are in threshold range)
        if self.feedback == FeedbackAxis.EASY and true_online:
            response["hint"] = self._get_partial_feedback(params)

        self._log_event("set_params", {
            "params": params,
            "predicted": prediction,
            "actual": true_online,
            "match": match,
            "response": response,
        })

        return response

    def propose_rule(self, description: str, agent_fn: Callable) -> Dict[str, Any]:
        """Agent spends 1 token to propose the rule."""
        if self.tokens < 1:
            msg = "Insufficient tokens. Earn tokens via SET first."
            self._log_event("propose_failed_tokens", {"description": description})
            return {"error": msg}

        self.tokens -= 1
        counter_example = self.counter_example_generator_fn(agent_fn)

        if counter_example is None:
            response = {
                "result": "Accepted",
                "message": "Correct! You have discovered the hidden rule.",
            }
            self._log_event("propose_success", {"description": description, "response": response})
            return response

        self.failed_proposals_count += 1
        response: Dict[str, Any] = {"result": "Rejected"}

        provide_ce = False
        if self.feedback == FeedbackAxis.EASY:
            provide_ce = True
        elif self.feedback == FeedbackAxis.MEDIUM and self.failed_proposals_count == 1:
            provide_ce = True

        if provide_ce:
            ce_params, true_label = counter_example
            agent_label = not true_label
            response["counter_example"] = {
                "params": ce_params,
                "system_actual": "online" if true_label else "offline",
                "your_rule_predicted": "online" if agent_label else "offline",
                "message": "These parameter values produce a different result under your rule vs the true rule.",
            }

        self._log_event("propose_rejected", {"description": description, "response": response})
        return response

    # ------------------------------------------------------------------
    # Presentation helpers
    # ------------------------------------------------------------------

    def _get_goal_instruction(self) -> str:
        if self.goal == GoalAxis.EASY:
            return (
                "Your goal is to discover the hidden rule that determines when the system comes online. "
                "Set parameter values to test configurations, then propose the rule when you think you know it."
            )
        elif self.goal == GoalAxis.MEDIUM:
            return "There is a pattern here. Figure out what makes the system activate."
        else:  # HARD
            return ""

    def _get_mechanics_instruction(self) -> str:
        param_names = ", ".join(f"P{i+1}" for i in range(self.n_params))
        if self.mechanics == MechanicsAxis.EASY:
            return f"""You have {self.n_params} integer parameters ({param_names}), each ranging from {PARAM_MIN} to {PARAM_MAX}.
Set them to observe whether the system comes online (1) or stays offline (0). Correct predictions earn 1 token.
Spend 1 token to propose the rule.

Respond with ONLY a JSON block. No other text.

SET action:
{{
  "action": "SET",
  "params": {{"P1": 7, "P2": 3, "P3": 9}},
  "prediction": true
}}

PROPOSE action (costs 1 token):
{{
  "action": "PROPOSE",
  "rule_description": "P1 > 5 and P2 > 5",
  "rule_code": "def agent_eval_fn(p):\\n    return p['P1'] > 5 and p['P2'] > 5"
}}"""
        elif self.mechanics == MechanicsAxis.MEDIUM:
            return f"""Parameters: {param_names} (range {PARAM_MIN}-{PARAM_MAX}). Observe the system output. Propose the rule when ready.

Respond with ONLY a JSON block.
SET: {{"action": "SET", "params": {{"P1": 5}}, "prediction": true}}
PROPOSE: {{"action": "PROPOSE", "rule_description": "...", "rule_code": "def agent_eval_fn(p): ..."}}"""
        else:  # HARD
            return "Respond with a JSON object containing your action ('SET' or 'PROPOSE')."

    def _format_example(self, params: Dict[str, int], label: bool) -> Dict[str, Any]:
        """Format an example for presentation, respecting the World axis."""
        if self.world == WorldAxis.EASY:
            return {
                "representation": params,
                "representation_type": "dict",
                "label": "online" if label else "offline",
            }
        elif self.world == WorldAxis.MEDIUM:
            parts = ", ".join(f"{k}={v}" for k, v in params.items())
            return {
                "representation": f"System with {parts} is {'online' if label else 'offline'}.",
                "representation_type": "text",
                "label": "online" if label else "offline",
            }
        else:  # HARD — no labels, abstract names
            return {
                "representation": params,
                "representation_type": "dict",
                # No label revealed in HARD world
            }

    def _get_partial_feedback(self, params: Dict[str, int]) -> str:
        """EASY feedback: hint about which parameters seem to matter (heuristic)."""
        # Simple heuristic: highlight params that are above midpoint
        high = [k for k, v in params.items() if v > (PARAM_MAX - PARAM_MIN) / 2]
        if high:
            return f"Parameters above midpoint: {', '.join(high)}"
        return "All parameters are at or below midpoint."

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_event(self, event_type: str, data: Dict[str, Any]):
        entry = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "event": event_type,
            "data": data,
        }
        self.history.append(entry)

    def save_history(self) -> str:
        slug = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(self.artifacts_dir, f"pt_history_{slug}.json")
        with open(filename, "w") as f:
            json.dump(self.history, f, indent=2)
        return filename
