import json
import os
import random
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Reuse the same axis enums as Zendo for consistency
try:
    from visual_zendo import WorldAxis, GoalAxis, MechanicsAxis, FeedbackAxis
except ImportError:
    from zendo.visual_zendo import WorldAxis, GoalAxis, MechanicsAxis, FeedbackAxis


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARTIFACTS_DIR = os.path.join(MODULE_DIR, "game_logs")

N_PARAMS = 3          # Number of sliders
PARAM_MIN = 0
PARAM_MAX = 10


# ---------------------------------------------------------------------------
# Target registry — hidden configurations the agent must converge on
#
# Design: the system "comes online" when the agent's submitted parameters
# are within TOLERANCE of the hidden target on all dimensions.  The agent
# never sees the target directly; it navigates by proximity feedback.
# This is the lockpicking model Kaus described — the agent *feels its way*
# toward the target rather than inferring a logical rule.
# ---------------------------------------------------------------------------

TOLERANCE = 2        # Max per-parameter distance for the system to come online


def _make_params(n: int = N_PARAMS) -> Dict[str, int]:
    """Generate a random parameter configuration."""
    return {f"P{i+1}": random.randint(PARAM_MIN, PARAM_MAX) for i in range(n)}


def _make_target(n: int = N_PARAMS) -> Dict[str, int]:
    """Generate a hidden target configuration."""
    return {f"P{i+1}": random.randint(PARAM_MIN, PARAM_MAX) for i in range(n)}


def _is_online(params: Dict[str, int], target: Dict[str, int], tolerance: int = TOLERANCE) -> bool:
    """System comes online when every parameter is within tolerance of its target."""
    return all(abs(params[k] - target[k]) <= tolerance for k in target)


def _proximity_signal(params: Dict[str, int], target: Dict[str, int]) -> Dict[str, str]:
    """
    Per-parameter proximity signal (hot/warm/cold) for EASY feedback.
    Revealed only when the agent's prediction is correct — rewarding calibrated confidence.
    """
    signals = {}
    for k in target:
        dist = abs(params[k] - target[k])
        if dist == 0:
            signals[k] = "exact"
        elif dist <= TOLERANCE:
            signals[k] = "hot"       # within tolerance — would activate
        elif dist <= TOLERANCE * 2:
            signals[k] = "warm"      # close but not activating
        else:
            signals[k] = "cold"      # far away
    return signals


# Target registry: named scenarios with fixed seeds for reproducible evals.
# Each entry: (name, target_dict)
# The eval function is generated at runtime by ParameterTuningEnv.reset().
TARGETS_REGISTRY = [
    ("T1: single-dim target",   {"P1": 7, "P2": 5, "P3": 3}),
    ("T2: high-end target",     {"P1": 9, "P2": 8, "P3": 9}),
    ("T3: low-end target",      {"P1": 1, "P2": 2, "P3": 1}),
    ("T4: mixed target",        {"P1": 3, "P2": 8, "P3": 5}),
]


def get_rule_by_index(index: int) -> Tuple[str, Callable]:
    """Returns (target_name, eval_fn) for the given index."""
    if 0 <= index < len(TARGETS_REGISTRY):
        name, target = TARGETS_REGISTRY[index]
        fn = lambda p, t=target: _is_online(p, t)
        return name, fn
    raise ValueError(f"Target index {index} out of bounds (max {len(TARGETS_REGISTRY)-1}).")


def get_target_by_index(index: int) -> Dict[str, int]:
    """Returns the raw target dict (used by env for proximity feedback)."""
    if 0 <= index < len(TARGETS_REGISTRY):
        return dict(TARGETS_REGISTRY[index][1])
    raise ValueError(f"Target index {index} out of bounds.")


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
    """Returns a function that finds params where agent diverges from target."""
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
        self.world = self._coerce_axis_value(world, WorldAxis)
        self.goal = self._coerce_axis_value(goal, GoalAxis)
        self.mechanics = self._coerce_axis_value(mechanics, MechanicsAxis)
        self.feedback = self._coerce_axis_value(feedback, FeedbackAxis)
        self.artifacts_dir = os.path.abspath(artifacts_dir)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.n_params = n_params

        # State
        self.tokens = 0
        self.true_rule_name = None
        self.true_rule_fn = None
        self.hidden_target: Optional[Dict[str, int]] = None  # for proximity feedback
        self.counter_example_generator_fn = None
        self.failed_proposals_count = 0
        self.history = []

    def _coerce_axis_value(self, value: Any, axis_type):
        if isinstance(value, axis_type):
            return value
        if isinstance(value, str):
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
                raise ValueError(
                    f"Invalid {axis_type.__name__} value '{value}'. Expected one of: {valid_values}, high, low."
                ) from exc
        raise TypeError(f"Unsupported {axis_type.__name__} value type: {type(value).__name__}")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        initial_examples: List[Tuple[Dict, bool]],
        true_rule_name: str,
        true_rule_fn: Callable,
        counter_example_generator_fn: Callable,
        hidden_target: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        self.tokens = 0
        self.true_rule_name = true_rule_name
        self.true_rule_fn = true_rule_fn
        self.hidden_target = hidden_target
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

        # Feedback axis: EASY gives proximity; HARD gives binary only.
        if self.hidden_target is not None and self.feedback == FeedbackAxis.EASY and match:
            response["proximity"] = _proximity_signal(params, self.hidden_target)

        self._log_event("set_params", {
            "params": params,
            "predicted": prediction,
            "actual": true_online,
            "match": match,
            "response": response,
        })

        return response

    def propose_target(self, guessed_target: Dict[str, int]) -> Dict[str, Any]:
        """
        Agent spends 1 token to propose the hidden target configuration directly.

        The agent submits a dict like {"P1": 7, "P2": 5, "P3": 3} — their best
        guess at the target values. Accepted if every param is within TOLERANCE.
        On rejection, EASY feedback reveals which params are off and by how much
        (direction only — "too high" / "too low" / "exact"), not the actual values.
        """
        if self.tokens < 1:
            msg = "Insufficient tokens. Earn tokens via SET first."
            self._log_event("propose_failed_tokens", {"guessed_target": guessed_target})
            return {"error": msg}

        self.tokens -= 1
        guessed_target = {k: int(v) for k, v in guessed_target.items()}

        if self.hidden_target is None:
            # Fallback: accept if all params are within tolerance of each other
            # (should not happen in normal use)
            accepted = False
        else:
            accepted = _is_online(guessed_target, self.hidden_target)

        if accepted:
            response: Dict[str, Any] = {
                "result": "Accepted",
                "message": "Correct! You have found the hidden target configuration.",
            }
            self._log_event("propose_success", {"guessed_target": guessed_target, "response": response})
            return response

        self.failed_proposals_count += 1
        response = {"result": "Rejected"}

        # Directional hints — EASY always, HARD never.
        provide_hint = self.feedback == FeedbackAxis.EASY
        if provide_hint and self.hidden_target is not None:
            hints = {}
            for k in self.hidden_target:
                g = guessed_target.get(k, 0)
                t = self.hidden_target[k]
                if abs(g - t) <= TOLERANCE:
                    hints[k] = "close"
                elif g < t:
                    hints[k] = "too low"
                else:
                    hints[k] = "too high"
            response["hints"] = hints
            response["message"] = "Not quite. Adjust and try again."

        self._log_event("propose_rejected", {"guessed_target": guessed_target, "response": response})
        return response

    # Keep propose_rule as a legacy alias so existing tests/ablation code doesn't break
    def propose_rule(self, description: str, agent_fn: Callable) -> Dict[str, Any]:
        """Legacy: accepts a callable and tests it against the true function.
        Use propose_target() for the convergence model."""
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
        response_r: Dict[str, Any] = {"result": "Rejected"}
        if self.feedback == FeedbackAxis.EASY:
            ce_params, true_label = counter_example
            response_r["counter_example"] = {
                "params": ce_params,
                "system_actual": "online" if true_label else "offline",
                "your_rule_predicted": "online" if not true_label else "offline",
            }
        self._log_event("propose_rejected", {"description": description, "response": response_r})
        return response_r

    # ------------------------------------------------------------------
    # Presentation helpers
    # ------------------------------------------------------------------

    def _get_goal_instruction(self) -> str:
        if self.goal == GoalAxis.EASY:
            return (
                "Your goal is to discover the hidden rule that determines when the system comes online. "
                "Set parameter values to test configurations, then propose the rule when you think you know it."
            )
        else:  # HARD
            return ""

    def _get_mechanics_instruction(self) -> str:
        param_names = ", ".join(f"P{i+1}" for i in range(self.n_params))
        if self.mechanics == MechanicsAxis.EASY:
            return f"""You have {self.n_params} integer parameters ({param_names}), each ranging from {PARAM_MIN} to {PARAM_MAX}.
Set them to observe whether the system comes online or offline. Correct predictions earn 1 token.
When you think you know the hidden target configuration, spend 1 token to propose it.

Respond with ONLY a JSON block. No other text.

SET action:
{{
  "action": "SET",
  "params": {{"P1": 7, "P2": 3, "P3": 9}},
  "prediction": true
}}

PROPOSE action (costs 1 token — submit your best guess at the target values):
{{
  "action": "PROPOSE",
  "target": {{"P1": 7, "P2": 5, "P3": 3}}
}}"""
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
        else:  # HARD — no labels, abstract names
            return {
                "representation": params,
                "representation_type": "dict",
                # No label revealed in HARD world
            }

    # _get_partial_feedback removed — proximity is now handled by _proximity_signal()
    # which operates on the hidden target directly.

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
