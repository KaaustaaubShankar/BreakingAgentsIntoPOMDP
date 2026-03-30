import json
import os
from datetime import datetime
from enum import Enum, auto
from typing import List, Dict, Callable, Any, Tuple, Optional, Type
from dataclasses import dataclass, asdict

# Optional PIL import for the Hard World Axis (Image rendering)
try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class WorldAxis(Enum):
    EASY = auto()    # Structured JSON table
    MEDIUM = auto()  # Natural language prose
    HARD = auto()    # Raw images only

class GoalAxis(Enum):
    EASY = auto()    # Explicit goal stated
    MEDIUM = auto()  # "There is a pattern here. Figure it out."
    HARD = auto()    # Nothing stated

class MechanicsAxis(Enum):
    EASY = auto()    # Full explanation
    MEDIUM = auto()  # Told only "query arrangements or propose a rule"
    HARD = auto()    # No instructions

class FeedbackAxis(Enum):
    EASY = auto()    # Binary result + counter-example always
    MEDIUM = auto()  # Binary always, counter-example only on first failure
    HARD = auto()    # Binary only, no counter-examples


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARTIFACTS_DIR = os.path.join(MODULE_DIR, "game_logs")


@dataclass
class Shape:
    color: str
    size: str
    type_: str  # e.g., 'triangle', 'circle', 'square'

@dataclass
class Arrangement:
    shapes: List[Shape]

    def to_json_dict(self):
        return [asdict(s) for s in self.shapes]

    def to_natural_language(self):
        desc = "An arrangement with "
        shape_descs = [f"a {s.size} {s.color} {s.type_}" for s in self.shapes]
        if len(shape_descs) == 0:
            return "An empty arrangement."
        elif len(shape_descs) == 1:
            return desc + shape_descs[0] + "."
        elif len(shape_descs) == 2:
            return desc + shape_descs[0] + " next to " + shape_descs[1] + "."
        else:
            return desc + ", ".join(shape_descs[:-1]) + ", and " + shape_descs[-1] + "."


class LithicArrayEnv:
    def __init__(
        self,
        world: WorldAxis = WorldAxis.EASY,
        goal: GoalAxis = GoalAxis.EASY,
        mechanics: MechanicsAxis = MechanicsAxis.EASY,
        feedback: FeedbackAxis = FeedbackAxis.EASY,
        artifacts_dir: str = DEFAULT_ARTIFACTS_DIR
    ):
        self.world = self._coerce_axis_value(world, WorldAxis)
        self.goal = self._coerce_axis_value(goal, GoalAxis)
        self.mechanics = self._coerce_axis_value(mechanics, MechanicsAxis)
        self.feedback = self._coerce_axis_value(feedback, FeedbackAxis)
        
        self.artifacts_dir = os.path.abspath(artifacts_dir)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
        # State variables
        self.tokens = 0
        self.true_rule_name = None
        self.true_rule_eval_fn = None
        self.counter_example_generator_fn = None
        self.failed_proposals_count = 0
        self.history = []

    def _coerce_axis_value(self, value: Any, axis_type: Type[Enum]) -> Enum:
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

    def _current_datetime(self) -> datetime:
        return datetime.now().astimezone()

    def _timestamp_string(self, dt: Optional[datetime] = None) -> str:
        dt = dt or self._current_datetime()
        return dt.isoformat(timespec="milliseconds")

    def _timestamp_slug(self, dt: Optional[datetime] = None) -> str:
        dt = dt or self._current_datetime()
        return dt.strftime("%Y%m%d_%H%M%S_%f")
        
    def reset(self, initial_examples: List[Tuple[Arrangement, bool]], true_rule_name: str, true_rule_eval_fn: Callable, counter_example_generator_fn: Callable):
        self.tokens = 0
        self.true_rule_name = true_rule_name
        self.true_rule_eval_fn = true_rule_eval_fn
        self.counter_example_generator_fn = counter_example_generator_fn
        self.failed_proposals_count = 0
        self.history = []
        
        # Log episode start
        self._log_event("episode_start", {
            "axes": {
                "world": self.world.name,
                "goal": self.goal.name,
                "mechanics": self.mechanics.name,
                "feedback": self.feedback.name
            },
            "true_rule": self.true_rule_name
        })
        
        # Generate initial presentation
        reset_dt = self._current_datetime()
        reset_slug = self._timestamp_slug(reset_dt)
        presentation = {
            "instruction": self._get_goal_instruction(),
            "mechanics": self._get_mechanics_instruction(),
            "initial_examples": [
                self._format_arrangement(arr, label, filename=f"initial_{reset_slug}_{i}.png")
                for i, (arr, label) in enumerate(initial_examples)
            ]
        }
        
        self._log_event("initial_presentation", presentation)
        return presentation

    def _get_goal_instruction(self):
        if self.goal == GoalAxis.EASY:
            return "Your goal is to discover the hidden rule that classifies arrangements as Quartz or Shale. State the rule explicitly to win."
        elif self.goal == GoalAxis.MEDIUM:
            return "There is a pattern here. Figure it out."
        else: # HARD
            return ""

    def _get_mechanics_instruction(self):
        if self.mechanics == MechanicsAxis.EASY:
            return """You may query arrangements via Strata (submit arrangement + predicted classification). Matches earn 1 token. You can spend tokens to propose a rule. The Basalt provides counter-examples on failed proposals.

You must respond with only a JSON block containing your action. Do not include markdown or other text outside the JSON.
Strata Action Schema:
{
  "action": "STRATA",
  "arrangement": [{"color": "red", "size": "small", "type_": "circle"}],
  "prediction": true  // true for Quartz, false for Shale
}

Propose Action Schema (Requires 1 token):
{
  "action": "PROPOSE",
  "rule_description": "All pieces are red",
  "rule_code": "def agent_eval_fn(arr):\n    # arr is an Arrangement obj. You can access shapes via arr.shapes. Each shape has .color, .size, .type_\n    return all(s.color == 'red' for s in arr.shapes)"
}"""
        elif self.mechanics == MechanicsAxis.MEDIUM:
            return """You can query arrangements or propose a rule.

You must respond with only a JSON block containing your action. Do not include markdown or other text.
Action Schemas:
STRATA: {"action": "STRATA", "arrangement": [{"color": "red", "size": "small", "type_": "circle"}], "prediction": true}
PROPOSE: {"action": "PROPOSE", "rule_description": "...", "rule_code": "def agent_eval_fn(arr): ..."}"""
        else: # HARD
            return "Respond with a JSON object containing your action ('STRATA' or 'PROPOSE')."

    def _format_arrangement(self, arrangement: Arrangement, label: Optional[bool] = None, filename: str = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.world == WorldAxis.EASY:
            result["representation"] = arrangement.to_json_dict()
            result["representation_type"] = "json"
        elif self.world == WorldAxis.MEDIUM:
            result["representation"] = arrangement.to_natural_language()
            result["representation_type"] = "text"
        else: # HARD
            image_filename = filename or f"arrangement_{self._timestamp_slug()}.png"
            image_path = os.path.abspath(os.path.join(self.artifacts_dir, image_filename))
            if PIL_AVAILABLE:
                self._render_image(arrangement, image_path)
                result["representation"] = image_path
                result["representation_type"] = "image"
                result["image_path"] = image_path
            else:
                result["representation"] = arrangement.to_json_dict()
                result["representation_type"] = "json"
                result["fallback_reason"] = "PIL not available for image rendering"
                
        if label is not None:
            result["label"] = "Quartz" if label else "Shale"
        
        return result

    def _render_image(self, arrangement: Arrangement, filepath: str):
        # A simple rendering mapping. In a real environment, you'd want beautiful procedural SVGs or PIL drawing.
        if not PIL_AVAILABLE:
            return
            
        img = Image.new("RGB", (300, 100), "white")
        draw = ImageDraw.Draw(img)
        
        x_offset = 10
        for shape in arrangement.shapes:
            # Map size
            size_val = 20 if shape.size == "small" else 40 if shape.size == "medium" else 60
            
            # Map color
            color_map = {"red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0), "yellow": (255, 255, 0), "black": (0, 0, 0)}
            fill_color = color_map.get(shape.color, (128, 128, 128))
            
            bbox = [x_offset, 50 - size_val // 2, x_offset + size_val, 50 + size_val // 2]
            if shape.type_ == "circle":
                draw.ellipse(bbox, fill=fill_color, outline="black")
            elif shape.type_ == "square":
                draw.rectangle(bbox, fill=fill_color, outline="black")
            elif shape.type_ == "triangle":
                draw.polygon([(x_offset + size_val // 2, 50 - size_val // 2), 
                              (x_offset, 50 + size_val // 2), 
                              (x_offset + size_val, 50 + size_val // 2)], 
                             fill=fill_color, outline="black")
            
            x_offset += size_val + 10
            
        img.save(filepath)

    def strata(self, arrangement: Arrangement, predicted_quartz: bool) -> Dict[str, Any]:
        """Agent predicts if an arrangement is quartz."""
        true_quartz = self.true_rule_eval_fn(arrangement)
        match = (predicted_quartz == true_quartz)
        
        if match:
            self.tokens += 1
            feedback_msg = "Match! You earned 1 token."
        else:
            feedback_msg = "Mismatch. No token earned."
        
        response = {
            "correct_label": "Quartz" if true_quartz else "Shale",
            "feedback": feedback_msg,
            "tokens_current": self.tokens
        }
        
        self._log_event("strata_query", {
            "arrangement": arrangement.to_json_dict(),
            "predicted": predicted_quartz,
            "actual": true_quartz,
            "match": match,
            "response": response
        })
        
        return response

    def propose_rule(self, proposed_rule: str, agent_eval_fn: Callable) -> Dict[str, Any]:
        """Agent spends 1 token to propose a rule."""
        if self.tokens < 1:
            msg = "Insufficent tokens to propose a rule. Earn tokens via Strata first."
            self._log_event("propose_rule_failed_tokens", {"proposed_rule": proposed_rule})
            return {"error": msg}
            
        self.tokens -= 1
        
        # The Basalt generates a counter-example if the agent's proposed rule does not match the true rule.
        # This requires the agent's evaluation function as input to test against the basalt's true rule.
        counter_example = self.counter_example_generator_fn(agent_eval_fn)
        
        if counter_example is None:
            # Rule is perfectly correct (no counter example exists)
            response = {"result": "Accepted", "message": "Congratulations! You have discovered the hidden rule."}
            self._log_event("propose_rule_success", {"proposed_rule": proposed_rule, "response": response})
            return response
        else:
            self.failed_proposals_count += 1
            
            # Incorrect rule
            response = {"result": "Rejected"}
            
            # Determine if we should provide the counter-example based on Feedback Axis
            provide_ce = False
            if self.feedback == FeedbackAxis.EASY:
                provide_ce = True
            elif self.feedback == FeedbackAxis.MEDIUM and self.failed_proposals_count == 1:
                provide_ce = True
            elif self.feedback == FeedbackAxis.HARD:
                provide_ce = False
                
            if provide_ce:
                arr_obj, true_label = counter_example
                # Agent's label on this counter-example would be the opposite of the true label
                agent_label = not true_label 
                filename = f"ce_{self._timestamp_slug()}.png"
                formatted_arr = self._format_arrangement(arr_obj, filename=filename)
                
                response["counter_example"] = {
                    "arrangement": formatted_arr["representation"],
                    "arrangement_details": formatted_arr,
                    "basalt_label": "Quartz" if true_label else "Shale",
                    "your_rule_predicted": "Quartz" if agent_label else "Shale",
                    "message": "This arrangement satisfies the Basalt's rule but not yours, or vice versa."
                }
                
            self._log_event("propose_rule_rejected", {"proposed_rule": proposed_rule, "response": response})
            return response

    def _log_event(self, event_type: str, data: Dict[str, Any]):
        event_dt = self._current_datetime()
        entry = {
            "timestamp": self._timestamp_string(event_dt),
            "event": event_type,
            "data": data
        }
        self.history.append(entry)

    def save_history(self) -> str:
        filename = os.path.join(self.artifacts_dir, f"history_{self._timestamp_slug()}.json")
        with open(filename, "w") as f:
            json.dump(self.history, f, indent=2)
        return filename
