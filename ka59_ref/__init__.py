"""KA59 Reference Simulator package."""
from .engine import Obj, KA59State, STEP, TAG_WALL_TRANSFER, TAG_WALL_SOLID, TAG_SELECTED, TAG_BLOCK
from .env import KA59BlindEnv, ObjectView, StepResult, Action, SELECT, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT

__all__ = [
    # Engine (reference / rule-aware)
    "Obj",
    "KA59State",
    "STEP",
    "TAG_WALL_TRANSFER",
    "TAG_WALL_SOLID",
    "TAG_SELECTED",
    "TAG_BLOCK",
    # Env (agent-facing / blinded)
    "KA59BlindEnv",
    "ObjectView",
    "StepResult",
    "Action",
    "SELECT",
    "MOVE_UP",
    "MOVE_DOWN",
    "MOVE_LEFT",
    "MOVE_RIGHT",
]
