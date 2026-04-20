"""KA59 Reference Simulator package."""
from .engine import Obj, KA59State, STEP, TAG_WALL_TRANSFER, TAG_WALL_SOLID, TAG_SELECTED, TAG_BLOCK
from .env    import KA59BlindEnv, ObjectView, StepResult, Action, SELECT, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT
from .runner    import run_episode, EpisodeTrace, EpisodeStep, InvalidActionError
from .discovery  import MinimalHypothesisAgent
from .scenarios  import SCENARIOS
from .benchmark  import evaluate_agent, BenchmarkResult, ScenarioResult

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
    # Runner
    "run_episode",
    "EpisodeTrace",
    "EpisodeStep",
    "InvalidActionError",
    # Discovery agent
    "MinimalHypothesisAgent",
    # Scenarios + benchmark
    "SCENARIOS",
    "evaluate_agent",
    "BenchmarkResult",
    "ScenarioResult",
    "ObjectView",
    "StepResult",
    "Action",
    "SELECT",
    "MOVE_UP",
    "MOVE_DOWN",
    "MOVE_LEFT",
    "MOVE_RIGHT",
]
