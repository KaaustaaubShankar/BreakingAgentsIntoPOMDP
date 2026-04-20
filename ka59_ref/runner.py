"""
KA59 Episode Runner
====================
Thin harness that drives a policy function through one episode of KA59BlindEnv
and returns a complete trajectory.

Separation of concerns
-----------------------
  engine.py  — game rules (internal)
  env.py     — blinded observation/action API (agent-facing)
  runner.py  — THIS FILE: episode loop + trajectory capture

The policy function is given only the blinded observation.  No engine state,
wall tags, or rule explanations pass through the runner boundary.

Usage
-----
    from ka59_ref.runner import run_episode
    from ka59_ref.env    import MOVE_RIGHT

    def my_policy(obs):          # obs: list[ObjectView]
        return MOVE_RIGHT        # → Action

    trace = run_episode(level_spec, my_policy, max_steps=50)

    print(trace.termination)    # "done" | "max_steps"
    print(len(trace.steps))     # number of actions taken
    for step in trace.steps:
        print(step.step_num, step.action, step.moved, step.steps_remaining)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from .env import KA59BlindEnv, ObjectView, StepResult, Action


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class InvalidActionError(ValueError):
    """
    Raised when a policy function returns something that is not an Action.

    Attributes
    ----------
    step_num   : 1-indexed step at which the bad action was returned
    bad_action : the value actually returned by the policy
    """

    def __init__(self, step_num: int, bad_action: object) -> None:
        self.step_num   = step_num
        self.bad_action = bad_action
        super().__init__(
            f"Policy returned invalid action at step {step_num}: "
            f"{bad_action!r} (expected Action instance)"
        )


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpisodeStep:
    """
    Record of a single step in the episode.

    Fields
    ------
    step_num        : 1-indexed position in the episode
    action          : the Action the policy returned
    obs             : observation AFTER the action was applied
    moved           : whether the selected piece changed position
    steps_remaining : stamina remaining after this action
    done            : True if the episode is over (stamina == 0)
    """
    step_num:        int
    action:          Action
    obs:             Tuple[ObjectView, ...]
    moved:           bool
    steps_remaining: int
    done:            bool


@dataclass(frozen=True)
class EpisodeTrace:
    """
    Complete trajectory for one episode.

    Fields
    ------
    initial_obs  : observation from reset() — state before any action
    steps        : ordered sequence of EpisodeStep records
    termination  : "done"      — env.done became True (stamina exhausted)
                   "max_steps" — max_steps cap hit before env was done
    """
    initial_obs:  Tuple[ObjectView, ...]
    steps:        Tuple[EpisodeStep, ...]
    termination:  str   # "done" | "max_steps"


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    level_spec: dict,
    policy_fn: Callable[[List[ObjectView]], Action],
    max_steps: Optional[int] = None,
) -> EpisodeTrace:
    """
    Run one episode of KA59BlindEnv driven by *policy_fn*.

    Parameters
    ----------
    level_spec  : level configuration passed to KA59BlindEnv.reset()
    policy_fn   : callable(obs: list[ObjectView]) -> Action
                  Receives only the blinded observation; must return an Action.
    max_steps   : optional hard cap on the number of steps taken.
                  None means run until env.done is True.

    Returns
    -------
    EpisodeTrace with the full trajectory.

    Raises
    ------
    InvalidActionError  if policy_fn returns a non-Action value.
    RuntimeError        if called on an uninitialised env (should not occur).
    """
    env = KA59BlindEnv()
    initial_obs = tuple(env.reset(level_spec))

    # If the env starts already done (steps=0 or immediate terminal), return early.
    if env._state is not None and env._state.steps <= 0:
        return EpisodeTrace(
            initial_obs = initial_obs,
            steps       = (),
            termination = "done",
        )

    steps_taken:  List[EpisodeStep] = []
    current_obs:  Tuple[ObjectView, ...] = initial_obs
    termination:  str = "done"   # default; overwritten if max_steps fires first

    step_num = 0
    while True:
        step_num += 1

        # -- Check max_steps cap BEFORE calling policy -----------------------
        if max_steps is not None and step_num > max_steps:
            termination = "max_steps"
            break

        # -- Ask the policy --------------------------------------------------
        action = policy_fn(current_obs)
        if not isinstance(action, Action):
            raise InvalidActionError(step_num, action)

        # -- Apply action ----------------------------------------------------
        result: StepResult = env.step(action)
        new_obs = tuple(result.obs)

        steps_taken.append(EpisodeStep(
            step_num        = step_num,
            action          = action,
            obs             = new_obs,
            moved           = result.moved,
            steps_remaining = result.steps_remaining,
            done            = result.done,
        ))

        current_obs = new_obs

        # -- Check termination -----------------------------------------------
        if result.done:
            termination = "done"
            break

    return EpisodeTrace(
        initial_obs = initial_obs,
        steps       = tuple(steps_taken),
        termination = termination,
    )
