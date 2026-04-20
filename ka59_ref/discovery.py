"""
KA59 Minimal Hypothesis Agent
================================
A tiny reflexive discovery agent for KA59BlindEnv.

The agent interacts exclusively through the blinded env surface:
  Inputs  : list[ObjectView]  (positions, sizes, opaque kinds)
  Outputs : Action            (MOVE_UP/DOWN/LEFT/RIGHT)

No engine internals (tag names, wall rules, collision logic) are imported
or referenced.  The agent builds its world-model purely from transition
observations.

Hypotheses maintained
---------------------
  blocked_count[dir]    How many times a move in <dir> failed to move the
                        selected piece.  Increments whether the block is a
                        direct wall hit or a push-into-hard-wall result —
                        the agent can't tell the difference (which is fine).

  push_success[dir]     How many times a move in <dir> succeeded AND at
                        least one block was displaced.  Signals that pushing
                        is working in this direction.

  passable_walls        Set of wall object IDs where a pushed block was
                        subsequently observed at the wall's position.
                        Empirical signal for WALL_TRANSFER-type objects,
                        discovered purely from transitions.

Scoring strategy
----------------
  score(d) = blocked_count[d]  −  push_success[d] × 2

  Lower score = more preferred.
  Push successes are weighted ×2 because they're a stronger positive signal
  than a single block is negative.
  Ties broken by initial preference order (right → down → left → up).

Minimalism note
---------------
  The agent only moves the currently selected piece.  It never issues SELECT
  because selection management would add complexity beyond the current scope.
  Extending to multi-piece selection is the natural next step.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple

from .env import Action, ObjectView, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT


# ---------------------------------------------------------------------------
# Module-level constants (direction only — no engine tags)
# ---------------------------------------------------------------------------

# Preference order for tie-breaking; agent adapts scores from here.
_PREF: Tuple[str, ...] = ("right", "down", "left", "up")

_DIR_TO_ACTION: Dict[str, Action] = {
    "right": MOVE_RIGHT,
    "down":  MOVE_DOWN,
    "left":  MOVE_LEFT,
    "up":    MOVE_UP,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _selected(obs: List[ObjectView]) -> Optional[ObjectView]:
    """Return the currently selected ObjectView, or None if not found."""
    for v in obs:
        if v.is_selected:
            return v
    return None


def _by_id(obs: List[ObjectView], obj_id: str) -> Optional[ObjectView]:
    """Look up an ObjectView by id."""
    for v in obs:
        if v.id == obj_id:
            return v
    return None


# ---------------------------------------------------------------------------
# MinimalHypothesisAgent
# ---------------------------------------------------------------------------

class MinimalHypothesisAgent:
    """
    Minimal reflexive discovery agent for KA59BlindEnv.

    Public interface
    ----------------
    reset()                 — clear all hypotheses and episode state
    act(obs) -> Action      — observe, update hypotheses, choose action

    Public hypothesis state (readable after each step)
    ---------------------------------------------------
    blocked_count  : dict[str, int]   direction → block count
    push_success   : dict[str, int]   direction → push-success count
    passable_walls : set[str]         wall IDs observed to be passable
    step_count     : int              total act() calls since last reset
    """

    def __init__(self) -> None:
        self.reset()

    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all hypotheses and episode-local state."""
        self.step_count:     int                   = 0
        self.last_obs:       Optional[List[ObjectView]] = None
        self.last_action:    Optional[Action]       = None
        self.blocked_count:  Dict[str, int]         = {}
        self.push_success:   Dict[str, int]         = {}
        self.passable_walls: Set[str]               = set()

    # -------------------------------------------------------------------------

    def act(self, obs: List[ObjectView]) -> Action:
        """
        Given the current blinded observation, return an Action.

        Step 1: if this isn't the first call, analyse the transition
                (last_obs → last_action → obs) and update hypotheses.
        Step 2: choose the best-scoring direction.
        Step 3: record state for the next call.

        Parameters
        ----------
        obs : blinded observation from KA59BlindEnv.step() or reset()

        Returns
        -------
        Action — one of MOVE_UP / MOVE_DOWN / MOVE_LEFT / MOVE_RIGHT
        """
        if self.last_action is not None and self.last_obs is not None:
            self._update_hypotheses(self.last_obs, self.last_action, obs)

        action = self._choose(obs)

        self.last_obs    = list(obs)   # snapshot (obs is a list/tuple from runner)
        self.last_action = action
        self.step_count += 1
        return action

    # -------------------------------------------------------------------------
    # Hypothesis update
    # -------------------------------------------------------------------------

    def _update_hypotheses(
        self,
        prev:   List[ObjectView],
        action: Action,
        curr:   List[ObjectView],
    ) -> None:
        """
        Analyse one transition and update blocked_count / push_success /
        passable_walls.

        Only directional actions produce positional evidence; SELECT is ignored.
        """
        if action.type not in _DIR_TO_ACTION:
            return

        sel_prev = _selected(prev)
        sel_curr = _selected(curr)
        if sel_prev is None or sel_curr is None:
            return

        selected_moved = (sel_prev.x != sel_curr.x) or (sel_prev.y != sel_curr.y)
        d = action.type

        if not selected_moved:
            # ── Blocked ──────────────────────────────────────────────────────
            # Either a direct wall hit, or a push into a hard wall.
            # The agent cannot distinguish them; both are "this direction failed".
            self.blocked_count[d] = self.blocked_count.get(d, 0) + 1

        else:
            # ── Move committed ───────────────────────────────────────────────
            # Check whether any block changed position.
            for pv in prev:
                if pv.kind != "block":
                    continue
                cv = _by_id(curr, pv.id)
                if cv is None:
                    continue
                if pv.x == cv.x and pv.y == cv.y:
                    continue                       # block didn't move

                # A block was pushed in this direction.
                self.push_success[d] = self.push_success.get(d, 0) + 1

                # Did the block land at a wall position?
                # If so, that wall is empirically passable (WALL_TRANSFER signal).
                for wv in curr:
                    if wv.kind == "wall" and wv.x == cv.x and wv.y == cv.y:
                        self.passable_walls.add(wv.id)

    # -------------------------------------------------------------------------
    # Action selection
    # -------------------------------------------------------------------------

    def _score(self, direction: str) -> int:
        """
        Score a direction.  Lower = more preferred.

            score = blocked_count[d]  −  push_success[d] × 2

        Rationale:
          • Each blocked attempt adds 1 (mild penalty — maybe temporary obstacle).
          • Each push success subtracts 2 (strong positive signal — this direction
            has shown it can make progress).
        """
        return (
            self.blocked_count.get(direction, 0)
            - self.push_success.get(direction, 0) * 2
        )

    def _choose(self, obs: List[ObjectView]) -> Action:
        """
        Pick the direction with the lowest score.
        Ties are broken by _PREF order (right → down → left → up).
        """
        best = min(_PREF, key=self._score)
        return _DIR_TO_ACTION[best]


# ---------------------------------------------------------------------------
# NaiveRightAgent
# ---------------------------------------------------------------------------

class NaiveRightAgent:
    """
    Always returns MOVE_RIGHT.  No state, no learning.

    Usefulness as a baseline
    ------------------------
    Establishes the floor for agents that never adapt.  In scenarios where
    right is permanently blocked after the first step, moved_count = 1 (or 0
    for solid walls) regardless of episode length.  Any agent that learns
    should outperform this on most scenarios except open_move_right where
    all agents are equivalent.
    """

    def reset(self) -> None:
        """No-op; NaiveRightAgent has no state to clear."""

    def act(self, obs: List[ObjectView]) -> Action:
        """Always return MOVE_RIGHT, ignoring obs entirely."""
        return MOVE_RIGHT


# ---------------------------------------------------------------------------
# RotateOnBlockAgent
# ---------------------------------------------------------------------------

class RotateOnBlockAgent:
    """
    Simple rotation fallback: stay on the current direction until blocked,
    then advance to the next direction in the cycle (right → down → left → up).

    Usefulness as a baseline
    ------------------------
    Performs better than NaiveRightAgent in obstructed scenarios because it
    eventually escapes blocked directions.  Unlike MinimalHypothesisAgent it
    does NOT track push success, so it cannot distinguish WALL_TRANSFER from
    WALL_SOLID and cannot preferentially revisit directions where pushing
    worked.  passable_walls is not tracked (always 0).

    State
    -----
    current_dir_idx : index into _PREF pointing at the current direction
    step_count      : total act() calls since last reset
    last_obs        : previous observation snapshot (for transition analysis)
    last_action     : previous action taken
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.step_count:   int                        = 0
        self.current_dir_idx: int                     = 0
        self.last_obs:     Optional[List[ObjectView]] = None
        self.last_action:  Optional[Action]           = None

    def act(self, obs: List[ObjectView]) -> Action:
        """
        Observe the transition from the previous step, rotate if blocked,
        then return the current direction.
        """
        if self.last_action is not None and self.last_obs is not None:
            self._maybe_rotate(self.last_obs, self.last_action, obs)

        action = _DIR_TO_ACTION[_PREF[self.current_dir_idx]]
        self.last_obs    = list(obs)
        self.last_action = action
        self.step_count += 1
        return action

    def _maybe_rotate(
        self,
        prev:   List[ObjectView],
        action: Action,
        curr:   List[ObjectView],
    ) -> None:
        """Rotate to the next direction if the selected piece did not move."""
        if action.type not in _DIR_TO_ACTION:
            return
        sel_prev = _selected(prev)
        sel_curr = _selected(curr)
        if sel_prev is None or sel_curr is None:
            return
        if sel_prev.x == sel_curr.x and sel_prev.y == sel_curr.y:
            # Blocked: advance to next direction in cycle
            self.current_dir_idx = (self.current_dir_idx + 1) % len(_PREF)
