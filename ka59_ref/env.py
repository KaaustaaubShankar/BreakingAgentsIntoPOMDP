"""
KA59 Blinded Agent Environment
================================
Agent-facing wrapper around KA59State.

Separation of concerns
-----------------------
  ka59_ref/engine.py   — knows the rules (wall tags, push asymmetry, etc.)
  ka59_ref/env.py      — THIS FILE: agent sees only positions, sizes, opaque kinds

What the agent sees (ObjectView.kind):
  "wall"        — boundary object; both WALL_TRANSFER and WALL_SOLID map here.
                  The agent cannot tell them apart by observation alone; it must
                  discover the behavioural difference through transitions.
  "block"       — pushable object (TAG_BLOCK, TAG_CROSS)
  "controllable"— a piece the agent can select and move (TAG_SELECTED)

What the agent does NOT see:
  - Internal tag strings (0015qniapgwsvb, 0029ifoxxfvvvs, etc.)
  - Which wall type is which
  - Why certain moves succeed or fail

Actions
-------
  MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT — directional moves
  SELECT(target_id)                         — switch active selection

Level spec  (used by reset(), NOT shown to the agent)
----------
  {
    "steps":   <int>,
    "objects": [
      {"id": <str>, "x": <int>, "y": <int>, "w": <int>, "h": <int>,
       "kind": <"selected" | "block" | "cross" | "wall_transfer" | "wall_solid">},
      ...
    ]
  }
  IDs are arbitrary strings; they persist across steps within one episode.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .engine import (
    Obj,
    KA59State,
    STEP,
    TAG_WALL_TRANSFER,
    TAG_WALL_SOLID,
    TAG_SELECTED,
    TAG_CROSS,
    TAG_BLOCK,
)


# ---------------------------------------------------------------------------
# Spec → engine tag mapping  (internal, never exposed to agent)
# ---------------------------------------------------------------------------

_SPEC_KIND_TO_TAG: Dict[str, str] = {
    "selected":      TAG_SELECTED,
    "block":         TAG_BLOCK,
    "cross":         TAG_CROSS,
    "wall_transfer": TAG_WALL_TRANSFER,
    "wall_solid":    TAG_WALL_SOLID,
}

# Observation kind visible to agent (both wall types collapse to "wall")
_TAG_TO_OBS_KIND: Dict[str, str] = {
    TAG_SELECTED:      "controllable",
    TAG_CROSS:         "block",
    TAG_BLOCK:         "block",
    TAG_WALL_TRANSFER: "wall",
    TAG_WALL_SOLID:    "wall",
}


def _obs_kind(obj: Obj) -> str:
    """Derive the agent-visible kind string from internal tags."""
    for tag, kind in _TAG_TO_OBS_KIND.items():
        if tag in obj.tags:
            return kind
    return "wall"   # safe default for any unrecognised boundary object


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectView:
    """
    Agent-visible snapshot of a single game object.
    Contains NO internal tag names or rule explanations.
    """
    id: str
    x: int
    y: int
    w: int
    h: int
    kind: str           # "wall" | "block" | "controllable"
    is_selected: bool   # True iff this is the currently controlled piece


@dataclass(frozen=True)
class StepResult:
    """Result returned by KA59BlindEnv.step()."""
    obs:             Tuple[ObjectView, ...]  # current observation after the action
    moved:           bool                    # did the selected piece change position?
    steps_remaining: int                     # stamina left
    done:            bool                    # True when steps_remaining == 0


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    """An agent action.  Use the module-level constants; don't construct directly."""
    type:      str
    target_id: Optional[str] = None


# Module-level action constants
MOVE_UP    = Action("up")
MOVE_DOWN  = Action("down")
MOVE_LEFT  = Action("left")
MOVE_RIGHT = Action("right")


def SELECT(target_id: str) -> Action:
    """Create a SELECT action targeting the object with the given id."""
    return Action("select", target_id=target_id)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class KA59BlindEnv:
    """
    Blinded agent-facing wrapper around KA59State.

    The agent interacts solely through:
      reset(level_spec) → list[ObjectView]
      step(action)      → StepResult

    Internal mechanics — tag names, wall type distinction, push rules —
    are fully hidden.  The agent must infer behaviour from transitions.
    """

    def __init__(self) -> None:
        self._state: Optional[KA59State] = None
        self._id_map: Dict[str, Obj] = {}   # id string → Obj
        self._obj_ids: Dict[int, str] = {}   # id(Obj) → id string  (by Python identity)

    # -- reset ---------------------------------------------------------------

    def reset(self, level_spec: dict) -> List[ObjectView]:
        """
        Initialise (or re-initialise) the environment from a level spec.

        Returns the initial observation as a list of ObjectView, sorted by id.
        The agent does not receive the spec itself or any explanation of rules.
        """
        objects: List[Obj] = []
        id_map: Dict[str, Obj] = {}
        first_selected: Optional[Obj] = None

        for entry in level_spec["objects"]:
            spec_kind = entry["kind"]
            if spec_kind not in _SPEC_KIND_TO_TAG:
                raise ValueError(f"Unknown kind in level spec: {spec_kind!r}")
            tag = _SPEC_KIND_TO_TAG[spec_kind]
            obj = Obj(
                x=entry["x"],
                y=entry["y"],
                w=entry["w"],
                h=entry["h"],
                tags=(tag,),
            )
            obj_id = entry["id"]
            id_map[obj_id] = obj
            objects.append(obj)
            if tag == TAG_SELECTED and first_selected is None:
                first_selected = obj

        if first_selected is None:
            raise ValueError("Level spec must contain at least one 'selected' object")

        self._id_map  = id_map
        self._obj_ids = {id(obj): obj_id for obj_id, obj in id_map.items()}
        self._state   = KA59State(
            objects  = objects,
            selected = first_selected,
            steps    = level_spec.get("steps", 0),
        )

        return self._observe()

    # -- step ----------------------------------------------------------------

    def step(self, action: Action) -> StepResult:
        """
        Apply *action* to the current state and return a StepResult.

        Raises RuntimeError if called before reset().
        Raises ValueError/KeyError if a SELECT target_id is unknown.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        state = self._state
        before_x = state.selected.x
        before_y = state.selected.y

        if action.type == "select":
            if action.target_id not in self._id_map:
                raise KeyError(f"No object with id={action.target_id!r}")
            new_sel = self._id_map[action.target_id]
            state.action_select(new_sel)
            moved = False   # selection switch: piece didn't translate

        elif action.type in ("up", "down", "left", "right"):
            dx, dy = _action_to_delta(action.type)
            moved = state.action_move(state.selected, dx, dy)

        else:
            raise ValueError(f"Unknown action type: {action.type!r}")

        done = state.steps <= 0
        return StepResult(
            obs             = tuple(self._observe()),
            moved           = moved,
            steps_remaining = state.steps,
            done            = done,
        )

    # -- internal observation builder ----------------------------------------

    def _observe(self) -> List[ObjectView]:
        """
        Build the current observation from engine state.

        Rules deliberately NOT encoded here:
          - Which walls block what
          - Why push succeeded/failed
          - Internal tag names

        Objects are sorted by id for determinism.
        """
        state = self._state
        views = []
        for obj_id, obj in self._id_map.items():
            views.append(ObjectView(
                id          = obj_id,
                x           = obj.x,
                y           = obj.y,
                w           = obj.w,
                h           = obj.h,
                kind        = _obs_kind(obj),
                is_selected = (obj is state.selected),
            ))
        views.sort(key=lambda v: v.id)
        return views


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_delta(action_type: str) -> Tuple[int, int]:
    return {
        "up":    ( 0,    -STEP),
        "down":  ( 0,    +STEP),
        "left":  (-STEP,  0),
        "right": (+STEP,  0),
    }[action_type]
