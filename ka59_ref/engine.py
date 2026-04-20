"""
KA59 Faithful Simulator — Core Engine
=====================================
Faithful, readable reimplementation of the movement semantics from ka59.py
(ARC Prize 2026 game source).

Source constants & obfuscated names mapped to readable names:
  zsqdfmgyjo  = 3     → STEP
  zuhizjvlpo  = 5     → MAX_PUSH_TRACK_STEPS
  0015qniapgwsvb      → TAG_WALL_TRANSFER
  0029ifoxxfvvvs      → TAG_WALL_SOLID
  0022vrxelxosfy      → TAG_SELECTED   (controllable/selectable block)
  0001uqqokjrptk      → TAG_CROSS      (cross/plus pushable)
  0003umnkyodpjp      → TAG_BLOCK      (plain pushable block)

Critical Wall Asymmetry
-----------------------
``direct_move`` (mirrors ``loydmqkgjw`` in source):
  Checks collisions against BOTH TAG_WALL_TRANSFER and TAG_WALL_SOLID.
  The selected piece cannot cross either wall type directly.

``push`` (mirrors ``ifoelczjjh`` in source):
  Checks collisions against TAG_WALL_SOLID ONLY.
  Pushed objects are *not* blocked by TAG_WALL_TRANSFER — they can be
  pushed through "transfer gate" walls that the player cannot enter.

This asymmetry is why objects can end up on the far side of a transfer
boundary while the selected/player piece remains on the near side.

Ambiguity note
--------------
The source uses pixel-accurate mask collision (arcengine Sprite.collides_with).
This simulator uses axis-aligned bounding-box (AABB) collision instead, which
is correct for all rectangular, fully-opaque objects. Sprites with transparent
pixels (e.g. cross-shaped sprites) would diverge from source in edge cases,
but the core wall-asymmetry mechanic is unaffected.
"""

from __future__ import annotations
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Constants (from source)
# ---------------------------------------------------------------------------

STEP: int = 3               # pixels per movement action (zsqdfmgyjo)
MAX_PUSH_TRACK_STEPS: int = 5  # steps before a pushed object is released (zuhizjvlpo)

# Wall tags
TAG_WALL_TRANSFER: str = "0015qniapgwsvb"  # one-way gate: blocks direct move, not push
TAG_WALL_SOLID: str    = "0029ifoxxfvvvs"  # hard wall: blocks everything

# Pushable-object tags
TAG_SELECTED: str = "0022vrxelxosfy"  # currently selectable/controllable block
TAG_CROSS: str    = "0001uqqokjrptk"  # cross-shaped pushable
TAG_BLOCK: str    = "0003umnkyodpjp"  # plain rectangular pushable

PUSHABLE_TAGS = frozenset({TAG_SELECTED, TAG_CROSS, TAG_BLOCK})
WALL_TAGS_ANY = frozenset({TAG_WALL_TRANSFER, TAG_WALL_SOLID})


# ---------------------------------------------------------------------------
# Obj: game object
# ---------------------------------------------------------------------------

class Obj:
    """
    A game object with a rectangular bounding box and a set of tags.

    Position (x, y) is the top-left pixel.
    The object occupies pixels [x, x+w) × [y, y+h).
    """

    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        tags: tuple | list | frozenset = (),
    ) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.tags = frozenset(tags)

    # -- Collision -----------------------------------------------------------

    def collides_with(self, other: "Obj") -> bool:
        """
        AABB overlap check (exclusive upper bound, matching arcengine semantics).
        Two objects collide when their bounding boxes share at least one pixel.

        Identity guard: a sprite never collides with itself (mirrors arcengine
        Sprite.collides_with which skips self-comparison).
        """
        if other is self:
            return False
        return (
            self.x < other.x + other.w
            and other.x < self.x + self.w
            and self.y < other.y + other.h
            and other.y < self.y + self.h
        )

    # -- Helpers -------------------------------------------------------------

    def is_pushable(self) -> bool:
        return bool(self.tags & PUSHABLE_TAGS)

    def has_any_tag(self, tag_set: frozenset) -> bool:
        return bool(self.tags & tag_set)

    def __repr__(self) -> str:
        return f"Obj(x={self.x}, y={self.y}, w={self.w}, h={self.h}, tags={set(self.tags)})"


# ---------------------------------------------------------------------------
# KA59State: game state
# ---------------------------------------------------------------------------

class KA59State:
    """
    Minimal KA59 game state.

    Attributes
    ----------
    objects  : all Obj instances in the level (walls, blocks, selected piece, …)
    selected : the currently controlled piece
    steps    : remaining stamina (decremented by actions 1/2/3/4/6)
    """

    def __init__(
        self,
        objects: List[Obj],
        selected: Obj,
        steps: int = 0,
    ) -> None:
        self.objects = list(objects)
        self.selected = selected
        self.steps = steps

    # -- Object queries ------------------------------------------------------

    def _walls_any(self) -> List[Obj]:
        """All wall objects (both WALL_TRANSFER and WALL_SOLID)."""
        return [o for o in self.objects if o.has_any_tag(WALL_TAGS_ANY)]

    def _walls_solid(self) -> List[Obj]:
        """Solid walls only (TAG_WALL_SOLID)."""
        return [o for o in self.objects if TAG_WALL_SOLID in o.tags]

    def _pushables(self) -> List[Obj]:
        """All pushable objects."""
        return [o for o in self.objects if o.is_pushable()]

    # -- Core movement: mirrors loydmqkgjw ----------------------------------

    def direct_move(
        self, sprite: Obj, dx: int, dy: int
    ) -> Tuple[bool, List[Obj]]:
        """
        Attempt to move *sprite* by (dx, dy).  Mirrors ``loydmqkgjw``.

        Logic (exact source order):
          1. Tentatively move sprite.
          2. Check collisions with ANY wall (WALL_TRANSFER or WALL_SOLID).
             → If hit: revert; return (False, []).
          3. Check collisions with pushable objects.
             → If hit: revert; return (False, [colliding_pushables]).
          4. No collision: sprite stays at new position; return (True, []).

        Returns
        -------
        (moved: bool, push_candidates: list[Obj])
            moved=True   → sprite moved freely.
            moved=False, push_candidates=[]  → wall blocked, no push possible.
            moved=False, push_candidates=[…] → pushable objects are in the way.
        """
        orig_x, orig_y = sprite.x, sprite.y
        sprite.x += dx
        sprite.y += dy

        # Step 1: check BOTH wall types (critical: WALL_TRANSFER also blocks)
        for wall in self._walls_any():
            if sprite.collides_with(wall):
                sprite.x, sprite.y = orig_x, orig_y
                return False, []

        # Step 2: check pushable objects
        hits = [o for o in self._pushables() if sprite.collides_with(o)]
        if hits:
            sprite.x, sprite.y = orig_x, orig_y
            return False, hits

        # No collision: move committed
        return True, []

    # -- Core push propagation: mirrors ifoelczjjh --------------------------

    def push(self, sprite: Obj, dx: int, dy: int) -> bool:
        """
        Recursively attempt to push *sprite* by (dx, dy).  Mirrors ``ifoelczjjh``.

        Logic (exact source order):
          1. Tentatively move sprite.
          2. Check collisions with WALL_SOLID ONLY (NOT WALL_TRANSFER!).
             → If hit: revert; return True (blocked).
          3. Check collisions with pushable objects; for each, recurse.
             → If any recursive push blocked: revert this sprite; return True.
          4. No block: sprite stays at new position; return False (success).

        The key asymmetry: WALL_TRANSFER is *not* checked here, so pushed
        objects can enter WALL_TRANSFER space that the selected piece cannot.

        Returns
        -------
        True  → push blocked; sprite is at its original position.
        False → push succeeded; sprite (and any chain) moved.
        """
        orig_x, orig_y = sprite.x, sprite.y
        sprite.x += dx
        sprite.y += dy

        # Step 1: only WALL_SOLID blocks push (WALL_TRANSFER does NOT)
        for wall in self._walls_solid():
            if sprite.collides_with(wall):
                sprite.x, sprite.y = orig_x, orig_y
                return True

        # Step 2: recursively push anything in the way
        for other in self._pushables():
            if sprite.collides_with(other):
                if self.push(other, dx, dy):
                    sprite.x, sprite.y = orig_x, orig_y
                    return True

        # Push succeeded
        return False

    # -- Action wrappers (consume stamina) -----------------------------------

    def action_move(self, sprite: Obj, dx: int, dy: int) -> bool:
        """
        Execute a directional move action (actions 1/2/3/4 in source).

        Consumes one step of stamina regardless of outcome.
        If pushable objects are in the way, attempts to push each one
        before retrying the move.

        Returns True if the selected piece ended up at a new position.
        """
        self.steps = max(0, self.steps - 1)

        moved, push_candidates = self.direct_move(sprite, dx, dy)
        if moved:
            return True

        if push_candidates:
            # Attempt to push all blocking pushables
            any_blocked = any(
                self.push(obj, dx, dy) for obj in push_candidates
            )
            if not any_blocked:
                # Pushables cleared — retry the move
                moved2, _ = self.direct_move(sprite, dx, dy)
                return moved2

        return False

    def action_select(self, new_selected: Obj) -> None:
        """
        Switch active selection to *new_selected* (action 6 in source).
        Consumes one step of stamina.
        """
        self.steps = max(0, self.steps - 1)
        self.selected = new_selected
