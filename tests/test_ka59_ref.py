"""
KA59 Reference Simulator Tests
================================
Tests for the faithful reimplementation of KA59 movement mechanics.

Critical asymmetry being verified:
  - Direct move (loydmqkgjw) blocks on BOTH WALL_TRANSFER and WALL_SOLID
  - Push propagation (ifoelczjjh) blocks on WALL_SOLID ONLY
  → Objects can be pushed *through* WALL_TRANSFER boundaries that the
    selected piece cannot directly cross.

Run with:
    python -m pytest tests/test_ka59_ref.py -v
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ka59_ref.engine import (
    Obj,
    KA59State,
    TAG_WALL_TRANSFER,
    TAG_WALL_SOLID,
    TAG_SELECTED,
    TAG_BLOCK,
    STEP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(objects, selected, steps=10):
    """Build a KA59State from a flat list of Obj instances."""
    return KA59State(objects=objects, selected=selected, steps=steps)


def wall_transfer(x, y, w=3, h=3):
    return Obj(x, y, w, h, tags=[TAG_WALL_TRANSFER])


def wall_solid(x, y, w=3, h=3):
    return Obj(x, y, w, h, tags=[TAG_WALL_SOLID])


def selected_obj(x, y, w=3, h=3):
    return Obj(x, y, w, h, tags=[TAG_SELECTED])


def pushable(x, y, w=3, h=3):
    return Obj(x, y, w, h, tags=[TAG_BLOCK])


# ---------------------------------------------------------------------------
# 1. Basic free movement (no walls, no obstacles)
# ---------------------------------------------------------------------------

class TestFreeMovement:
    def test_selected_moves_right(self):
        sel = selected_obj(0, 0)
        state = make_state([sel], sel)
        moved, _ = state.direct_move(sel, STEP, 0)
        assert moved is True
        assert sel.x == STEP

    def test_selected_moves_left(self):
        sel = selected_obj(6, 0)
        state = make_state([sel], sel)
        moved, _ = state.direct_move(sel, -STEP, 0)
        assert moved is True
        assert sel.x == 3

    def test_selected_moves_up(self):
        sel = selected_obj(0, 6)
        state = make_state([sel], sel)
        moved, _ = state.direct_move(sel, 0, -STEP)
        assert moved is True
        assert sel.y == 3

    def test_selected_moves_down(self):
        sel = selected_obj(0, 0)
        state = make_state([sel], sel)
        moved, _ = state.direct_move(sel, 0, STEP)
        assert moved is True
        assert sel.y == STEP


# ---------------------------------------------------------------------------
# 2. Direct-move wall collision (both wall types block the selected piece)
# ---------------------------------------------------------------------------

class TestDirectMoveWallBlocking:
    def test_direct_blocked_by_transfer_wall(self):
        """
        Selected cannot move directly into a WALL_TRANSFER tile.
        This is the first half of the critical asymmetry.
        """
        sel = selected_obj(0, 0)
        wt = wall_transfer(STEP, 0)          # wall sits exactly one STEP ahead
        state = make_state([sel, wt], sel)

        moved, collisions = state.direct_move(sel, STEP, 0)

        assert moved is False, "Direct move should be blocked by WALL_TRANSFER"
        assert len(collisions) == 0, "Wall blocks return empty collision list"
        assert sel.x == 0, "Selected sprite must be reverted to original position"

    def test_direct_blocked_by_solid_wall(self):
        """Selected cannot move directly into a WALL_SOLID tile."""
        sel = selected_obj(0, 0)
        ws = wall_solid(STEP, 0)
        state = make_state([sel, ws], sel)

        moved, collisions = state.direct_move(sel, STEP, 0)

        assert moved is False
        assert len(collisions) == 0
        assert sel.x == 0

    def test_direct_succeeds_when_no_wall(self):
        """Sanity: no wall → direct move succeeds."""
        sel = selected_obj(0, 0)
        state = make_state([sel], sel)

        moved, _ = state.direct_move(sel, STEP, 0)
        assert moved is True
        assert sel.x == STEP


# ---------------------------------------------------------------------------
# 3. Push-propagation wall asymmetry (THE CRITICAL TEST)
# ---------------------------------------------------------------------------

class TestPushWallAsymmetry:
    def test_push_NOT_blocked_by_transfer_wall(self):
        """
        A pushable object IS allowed to be pushed through WALL_TRANSFER.
        (ifoelczjjh only checks WALL_SOLID, not WALL_TRANSFER)
        This is the core wall-transfer mechanic.
        """
        pb = pushable(3, 0)
        wt = wall_transfer(6, 0)             # WALL_TRANSFER is where pushable lands
        state = make_state([pb, wt], selected_obj(100, 100))  # selected far away

        blocked = state.push(pb, STEP, 0)

        assert blocked is False, (
            "Push propagation must NOT be blocked by WALL_TRANSFER "
            "(ifoelczjjh only blocks on WALL_SOLID)"
        )
        assert pb.x == 6, "Pushable should have moved into WALL_TRANSFER tile"

    def test_push_IS_blocked_by_solid_wall(self):
        """
        A pushable object is stopped by WALL_SOLID.
        This confirms the asymmetry is specific to WALL_TRANSFER.
        """
        pb = pushable(3, 0)
        ws = wall_solid(6, 0)
        state = make_state([pb, ws], selected_obj(100, 100))

        blocked = state.push(pb, STEP, 0)

        assert blocked is True, "Push propagation must be blocked by WALL_SOLID"
        assert pb.x == 3, "Pushable must be reverted when push is blocked"

    def test_direct_blocked_transfer_but_push_succeeds_same_wall(self):
        """
        Definitive asymmetry test in one scenario:
        - Direct move of selected into position covered by WALL_TRANSFER → blocked
        - Push of a pushable into the same WALL_TRANSFER position → succeeds
        """
        wt = wall_transfer(6, 0, w=3, h=3)

        # Direct move test
        sel = selected_obj(3, 0)
        state = make_state([sel, wt], sel)
        moved, _ = state.direct_move(sel, STEP, 0)
        assert moved is False, "Selected must not cross WALL_TRANSFER directly"
        assert sel.x == 3

        # Push test (reset pushable to just before the wall)
        pb = pushable(3, 0)
        far_sel = selected_obj(0, 0)
        state2 = make_state([pb, wt, far_sel], far_sel)
        blocked = state2.push(pb, STEP, 0)
        assert blocked is False, "Pushed object can occupy WALL_TRANSFER space"
        assert pb.x == 6


# ---------------------------------------------------------------------------
# 4. Push chain propagation
# ---------------------------------------------------------------------------

class TestPushChain:
    def test_simple_push_chain(self):
        """A → B: pushing A forces B out of the way."""
        a = pushable(0, 0)
        b = pushable(STEP, 0)       # adjacent to a
        state = make_state([a, b], selected_obj(100, 100))

        blocked = state.push(a, STEP, 0)

        assert blocked is False
        assert a.x == STEP
        assert b.x == STEP * 2     # b got pushed one further step

    def test_chain_blocked_when_solid_wall_at_end(self):
        """A → B → WALL_SOLID: entire chain is blocked."""
        a = pushable(0, 0)
        b = pushable(STEP, 0)
        ws = wall_solid(STEP * 2, 0)
        state = make_state([a, b, ws], selected_obj(100, 100))

        blocked = state.push(a, STEP, 0)

        assert blocked is True
        assert a.x == 0     # reverted
        assert b.x == STEP  # reverted

    def test_chain_through_transfer_wall(self):
        """A → B, WALL_TRANSFER behind B: chain can still pass through."""
        a = pushable(0, 0)
        b = pushable(STEP, 0)
        wt = wall_transfer(STEP * 2, 0)
        state = make_state([a, b, wt], selected_obj(100, 100))

        blocked = state.push(a, STEP, 0)

        assert blocked is False
        assert a.x == STEP
        assert b.x == STEP * 2     # b moved into WALL_TRANSFER space


# ---------------------------------------------------------------------------
# 5. Step counter / stamina
# ---------------------------------------------------------------------------

class TestStepCounter:
    def test_step_decrements_on_move(self):
        sel = selected_obj(0, 0)
        state = make_state([sel], sel, steps=5)

        state.action_move(sel, STEP, 0)
        assert state.steps == 4

    def test_step_decrements_on_blocked_move(self):
        """Stamina is consumed even if the move is blocked."""
        sel = selected_obj(0, 0)
        ws = wall_solid(STEP, 0)
        state = make_state([sel, ws], sel, steps=5)

        state.action_move(sel, STEP, 0)
        assert state.steps == 4

    def test_step_does_not_go_negative(self):
        sel = selected_obj(0, 0)
        state = make_state([sel], sel, steps=1)

        state.action_move(sel, STEP, 0)
        state.action_move(sel, STEP, 0)
        assert state.steps == 0

    def test_step_decrements_on_selection_switch(self):
        """Action 6 (selection switch) also consumes a step."""
        sel1 = selected_obj(0, 0)
        sel2 = selected_obj(9, 9)
        state = make_state([sel1, sel2], sel1, steps=5)

        state.action_select(sel2)
        assert state.steps == 4
        assert state.selected is sel2

    def test_initial_steps_preserved(self):
        sel = selected_obj(0, 0)
        state = make_state([sel], sel, steps=20)
        assert state.steps == 20


# ---------------------------------------------------------------------------
# 6. Selection switching
# ---------------------------------------------------------------------------

class TestSelectionSwitch:
    def test_select_switches_active_piece(self):
        sel1 = selected_obj(0, 0)
        sel2 = selected_obj(9, 9)
        state = make_state([sel1, sel2], sel1)

        state.action_select(sel2)
        assert state.selected is sel2

    def test_move_after_switch_moves_new_selected(self):
        sel1 = selected_obj(0, 0)
        sel2 = selected_obj(9, 0)
        state = make_state([sel1, sel2], sel1)

        state.action_select(sel2)
        state.action_move(sel2, STEP, 0)

        assert sel2.x == 9 + STEP
        assert sel1.x == 0   # old selected did not move


# ---------------------------------------------------------------------------
# 7. AABB collision edge cases
# ---------------------------------------------------------------------------

class TestCollision:
    def test_adjacent_no_overlap(self):
        """Two 3×3 objects side-by-side do NOT collide (exclusive bound)."""
        a = Obj(0, 0, 3, 3)
        b = Obj(3, 0, 3, 3)
        assert a.collides_with(b) is False
        assert b.collides_with(a) is False

    def test_overlapping_objects_collide(self):
        a = Obj(0, 0, 4, 4)
        b = Obj(3, 3, 4, 4)
        assert a.collides_with(b) is True

    def test_same_position_collides(self):
        a = Obj(5, 5, 3, 3)
        b = Obj(5, 5, 3, 3)
        assert a.collides_with(b) is True
