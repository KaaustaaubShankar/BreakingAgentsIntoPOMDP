"""
epistemic.py — BeliefState & RuleCandidate tracking for BreakingAgentsIntoPOMDP

Instruments the existing Zendo agent loop with epistemic lineage tracking.
Zero changes required to LithicArrayEnv, rules.py, or the ablation harness.

Usage:
    from epistemic import BeliefStateTracker, extract_candidate_from_text

See: PAPER1_MINIMAL_INTERFACE_SPEC.md for full design rationale.
"""

import math
import time
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------

@dataclass
class EvidenceRef:
    turn: int
    event_type: str          # "strata_match" | "strata_mismatch" | "counter_example"
    arrangement_hash: str    # Deterministic hash of the arrangement for dedup
    detail: str              # Brief description


@dataclass
class StatusChange:
    from_status: str
    to_status: str
    turn: int
    reason: str


# ---------------------------------------------------------------------------
# RuleCandidate
# ---------------------------------------------------------------------------

class RuleCandidate:
    """
    A first-class rule hypothesis with lifecycle and evidence tracking.

    Lifecycle states: candidate → supported | rejected
    Confidence: Laplace-smoothed MLE from evidence ledger.
    """

    N_SUPPORT: int = 3       # Turns until candidate → supported
    K_SMOOTH: float = 2.0    # Confidence smoothing constant

    def __init__(self, id: str, rule_text: str, rule_code: Optional[str], created_turn: int):
        self.id = id
        self.rule_text = rule_text
        self.rule_code = rule_code
        self.status = "candidate"
        self.created_turn = created_turn
        self.last_updated_turn = created_turn
        self.supporting_evidence: List[EvidenceRef] = []
        self.contradicting_evidence: List[EvidenceRef] = []
        self.promotion_history: List[StatusChange] = []
        self.proposal_attempts: int = 0
        self.accepted: bool = False

    @property
    def confidence(self) -> float:
        s = len(self.supporting_evidence)
        c = len(self.contradicting_evidence)
        return s / (s + c + self.K_SMOOTH)

    def add_support(self, evidence: EvidenceRef):
        self.supporting_evidence.append(evidence)
        self.last_updated_turn = evidence.turn
        if self.status == "candidate" and len(self.supporting_evidence) >= self.N_SUPPORT:
            self._transition("supported", evidence.turn,
                             f"{self.N_SUPPORT}th supporting evidence")

    def add_contradiction(self, evidence: EvidenceRef):
        self.contradicting_evidence.append(evidence)
        self.last_updated_turn = evidence.turn
        if self.status != "rejected":
            self._transition("rejected", evidence.turn,
                             f"Contradicted by {evidence.event_type} at turn {evidence.turn}")

    def record_proposal(self, accepted: bool, turn: int):
        self.proposal_attempts += 1
        if accepted:
            self.accepted = True
            if self.status != "supported":
                self._transition("supported", turn, "Accepted by environment")

    def _transition(self, new_status: str, turn: int, reason: str):
        self.promotion_history.append(StatusChange(
            from_status=self.status,
            to_status=new_status,
            turn=turn,
            reason=reason,
        ))
        self.status = new_status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "rule_text": self.rule_text,
            "rule_code": self.rule_code,
            "status": self.status,
            "confidence": self.confidence,
            "created_turn": self.created_turn,
            "last_updated_turn": self.last_updated_turn,
            "supporting_evidence": [asdict(e) for e in self.supporting_evidence],
            "contradicting_evidence": [asdict(e) for e in self.contradicting_evidence],
            "promotion_history": [asdict(s) for s in self.promotion_history],
            "proposal_attempts": self.proposal_attempts,
            "accepted": self.accepted,
        }


# ---------------------------------------------------------------------------
# BeliefState
# ---------------------------------------------------------------------------

@dataclass
class BeliefState:
    turn: int
    timestamp: float
    active_candidates: List[str]
    top_candidate_id: Optional[str]
    top_confidence: float
    entropy: float
    num_candidates: int
    total_evidence: int
    last_event: str
    delta_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BeliefStateTracker
# ---------------------------------------------------------------------------

class BeliefStateTracker:
    """
    Maintains and snapshots the agent's belief state across a game.
    Call update() after every STRATA or PROPOSE interaction.
    Call save() at game end to merge traces into the history JSON.
    """

    def __init__(self):
        self.snapshots: List[BeliefState] = []
        self.candidates: Dict[str, RuleCandidate] = {}
        self.total_evidence: int = 0
        self._candidate_counter: int = 0

    # ------------------------------------------------------------------
    # Candidate management
    # ------------------------------------------------------------------

    def register_candidate(self, rule_text: str, rule_code: Optional[str] = None,
                           turn: int = 0) -> RuleCandidate:
        """Create and register a new hypothesis. Returns the candidate."""
        self._candidate_counter += 1
        cid = f"rc_{self._candidate_counter:03d}"
        candidate = RuleCandidate(id=cid, rule_text=rule_text,
                                  rule_code=rule_code, created_turn=turn)
        self.candidates[cid] = candidate
        return candidate

    def find_candidate_by_text(self, rule_text: str) -> Optional[RuleCandidate]:
        """Look up an active candidate by its rule text (case-insensitive)."""
        needle = rule_text.strip().lower()
        for c in self.candidates.values():
            if c.status != "rejected" and c.rule_text.strip().lower() == needle:
                return c
        return None

    # ------------------------------------------------------------------
    # Evidence updates
    # ------------------------------------------------------------------

    def evaluate_strata(self, arrangement_data: Any, predicted: bool,
                        actual_result: bool, turn: int):
        """
        Call after each STRATA result.
        predicted: what the agent predicted (True = Quartz, False = Shale).
        actual_result: True if the arrangement IS a valid example (Quartz).
        """
        correct_prediction = (predicted == actual_result)
        arr_hash = str(hash(json.dumps(arrangement_data, sort_keys=True)))[:8]

        for candidate in self.candidates.values():
            if candidate.status == "rejected":
                continue
            ev = EvidenceRef(
                turn=turn,
                event_type="strata_match" if correct_prediction else "strata_mismatch",
                arrangement_hash=arr_hash,
                detail=f"predicted={predicted}, actual={actual_result}",
            )
            if correct_prediction:
                candidate.add_support(ev)
            else:
                candidate.add_contradiction(ev)

    def evaluate_propose(self, rule_text: str, accepted: bool, turn: int):
        """Call after each PROPOSE result."""
        candidate = self.find_candidate_by_text(rule_text)
        if candidate is None:
            candidate = self.register_candidate(rule_text, turn=turn)
        candidate.record_proposal(accepted=accepted, turn=turn)
        if not accepted:
            ev = EvidenceRef(
                turn=turn,
                event_type="counter_example",
                arrangement_hash="propose",
                detail="Proposal rejected by environment",
            )
            candidate.add_contradiction(ev)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def update(self, event: str, turn: int) -> BeliefState:
        """
        Take a belief snapshot. Call after every STRATA or PROPOSE interaction.
        event: "strata_match" | "strata_mismatch" | "propose_accepted" | "propose_rejected"
        """
        self.total_evidence += 1

        active = {k: v for k, v in self.candidates.items() if v.status != "rejected"}

        if not active:
            top_id, top_conf = None, 0.0
            entropy = float("nan")
        else:
            ranked = sorted(active.values(), key=lambda c: c.confidence, reverse=True)
            top_id = ranked[0].id
            top_conf = ranked[0].confidence
            confs = [c.confidence for c in ranked]
            total = sum(confs) or 1.0
            probs = [c / total for c in confs]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        prev_top = self.snapshots[-1].top_confidence if self.snapshots else 0.0

        snap = BeliefState(
            turn=turn,
            timestamp=time.time(),
            active_candidates=[c.id for c in sorted(active.values(),
                               key=lambda c: c.confidence, reverse=True)],
            top_candidate_id=top_id,
            top_confidence=top_conf,
            entropy=entropy,
            num_candidates=len(active),
            total_evidence=self.total_evidence,
            last_event=event,
            delta_confidence=top_conf - prev_top,
            metadata={},
        )
        self.snapshots.append(snap)
        return snap

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_log(self) -> Dict[str, Any]:
        """Return a dict suitable for merging into the history JSON."""
        return {
            "belief_trace": [asdict(s) for s in self.snapshots],
            "rule_candidates": {cid: c.to_dict() for cid, c in self.candidates.items()},
        }

    def save(self, history_file: str):
        """Merge epistemic traces into an existing history JSON file.
        
        The history file may be a list (event log) or a dict. We wrap list
        format in a dict so we can add our keys alongside it.
        """
        try:
            with open(history_file, "r") as f:
                raw = json.load(f)
        except Exception:
            raw = {}

        # Cal's format is a list of event dicts — wrap it to add our keys
        if isinstance(raw, list):
            data = {"events": raw}
        else:
            data = raw

        data.update(self.to_log())

        with open(history_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[epistemic] Saved belief trace ({len(self.snapshots)} snapshots, "
              f"{len(self.candidates)} candidates) → {history_file}")


# ---------------------------------------------------------------------------
# Candidate extraction from LLM text
# ---------------------------------------------------------------------------

_HYPOTHESIS_PATTERNS = [
    r"(?:i think|i believe|my hypothesis is|the rule (?:is|might be|could be)|"
    r"rule:\s*|hypothesis:\s*|i(?:'m| am) guessing)\s+(.+?)(?:\.|$)",
    r"rule_description[\"']?\s*:\s*[\"'](.+?)[\"']",
]

def extract_candidate_from_text(text: str) -> Optional[str]:
    """
    Heuristic extraction of a rule hypothesis from LLM natural-language output.
    Returns the rule text if found, else None.
    """
    for pattern in _HYPOTHESIS_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            candidate_text = m.group(1).strip().rstrip(".,;\"'")
            if len(candidate_text) > 5:
                return candidate_text
    return None
