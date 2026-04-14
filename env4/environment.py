"""
environment.py — ChiralityEnv: sequence classification environment for Env4.

The agent sees (sequence, label) examples and must discover the hidden rule
that determines whether a sequence is "R" or "L".
"""

import random
import string
from dataclasses import dataclass, field
from typing import Optional
from env4.rules import get_rule, Label

SYMBOLS = list("ABCDE")
SEQ_LEN = 5


def random_sequence(rng: random.Random) -> str:
    return "".join(rng.choices(SYMBOLS, k=SEQ_LEN))


def generate_examples(rule_index: int, n: int, rng: random.Random) -> list[tuple[str, Label]]:
    """Generate n labeled (sequence, label) pairs for a given rule."""
    rule = get_rule(rule_index)
    examples = []
    seen = set()
    attempts = 0
    while len(examples) < n and attempts < n * 20:
        seq = random_sequence(rng)
        if seq not in seen:
            seen.add(seq)
            examples.append((seq, rule(seq)))
        attempts += 1
    return examples


@dataclass
class ChiralityEnv:
    rule_index: int
    seed: int = 42

    _rng: random.Random = field(init=False, repr=False)
    _rule_fn: object = field(init=False, repr=False)
    _turn: int = field(default=0, init=False)
    _queries_used: int = field(default=0, init=False)
    _classify_attempts: int = field(default=0, init=False)
    _correct: Optional[bool] = field(default=None, init=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        self._rule_fn = get_rule(self.rule_index)

    def get_initial_examples(self, n: int) -> list[tuple[str, Label]]:
        return generate_examples(self.rule_index, n, self._rng)

    def query(self) -> tuple[str, Label]:
        """QUERY action: agent requests one more labeled example (costs 1 token)."""
        self._queries_used += 1
        seq = random_sequence(self._rng)
        return seq, self._rule_fn(seq)

    def classify(self, sequence: str, prediction: Label) -> dict:
        """CLASSIFY action: agent submits a classification for a sequence."""
        self._classify_attempts += 1
        self._turn += 1
        true_label = self._rule_fn(sequence)
        correct = prediction == true_label
        self._correct = correct
        return {
            "correct": correct,
            "true_label": true_label,
            "prediction": prediction,
            "sequence": sequence,
            "turn": self._turn,
        }

    @property
    def turns_taken(self) -> int:
        return self._turn

    @property
    def queries_used(self) -> int:
        return self._queries_used
