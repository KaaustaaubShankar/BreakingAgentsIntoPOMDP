"""
rules.py — Synthetic chirality-inspired classification rules for Env4.

Each rule maps a 5-symbol sequence (letters A-E) to "R" or "L".
Rules are designed to be structurally plausible but not pre-trained knowledge.
"""

from typing import Literal

Label = Literal["R", "L"]

VOWELS = {"A", "E"}


def rule_1(seq: str) -> Label:
    """R if the middle symbol (index 2) is alphabetically earlier than the first symbol."""
    return "R" if seq[2] < seq[0] else "L"


def rule_2(seq: str) -> Label:
    """R if sequence is not a palindrome AND first symbol alphabetically precedes last."""
    not_palindrome = seq != seq[::-1]
    first_before_last = seq[0] < seq[-1]
    return "R" if (not_palindrome and first_before_last) else "L"


def rule_3(seq: str) -> Label:
    """R if exactly one symbol appears more than once (one pair, rest unique)."""
    counts = [seq.count(c) for c in set(seq)]
    return "R" if counts.count(2) == 1 and all(c <= 2 for c in counts) else "L"


def rule_4(seq: str) -> Label:
    """R if the count of vowels (A, E) in the sequence is even (including zero)."""
    vowel_count = sum(1 for c in seq if c in VOWELS)
    return "R" if vowel_count % 2 == 0 else "L"


def rule_5(seq: str) -> Label:
    """R if the first symbol has a higher ASCII value than the last symbol."""
    return "R" if seq[0] > seq[-1] else "L"


def rule_6(seq: str) -> Label:
    """L if the sequence contains two adjacent identical symbols."""
    return "L" if any(seq[i] == seq[i+1] for i in range(len(seq)-1)) else "R"


def rule_7(seq: str) -> Label:
    """R if exactly 2 adjacent pairs are in ascending alphabetical order."""
    ascending_pairs = sum(seq[i] < seq[i+1] for i in range(len(seq)-1))
    return "R" if ascending_pairs == 2 else "L"


def rule_8(seq: str) -> Label:
    """L if the number of symbol-type transitions (vowel<->consonant) is odd."""
    transitions = sum(
        1 for i in range(len(seq)-1)
        if (seq[i] in VOWELS) != (seq[i+1] in VOWELS)
    )
    return "L" if transitions % 2 == 1 else "R"


def rule_9(seq: str) -> Label:
    """R if the second symbol has a higher ASCII value than the fourth symbol."""
    return "R" if seq[1] > seq[3] else "L"


def rule_10(seq: str) -> Label:
    """R if the 2nd character (index 1) is alphabetically earlier than the 4th character (index 3)."""
    return "R" if seq[1] < seq[3] else "L"


RULES = {
    1: rule_1,
    2: rule_2,
    3: rule_3,
    4: rule_4,
    5: rule_5,
    6: rule_6,
    7: rule_7,
    8: rule_8,
    9: rule_9,
    10: rule_10,
}


def get_rule(index: int):
    return RULES[index]


def get_rule_name(index: int) -> str:
    return RULES[index].__doc__.strip().split("\n")[0]
