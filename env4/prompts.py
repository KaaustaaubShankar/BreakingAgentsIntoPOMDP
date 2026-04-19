"""
prompts.py — POMDP axis prompts for ChiralityEnv (Env4).

Four axes: World, Goal, Mechanics, Feedback.
Each has EASY and HARD variants following the same design as PT (visual_zendo.py).
"""

from enum import Enum


class AxisLevel(str, Enum):
    EASY = "EASY"
    HARD = "HARD"


# ---------------------------------------------------------------------------
# World axis — richness of initial labeled examples
# ---------------------------------------------------------------------------

WORLD_EASY = """\
You are shown {n_examples} labeled example sequences to help you learn the rule:

{examples}

Use these examples to form an initial hypothesis about the hidden rule."""

WORLD_HARD = """\
You receive no examples. You must begin classifying immediately."""


# ---------------------------------------------------------------------------
# Goal axis — clarity of objective
# ---------------------------------------------------------------------------

GOAL_EASY = """\
Your goal is to discover the hidden rule that determines whether a sequence is \
classified as "R" or "L". Once you understand the rule, you can classify new \
sequences correctly. You succeed when you correctly classify a sequence."""

GOAL_HARD = """\
Return the correct output for each input."""


# ---------------------------------------------------------------------------
# Mechanics axis — explanation of available actions
# ---------------------------------------------------------------------------

MECHANICS_EASY = """\
You have two actions available:

CLASSIFY: Submit your classification for a given sequence.
  Format: {"action": "classify", "sequence": "<5-letter sequence>", "label": "R" or "L"}
  Use this when you are ready to commit to a classification.

QUERY: Request one additional labeled example to refine your hypothesis.
  Format: {"action": "query"}
  Each QUERY costs 1 token from your budget. Use it when you need more data."""

MECHANICS_HARD = """\
Format your response as JSON."""


# ---------------------------------------------------------------------------
# Feedback axis — information returned after CLASSIFY
# ---------------------------------------------------------------------------

FEEDBACK_EASY = """\
After each CLASSIFY attempt:
- If correct: you will be told "Correct. The label for <sequence> is <label>."
- If incorrect: you will be told "Incorrect. The correct label for <sequence> was <label>."
Use this feedback to refine your understanding of the rule."""

FEEDBACK_HARD = """\
You will not receive feedback on your classifications."""


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def build_system_prompt(
    world: AxisLevel,
    goal: AxisLevel,
    mechanics: AxisLevel,
    feedback: AxisLevel,
    examples: list[tuple[str, str]],
    n_easy_examples: int = 5,
) -> str:
    example_str = "\n".join(f"  {seq} -> {label}" for seq, label in examples)

    if world == AxisLevel.EASY:
        world_block = WORLD_EASY.format(n_examples=n_easy_examples, examples=example_str)
    else:
        # HARD: zero examples, cold start
        world_block = WORLD_HARD

    goal_block = GOAL_EASY if goal == AxisLevel.EASY else GOAL_HARD
    mechanics_block = MECHANICS_EASY if mechanics == AxisLevel.EASY else MECHANICS_HARD
    feedback_block = FEEDBACK_EASY if feedback == AxisLevel.EASY else FEEDBACK_HARD

    return "\n\n".join([world_block, goal_block, mechanics_block, feedback_block])


def format_classify_result(result: dict, feedback_level: AxisLevel) -> str:
    seq = result["sequence"]
    label = result["true_label"]
    correct = result["correct"]
    turn = result.get("turn", "?")
    if feedback_level == AxisLevel.HARD:
        return f'Turn {turn} complete.'
    if correct:
        return f'Correct. The label for {seq} is {label}.'
    else:
        return f'Incorrect. The correct label for {seq} was {label}.'


def format_query_result(seq: str, label: str) -> str:
    return f'Example: {seq} -> {label}'
