import random
from typing import Callable, Tuple, Optional
from visual_zendo import Shape, Arrangement

COLORS = ["red", "blue", "green", "yellow", "black"]
SIZES = ["small", "medium", "large"]
TYPES = ["triangle", "circle", "square"]

def generate_random_shape() -> Shape:
    return Shape(
        color=random.choice(COLORS),
        size=random.choice(SIZES),
        type_=random.choice(TYPES)
    )

def generate_random_arrangement(min_shapes=1, max_shapes=5) -> Arrangement:
    num_shapes = random.randint(min_shapes, max_shapes)
    shapes = [generate_random_shape() for _ in range(num_shapes)]
    return Arrangement(shapes=shapes)

# A registry of predefined rules
# Each rule is a tuple: (name, eval_fn)
# Ensure deterministic rule evaluation
RULES_REGISTRY = [
    (
        "At least one red triangle",
        lambda arr: any(s.color == "red" and s.type_ == "triangle" for s in arr.shapes)
    ),
    (
        "All pieces are the same color",
        lambda arr: len(set(s.color for s in arr.shapes)) == 1 if arr.shapes else False
    ),
    (
        "Exactly two large shapes",
        lambda arr: sum(1 for s in arr.shapes if s.size == "large") == 2
    ),
    (
        "No blue squares",
        lambda arr: not any(s.color == "blue" and s.type_ == "square" for s in arr.shapes)
    )
]

def get_rule_by_index(index: int) -> Tuple[str, Callable]:
    """Returns rule name and evaluation function by index."""
    if 0 <= index < len(RULES_REGISTRY):
        return RULES_REGISTRY[index]
    raise ValueError(f"Rule index {index} out of bounds.")

def create_counter_example_generator(basalt_eval_fn: Callable, max_attempts=1000) -> Callable:
    """
    Returns a function that takes an agent's evaluation function and tries to find a counter-example.
    A counter-example is an arrangement where basalt_eval_fn(arr) != agent_eval_fn(arr).
    If no counter-example is found after max_attempts, assumes the rules are equivalent and returns None.
    """
    def generator(agent_eval_fn: Callable) -> Optional[Tuple[Arrangement, bool]]:
        for _ in range(max_attempts):
            arr = generate_random_arrangement()
            try:
                basalt_label = basalt_eval_fn(arr)
                agent_label = agent_eval_fn(arr)
                if basalt_label != agent_label:
                    return arr, basalt_label
            except Exception:
                # If agent eval function crashes on this input, it fails to handle valid arrangements
                # We can treat this as a counter-example where the basalt succeeds but agent fails
                return arr, basalt_eval_fn(arr)
        return None
        
    return generator

def generate_initial_examples(basalt_eval_fn: Callable, num_quartz=3, num_shale=3) -> list:
    """Generates initial labeled examples to present to the user."""
    examples = []
    q_count, s_count = 0, 0
    while q_count < num_quartz or s_count < num_shale:
        arr = generate_random_arrangement()
        label = basalt_eval_fn(arr)
        if label and q_count < num_quartz:
            examples.append((arr, label))
            q_count += 1
        elif not label and s_count < num_shale:
            examples.append((arr, label))
            s_count += 1
    
    # Shuffle so they aren't completely ordered
    random.shuffle(examples)
    return examples
