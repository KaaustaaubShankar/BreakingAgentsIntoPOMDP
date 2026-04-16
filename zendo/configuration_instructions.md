# Zendo Configuration Instructions

This document shows the instruction text produced by the current Zendo prompt configuration.

The system prompt is assembled as:

```text
{goal_instruction}
{mechanics_instruction}

Valid colors: red, blue, green, yellow, black. Sizes: small, medium, large. Types: triangle, circle, square.
```

The four ablation configurations are:

- `baseline`
- `world_HARD`
- `goal_HARD`
- `mechanics_HARD`
- `feedback_HARD`

## Baseline

Axes:

- `world=EASY`
- `goal=EASY`
- `mechanics=EASY`
- `feedback=EASY`

Goal instruction:

```text
Your goal is to discover the hidden rule that classifies arrangements as Quartz or Shale. State the rule explicitly to win.
```

Mechanics instruction:

```text
You have two actions available. STRATA submits an arrangement together with your predicted classification. Matching the Basalt's classification earns 1 token. PROPOSE costs 1 token. If a PROPOSE attempt is rejected, the Basalt provides a counter-example.

You must respond with only a JSON block containing your action. Do not include markdown or other text outside the JSON.
Strata Action Schema:
{
  "action": "STRATA",
  "arrangement": [{"color": "red", "size": "small", "type_": "circle"}],
  "prediction": true  // true for Quartz, false for Shale
}

Propose Action Schema (Requires 1 token):
{
  "action": "PROPOSE",
  "rule_description": "All pieces are red",
  "rule_code": "def agent_eval_fn(arr):\n    # arr is an Arrangement obj. You can access shapes via arr.shapes. Each shape has .color, .size, .type_\n    return all(s.color == 'red' for s in arr.shapes)"
}
```

Full system prompt:

```text
Your goal is to discover the hidden rule that classifies arrangements as Quartz or Shale. State the rule explicitly to win.
You have two actions available. STRATA submits an arrangement together with your predicted classification. Matching the Basalt's classification earns 1 token. PROPOSE costs 1 token. If a PROPOSE attempt is rejected, the Basalt provides a counter-example.

You must respond with only a JSON block containing your action. Do not include markdown or other text outside the JSON.
Strata Action Schema:
{
  "action": "STRATA",
  "arrangement": [{"color": "red", "size": "small", "type_": "circle"}],
  "prediction": true  // true for Quartz, false for Shale
}

Propose Action Schema (Requires 1 token):
{
  "action": "PROPOSE",
  "rule_description": "All pieces are red",
  "rule_code": "def agent_eval_fn(arr):\n    # arr is an Arrangement obj. You can access shapes via arr.shapes. Each shape has .color, .size, .type_\n    return all(s.color == 'red' for s in arr.shapes)"
}

Valid colors: red, blue, green, yellow, black. Sizes: small, medium, large. Types: triangle, circle, square.
```

## world_HARD

Axes:

- `world=HARD`
- `goal=EASY`
- `mechanics=EASY`
- `feedback=EASY`

Instruction text:

The instruction text is identical to `baseline`.

What changes:

- Initial examples are presented as image paths instead of inline JSON shape tables when image rendering is available.

## goal_HARD

Axes:

- `world=EASY`
- `goal=HARD`
- `mechanics=EASY`
- `feedback=EASY`

Goal instruction:

```text

```

Mechanics instruction:

```text
You have two actions available. STRATA submits an arrangement together with your predicted classification. Matching the Basalt's classification earns 1 token. PROPOSE costs 1 token. If a PROPOSE attempt is rejected, the Basalt provides a counter-example.

You must respond with only a JSON block containing your action. Do not include markdown or other text outside the JSON.
Strata Action Schema:
{
  "action": "STRATA",
  "arrangement": [{"color": "red", "size": "small", "type_": "circle"}],
  "prediction": true  // true for Quartz, false for Shale
}

Propose Action Schema (Requires 1 token):
{
  "action": "PROPOSE",
  "rule_description": "All pieces are red",
  "rule_code": "def agent_eval_fn(arr):\n    # arr is an Arrangement obj. You can access shapes via arr.shapes. Each shape has .color, .size, .type_\n    return all(s.color == 'red' for s in arr.shapes)"
}
```

Full system prompt:

```text

You have two actions available. STRATA submits an arrangement together with your predicted classification. Matching the Basalt's classification earns 1 token. PROPOSE costs 1 token. If a PROPOSE attempt is rejected, the Basalt provides a counter-example.

You must respond with only a JSON block containing your action. Do not include markdown or other text outside the JSON.
Strata Action Schema:
{
  "action": "STRATA",
  "arrangement": [{"color": "red", "size": "small", "type_": "circle"}],
  "prediction": true  // true for Quartz, false for Shale
}

Propose Action Schema (Requires 1 token):
{
  "action": "PROPOSE",
  "rule_description": "All pieces are red",
  "rule_code": "def agent_eval_fn(arr):\n    # arr is an Arrangement obj. You can access shapes via arr.shapes. Each shape has .color, .size, .type_\n    return all(s.color == 'red' for s in arr.shapes)"
}

Valid colors: red, blue, green, yellow, black. Sizes: small, medium, large. Types: triangle, circle, square.
```

## mechanics_HARD

Axes:

- `world=EASY`
- `goal=EASY`
- `mechanics=HARD`
- `feedback=EASY`

Goal instruction:

```text
Your goal is to discover the hidden rule that classifies arrangements as Quartz or Shale. State the rule explicitly to win.
```

Mechanics instruction:

```text
Respond with a JSON object containing your action ('STRATA' or 'PROPOSE').
```

Full system prompt:

```text
Your goal is to discover the hidden rule that classifies arrangements as Quartz or Shale. State the rule explicitly to win.
Respond with a JSON object containing your action ('STRATA' or 'PROPOSE').

Valid colors: red, blue, green, yellow, black. Sizes: small, medium, large. Types: triangle, circle, square.
```

## feedback_HARD

Axes:

- `world=EASY`
- `goal=EASY`
- `mechanics=EASY`
- `feedback=HARD`

Instruction text:

The instruction text is identical to `baseline`.

What changes:

- Failed `PROPOSE` attempts no longer include a counter-example in the environment response.

## Per-turn user prompt

The user prompt sent each turn is the same across all configurations:

```text
Current Game State History:
{history_log}

Choose your next action (STRATA or PROPOSE).
```
