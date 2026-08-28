# KA59-Simple Accepted-Data Protocol Lock

This lock describes the protocol that produced the accepted-paper raw data. It
separates direct raw observations from metadata recovered through Git history.
Where the evidence conflicts or is incomplete, the field is marked **AMBIGUOUS**.

## Environment

- Environment ID observed in raw files: `ka59simple`.
- Recovered source path: `environment_files/ka59simple/20260430`.
- Environment source SHA-256: `6696b048da8dc6c4dabe7ab3839c8051e1b37270b65c6b7282974e5242af080e`.
- Canonical KA59 dependency: `environment_files/ka59/38d34dbb`.
- Raw-trial environment Git SHA: **AMBIGUOUS**. It was not logged. The dated
  environment directory and stable Git history support the recovered path, but
  that is not a direct per-trial observation.

## Accepted condition matrix

| Condition | World | Goal | Mechanics | Feedback |
|---|---|---|---|---|
| `baseline` | EASY | EASY | EASY | EASY |
| `world_hard` | HARD | EASY | EASY | EASY |
| `mechanics_hard` | EASY | EASY | HARD | EASY |
| `mechanics_hard_format_only` / `norules` | EASY | EASY | `HARD_FORMAT_ONLY` tag | EASY |
| `feedback_hard` | EASY | EASY | EASY | HARD |

`goal_hard` exists in historical scaffolding but is not part of the accepted
matrix and is not exposed by the camera-ready runner.

## Turn and level-attempt policy

- One KA59-Simple level per trial.
- 64 turns per attempt.
- Two attempts at the same level, for a maximum of 128 turns per trial.
- On attempt failure, the environment is reset to the same level.
- The failed attempt remains in the model's action-history context.
- A win on either attempt scores the trial as a win.

The 64-turn attempt budget and two-attempt history are directly visible in
valid raw histories (`attempt`, `attempt_turn`, and `resources.step_budget`).
They are recovered from the historical runner only for all-error files that
never produced an action event.

## Retry and failure semantics

Historical per-turn behavior:

1. Request a JSON response.
2. If JSON parsing fails, issue one immediate format-retry request.
3. If the request or retry still fails, the historical runner logged an error,
   consumed that turn, and continued.

Historical trial-level `MAX_TRIAL_RETRIES=2` only caught exceptions escaping
`run_agent`; because `run_agent` swallowed most API/parse errors, many failed
requests were incorrectly retained as losses. This is the source of the false
clean-N denominators.

Camera-ready behavior deliberately changes only error accounting: a final API,
authentication, timeout, provider, parse, or environment error aborts the trial,
saves it as excluded evidence, and does not increment valid N. The orchestrator
continues until target valid N or stops after three infrastructure failures.

## Action and prompt semantics

- EASY mechanics supplies pixel-coordinate conventions, movement actions,
  pushing, wall uncertainty, CLICK selection, structured state, and the JSON
  response protocol.
- HARD mechanics supplies only a minimal JSON action schema.
- WORLD HARD removes non-player object positions and exposes the selected
  player, blocked-direction flags, level/resources, object counts, and game
  state.
- FEEDBACK EASY returns a natural-language state delta; FEEDBACK HARD returns
  the constant `Ok.`.
- Valid actions are MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN, and CLICK with a
  target position when the action protocol is present.

### Format-only/norules conflict

The accepted raw files carry `mechanics: HARD_FORMAT_ONLY`, but the historical
`build_system_prompt` had no `HARD_FORMAT_ONLY` branch before commit `412ba5f`.
It therefore fell through to the ordinary HARD prompt. The accepted raw
`mechanics_hard` and `mechanics_hard_format_only` conditions had the same system
prompt.

The paper describes a different intended control that retains the complete
action list and CLICK protocol. That branch was added later. Therefore:

- exact accepted-raw reproduction = fall through to the HARD prompt;
- paper-intended format-only control = later, incompatible protocol;
- pooling those two protocols is forbidden;
- whether to remove/relabel the accepted control or run the intended control is
  **AMBIGUOUS** and requires an author decision.

## Model, provider, and reasoning semantics

### GPT-5.2

- Model slug observed: `openai/gpt-5.2`.
- Provider observed: `openrouter`.
- Reasoning efforts: `none` and `medium`.
- Main-cell effort metadata in restored per-trial files is historically
  recovered (`_effort_recovered: true`), not directly logged.
- No-rules effort is recovered from distinct run dates.
- OpenRouter upstream/provider pin: **AMBIGUOUS**. It was not logged and the
  historical client did not pin an upstream provider.

### DeepSeek-V4-Pro

- Model slug observed: `deepseek-v4-pro`.
- Per-trial provider observed: `deepseek`.
- Reasoning efforts: `none` and `medium`.
- Reasoning effort is directly logged in these files.
- The medium aggregate says `provider: pooled`, while its selected per-trial
  records say `deepseek`; exact endpoint composition is **AMBIGUOUS** and must
  not be pooled with a new provider identity without explicit provenance.

## Frozen execution identity

The camera-ready runner hashes environment revision, prompt lineage, turn
budget, attempts, context retention, format-only behavior, provider, model,
reasoning effort, and config into `protocol_id`. Existing result files with a
different identity cause a hard refusal rather than silent pooling.
