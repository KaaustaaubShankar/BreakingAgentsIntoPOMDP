# KA59 branch status

This note is the fastest way to understand what the `ka59` branch adds and how it fits the broader benchmark work.

## What this branch is for

KA59 is our **smallest clean mechanics-discovery slice**.

It is **not** meant to replace Kaus's `env4` / BP35 work.
It complements that lane by giving us a controlled environment where hidden transition rules can be tested, compared across simple agents, and reported cleanly.

## What is implemented

- faithful simulator in `ka59_ref/engine.py`
- blind agent-facing environment in `ka59_ref/env.py`
- episode runner in `ka59_ref/runner.py`
- three baseline agents in `ka59_ref/discovery.py`
  - `NaiveRightAgent`
  - `RotateOnBlockAgent`
  - `MinimalHypothesisAgent`
- canonical scenarios in `ka59_ref/scenarios.py`
- benchmark evaluator in `ka59_ref/benchmark.py`
- runnable report in `scripts/run_ka59_report.py`

## What KA59 currently claims

- **Mechanics:** strong
- **World:** partial
- **Goal:** none
- **Feedback:** none

This is intentional. KA59 is currently a **Mechanics-heavy benchmark**, not a full World / Goal / Mechanics / Feedback matrix.

## Current result

Run:

```bash
python3 scripts/run_ka59_report.py
```

Key signal:

- in `tw_push`, only `MinimalHypothesisAgent` discovers the hidden wall asymmetry
- in `tw_push_world_blind`, that discovery disappears

That gives us a compact paper-facing result:
- discovery baseline succeeds where non-adaptive baselines do not
- reduced observability weakens discovery

## Representative baseline snapshot

`tw_push`

- `NaiveRight`: `moved=1 blocked=9 p_walls=0`
- `RotateOnBlock`: `moved=9 blocked=1 p_walls=0`
- `MinimalHypothesis`: `moved=7 blocked=3 p_walls=1`

`tw_push_world_blind`

- `NaiveRight`: `moved=1 blocked=9 p_walls=0`
- `RotateOnBlock`: `moved=9 blocked=1 p_walls=0`
- `MinimalHypothesis`: `moved=9 blocked=1 p_walls=0`

## How this complements the rest of the project

- **BP35 / env4:** game-facing implementation and harder planning task, led by Kaus
- **LS20:** another environment in the broader benchmark set
- **KA59:** smallest clean mechanics probe with readable discovery behavior

## What this branch is not trying to do

- not a competing `env4`
- not a replacement for BP35's JSON-heavy interface
- not a claim that KA59 already covers the full paper

## Verification

```bash
python3 -m pytest tests/ --tb=no -q
python3 scripts/run_ka59_report.py
```
