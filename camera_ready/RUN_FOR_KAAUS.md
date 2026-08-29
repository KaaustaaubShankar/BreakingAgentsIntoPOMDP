# KA59-Simple Camera-Ready Execution Appendix

Do not start paid runs from this file alone. Read `camera_ready/README.md` and
record the author decisions on reporting unit, target N, no-rules protocol, and
model/provider identity first.

## What changed

Historical summaries counted provider/parse failures as losses, displayed GPT
N=5 evidence through N=10 rows, and described a no-rules prompt branch that did
not yet exist. The manifest now separates valid wins/losses, errors, duplicates,
and incompatible protocols.

## Setup and zero-call checks

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m scripts.audit_ka59_camera_ready --check
.venv/bin/python -m scripts.run_ka59_camera_ready --smoke
```

`--smoke` initializes KA59-Simple locally and makes zero model calls.

## Provisional GPT-5.2 strict valid N and N=20-per-effort deficit

| Effort | baseline | world | mechanics | no-rules tag | feedback |
|---|---:|---:|---:|---:|---:|
| none valid N | 5 | 2 | 2 | 0 | 5 |
| none remaining | 15 | 18 | 18 | 20 | 15 |
| medium valid N | 0 | 3 | 3 | 0 | 0 |
| medium remaining | 20 | 17 | 17 | 20 | 20 |

These strict denominators exclude any trial with a provider/API/parse/environment
failure. They intentionally differ from the historical N=5/N=10 summary table.
N=20 here is an operational scenario, not a settled scientific requirement.
The reviewer/PI request to increase N still requires authors to choose whether
the target is per effort or pooled and which cells are primary.

## Plan GPT-5.2 (zero calls)

```bash
.venv/bin/python -m scripts.run_ka59_camera_ready \
  --provider openrouter \
  --model openai/gpt-5.2 \
  --reasoning-effort none \
  --target-n 20 \
  --plan

.venv/bin/python -m scripts.run_ka59_camera_ready \
  --provider openrouter \
  --model openai/gpt-5.2 \
  --reasoning-effort medium \
  --target-n 20 \
  --plan
```

## Execute and resume

Remove `--plan` to execute. Add `--resume` after any interrupted run:

```bash
.venv/bin/python -m scripts.run_ka59_camera_ready \
  --provider openrouter \
  --model openai/gpt-5.2 \
  --reasoning-effort none \
  --target-n 20 \
  --resume
```

Omit `--config` for all five accepted tags, or repeat it for selected cells,
for example `--config baseline --config feedback_hard`. `--target-n 20` runs
only the strict valid deficit, never 20 additional trials. Errors are saved but
excluded; three repeated infrastructure failures stop the run explicitly.

The no-rules tag exactly reproduces accepted raw behavior, which fell through
to the ordinary mechanics-hard prompt. Do not run or interpret it as the later
paper-intended format-only control without an author decision. If authors choose
the intended control, it needs a distinct protocol identity and runner change;
historical fallthrough trials cannot be pooled into it.

The runner currently plans against `strict_error_free` aliases. If the authors
choose `infrastructure_clean_scored` as the primary denominator, update and
review the planning policy before paid execution rather than assuming the
candidate 4/20 DeepSeek mechanics cell is already counted by this runner.

## Validate returned results

Re-run the same command with `--plan --resume`. `current_valid_n` must rise only
for error-free, identity-matching results and `deficit` must fall. Then run:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

## “5.6 Luna”

**BLOCKED ON EXACT MODEL/PROVIDER IDENTIFIER.** We still need the provider, API
endpoint/provider type, exact model slug, reasoning-effort semantics, and key
environment-variable name. Once supplied and supported by the provider layer,
use the same command with the exact `--provider` and `--model`. Do not put the
key in Git or on the command line.
