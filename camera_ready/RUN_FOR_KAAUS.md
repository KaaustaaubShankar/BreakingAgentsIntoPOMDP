# Run KA59-Simple for Camera-Ready

## What changed

Historical summaries counted provider/parse failures as losses, duplicated a
pooled GPT N=10 row under both efforts, and described a no-rules prompt branch
that did not yet exist. The manifest now separates valid wins/losses, errors,
duplicates, and incompatible protocols. Read `KA59_CAMERA_READY_TRUTH.md` before
spending compute.

## Setup and zero-call checks

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m scripts.audit_ka59_camera_ready --check
.venv/bin/python -m scripts.run_ka59_camera_ready --smoke
```

`--smoke` initializes KA59-Simple locally and makes zero model calls.

## Current GPT-5.2 strict valid N and N=20 deficit

| Effort | baseline | world | mechanics | no-rules tag | feedback |
|---|---:|---:|---:|---:|---:|
| none valid N | 5 | 2 | 2 | 0 | 5 |
| none remaining | 15 | 18 | 18 | 20 | 15 |
| medium valid N | 0 | 3 | 3 | 0 | 0 |
| medium remaining | 20 | 17 | 17 | 20 | 20 |

These strict denominators exclude any trial with a provider/API/parse/environment
failure. They intentionally differ from the historical N=5/N=10 summary table.

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
paper-intended format-only control without an author decision.

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
