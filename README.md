# BreakingAgentsIntoPOMDP

This branch currently contains two main workstreams:

- `zendo/` — earlier Zendo benchmark work
- `ka59_ref/` — KA59 mechanics-discovery benchmark slice

## If you're looking at the KA59 work

Start here:

- `KA59_STATUS.md` — concise status note, current claim, and how it fits the project
- `ka59_ref/README.md` — KA59 environment / benchmark details
- `scripts/run_ka59_report.py` — runnable comparison report

## Quick run

```bash
python3 -m pytest tests/ --tb=no -q
python3 scripts/run_ka59_report.py
```
