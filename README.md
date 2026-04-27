# BreakingAgentsIntoPOMDP

## Environments

`env3` contains the ARC-AGI `ls20` environment with the four-axis ablation setup.

`env4` adds a new ARC-AGI environment harness with the same ablation matrix and
defaults to task `tr87`. Unlike `env3`, it is task-agnostic at the observation
layer: Easy World exposes a public JSON summary derived from frame/action data,
while Hard World uses ASCII or vision from the raw frame. This is safer for new
ARC tasks whose private game internals differ from `ls20`.

### Running env4

From `env4/`:

```bash
python experiment.py --task tr87
python experiment.py --task tr87 --world HARD
python ablation.py --task tr87 --trials 5
```

Current assumption: `tr87` is run through ARC's public API and may expose a
different action set than `ls20`, so `env4` handles actions dynamically instead
of hardcoding directional movement only.
