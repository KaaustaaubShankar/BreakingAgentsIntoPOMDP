# BreakingAgentsIntoPOMDP

## Environments

`env3` contains the ARC-AGI `ls20` environment with the four-axis ablation setup.

`env4` adds a new ARC-AGI environment harness with the same ablation matrix and
defaults to task `bp35`. Unlike `env3`, it is task-agnostic at the observation
layer: Easy World exposes a public JSON summary derived from frame/action data,
while Hard World uses ASCII or vision from the raw frame. This is safer for new
ARC tasks whose private game internals differ from `ls20`.
