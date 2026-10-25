# KA59-Simple final results

Three folders, one per model, in the `bp35/results/` layout: all efforts live in
the same folder and are distinguished by the filename and each trial's
`reasoning_effort` field.

```
run_<fgmw flags>_<timestamp>_<config>_<effort>_t<N>.json
```

`t<N>` restarts at 1 within each (model, effort, config) cell.

| Folder | Trials | Cells |
|---|---:|---|
| `gpt-5.6-luna/` | 150 | none @ 20/cell, medium @ 10/cell |
| `gpt-5.2/` | 150 | none @ 20/cell, medium @ 10/cell |
| `deepseek-v4-pro/` | 100 | none @ 20/cell |

One `ablation_summary_<model>_<effort>.json` per matrix, in the schema of
`bp35/results/ablation_summary_*.json` field for field, with the KA59-Simple
metric set (`avg_wall_transfers`, `avg_object_pushes`,
`avg_max_goals_occupied`) in place of bp35's `avg_gravity_flips` and
`avg_undos`. `provider` reads `pooled` where a cell draws on more than one
transport. Each summary is derived only from the trials in this folder, and its
`run_files` cite them by repo-relative path.

## Cell outcomes (wins/N)

| Model | Effort | baseline | world_hard | mechanics_hard | mechanics_hard_format_only | feedback_hard |
|---|---|---:|---:|---:|---:|---:|
| gpt-5.6-luna | none | 17/20 | 0/20 | 16/20 | 0/20 | 16/20 |
| gpt-5.6-luna | medium | 4/10 | 0/10 | 1/10 | 2/10 | 5/10 |
| gpt-5.2 | none | 20/20 | 0/20 | 0/20 | 0/20 | 15/20 |
| gpt-5.2 | medium | 10/10 | 0/10 | 0/10 | 0/10 | 7/10 |
| deepseek-v4-pro | none | 12/20 | 0/20 | 4/20 | 4/20 | 15/20 |

## How trials were selected

Gathered from every tree that holds KA59-Simple trials
(`camera_ready/results/`, `ka59_game/results_clean/`, `ka59_game/results/`,
`ka59_game/results/camera_ready/`), then per cell:

1. de-duplicate by gameplay history, so the same trial reached through several
   trees is counted once;
2. drop infrastructure failures (`camera_ready_status` outside win/loss, and
   dead-key `credit_balance_exhausted` 429s) - never scored as losses;
3. prefer trials carrying a camera-ready `protocol_identity`, filling any
   shortfall with legacy trials, ties broken by timestamp;
4. take the cell's target N.

Surplus trials beyond target are left in their original trees and not copied
here: 29 for gpt-5.2 none, 30 for gpt-5.2 medium, 20 for deepseek none.

Where a cell had more candidates than its target, the identity-vs-legacy choice
changes nothing except in two gpt-5.2 medium cells, and there the
identity-bearing set is the camera-ready run: baseline 10/10 (identity) vs 4/5
(legacy), feedback_hard 7/10 vs 5/5. The ambiguous cells are neutral -
deepseek mechanics_hard_format_only is 4/20 either way, gpt-5.2 none
mechanics_hard_format_only is 0 either way.

DeepSeek cell outcomes here match `camera_ready/KA59_CAMERA_READY_TRUTH.md`
exactly (12/8, 0/20, 4/16, 4/16, 15/5).

## Copies, not moves

`camera_ready/results/<protocol_id>/` is named by identity hash and the runner
resolves its output directory by that hash, so those directories stay where
they are or `--resume` breaks. Everything here is a copy.

Not included: `deepseek-v4-pro @ medium` (104 trials exist, but
`KA59_CAMERA_READY_TRUTH.md` records every cell as fully infrastructure-excluded
or duplicate), and one `gpt-5.2` trial whose `reasoning_effort` is null.
