# Camera-ready trials, organised by model and effort

Browsable copies of the trials under `camera_ready/results/<protocol_id>/`, which
is named by identity hash and hard to read. The runner still resolves its own
directories by hash -- these are copies, so deleting or editing them is safe.

Kept under `camera_ready/` rather than flat beside `gpt-5.2/` and
`deepseek-v4-pro/` because those hold the accepted historical trials that
`ablation_summary_*.json` references by path, and `deepseek-v4-pro-medium`
would otherwise collide.

| Folder | Trials | Valid | W/L | Excluded |
|---|---:|---:|---:|---:|
| `deepseek-v4-pro-medium` | 3 | 1 | 1/0 | 2 |
| `deepseek-v4-pro-none` | 20 | 20 | 4/16 | 0 |
| `gpt-5.2-none` | 70 | 65 | 27/38 | 5 |

## Per-cell breakdown

**deepseek-v4-pro-medium**

- `baseline`: 1/1, 2 excluded

**deepseek-v4-pro-none**

- `mechanics_hard_format_only`: 4/20

**gpt-5.2-none**

- `baseline`: 15/15
- `feedback_hard`: 12/15
- `mechanics_hard_format_only`: 0/20
- `world_hard`: 0/15, 5 excluded

