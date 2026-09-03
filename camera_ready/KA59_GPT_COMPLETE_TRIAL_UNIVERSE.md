# KA59-Simple GPT-5.2 Complete Historical Trial Universe

## Answer first

PR #20's restored per-trial directory was an accepted-evidence subset, not the complete historical universe. The exhaustive Git-object audit recovered 21 additional protocol-compatible June 1 no-rules-none trials: 19 are infrastructure-clean losses and two have insufficient persisted parse provenance. It also corrected the May 13 world/mechanics effort mapping from the effort-major aggregate order. Main `none` world/mechanics are therefore 0/5, the corresponding `medium` cells have no clean trials, and historical no-rules-none pools to 0/19.

The May 13 N=1 baseline does **not** count: contemporaneous planning identifies it as reduced-turn preflight/smoke before the N=5 sweep. No outcome was used to make that decision.

## Outcome-blind eligibility rule

Protocol identity comes from `camera_ready/KA59_PROTOCOL_LOCK.md`: KA59-Simple dated environment, OpenRouter `openai/gpt-5.2`, effort-specific cell, 64 turns per attempt, two same-level attempts, retained failed-attempt context, historical per-turn retry semantics, and the contemporaneous prompt/config implementation. Classification is assigned before reading win/loss. Infrastructure errors are then removed; independent compatible batches are pooled.

The historical no-rules tag remains the pre-`412ba5f` fallthrough to ordinary mechanics-hard. These trials estimate that historical implementation only; they do not estimate the action-list-retaining control described in the paper.

## Batch inventory

A row is one recovered invocation/batch. Mixed batches report the exact trial-level category counts; every logical trial has exactly one category in the JSON inventory.

| Batch | Protocol identity | Effort | Config(s) | Raw N | Valid N | Wins | Eligibility | Exact exclusion reason |
|---|---|---|---|---:|---:|---:|---|---|
| `20260501T042613` | `PRE_LEVEL_ATTEMPTS_32T_UNKNOWN_EFFORT` | unknown | baseline | 2 | 0 | 0 | EXCLUDED_DIFFERENT_PROTOCOL=2 | pre-May-13 runner used one 32-turn attempt and lacks accepted effort provenance |
| `20260501T070936` | `PRE_LEVEL_ATTEMPTS_32T_UNKNOWN_EFFORT` | unknown | baseline, goal_hard, mechanics_hard, mechanics_ooda, world_hard | 25 | 0 | 0 | EXCLUDED_DIFFERENT_PROTOCOL=15, SEPARATE_EXPERIMENT=10 | config is outside the accepted paper matrix; pre-May-13 runner used one 32-turn attempt and lacks accepted effort provenance |
| `20260501T070938` | `PRE_LEVEL_ATTEMPTS_32T_UNKNOWN_EFFORT` | unknown | baseline, goal_hard, mechanics_hard, mechanics_ooda, world_hard | 25 | 0 | 0 | EXCLUDED_DIFFERENT_PROTOCOL=15, SEPARATE_EXPERIMENT=10 | config is outside the accepted paper matrix; pre-May-13 runner used one 32-turn attempt and lacks accepted effort provenance |
| `20260501T144336` | `PRE_LEVEL_ATTEMPTS_32T_UNKNOWN_EFFORT` | unknown | baseline, feedback_hard, goal_hard, mechanics_hard, world_hard | 25 | 0 | 0 | EXCLUDED_DIFFERENT_PROTOCOL=20, SEPARATE_EXPERIMENT=5 | config is outside the accepted paper matrix; pre-May-13 runner used one 32-turn attempt and lacks accepted effort provenance |
| `20260501T145112` | `PRE_LEVEL_ATTEMPTS_32T_UNKNOWN_EFFORT` | unknown | baseline, feedback_hard, goal_hard, mechanics_hard, world_hard | 25 | 0 | 0 | EXCLUDED_DIFFERENT_PROTOCOL=20, SEPARATE_EXPERIMENT=5 | config is outside the accepted paper matrix; pre-May-13 runner used one 32-turn attempt and lacks accepted effort provenance |
| `20260501T145553` | `PRE_LEVEL_ATTEMPTS_32T_UNKNOWN_EFFORT` | unknown | feedback_hard, mechanics_ooda_f | 10 | 0 | 0 | EXCLUDED_DIFFERENT_PROTOCOL=5, SEPARATE_EXPERIMENT=5 | config is outside the accepted paper matrix; pre-May-13 runner used one 32-turn attempt and lacks accepted effort provenance |
| `20260505T003219` | `PRE_LEVEL_ATTEMPTS_32T_UNKNOWN_EFFORT` | unknown | goal_hard | 5 | 0 | 0 | SEPARATE_EXPERIMENT=5 | config is outside the accepted paper matrix |
| `20260505T003507` | `PRE_LEVEL_ATTEMPTS_32T_UNKNOWN_EFFORT` | unknown | goal_hard | 5 | 0 | 0 | SEPARATE_EXPERIMENT=5 | config is outside the accepted paper matrix |
| `20260513T043402_656079` | `MAY13_EXPLICIT_PREFLIGHT_REDUCED_TURN_BUDGET` | none | baseline | 1 | 0 | 0 | EXCLUDED_EXPLICIT_SMOKE_OR_DEBUG=1 | explicit one-cell preflight; 8 turns per attempt |
| `20260513T043715_RAW_ONLY` | `MAY13_EXPLICIT_PREFLIGHT_REDUCED_TURN_BUDGET` | medium, none | baseline, mechanics_hard | 4 | 0 | 0 | EXCLUDED_EXPLICIT_SMOKE_OR_DEBUG=4 | contemporaneous plan explicitly calls this the end-to-end smoke slice; 16 turns per attempt |
| `20260513T055440_493226` | `ACCEPTED_20260513_TWO_ATTEMPT_64T` | none, medium | baseline, world_hard, goal_hard, mechanics_hard, feedback_hard, baseline, world_hard, goal_hard, mechanics_hard, feedback_hard | 50 | 20 | 8 | EXCLUDED_INFRASTRUCTURE=20, INCLUDED_SAME_PROTOCOL=20, SEPARATE_EXPERIMENT=10 | goal_hard is outside the accepted paper matrix; provider empty-content failures occurred during the trial |
| `20260601T020500_243047` | `ACCEPTED_HISTORICAL_NORULES_FALLTHROUGH_NONE` | none | mechanics_hard_format_only | 5 | 0 | 0 | EXCLUDED_INFRASTRUCTURE=5 | HTTP 402 insufficient-credit failures consumed the entire trial |
| `20260601T021239_568866` | `ACCEPTED_HISTORICAL_NORULES_FALLTHROUGH_NONE` | none | mechanics_hard_format_only | 5 | 0 | 0 | EXCLUDED_INFRASTRUCTURE=5 | HTTP 402 insufficient-credit failures consumed the entire trial |
| `20260601T021314_763237` | `ACCEPTED_HISTORICAL_NORULES_FALLTHROUGH_NONE` | none | mechanics_hard_format_only | 5 | 5 | 0 | INCLUDED_SAME_PROTOCOL=5 | none |
| `20260601T021411_105948` | `ACCEPTED_HISTORICAL_NORULES_FALLTHROUGH_NONE` | none | mechanics_hard_format_only | 5 | 4 | 0 | EXCLUDED_PROVENANCE_INSUFFICIENT=1, INCLUDED_SAME_PROTOCOL=4 | persisted NoneType parse error has no raw response; model-versus-harness cause cannot be recovered |
| `20260601T021458_230073` | `ACCEPTED_HISTORICAL_NORULES_FALLTHROUGH_NONE` | none | mechanics_hard_format_only | 1 | 1 | 0 | INCLUDED_SAME_PROTOCOL=1 | none |
| `20260601T021821_835569` | `ACCEPTED_HISTORICAL_NORULES_FALLTHROUGH_NONE` | none | mechanics_hard_format_only | 5 | 4 | 0 | EXCLUDED_PROVENANCE_INSUFFICIENT=1, INCLUDED_SAME_PROTOCOL=4 | persisted NoneType parse error has no raw response; model-versus-harness cause cannot be recovered |
| `20260601T021844_403409` | `ACCEPTED_HISTORICAL_NORULES_FALLTHROUGH_NONE` | none | mechanics_hard_format_only | 5 | 5 | 0 | INCLUDED_SAME_PROTOCOL=5 | none |
| `20260609T195446_323788` | `PROTOCOL_LABEL_MATCHES_RAW_PROVENANCE_MISSING` | none | baseline | 1 | 0 | 0 | EXCLUDED_PROVENANCE_INSUFFICIENT=1 | aggregate-only all-zero record; raw trial and error string are absent, so protocol completion and failure cause cannot be proven |
| `20260620T213830_707671_p35675` | `ACCEPTED_HISTORICAL_NORULES_FALLTHROUGH_MEDIUM` | medium | mechanics_hard_format_only | 2 | 0 | 0 | EXCLUDED_INFRASTRUCTURE=2 | OpenRouter empty-content failures occurred during the trial |
| `20260620T213830_707671_p35677` | `ACCEPTED_HISTORICAL_NORULES_FALLTHROUGH_MEDIUM` | medium | mechanics_hard_format_only | 2 | 0 | 0 | EXCLUDED_INFRASTRUCTURE=2 | OpenRouter empty-content failures occurred during the trial |
| `20260620T213830_707672_p35674` | `ACCEPTED_HISTORICAL_NORULES_FALLTHROUGH_MEDIUM` | medium | mechanics_hard_format_only | 2 | 0 | 0 | EXCLUDED_INFRASTRUCTURE=2 | OpenRouter empty-content failures occurred during the trial |
| `20260620T213830_707676_p35673` | `ACCEPTED_HISTORICAL_NORULES_FALLTHROUGH_MEDIUM` | medium | mechanics_hard_format_only | 2 | 0 | 0 | EXCLUDED_INFRASTRUCTURE=2 | OpenRouter empty-content failures occurred during the trial |
| `20260620T213830_707882_p35676` | `ACCEPTED_HISTORICAL_NORULES_FALLTHROUGH_MEDIUM` | medium | mechanics_hard_format_only | 2 | 0 | 0 | EXCLUDED_INFRASTRUCTURE=2 | OpenRouter empty-content failures occurred during the trial |

## Final eligible cells

`Historical candidate N` includes known-effort candidates, including smoke and later excluded records; the older May 1 default/unknown-effort runs remain visible in the batch table but are not silently assigned to `none` or `medium`. `Same-protocol candidate N` is the locked-protocol universe before failure/provenance exclusions.

| Effort | Config | Historical candidate N | Same-protocol candidate N | Infrastructure-clean included N | Wins | Losses | Excluded by reason | Final pooled estimate |
|---|---|---:|---:|---:|---:|---:|---|---|
| none | baseline | 8 | 5 | 5 | 5 | 0 | EXCLUDED_EXPLICIT_SMOKE_OR_DEBUG=2, EXCLUDED_PROVENANCE_INSUFFICIENT=1 | 5/5 (100%) |
| none | world_hard | 5 | 5 | 5 | 0 | 5 | none | 0/5 (0%) |
| none | mechanics_hard | 6 | 5 | 5 | 0 | 5 | EXCLUDED_EXPLICIT_SMOKE_OR_DEBUG=1 | 0/5 (0%) |
| none | mechanics_hard_format_only | 31 | 31 | 19 | 0 | 19 | EXCLUDED_INFRASTRUCTURE=10, EXCLUDED_PROVENANCE_INSUFFICIENT=2 | 0/19 (0%) |
| none | feedback_hard | 5 | 5 | 5 | 3 | 2 | none | 3/5 (60%) |
| medium | baseline | 6 | 5 | 0 | 0 | 0 | EXCLUDED_EXPLICIT_SMOKE_OR_DEBUG=1, EXCLUDED_INFRASTRUCTURE=5 | NO ELIGIBLE TRIALS |
| medium | world_hard | 5 | 5 | 0 | 0 | 0 | EXCLUDED_INFRASTRUCTURE=5 | NO ELIGIBLE TRIALS |
| medium | mechanics_hard | 6 | 5 | 0 | 0 | 0 | EXCLUDED_EXPLICIT_SMOKE_OR_DEBUG=1, EXCLUDED_INFRASTRUCTURE=5 | NO ELIGIBLE TRIALS |
| medium | mechanics_hard_format_only | 10 | 10 | 0 | 0 | 0 | EXCLUDED_INFRASTRUCTURE=10 | NO ELIGIBLE TRIALS |
| medium | feedback_hard | 5 | 5 | 0 | 0 | 0 | EXCLUDED_INFRASTRUCTURE=5 | NO ELIGIBLE TRIALS |

## Per-batch heterogeneity for pooled cells

| Effort/config | Included batch | Included result |
|---|---|---:|
| none/baseline | `20260513T055440_493226` | 5/5 |
| none/feedback_hard | `20260513T055440_493226` | 3/5 |
| none/mechanics_hard | `20260513T055440_493226` | 0/5 |
| none/mechanics_hard_format_only | `20260601T021314_763237` | 0/5 |
| none/mechanics_hard_format_only | `20260601T021411_105948` | 0/4 |
| none/mechanics_hard_format_only | `20260601T021458_230073` | 0/1 |
| none/mechanics_hard_format_only | `20260601T021821_835569` | 0/4 |
| none/mechanics_hard_format_only | `20260601T021844_403409` | 0/5 |
| none/world_hard | `20260513T055440_493226` | 0/5 |

All five included no-rules-none batches are 0% individually (0/5, 0/4, 0/1, 0/4, 0/5), so the pooled 0/19 does not conceal directional batch heterogeneity. The two 4/5 batches each exclude one indeterminate parse-provenance trial.
The June 1 N=1 batch (`20260601T021458_230073`) is included: it has the same observed 64-turn/two-attempt identity as the surrounding parallel batches, and no preserved plan, name, or log marks it as smoke/debug. N=1 alone is not an exclusion criterion, and its loss was not used in the decision.

## Provenance correction to PR #20

The May 13 aggregate preserves effort-major runner order: all `none` configurations precede all `medium` configurations. PR #20's filenames instead interleaved effort labels within world-hard and mechanics-hard. Using aggregate order plus raw timestamps restores the correct membership: `none` world-hard and mechanics-hard each have five clean trials; `medium` world-hard and mechanics-hard each have five provider-empty trials and no clean denominator. The raw outcomes did not determine this remapping.

## Exhaustiveness proof and limits

The audit scanned all objects reachable from all local/remote refs under the three result roots, then checked all local worktrees. It found 346 GPT-5.2 KA59-Simple JSON artifacts with 346 unique blobs and no untracked GPT result artifact. Every logical trial represented by an aggregate/sidecar or raw-only smoke file is accounted for here; restored PR #20 copies are linked as duplicates rather than counted again. The June 9 N=1 sidecar has no raw trial and is therefore provenance-insufficient, not converted into a model loss.

This closes historical reconciliation without compute. Additional runs are a separate author decision: they are necessary only if the camera-ready estimand requires larger per-effort N, a pinned fresh matrix, or the paper-intended no-rules control.
