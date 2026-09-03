# KA59-Simple Camera-Ready Owner Brief

## The state in one minute

The audit and rerun machinery in this PR are useful and should remain. The
accepted KA59-Simple reporting, however, is not camera-ready as written.

- The accepted Figure 3 does not have independent GPT-5.2 N=10 samples for
  `none` and `medium`. The source run is N=5 per effort, and the final figure
  mixes a pooled row with rates scaled from N=5 to an N=10 display.
- PR #20 restored a selected GPT per-trial subset, not the complete historical
  universe. Exhaustive recovery adds 21 compatible no-rules-none trials and
  corrects the May 13 effort mapping for world/mechanics.
- Provider and account failures were historically allowed to consume turns and
  then enter summaries as losses. Those records measure degraded infrastructure,
  not clean model behavior.
- The accepted `MECHANICS HARD NORULES` runs used the same prompt as
  `MECHANICS HARD`. They did not implement the action-protocol-retaining control
  described in the paper.
- DeepSeek `none` supplies a clean N=20 matrix. The accepted DeepSeek `medium`
  files do not supply a clean performance denominator under either audit policy.
- The existing data still support a narrower directional story, especially for
  DeepSeek `none`. They do not support every model-by-effort percentage or the
  claimed no-rules causal comparison.

The remaining blockers are mostly author choices: what the paper will estimate,
which control it will claim, which models/efforts must be rerun, and whether the
single recoverable model-protocol failure is scored in the main denominator.

## How this happened

1. On 2026-05-13, GPT-5.2 ran five trials per effort for each main KA59-Simple
   condition. Historical reporting combined or rescaled those cells to show N=10.
2. The final accepted figure retained pooled `none` baseline/feedback values
   (9/10 and 8/10), but changed the `medium` percentages to the N=5 medium rates
   while still printing N=10 (8/10 from 4/5; 10/10 from 5/5). There was no second
   N=5 batch for either effort.
3. DeepSeek `none` produced a clean N=20 matrix. Attempts to obtain DeepSeek
   `medium` N=20 encountered provider-empty responses and HTTP 402 insufficient
   balance. Later summaries selected nominal N=20 files and treated disrupted
   trials as losses.
4. The historical runner swallowed many provider/parse errors, consumed the
   affected turn, and continued. Aggregate completion therefore did not imply a
   clean model trial.
5. The `HARD_FORMAT_ONLY` prompt branch was implemented only after the accepted
   no-rules runs. The tag in the raw files and the control described in the paper
   refer to different protocols.
6. PR #20 restored selected GPT per-trial files, but it was not an exhaustive
   universe definition. Its reconstructed filenames also interleaved `none`
   and `medium` for May 13 world/mechanics despite the aggregate's effort-major
   execution order.
7. The complete-history audit accounts for 219 logical trials plus 60 restored
   duplicate artifacts. It finds five additional compatible June 1 no-rules
   batches: 19 clean losses and two provenance-insufficient parse records.

## What the audit changes conceptually

The audit separates three layers that historical reporting blurred:

| Layer | Question | Rule |
|---|---|---|
| Historical display | What appeared in the paper/Sheet? | Preserve it as provenance, not evidence. |
| Complete historical universe | What GPT trials or aggregates can be recovered anywhere in reachable history/local preservation? | Inventory every candidate outcome-blind; do not treat PR #20 selection as the universe. |
| Final eligible evidence | Which candidates match the locked protocol and survive failure/provenance exclusions? | Pool every independent compatible trial; exclude infrastructure, indeterminate, incompatible, smoke, and duplicate records. |
| Camera-ready experiment | What should be run and reported now? | Use one frozen protocol identity and count only valid completed trials toward the chosen target N. |

Infrastructure failures include authentication, insufficient balance, timeout,
provider-empty response, environment failure, and harness failure. They are not
losses. A model-produced malformed response is different: it is model behavior
if the provider request succeeded. The audit exposes both a candidate main view
and a stricter sensitivity view rather than hiding that choice.

## What the existing evidence supports

The compact table below uses the candidate `infrastructure_clean_scored` view.
The strict view differs in only one place, noted below.

| Evidence slice | Clean result | Defensible use |
|---|---|---|
| DeepSeek `none` baseline | 12/20 | Supported baseline estimate. |
| DeepSeek `none` world-hard | 0/20 | Strong support that removing world state is disruptive for this slice. |
| DeepSeek `none` mechanics-hard | 4/20 | Supported; strict sensitivity is 4/19 because one malformed model response is excluded. |
| DeepSeek `none` historical no-rules tag | 4/20 | Outcome is reproducible, but only as another run of the ordinary mechanics-hard prompt, not the paper-described control. |
| DeepSeek `none` feedback-hard | 15/20 | Supported evidence that feedback removal is comparatively survivable for this slice. |
| GPT-5.2 `none` main cells | baseline 5/5; world 0/5; mechanics 0/5; feedback 3/5 | Complete-history, effort-corrected small-N evidence. |
| GPT-5.2 `none` historical no-rules tag | 0/19 across five batches | Supported only as the historical mechanics-hard fallthrough, not the paper-described control. |
| GPT-5.2 `medium` | no clean cell | All five candidates in each main cell and all ten no-rules candidates contain provider-empty failures. |
| DeepSeek `medium` | no clean cell | Supports an infrastructure-failure diagnosis, not the published model rates. |

The existing data support the narrower conclusion that world information is
important and that mechanics removal is highly disruptive in the audited slices.
They also support relative feedback robustness for DeepSeek `none` and GPT-5.2
`none`. They do not support the paper's universal wording across every model and
effort, the exact DeepSeek-medium percentages, or the no-rules causal claim.

## Why increasing N is not just a rerun command

The stated reviewer/PI request to increase N changes the camera-ready goal from
"explain the pilot" to "produce a clean estimand." It does not by itself decide:

- whether N is per effort or pooled across efforts;
- whether all cells or only a prespecified comparison need N=20;
- whether the historical prompt or the paper-intended no-rules control is the
  target protocol;
- whether old trials can be topped up or a fresh, provider-pinned matrix is
  required; or
- which model/provider identity is in scope.

The repository and PR discussion do not record the exact requested scope of the
reviewer/PI feedback. Until the authors make those choices, `--target-n 20` is a
safe operational capability, not an automatic scientific recommendation.

## Author decisions required

| Decision | Alternatives | Consequence |
|---|---|---|
| Reporting unit | Separate `none` and `medium`, or one explicitly pooled exploratory row | Separate rows require honest per-effort N. A pool must appear once and needs a scientific rationale; it cannot be duplicated under both labels. |
| Response to increased-N feedback | Correct and narrow the existing claims, or run a clean target-N matrix | The first is cheaper but remains small-N/exploratory. The second is the defensible route if the camera-ready paper will make model-by-effort or confirmatory claims. |
| No-rules control | Relabel/remove the historical condition, or run the later intended action-protocol-retaining control | A new intended-control run is a new protocol and cannot be pooled with historical fallthrough data. |
| Main denominator | `infrastructure_clean_scored`, with strict sensitivity; or strict-only | This changes only DeepSeek `none` mechanics-hard: 4/20 (20%) versus 4/19 (21%). |
| DeepSeek `medium` | Omit/mark unsupported, or rerun under a pinned identity | Existing accepted files cannot supply the published percentages. |
| Rerun identity and scope | Choose exact model slug, provider/upstream, reasoning semantics, cells, and target N | Engineering should not guess these. The same choice is still missing for the requested “5.6 Luna” run. |

## Recommended camera-ready end state

If the authors intend to answer the increased-N feedback with new evidence, the
clean end state is:

1. Freeze the reporting unit, primary comparison(s), target N, denominator, and
   exact model/provider identities in writing.
2. Decide the no-rules estimand. Keep historical reproduction and the intended
   control as distinct protocol IDs.
3. Run or top up only protocol-compatible cells. Infrastructure failures remain
   saved for audit but never increment N or losses.
4. Regenerate the manifest and produce the paper table/figure from audited
   counts rather than hand-entered percentages.
5. Rewrite the Methods, Figure 3 caption/cells, and Results claims to match the
   chosen evidence. Label residual small-N cells exploratory.
6. Keep the historical display and excluded trials in the evidence appendix so
   the correction remains reproducible.

If no new runs are authorized, the defensible end state is a narrower paper:
report the clean existing denominators, remove unsupported DeepSeek-medium exact
rates, remove or relabel the no-rules causal interpretation, and soften universal
model-by-effort statements.

## Drill-down map

- `PAPER_RECONCILIATION.md`: claim-by-claim manuscript correction checklist.
- `KA59_GPT_COMPLETE_TRIAL_UNIVERSE.md`: exhaustive GPT batch inventory,
  eligibility decisions, and final pooled cells.
- `ka59_gpt_complete_trial_universe.json`: trial-level source paths, protocol
  metadata, category, failures, outcomes, and duplicate relationships.
- `KA59_CAMERA_READY_TRUTH.md`: generated cell table and parse review.
- `ka59_camera_ready_manifest.json`: candidate-level files, hashes,
  classifications, and denominator membership.
- `KA59_PROTOCOL_LOCK.md`: accepted-data protocol and known ambiguities.
- `RUN_FOR_KAAUS.md`: execution appendix after author decisions are recorded.
- `LOCAL_STATE_BEFORE.md`: repository/worktree provenance only.
