# KA59-Simple Paper Reconciliation

Start with `camera_ready/README.md`. This file is the manuscript-edit checklist,
not the project overview.

Authority reviewed: the accepted 16-page paper PDF, merged PR #18/#20 raw
artifacts, the complete GPT historical-universe audit, Git history, and the
generated camera-ready manifest. The manuscript source is not committed to
`anon-submission`, so this PR identifies required edits but does not silently
choose scientific wording for the authors.

## Required corrections

| Paper location | Accepted statement/display | Evidence status | Camera-ready action |
|---|---|---|---|
| Methods, lines 198-205 | GPT-5.2 has 10 trials per configuration | False as an effort-specific KA59-Simple claim. Main raw cells are N=5 per effort before exclusions. | State N per cell/effort, or show one explicitly pooled exploratory row once. |
| Figure 3, KA59-Simple GPT rows | `none` and `medium` are both displayed at N=10 | The final `none` baseline/feedback values are pooled (9/10, 8/10); the `medium` values scale raw 4/5 and 5/5 rates to 8/10 and 10/10. Neither row is an independent N=10 effort-specific sample. | Replace with audited counts, or merge into one clearly labeled pool if authors justify pooling. |
| Methods/Table 1/Appendix A.2 | `MECHANICS HARD NORULES` retains the detailed action protocol | False for accepted raw runs. `HARD_FORMAT_ONLY` fell through to the ordinary `MECHANICS_HARD` prompt. | Remove/relabel the historical control, or run the later intended control as a new protocol. |
| Figure 3, DeepSeek `medium` row | Clean N=20 rates of 20/0/0/0/10 | Unsupported. Selected trials contain provider-empty/HTTP 402 failures that historically consumed turns and became losses. | Remove exact rates or replace them with a clean rerun under a pinned identity. |
| World result, lines 209-217 | Every KA59-Simple model falls to 0% except DeepSeek `medium` | Directionally consistent in available clean GPT/DeepSeek-none slices, but not at the displayed N=10/N=20 matrix. | State the actual denominators and avoid universal wording until rerun. |
| Mechanics result, lines 218-231 | Full and no-rules mechanics conditions establish rule-content causality | The historical prompts were identical, so the comparison does not isolate action-protocol discovery. | Remove the causal decomposition unless the intended control is run. |
| Feedback result, lines 234-240 | GPT configurations fall 90% to 80%; DeepSeek `medium` falls 20% to 10% | GPT sentence describes historical pooled/display operations, not two clean effort cells. DeepSeek exact rates are unsupported. | Replace with audited slice-specific claims. |
| Low-baseline result, lines 242-248 | DeepSeek `medium` 20% versus GPT-5.2 90% establishes a capability threshold | Both numbers depend on invalid/pooling conventions. | Remove the numeric comparison or re-establish it with a clean comparable matrix. |
| `plots/plot.py` | Hand-entered percentages are converted back to counts | Not an evidence pipeline. | Generate camera-ready figure inputs from the audited manifest after author decisions. |

## Claims that remain supportable now

- DeepSeek `none`: baseline 12/20, world-hard 0/20, mechanics-hard 4/20
  under the candidate denominator, and feedback-hard 15/20.
- DeepSeek `none` mechanics-hard is 4/19 in the strict sensitivity view. This
  is the only numeric difference between the two denominator policies.
- GPT-5.2 `none`: baseline 5/5, world-hard 0/5, mechanics-hard 0/5,
  feedback-hard 3/5, and historical no-rules fallthrough 0/19.
- GPT-5.2 `medium`: no clean denominator in any main or no-rules cell. The
  complete candidate cells are N=5 for each main condition and N=10 for
  no-rules; every trial contains provider-empty failures.
- The high-level world/mechanics disruption pattern is directionally supported
  in these slices. It must not be presented as a uniformly powered full matrix.

The historical no-rules-tagged DeepSeek `none` outcome is 4/20, but it is only
supportable as another run of the ordinary mechanics-hard prompt. It is not
evidence for the paper-described control.

The GPT no-rules-none 0/19 pool has the same interpretation boundary. Its five
included batches are independently 0/5, 0/4, 0/1, 0/4, and 0/5; this is a
replicated result for the historical fallthrough prompt, not validation of the
paper-described format-only control.

## Decisions the manuscript cannot make automatically

1. Per-effort reporting versus one explicitly pooled exploratory GPT row.
2. Existing-data correction versus a new target-N matrix in response to the
   reviewer/PI request to increase N.
3. Historical no-rules relabel/removal versus a new intended-control run.
4. Candidate main denominator (4/20) with strict sensitivity (4/19), versus
   strict-only reporting for DeepSeek `none` mechanics-hard.
5. Omit DeepSeek `medium` or rerun it under a fixed provider/model identity.
6. Exact model/provider/upstream identity, reasoning semantics, target cells,
   and target N for any new GPT-5.2 or “5.6 Luna” work.

## Parse review boundary

Of 49 parse-bearing records now in scope, 46 explicitly record DeepSeek
empty content and are infrastructure failures. One DeepSeek `none`
mechanics-hard trial contains the nonempty malformed fragment `'{"'` at turn 100,
then completes 27 later action events and ends in a normal loss. The candidate
denominator treats that as recoverable model behavior; the strict view excludes
it. Two newly recovered GPT no-rules-none trials persist a `NoneType` parse
exception without the raw response, so they are indeterminate and excluded.
No reviewed trial proves a harness parse failure.
