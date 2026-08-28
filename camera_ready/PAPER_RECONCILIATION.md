# KA59-Simple Paper Reconciliation

Source reviewed: the 16-page accepted-paper PDF at
`/Users/edward/Downloads/final.pdf`, the merged PR #18/#20 raw artifacts, and the
generated camera-ready manifest. The repository does not contain manuscript
LaTeX, so this report identifies edits but does not rewrite the paper.

## MUST FIX

1. **Figure 3 denominators and several counts are not raw-data valid.** The
   figure says GPT N=10 per effort and DeepSeek medium N=20. The manifest now
   reports both `infrastructure_clean_scored` and `strict_error_free`; neither
   view supports uniformly clean 10/20 denominators.
2. **Methods, lines 198-205:** “10 trials for GPT-5.2” is false as an
   effort-specific claim for KA59-Simple. The four main cells have five source
   trials per effort before error exclusion; the reported N=10 is a none+medium
   pool displayed under both effort labels.
3. **Methods/Table 1/Appendix A.2:** the claim that
   `MECHANICS HARD NORULES` retained the detailed action protocol is false for
   accepted raw runs. Historical prompt code made it identical to
   `MECHANICS HARD`; the retaining branch was added later.
4. **Feedback result, lines 234-240:** “GPT-5.2 configurations fall from 90% to
   80%” describes the pooled row, not two effort-specific configurations.
   “Deepseek medium falls from 20% to 10%” treats error-tainted trials as model
   losses and is supported by neither corrected denominator.
5. **Low-baseline paragraph, lines 242-248:** DeepSeek medium “20% baseline” and
   the comparison to GPT-5.2 “90% baseline” use the same invalid/pooling
   conventions. The capability-threshold interpretation cannot be numerically
   grounded in those cells until the denominator decision is resolved.
6. **Mechanics paragraph, lines 218-231:** DeepSeek none mechanics-hard is 4/20
   under `infrastructure_clean_scored` and 4/19 under `strict_error_free`. The
   differing denominator is a reporting-policy decision, not evidence that the
   turn-100 malformed response was a harness failure. DeepSeek medium and the
   no-rules rows have no infrastructure-clean or strict accepted denominator
   and cannot be stated as clean 0% results.
7. **World paragraph, lines 209-217:** directionally observed 0% cells still have
   smaller strict valid denominators for several GPT cells. “Every model” must
   not imply the displayed N=10/N=20 evidence exists for every row.
8. **`plots/plot.py`:** KA59 values remain hand-entered percentages and
   back-calculated counts. They must not be regenerated as evidence until the
   authors choose a corrected reporting policy based on the manifest.

## NEEDS HUMAN DECISION

1. Report GPT effort-specific results with honest small N, or report one pooled
   exploratory row. A pool must not be duplicated under both effort labels.
2. Remove/relabel the no-rules control as an implementation duplicate of
   mechanics-hard, or run the later paper-intended control. Future data must not
   be mixed with accepted-raw fallthrough data.
3. Decide whether camera-ready numeric claims use the candidate
   `infrastructure_clean_scored` denominator or the ultra-conservative
   `strict_error_free` sensitivity view. This changes only DeepSeek none
   mechanics-hard: 4/20 (20%) versus 4/19 (21%). All other cells are identical
   across the two policies. Hypothetical future runs are not included.
4. Decide how to describe DeepSeek medium. HTTP 402/insufficient-balance and
   provider-empty responses are infrastructure failures under both policies.
   The merged summary's nominal N=20 is not a clean model denominator, and
   neither accepted set supports the published exact percentages.
5. Confirm the desired OpenRouter upstream/provider pin for future GPT-5.2 runs.
   Historical raw metadata does not determine it.

## ALREADY CONSISTENT

- KA59-Simple is the accepted one-level environment and canonical `goal_hard`
  is not part of the reported matrix.
- The paper's 128-turn maximum matches two 64-turn attempts.
- The statement that a second attempt retains information from the first is
  consistent with raw history and historical code.
- DeepSeek none baseline 12/20, world-hard 0/20, no-rules-tagged 4/20, and
  feedback-hard 15/20 reproduce from strict raw trials.
- GPT none baseline 5/5 and feedback-hard 3/5 reproduce from strict raw trials.
- GPT medium baseline and feedback have no scored denominator under either
  policy because every nominal trial contains provider-empty responses. GPT
  world/mechanics cells also have the same reduced denominators under both
  policies; their empty responses remain infrastructure exclusions.
- The high-level finding that world information is important is directionally
  consistent with valid trials, but its displayed denominators still need
  correction.

## Parse-error classification correction

The 47 trials previously labelled `parse_error` were reviewed individually in
the manifest. Forty-six are DeepSeek-medium provider-empty records and are now
classified as infrastructure failures. One DeepSeek-none mechanics record
persisted the nonempty malformed response fragment `'{"'` at turn 100. The
historical parser correctly rejected it, and the runner subsequently completed
27 valid action events before a normal turn-128 loss. It is therefore a
recoverable `model_protocol_failure`, retained in
`infrastructure_clean_scored` and excluded only from `strict_error_free`. No
reviewed trial supports `harness_parse_failure` or `indeterminate`.

## Numeric-edit status

No manuscript numeric edit is committed because the accepted manuscript source
is absent from `anon-submission`, and several corrections depend on the author
decisions above. This report is the separate review unit; it does not modify
scientific framing or substitute hypothetical future results.
