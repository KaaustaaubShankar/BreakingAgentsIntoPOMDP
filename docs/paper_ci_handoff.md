# Paper CI / error-bar work — handoff (resume tomorrow)

_Session ending 2026-06-17. COLM submission deadline: **June 23, 2026 (AoE)**._

## ⏯️ RESUME AFTER RESTART (2026-06-20): top up DeepSeek KA59-Simple MEDIUM to clean N=20
**Goal:** Edward wants the DeepSeek-V4-Pro `medium` KA59-Simple cells at an exact N=20 (currently pooled 14–23). Everything else DeepSeek is already clean N=20 (LS20 none+medium, KA59-Simple none).

**Current real (non-zero-token) N per medium cell → gap to 20:**
- baseline 16 → **+4**
- world_hard 23 → done (skip)
- mechanics_hard 14 → **+6**
- mechanics_hard_format_only 15 → **+5**
- feedback_hard 18 → **+2**
(~17 trials; run +2 buffer each since some may error / dedup.)

**🚧 PLAN (decided by Edward): run on OpenRouter with the PROVIDER PINNED.** The DeepSeek direct API balance is empty (402), and OpenRouter's default *pooled* routing was ~3× the direct cost (it load-balances across Alibaba/Together/etc. that charge more for DeepSeek's heavy token output). So redeploy via `--provider openrouter --model deepseek/deepseek-v4-pro` but **pin to DeepSeek's first-party provider** so the rate ≈ direct and doesn't vary.

**Code change needed (one-time):** `ka59_game/llm_client.py` `_generate_openrouter` currently sets `extra_body={"reasoning":{"effort":...}}`. Add a provider pin so it becomes:
```python
extra_body = {"reasoning": {"effort": self.reasoning_effort},
              "provider": {"order": ["DeepSeek"], "allow_fallbacks": False}}
```
(Verify the slug: `curl https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-pro/endpoints` to see provider names; if "DeepSeek" first-party isn't listed, use `{"sort": "price"}` for cheapest. Make it deepseek-only so other models' runs are unaffected.)

**Gate-check before the full run:** one call → confirm tokens>0 AND inspect cost/provider (OpenRouter returns the serving provider; log it) so we know the pin took and the rate is the cheap one. Monitor first trials for `tok=0in` — abort if 402/zero-token/expensive-provider.

**Budget/settings:** default 128-turn (matches existing medium), `--reasoning-effort medium`, rates 0.30/0.90 for reporting. Run per-cell (gap+2 buffer):
```
for spec in "baseline 6" "mechanics_hard 8" "mechanics_hard_format_only 7" "feedback_hard 4"; do
  set -- $spec
  .venv/bin/python -m scripts.run_real_ablation --env ka59simple --provider openrouter \
    --model deepseek/deepseek-v4-pro --reasoning-effort medium --configs "$1" --trials "$2" \
    --input-cost-per-m 0.30 --output-cost-per-m 0.90 > /tmp/ka59s_medtop_$1.log 2>&1 &
done; wait
```
(world_hard already ≥20 — don't rerun.) Slow (~2–3h/trial) → ~half a day. NOTE: pooling will then mix OpenRouter-pinned + earlier direct trials — same deepseek-v4-pro/medium/128-turn, so fine; keep the endpoint-pool footnote.

**AFTER it finishes:**
1. Re-pool: real (non-zero-token) medium trials, dedup, take exactly 20/cell. Recompute win rates + Wilson CIs + Fisher (`scripts/wilson_cis.py`).
2. Update `dashboard/jkj results - Detailed_Results.csv` + Overview (and push to VM `/home/exedev/`, restart `python3 app.py`).
3. Update `results/figure_data/winrate_ci.csv` + regen `docs/ci_table.txt`.
4. Update the **Google sheet** (source of truth) with the clean N=20 medium numbers.
5. Heads-up to give team: until this finishes, KA59-Simple medium DeepSeek is 14–23, NOT a clean 20 (correct any "all done at 20" statement).
Recipe + history: [[paper-ci-work-state]] memory. Do NOT touch Kaaus's paper prose ([[dont-ai-rewrite-coauthor-prose]]).

---

## TL;DR — where we are
- LS20 (env3) deepseek-v4-pro N=20 sweep: **DONE**, committed, dashboard updated (see `results/ls20_real_ablation/`, `dashboard/`).
- CI tooling: `scripts/wilson_cis.py` **rewritten + fixed** (it was pointing at a non-existent CSV). Now reads the canonical `dashboard/jkj results - Detailed_Results.csv`, prints (1) a Wilson 95% CI table for every cell and (2) Fisher's-exact none-vs-medium primary comparisons. Pure stdlib (no scipy) — runs after a plain `git pull`.
- Output captured at `docs/ci_table.txt` (regenerate: `python scripts/wilson_cis.py`).

## Why N=20 matters (the framing change for the COLM paper)
`colm2026_conference.tex` currently says "n=5 pilot… qualitative patterns, not significance." With our N=20 deepseek data that's outdated. The headline reasoning-hurts-mechanics result is now **confirmatory**:

| cell | none | medium | Fisher exact |
|---|---|---|---|
| n=5 (old) | 2/5 (40%) [12,77] | 0/5 (0%) [0,43] | p=0.44 — ns |
| **n=20 (now)** | **9/20 (45%) [26,66]** | **0/20 (0%) [0,16]** | **p=0.0012 \*\*** |

→ The COLM version can report these as real effects, not just "qualitative."

## How CIs / error bars go into the paper (methodology, agreed)
1. Per-cell: **Wilson 95%** (binary win/loss; correct at 0%/100% boundaries). `wilson_cis.py`.
2. Tables: `k/n [lo, hi]` per cell.
3. Figures: win-rate bars with **asymmetric** error bars = Wilson [lo, hi].
4. Condition comparisons (none vs medium, GPT vs Grok): **Fisher's exact** on the 2x2; report p (+ odds ratio). Non-overlapping Wilson CIs is a conservative sufficient signal.
5. Mixed N across cells — report N per cell; designate primary comparisons confirmatory, rest exploratory (or Holm-correct).

## ka59simple deepseek — DATA IS CONFOUNDED, needs a clean re-run (2026-06-17 update)
Tried to "add ka59simple fully" and found three problems making the existing deepseek ka59simple data NOT paper-usable as-is:
1. **`medium` is ~75% zero-token contaminated.** Team's older OpenRouter run: ~20/26 trials per cell were silent 401/zero-token failures counted as losses (raw `medium::baseline`=4% is an artifact; real ~17% on N≈6). Same silent-failure bug class as the arc_agi/.env one.
2. **Turn-budget mismatch.** deepseek ka59simple ran at **128 turns** (1 level x turns_per_level=128) while gpt/grok + paper.tex use a **32-turn budget**. 4x larger budget → not comparable across models.
3. **Endpoint split.** deepseek `none` = direct DeepSeek API; `medium` = OpenRouter. (LS20 was all-OpenRouter = clean.)
4. (Fixed) **Label inconsistency** `no-R`/`medium-R`/`default-R` — now normalized non-destructively in `wilson_cis.py` on read (ka59simple gpt-5.2 now pairs in Fisher output).

### UPDATE 5 (2026-06-18): POOLED medium old+new -> N=14-23 (Edward's combine idea, $0)
Disk-full scare was a transient on the data volume (now 322Gi free; results/ka59simple_game is only 193MB/864 files — not the cause, nothing deleted). Then pooled the REAL (non-zero-token) deepseek-v4-pro medium trials from the old OpenRouter batch + the new direct-API run (both medium, 128-turn): N = baseline 16, world_hard 23, mech 14, format 15, feedback 18. Updated Detailed + Overview medium rows (pooled) in dashboard CSVs + VM; regen ci_table.
**Reasoning-hurts now significant on TWO ka59simple cells: baseline 60->19% (Fisher p=0.019 *) and feedback_hard 75->22% (p=0.003 **)** — baseline crossed into significance vs the earlier N=10. Endpoint-pool footnote needed (OpenRouter + direct, same model/effort/budget).
STRATEGY (agreed): combine old+new (done, free); SKIP GPT-5.2 N=20 (older model, would eat the ~$90 budget, deepseek already carries the confirmatory headline); spend remaining time + $90 on the WRITE-UP (CI framing, error-bar figs, cost-correction, Background). Optional ~$10/~17-trial direct-API top-up could even medium to exactly N=20 but not needed.

### UPDATE 4 (2026-06-18): medium added at N=9-13 (DeepSeek balance ran out) [superseded by Update 5 pooling]
The direct-API medium run hit **402 Insufficient Balance** partway — the DeepSeek account drained mid-run (these 452k-output trials burn balance fast). ~10-11 of 20/cell failed with 402 (zero-token). Edward chose: **use the clean survivors as-is** (no top-up). Added clean deepseek ka59simple MEDIUM rows at **N=9-13** (402/zero-token dropped) to dashboard CSVs + VM; added clean medium Overview row; dashboard restarted; ci_table regenerated.
Final deepseek ka59simple (direct API, 128-turn budget): none baseline 60/world 0/mech 20/format 20/feedback 75 (N=20); medium baseline 20%(2/10)/world 0%(0/13)/mech 0%(0/9)/format 0%(0/9)/feedback 10%(1/10).
**Result: reasoning-hurts is significant on feedback_hard (75->10%, Fisher p=0.0014 **), borderline on baseline (60->20%, p=0.058); corroborates LS20** even at small medium N. CAVEATS for paper: (a) 128-turn budget vs gpt/grok 32 — not gpt-comparable; (b) medium N=9-13 (small) — footnote; (c) cost-reporting correction still owed (see below).
DONE with the ka59simple data step. Remaining paper work: flip COLM framing + insert CI table, Wilson error-bar figures, Background, cost-correction.

### (history) UPDATE 3: switched medium to DIRECT DeepSeek API (Kaaus cost catch, 2026-06-17)
Kaaus flagged OpenRouter is mispricing DeepSeek: its displayed top price isn't real — it routes across providers (Alibaba etc.) so the true cost is the **weighted average** (page bottom), and since DeepSeek emits ~5x GPT's tokens, OpenRouter is **~$1.50/trial vs ~$0.33/trial on the direct DeepSeek API** (and direct is faster). So:
- **Killed the OpenRouter medium top-up.** Relaunched **fresh N=20 medium on the DIRECT API** (provider=deepseek, default 128-turn budget, 5 configs) — verified healthy, no auth errors. This makes ka59simple fully endpoint-consistent (none+medium both direct API) and is ~3x cheaper. Logs `/tmp/ka59s_medD_<cfg>.log`. **Fresh N=20 = NO merge needed** (supersedes Update 2's top-up/merge plan and the OpenRouter clean medium).
- **WHEN DONE:** aggregate the new `ablation_deepseek_*` medium files (dedup, drop tok=0) -> N=20; add medium Detailed rows + clean medium Overview row to dashboard CSVs + VM; restart dashboard; regen `docs/ci_table.txt`.

### COST-REPORTING CORRECTION NEEDED (paper) — from Kaaus's catch
The reported costs (LS20 $20.10; all OpenRouter cells) were computed at $0.30/$0.90 per M, which are OpenRouter *estimates that understate* the real weighted-average (~$1.50/trial empirically). For the paper's cost analysis: recompute with accurate per-endpoint rates — OpenRouter weighted-avg for the LS20/old data, real direct DeepSeek API rates for ka59simple none + new medium. Get real rates from Kaaus / OpenRouter+DeepSeek billing. Cost is a selling point ([[paper-revision-bens-findings]] budget angle), so the numbers must be right.

### (superseded) UPDATE 2: medium IS real — top-up plan (replaced by Update 3 direct re-run)
Re-examined: the medium run DID work (slow ~500k-reasoning trials are genuine). What I'd called "contamination" was dead-key **zero-token instant failures mixed in** (~15-20/cell). Clean medium = N=5-10/cell at the **same 128-turn budget as none** → deepseek none-vs-medium IS internally comparable, and shows the SAME reasoning-hurts pattern as LS20 (baseline 60->17, mech 20->0, feedback 75->38).
Decision (Edward): **top up to N=20.** LAUNCHED 5 cells (medium, OpenRouter, DEFAULT budget=128, deltas baseline+15/world+11/mech+16/format+15/feedback+13) — verified OpenRouter live, no auth errors. ~2 days for slow cells. Logs `/tmp/ka59s_medtopup_<cfg>.log`.
**WHEN DONE:** merge existing-clean medium (non-zero-token, deduped) + new trials -> exactly N=20/cell; compute win rates; add 5 medium Detailed rows + restore a CLEAN medium Overview row (deepseek ka59simple) to `dashboard/` CSVs + VM; restart dashboard; regenerate `docs/ci_table.txt`. Budget caveat (128 vs gpt 32) still applies to the whole deepseek ka59simple row.
Dedup/merge recipe: gather medium trials from `results/ka59simple_real_ablation/ablation_openrouter_deepseek_*` (existing) + the new top-up files; drop `input_tokens==0`; md5-dedup; take 20/cell.

### RESOLUTION (2026-06-17, SUPERSEDED by Update 2 for medium): added clean none, skipped medium
Decision: the direct-API `none` (N=20, clean, zero contamination) is good data — added it as-is. `medium` has no clean version (only the contaminated OpenRouter batch) and was **skipped** (no re-run). The wrong-budget re-runs (128 then 32) were both killed.
- Added 5 `none` Detailed rows + kept the correct Overview none row (row 29) in `dashboard/` CSVs AND on the VM; **removed the contaminated `medium` Overview row** (was 75% zero-token garbage).
- `docs/ci_table.txt` regenerated — deepseek ka59simple none now has Wilson CIs (baseline 60% [39,78], world_hard 0% [0,16], mechanics_hard 20% [8,42], mech_format 20%, feedback_hard 75% [53,89]).
- **CAVEAT for the paper:** deepseek ka59simple ran at a **128-turn budget** vs gpt/grok's 32 → footnote this; deepseek-vs-gpt on ka59simple is NOT apples-to-apples. No deepseek `medium` row for ka59simple.
- The clean cross-condition (none vs medium) deepseek story lives in **LS20** (both N=20, OpenRouter, same budget) — that's the headline.

### (superseded) earlier corrected plan — re-run at 32 — NOT pursued
Re-run deepseek ka59simple **none + medium together, OpenRouter, `--max-turns 32`, N=20, 5 configs**
(baseline/world_hard/mechanics_hard/mechanics_hard_format_only/feedback_hard; no goal_hard).
→ 32-turn budget matches gpt/grok/paper; OpenRouter matches LS20; fresh run kills contamination; ~4x cheaper
than 128 (≈ $15 / a few hours, not $50 / 2 days). Command:
```
for eff in none medium; do for cfg in baseline world_hard mechanics_hard mechanics_hard_format_only feedback_hard; do
  python -m scripts.run_real_ablation --env ka59simple --provider openrouter --model deepseek/deepseek-v4-pro \
    --reasoning-effort $eff --configs $cfg --trials 20 --max-turns 32 \
    --input-cost-per-m 0.30 --output-cost-per-m 0.90 > /tmp/ka59s_${eff}_${cfg}.log 2>&1 &
done; done; wait
```
**STATUS: LAUNCHED 2026-06-17 ~03:30 UTC** — all 10 cells (none+medium x 5 configs) running at 32-turn budget on OpenRouter, verified no auth errors / real tokens (gate-checked). Running overnight (~hours). Logs `/tmp/ka59s_<eff>_<cfg>.log`; results land in `results/ka59simple_real_ablation/ablation_openrouter_deepseek_*` + per-trial `results/ka59simple_game/`. The wrong-budget (128) medium run was killed.

### TOMORROW — when the re-run is done
1. Verify: no tok=0 trials; N=20/cell; budget=/32.
2. Aggregate the new deepseek ka59simple none+medium win rates (dedupe like the LS20 merge).
3. Add deepseek ka59simple rows to `dashboard/jkj results - Detailed_Results.csv` (Detailed) + fix Overview rows 29/30; push the dashboard CSVs to the VM (`breaking-agents.exe.xyz:/home/exedev/`) and restart `python3 app.py`.
4. `python scripts/wilson_cis.py > docs/ci_table.txt` — confirm ka59simple deepseek none-vs-medium Fisher p appears.
5. Then the paper steps: flip COLM framing + CI table, Wilson error-bar figures, Background.
NOTE: the gpt/grok ka59simple data may ALSO have zero-token contamination worth auditing (see [[paper-data-reconciliation-conflicts]] — needs Edward's calls).

## Next steps (recommended order)
1. **Reconcile the CSV** (the two blockers above) → re-run `python scripts/wilson_cis.py > docs/ci_table.txt`. Confirm ka59simple deepseek none-vs-medium now appears with Fisher p.
2. **Flip the COLM framing**: edit `colm2026_conference.tex` §5 (Preliminary Results) + §6 (Discussion) from "qualitative pilot" to CI-backed; insert the Wilson CI table (`paper.tex` already has the table format to copy) and the significant Fisher results.
3. **Error-bar figures**: matplotlib win-rate bars with Wilson asymmetric `yerr`. New script e.g. `scripts/plot_winrate_cis.py` reading the same CSV.
4. **Background**: tighten `colm2026_conference.tex` §3 (POMDP→knockout mapping) and §2 (Related Work — position the verbal/behavioral discovery decoupling against self-report eval literature). `paper.tex` has fuller prose to draw from.

## Files touched this session (all committed/pushed on branch `levi/ka59simple-level-attempts`)
- `scripts/wilson_cis.py` (rewritten), `docs/ci_table.txt` (new)
- `env3/{llm_client,experiment,ablation}.py`, `env4/llm_client.py` (LS20 parity + arc_agi auth fix)
- `ka59_game/llm_client.py`, `scripts/run_real_ablation.py` (DeepSeek provider + pid fix)
- `results/ls20_real_ablation/` (LS20 N=20 data), `dashboard/` (Flask app + CSVs)

## Resume command
```
git pull && python scripts/wilson_cis.py | less
```
