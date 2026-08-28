# Local State Before Camera-Ready Work

Captured before `git fetch`, branch creation, switching, merging, stashing,
resetting, cleaning, or deleting in the source checkout.

- Source checkout: `/Users/edward/Projects/BreakingAgentsIntoPOMDP`
- Branch: `levi/ka59simple-level-attempts`
- HEAD: `c9063f293f93d743a648b969d5dfbffe93b9532d`
- Upstream: `origin/levi/ka59simple-level-attempts`
- Relationship: ahead by one commit
- Tracked unstaged changes: none
- Staged changes: none
- Untracked files: `tmp/pdfs/final_review/final.txt`

The untracked file and the local-only commit were left untouched. No stash,
reset, clean, checkout, merge, or formatting operation was performed in the
source checkout.

## `git status --short --branch`

```text
## levi/ka59simple-level-attempts...origin/levi/ka59simple-level-attempts [ahead 1]
?? tmp/
```

## `git diff`

No output.

## `git diff --cached`

No output.

## `git ls-files --others --exclude-standard`

```text
tmp/pdfs/final_review/final.txt
```

## `git log --oneline --decorate -20`

```text
c9063f2 (HEAD -> levi/ka59simple-level-attempts) wip: preserve ka59 level-attempt outputs and analysis
412ba5f (origin/levi/ka59simple-level-attempts) feat(ka59simple): track prompt cache usage
e04cd59 handoff: lock resume plan to OpenRouter+provider-pin (DeepSeek first-party) for the medium top-up
a5017c7 handoff: resume task — top up DeepSeek KA59-Simple medium to clean N=20 (blocker: DeepSeek balance empty)
6a3f140 docs: DeepSeek additions hand-off for Kaaus (optional, data-only; paper prose reverted to his version)
d60c2b0 figs(paper): win-rate + Wilson CI data CSV + figure specs for Kaaus
74b592f handoff: pooled medium N=14-23; strategy = combine done, skip GPT-5.2, focus write-up
60ae551 data(ka59simple): pool deepseek medium across endpoints -> N=14-23 (no new runs)
7abbeaf data(ka59simple): add clean deepseek medium N=9-13 (direct API; 402-failures dropped)
fc1b4d8 handoff: switch medium to direct DeepSeek API (Kaaus OpenRouter cost catch); flag cost-reporting correction
fcccea9 handoff: medium top-up launched (N=5-10 clean exists; topping to N=20 at 128-turn)
a2cdf94 data(ka59simple): add clean deepseek-v4-pro none N=20; drop contaminated medium
5d19577 handoff: ka59simple deepseek 32-turn re-run launched (running overnight)
ac8e2c1 stats(paper): label-normalize ka59simple in wilson_cis; document deepseek ka59simple confounds
3d7f2a9 stats(paper): fix wilson_cis.py + Fisher comparisons on canonical CSV; CI handoff
5d06f9a dashboard: LS20 DeepSeek-v4-pro N=20 results + COLM countdowns
34f06a8 data(ls20): DeepSeek-v4-pro N=20 ablation matrix (none + medium)
d6a444c feat(ls20): OpenRouter ablation parity + fix arc_agi .env.example auth clobber
a4787b5 feat(ka59): direct DeepSeek API provider + collision-safe ablation filenames
c52269f audit(bp35): verbal channel — metric non-transferable, but re-inference corroborates inferability spectrum
```

## `git reflog -20`

```text
c9063f2 HEAD@{0}: commit: wip: preserve ka59 level-attempt outputs and analysis
412ba5f HEAD@{1}: commit: feat(ka59simple): track prompt cache usage
e04cd59 HEAD@{2}: commit: handoff: lock resume plan to OpenRouter+provider-pin (DeepSeek first-party) for the medium top-up
a5017c7 HEAD@{3}: commit: handoff: resume task — top up DeepSeek KA59-Simple medium to clean N=20 (blocker: DeepSeek balance empty)
6a3f140 HEAD@{4}: commit: docs: DeepSeek additions hand-off for Kaaus (optional, data-only; paper prose reverted to his version)
d60c2b0 HEAD@{5}: commit: figs(paper): win-rate + Wilson CI data CSV + figure specs for Kaaus
74b592f HEAD@{6}: commit: pooled medium N=14-23; strategy = combine done, skip GPT-5.2, focus write-up
60ae551 HEAD@{7}: commit: data(ka59simple): pool deepseek medium across endpoints -> N=14-23 (no new runs)
7abbeaf HEAD@{8}: commit: data(ka59simple): add clean deepseek medium N=9-13 (direct API; 402-failures dropped)
fc1b4d8 HEAD@{9}: commit: handoff: switch medium to direct DeepSeek API (Kaaus OpenRouter cost catch); flag cost-reporting correction
fcccea9 HEAD@{10}: commit: medium top-up launched (N=5-10 clean exists; topping to N=20 at 128-turn)
a2cdf94 HEAD@{11}: commit: data(ka59simple): add clean deepseek-v4-pro none N=20; drop contaminated medium
5d19577 HEAD@{12}: commit: handoff: ka59simple deepseek 32-turn re-run launched (running overnight)
ac8e2c1 HEAD@{13}: commit: stats(paper): label-normalize ka59simple in wilson_cis; document deepseek ka59simple confounds
3d7f2a9 HEAD@{14}: commit: stats(paper): fix wilson_cis.py + Fisher comparisons on canonical CSV; CI handoff
5d06f9a HEAD@{15}: commit: dashboard: LS20 DeepSeek-v4-pro N=20 results + COLM countdowns
34f06a8 HEAD@{16}: commit: data(ls20): DeepSeek-v4-pro N=20 ablation matrix (none + medium)
d6a444c HEAD@{17}: commit: feat(ls20): OpenRouter ablation parity + fix arc_agi .env.example auth clobber
a4787b5 HEAD@{18}: commit: ka59 direct DeepSeek API provider + collision-safe ablation filenames
c52269f HEAD@{19}: commit: audit(bp35): verbal channel — metric non-transferable, but re-inference corroborates inferability spectrum
```

## Camera-ready worktree created afterward

- Worktree: `/Users/edward/Projects/BreakingAgentsIntoPOMDP-camera-ready`
- Branch: `camera-ready/ka59-truth`
- Base: `origin/anon-submission` at
  `996004f0ed1f28cdd24fe0a3722fa57c082c1df4`
