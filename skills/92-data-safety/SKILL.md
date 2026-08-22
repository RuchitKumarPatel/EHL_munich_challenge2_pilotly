---
name: 92-data-safety
description: Gate. Rules for the proprietary export and for this SHARED GitHub repo — what may never be copied, printed, committed or uploaded. Invoke before writing any file, printing any sample, or preparing any submission or slide.
---

# 92-data-safety

`export/` is a proprietary, challenge-use-only dataset. This repo is a SHARED GitHub repo that
other people are working in. Both facts constrain every write.

## 1. Preconditions
- None. This gate is always in force, including before `00-premise-freeze` reads a line.
- Know before you write: which files you own. Every other file belongs to another agent.

## 2. Procedure
1. Treat `export/` as READ-ONLY. Never modify, move, delete or re-write anything under it.
2. Never copy export contents into any file you write. Derived aggregates (counts, sums, token
   estimates) are fine; raw text is not.
3. Never print more than a ~60-character REDACTED snippet of any export line to any output —
   terminal, log, artifact, slide, chat.
4. Never upload the dataset or any excerpt anywhere: no network calls, no gists, no pastebins,
   no API calls, no artifact publish containing export text. The pipeline is fully offline.
5. Never run a git write command: no `git add`, no `git commit`, no `git push`, no `git rm`,
   no branch/tag mutation. Read-only git (`status`, `log`, `diff`) only.
6. Never touch the shared files: `scripts/`, `export/`, `site/`, `README.md`, `.gitignore`,
   `AGENTS.md`, `CLAUDE.md`. Other people are working in them.
7. Write ONLY the files listed as yours. Do not create `__init__` files, Makefiles, or docs
   that were not assigned to you.
8. Scratch work goes in `/tmp`. `results/` is generated output and is gitignored — writing there
   at runtime is fine.
9. Never emit a `PII_*` placeholder or a redacted entity token (`<PERSON_ROBERT>`, `<PROJECT_NAME>`,
   `<COMPANY_A>`, `<CHANNEL_NAME>`, `<ID_13>`) into any file, chart label or slide.
10. `python -m router.datasafety` scans every file under `results/` and every draft text for
    export-derived strings, `PII_*` tokens and `<ENTITY>` placeholders; exits non-zero on a hit.

## 3. Acceptance
- 0 files written outside the owned set; `git status` shows no modification to `scripts/`,
  `export/`, `site/`, `README.md`, `.gitignore`, `AGENTS.md`, `CLAUDE.md`.
- 0 git write commands executed in the session.
- 0 `PII_*` tokens and 0 `<ENTITY_>` placeholders in any file under `results/` or in any draft.
- 0 network calls; 0 API keys required; the pipeline runs offline on a laptop.
- 0 raw export lines copied into any written file; any printed snippet <= ~60 characters and redacted.
- `export/` byte-identical before and after the run; the 1000-line count unchanged.
- Exit code 0.

## 4. Banned moves
- Copying export text into `results/`, a slide, a README or an artifact — it is proprietary,
  challenge use only, no redistribution.
- `git add` / `git commit` / `git push` — this repo is shared; commits are a human decision.
- Editing `scripts/` to "fix" `group_trajectories()` — it is the organizers' tracked code and
  other people are working in it. Route around it (see 00-premise-freeze F2).
- Printing a full export line for debugging — use aggregates, or a <= 60-character redacted snippet.
- Uploading anything for a "quick check" — no network, no exceptions.
- Emitting a `PII_*` token — the placeholders are renumbered per request and are still
  export-derived content.
- Committing `export/` or the `.tar.gz` archive — both are gitignored; keep them there.

## 5. Postconditions
- Gate result recorded in `results/claims.json` under a stable key.
- Re-runs after every artifact write and before 91-verify-number, any submission, any slide.
- Red blocks every downstream skill including 95-make-frontier.

## 6. Escalate when
- A task appears to require modifying a shared file or committing anything. Stop and ask a human.
- An export line must be shown in full to explain a finding. Ask a human; do not decide alone.
- You find export-derived text already inside a tracked file. Do not fix it by committing;
  report it to a human.
