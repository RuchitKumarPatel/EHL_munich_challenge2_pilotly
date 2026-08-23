## Your job this turn: DECK — rebuild the 5-minute defense deck from current numbers

This runs every few turns so the slide set never drifts behind the work. Treat it as a rebuild
from source, not an edit: the numbers come from `results/claims.json`, always.

### 1. Refresh the source of truth

```bash
git checkout main
make all
```

Three minutes, and it guarantees `results/claims.json` and `results/frontier.png` match what is
on `main` right now. Then read `results/NUMBERS.md` — it is the prose version of the same
numbers and it already says what the argument is.

### 2. Build `presentation.html` at the repo root

Follow `skills/make-presentation/SKILL.md`. Start from `templates/presentation.html` and fill
its `<!-- FILL -->` slots. The slide order is fixed on purpose — do not reorder or add slides:

1. **Title** — team name, one-line claim
2. **Objective** — the trade-off chosen, one sentence
3. **The signal** — what the router looks at, on a real trace example
4. **The frontier** — the one chart
5. **Honesty** — the off-policy method and where it breaks
6. **Close** — one claim, one number, one weakness, next step

Two things the template does not do for you:

- **Embed the chart as a base64 `data:` URI.** The template points at `../results/frontier.png`,
  but the generated artifacts are gitignored, so a committed deck with that path shows a broken
  image on any teammate's machine. Read the PNG, base64 it, inline it.
- **Leave `[FILL]` markers visible** where the header above gave you `[FILL]` for team name or
  members. Do not invent a team name. A visible marker is a 5-second fix for the human in the
  morning; an invented name is an embarrassment on stage.

Brand: violet `#6748FD` on navy `#150079`, peach `#FFBD9E` accents — already in the template.
No extra colours, no hype words, short active sentences.

### 3. The rule that governs every number on a slide

Every figure, percentage and dollar amount on a slide must exist in `results/claims.json` under
a stable key. Check each one. If a number you want is not there, either it is not ours and it
comes off the slide, or it belongs in `router/report.py` and a future implement turn adds it.
Do not transcribe a number from prose.

Carry the qualifiers onto the slides, not just into the notes: tokens are **estimated**, dollars
rest on an **assumed price sheet**, and the outcome is **process friction, not answer quality**.
`docs/OPEN-QUESTIONS.md` Q1 means the *sign* of any saving is not established from this export
alone — a slide claiming "we cut spend by X%" without naming that assumption in the same
sentence is the fastest way to lose the evaluation-rigor criterion.

Slide 5 must name a real failure mode. The honest candidates are already measured: the router
does not beat a single-arm mixture at equal spend (`NUMBERS.md` #5), the MDE is 11.4 pp so no
point estimate is available, and two feature columns are outcome-derived and leak under
job-blocked folds. Pick the one the current state of `main` actually justifies.

### 4. Check it renders

Open it and step through. Nothing may overflow at 1920×1080. If a browser is available,
`tests/console_states.sh` shows the pattern this repo uses for driving a page; otherwise a
careful read of the markup plus a headless render is enough.

### 5. Commit

```bash
git add presentation.html
git commit -m "deck: rebuild slides from current claims (turn N)"
git push origin main
```

Do **not** add a backlog entry for the deck rebuild. The deck is on a fixed cadence, not in the
idea queue; adding it would send a presentation chore through evaluate → implement → merge for
no reason. The journal line and this commit are the record.

Your final output: one paragraph naming which numbers moved since the last deck build, or
stating that none did. If a slide had to drop a number because it was not in `claims.json`, say
which — that is the most useful thing the human can read in the morning.
