#!/usr/bin/env bash
# One-shot setup, run once before the first `loop/run.sh`.
#
# Why this exists as its own step: `main` is currently the pristine starter kit with no
# `router/` at all. Idea branches are cut from `main`, so a loop started against it would build
# every idea on top of nothing. This establishes `main` as the real integration branch -- but
# only after the gate proves the state it is about to promote actually works.
#
# It is also the one step that moves shared history, so it is deliberately separate from the
# loop, prints what it is about to do, and refuses rather than guesses.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
source "$ROOT/loop/config.env"

PY="$ROOT/.venv/bin/python"
BASE_BRANCH="${1:-tool-block-levers}"
LOGDIR="$ROOT/loop/logs"

# One implementation of "may this ref be pushed", shared with loop/run.sh. Sourcing it has
# no side effects; it only defines functions.
# shellcheck source=/dev/null
source "$ROOT/loop/push_gate.sh"

die() { printf '\n  REFUSING: %s\n\n' "$*" >&2; exit 1; }
step() { printf '\n=== %s ===\n' "$*"; }

step "preflight"

# Two agents in one working tree is how a night's work gets trampled. The loop assumes it is
# alone in this checkout.
OTHERS=$(pgrep -af 'claude-code/[0-9.]*/claude' 2>/dev/null | grep -v "$$" | wc -l)
if [ "$OTHERS" -gt 0 ]; then
  printf '  %s other Claude Code process(es) are running:\n' "$OTHERS"
  pgrep -af 'claude-code/[0-9.]*/claude' 2>/dev/null | cut -c1-100 | sed 's/^/    /'
  printf '\n  If any of them has THIS repository open, close it first.\n'
  printf '  Continue anyway? [y/N] '
  read -r ans
  [ "$ans" = "y" ] || die "aborted by operator"
fi

# The loop's two .gitignore lines are re-applied after the merge instead of being carried
# through it as a local modification: `main` and `$BASE_BRANCH` differ in this same file, and a
# fast-forward refuses to run over local changes it would overwrite.
if ! git diff --quiet -- .gitignore; then
  printf '  reverting local .gitignore edits (re-applied after the merge)\n'
  git checkout -- .gitignore
fi

git rev-parse --verify "$BASE_BRANCH" >/dev/null 2>&1 || die "no such branch: $BASE_BRANCH"
git rev-parse --verify "$INTEGRATION_BRANCH" >/dev/null 2>&1 || die "no such branch: $INTEGRATION_BRANCH"
[ -x "$PY" ] || die "no venv interpreter at $PY"

if ! git merge-base --is-ancestor "$INTEGRATION_BRANCH" "$BASE_BRANCH"; then
  die "$BASE_BRANCH is not a descendant of $INTEGRATION_BRANCH -- this would not be a
  fast-forward. Resolve by hand; bootstrap will not guess at shared history."
fi

AHEAD=$(git rev-list --count "$INTEGRATION_BRANCH..$BASE_BRANCH")
printf '  will fast-forward %s onto %s (+%s commits)\n' "$INTEGRATION_BRANCH" "$BASE_BRANCH" "$AHEAD"

step "gate: does $BASE_BRANCH actually work?"

git checkout -q "$BASE_BRANCH" || die "could not check out $BASE_BRANCH"

# `make all` runs FIRST on purpose. Two classes in tests/test_data_safety.py scan the
# generated artifacts and raise SkipTest when that directory does not exist yet, so a
# suite run before the pipeline has built anything is green for the wrong reason.
make all     || die "make all failed on $BASE_BRANCH -- not promoting a broken pipeline to main"
make test    || die "make test failed on $BASE_BRANCH -- not promoting a red branch to main"
"$PY" -m router.gates || die "router.gates failed on $BASE_BRANCH -- not promoting past the publication gate"

printf '  gate green on %s\n' "$BASE_BRANCH"

step "promote $BASE_BRANCH to $INTEGRATION_BRANCH"

git checkout -q "$INTEGRATION_BRANCH" || die "could not check out $INTEGRATION_BRANCH"
git merge --ff-only "$BASE_BRANCH"    || die "fast-forward failed"

step "commit the loop's own infrastructure onto $INTEGRATION_BRANCH"

# Now that the merge has landed, re-apply what the loop needs ignored.
for pat in 'loop/logs/' '.loop-stop'; do
  grep -qxF "$pat" .gitignore || printf '%s\n' "$pat" >> .gitignore
done

git add loop/run.sh loop/stop.sh loop/status.sh loop/bootstrap.sh \
        loop/backlog.py loop/config.env loop/README.md \
        loop/prompts docs/IDEAS.md .gitignore
git add loop/state 2>/dev/null || true
git add .entire/settings.json .entire/.gitignore 2>/dev/null || true
git commit -q -m "loop: unattended overnight improvement loop

One turn is one fresh headless process; state lives in git, loop/state/ and entire's
checkpoints, so a hung or derailed turn costs one slot rather than the night. The turn
type is chosen by a pure function of the queue, not by the model: deck on a fixed
cadence, then merge, implement, evaluate, and council only when the queue is dry.

main moves only through a merge turn that re-runs the full gate itself and reverts if
the merged state fails it." \
  || printf '  (nothing new to commit)\n'

step "push every branch"

# This loop is where the night-one leak happened: it pushed every branch with no check of
# any kind, and one of them carried raw identifiers out of export/ to a shared repository.
# It now goes through the same gate run.sh uses -- working tree once, then each branch
# against its own tree -- and refuses rather than guesses.
if [ "$PUSH" = "1" ]; then
  DIVERGED=0
  if push_all_branches; then
    printf '  pushed every branch past the data-safety gate\n'
  else
    die "the data-safety gate refused at least one branch -- see loop/logs/data-safety*.log.
  Nothing was force-pushed and nothing was deleted; fix the offending tree and re-run."
  fi
else
  printf '  PUSH=0 in loop/config.env -- staying local\n'
fi

step "ready"
"$PY" loop/backlog.py stats
printf '\n  next turn would be: %s\n' "$("$PY" loop/backlog.py plan --deck-every "$DECK_EVERY")"
printf '\n  start the loop with:\n    nohup loop/run.sh > loop/logs/supervisor.log 2>&1 &\n\n'
