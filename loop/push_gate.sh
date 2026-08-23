#!/usr/bin/env bash
# The data-safety gate on everything that leaves this machine.
#
# Sourced, never executed: `loop/run.sh` and `loop/bootstrap.sh` both pull it in so
# there is exactly one implementation of "may this ref be pushed". It defines
# functions and nothing else -- sourcing it has no side effects.
#
# WHY IT IS ITS OWN FILE
#   The gate used to live inline in run.sh and it scanned the WORKING TREE, then
#   pushed every ref under refs/heads. A branch that an earlier turn updated and
#   moved off was therefore pushed having never been scanned. That is the exact
#   incident the gate was written to prevent: on the loop's first night
#   bootstrap.sh pushed every branch before any check ran, and one of them carried
#   raw identifiers straight out of export/ to a shared repository. The dataset is
#   challenge-use-only and a push cannot be taken back.
#
# WHAT IT DOES NOW
#   1. The working tree is scanned once. Red suppresses the whole cycle, because a
#      red working tree usually means the turn itself produced something it should
#      not have.
#   2. Then EACH branch that the remote does not already have is scanned against
#      ITS OWN tree, via DATA_SAFETY_SCAN_REF, before that branch is pushed. A
#      branch that fails is skipped and named in the log; the clean ones still go.
#   3. Both scans FAIL CLOSED. The suite prints a receipt saying how many files it
#      enumerated, how many tests it ran and how many failed, and a branch is
#      refused unless that receipt agrees with a green exit code. Before this, a
#      suite in which every check skipped exited 0 and read as a clean branch --
#      see ds_receipt_ok.
#   4. Every branch NAME is validated before it becomes a git argument, and the
#      push is pinned to the sha that was scanned. A ref name is untrusted input
#      -- see ds_ref_name_ok -- and the gate used to hand it to `git push` bare.
#
# WHAT IT STILL DOES NOT COVER  (do not read more into this than it says)
#   - `entire/*` checkpoint refs are exempt by name -- backlog i0007.
#   - The prompts for the merge and deck turns push by hand; they are told to run
#     the suite first, but that is an instruction to a model, not a gate.
#   Both are recorded in docs/DATA-SAFETY-DEBT.md rather than implied away here.
#
# Callers supply: ROOT, PY, LOGDIR, REMOTE, PUSH, INTEGRATION_BRANCH, DIVERGED,
# and optionally TURN, TYPE and a `journal` function. Every one has a fallback so
# the file can be sourced and driven standalone by tests/test_push_gate.py.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-$ROOT/.venv/bin/python}"
LOGDIR="${LOGDIR:-$ROOT/loop/logs}"
REMOTE="${REMOTE:-origin}"
PUSH="${PUSH:-1}"
INTEGRATION_BRANCH="${INTEGRATION_BRANCH:-main}"
DIVERGED="${DIVERGED:-0}"

# run.sh defines both of these before sourcing; bootstrap.sh and the tests do not.
declare -F log     >/dev/null || log()     { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
declare -F journal >/dev/null || journal() { :; }

# A green exit code is not enough, and that is not a hypothetical.
# `python -m unittest` exits 0 when every test SKIPS, and every content check in
# the suite skipped whenever it could not enumerate files. So
# `DATA_SAFETY_SCAN_REF=refs/heads/no-such-branch` printed `OK (skipped=6)`,
# exit 0, and this gate pushed the branch having scanned nothing. Any git error
# reached the same place. The suite therefore prints a receipt saying what it
# actually did, and both scans below refuse unless the receipt agrees with the
# exit code. `-m tests.test_data_safety` (not `-m unittest`) is what prints it.
ds_receipt_ok() {
  local logfile="$1" line files ran failed
  [ -r "$logfile" ] || return 1
  line=$(grep -a '^DATA-SAFETY-RECEIPT ' "$logfile" | tail -1)
  [ -n "$line" ] || return 1
  files=$(printf '%s' "$line" | sed -n 's/.*[ ]files=\([0-9][0-9]*\).*/\1/p')
  ran=$(printf '%s' "$line" | sed -n 's/.*[ ]ran=\([0-9][0-9]*\).*/\1/p')
  failed=$(printf '%s' "$line" | sed -n 's/.*[ ]failed=\([0-9][0-9]*\).*/\1/p')
  [ -n "$files" ] && [ -n "$ran" ] && [ -n "$failed" ] || return 1
  [ "$files" -gt 0 ] || return 1        # enumerated nothing: not the same as clean
  [ "$ran" -gt 0 ] || return 1          # ran nothing: not the same as clean
  [ "$failed" = "0" ] || return 1
  return 0
}

# Is this a name we are willing to put on a git command line?
#
# WHY THIS EXISTS AT ALL
#   `git for-each-ref` is not a trusted source. `git branch` and `git checkout -b`
#   refuse an option-shaped name, but `git update-ref` accepts one, and so does a
#   file written straight into .git/refs/heads -- and every turn of this loop runs
#   an agent with unrestricted bash. Measured on git 2.55:
#     * `refs/heads/--receive-pack=/tmp/x/pwn.sh` is creatable with update-ref and
#       is listed by for-each-ref.
#     * `git rev-parse` echoes it back with exit 0, so the old `|| continue` after
#       rev-parse did not catch it.
#     * `git push -q origin '--receive-pack=/tmp/x/pwn.sh'` parses it as an OPTION
#       and spawns that program locally, as the operator's user, then pushes
#       whatever push.default says. That is arbitrary local code execution out of
#       a ref name, reached from every turn and from bootstrap.sh with PUSH=1.
#
# WHY AN ALLOWLIST AND NOT `git check-ref-format`
#   check-ref-format answers "is this a legal refname", which is a different
#   question: it returns 0 for `refs/heads/--receive-pack=x`, measured. It is kept
#   as a second gate, not as the first one. The first one is the character class,
#   because an ordinary branch name in this repo is `main`, `idea/<slug>`,
#   `entire/<hash>` or a plain hyphenated word, and nothing legitimate here needs
#   a leading dash, a space, an `=`, a glob character or a `~`.
ds_ref_name_ok() {
  local br="$1"
  [ -n "$br" ] || return 1
  case "$br" in
    -*|*[!A-Za-z0-9._/-]*) return 1 ;;
  esac
  # The argument is literally prefixed with `refs/heads/`, so it cannot itself be
  # option-shaped however hostile $br is.
  git -C "$ROOT" check-ref-format "refs/heads/$br" 2>/dev/null
}

# The suite, against the working tree. This is the only scan that sees untracked
# files and the generated artifacts -- neither is tracked, so neither can travel
# through a ref, but both are how a turn notices it has made a mess.
ds_scan_worktree() {
  local logfile="$LOGDIR/data-safety.log"
  mkdir -p "$LOGDIR"
  ( cd "$ROOT" && DATA_SAFETY_STRICT=1 "$PY" -m tests.test_data_safety ) \
    > "$logfile" 2>&1 || return 1
  ds_receipt_ok "$logfile"
}

# The suite, against one ref's tree. DATA_SAFETY_SCAN_REF makes the repo-file
# checks enumerate `git ls-tree` instead of `git ls-files`, restricts the run to
# the ref-aware classes, and implies STRICT -- so a git failure is RED rather
# than a skip. See the constants' docstrings in tests/test_data_safety.py.
ds_scan_ref() {
  local ref="$1" slug logfile
  # Callers pass the resolved sha as $1 and the branch name as $2: the scan must
  # pin the object it checked, while the log file still has to be findable by the
  # name a human knows. With one argument the two are the same, as before.
  slug="${2:-$1}"
  slug="${slug//\//-}"
  logfile="$LOGDIR/data-safety-$slug.log"
  mkdir -p "$LOGDIR"
  ( cd "$ROOT" && DATA_SAFETY_SCAN_REF="$ref" "$PY" -m tests.test_data_safety ) \
    > "$logfile" 2>&1 || return 1
  ds_receipt_ok "$logfile"
}

# Every branch goes to the remote every turn, so nothing the night produces exists
# only on this laptop -- including the branches whose ideas were rejected. Two
# exceptions: `main` while diverged, because pushing a diverged integration branch
# is how a shared repository gets damaged, and `entire/*`, the CLI's checkpoint
# refs. Returns 1 if any branch was refused, so a caller can report it.
push_all_branches() {
  [ "$PUSH" = "1" ] || return 0

  if ! ds_scan_worktree; then
    log "DATA SAFETY RED on the working tree -- pushing nothing this turn ($LOGDIR/data-safety.log)"
    journal "${TURN:-?}" "${TYPE:-?}" "push suppressed: tests.test_data_safety is red on the working tree"
    return 1
  fi

  local br local_sha remote_sha refused=0
  while read -r br; do
    case "$br" in
      entire/*) continue ;;
      "$INTEGRATION_BRANCH")
        [ "$DIVERGED" = "1" ] && continue ;;
    esac
    # Before the name is an argument to anything. An unsafe name in this repo is
    # never a mistake, so it stops that branch loudly instead of being pushed.
    if ! ds_ref_name_ok "$br"; then
      log "REFUSING a branch with an unsafe name -- it never reaches a git argument list"
      journal "${TURN:-?}" "${TYPE:-?}" "push refused: a ref name failed ds_ref_name_ok"
      refused=1
      continue
    fi
    # Resolved ONCE, to a commit, and everything downstream uses the sha. That is
    # both the option-shape defence and the fix for scanning a name and then
    # pushing whatever that name points at by the time the push runs.
    local_sha=$(git -C "$ROOT" rev-parse --verify --quiet --end-of-options "$br^{commit}") || continue
    # Skip what the remote already has. Fewer pushes is not just faster: a branch
    # the loop never touched has no business being re-offered to a shared repo
    # every twenty minutes -- and it saves a scan we already did last turn.
    remote_sha=$(git -C "$ROOT" rev-parse --verify --quiet --end-of-options "$REMOTE/$br^{commit}" || echo "")
    [ "$local_sha" = "$remote_sha" ] && continue

    if ! ds_scan_ref "$local_sha" "$br"; then
      log "DATA SAFETY RED or UNRUNNABLE on $br -- refusing that branch ($LOGDIR/data-safety-${br//\//-}.log)"
      journal "${TURN:-?}" "${TYPE:-?}" "push of $br refused: tests.test_data_safety is red on its own tree"
      refused=1
      continue
    fi
    git -C "$ROOT" push -q "$REMOTE" -- "$local_sha:refs/heads/$br" 2>/dev/null \
      || log "WARN: push of $br failed"
  done < <(git -C "$ROOT" for-each-ref --format='%(refname:short)' refs/heads)

  [ "$refused" = "1" ] && return 1
  return 0
}
