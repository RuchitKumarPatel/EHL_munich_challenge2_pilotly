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

# The suite, against the working tree. This is the only scan that sees untracked
# files and the generated artifacts -- neither is tracked, so neither can travel
# through a ref, but both are how a turn notices it has made a mess.
ds_scan_worktree() {
  mkdir -p "$LOGDIR"
  ( cd "$ROOT" && "$PY" -m unittest tests.test_data_safety ) \
    > "$LOGDIR/data-safety.log" 2>&1
}

# The suite, against one ref's tree. DATA_SAFETY_SCAN_REF makes the repo-file
# checks enumerate `git ls-tree` instead of `git ls-files`; see the constant's
# docstring in tests/test_data_safety.py.
ds_scan_ref() {
  local ref="$1" slug
  slug="${1//\//-}"
  mkdir -p "$LOGDIR"
  ( cd "$ROOT" && DATA_SAFETY_SCAN_REF="$ref" "$PY" -m unittest tests.test_data_safety ) \
    > "$LOGDIR/data-safety-$slug.log" 2>&1
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
    # Skip what the remote already has. Fewer pushes is not just faster: a branch
    # the loop never touched has no business being re-offered to a shared repo
    # every twenty minutes -- and it saves a scan we already did last turn.
    local_sha=$(git -C "$ROOT" rev-parse "$br" 2>/dev/null) || continue
    remote_sha=$(git -C "$ROOT" rev-parse "$REMOTE/$br" 2>/dev/null || echo "")
    [ "$local_sha" = "$remote_sha" ] && continue

    if ! ds_scan_ref "$br"; then
      log "DATA SAFETY RED on $br -- refusing that branch ($LOGDIR/data-safety-${br//\//-}.log)"
      journal "${TURN:-?}" "${TYPE:-?}" "push of $br refused: tests.test_data_safety is red on its own tree"
      refused=1
      continue
    fi
    git -C "$ROOT" push -q "$REMOTE" "$br" 2>/dev/null || log "WARN: push of $br failed"
  done < <(git -C "$ROOT" for-each-ref --format='%(refname:short)' refs/heads)

  [ "$refused" = "1" ] && return 1
  return 0
}
