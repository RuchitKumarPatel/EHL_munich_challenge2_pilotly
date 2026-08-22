#!/usr/bin/env bash
# Overnight loop supervisor.
#
# One iteration = one fresh `claude -p` process. Nothing is carried in context
# between iterations; everything the next turn needs is on disk (loop/state/,
# git, and entire's checkpoints). That is the whole point: a turn can die, hang
# or go off the rails without taking the night with it.
#
#   start:  nohup loop/run.sh > loop/logs/supervisor.log 2>&1 &
#   stop:   loop/stop.sh
#   watch:  loop/status.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

# shellcheck source=/dev/null
source "$ROOT/loop/config.env"

PY="$ROOT/.venv/bin/python"
BACKLOG=("$PY" "$ROOT/loop/backlog.py")
STOPFILE="$ROOT/.loop-stop"
LOGDIR="$ROOT/loop/logs"
JOURNAL="$ROOT/loop/JOURNAL.md"
STATEFILE="$ROOT/loop/state/state.json"
mkdir -p "$LOGDIR"

START_EPOCH=$(date +%s)
DEADLINE_EPOCH=$(( START_EPOCH + LOOP_HOURS * 3600 ))

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

state_field() { "$PY" -c "import json;print(json.load(open('$STATEFILE')).get('$1',$2))" 2>/dev/null || echo "$2"; }

# This account runs on an OAuth subscription with extra usage disabled, so hitting the limit is
# a hard stop, not an overage charge. The CLI keeps its own utilization snapshot and refreshes
# it on every call, which makes it the cheapest available read on where we stand.
usage_pct() {   # $1 = five_hour | seven_day
  "$PY" -c "
import json,os
try:
    d=json.load(open(os.path.expanduser('~/.claude.json')))
    w=((d.get('cachedUsageUtilization') or {}).get('utilization') or {}).get('$1') or {}
    print(int(w.get('utilization') or 0))
except Exception:
    print(0)
" 2>/dev/null || echo 0
}

quota_reset_epoch() {
  "$PY" -c "
import json,os,datetime
try:
    d=json.load(open(os.path.expanduser('~/.claude.json')))
    w=((d.get('cachedUsageUtilization') or {}).get('utilization') or {}).get('five_hour') or {}
    r=w.get('resets_at')
    print(int(datetime.datetime.fromisoformat(r).timestamp()) if r else 0)
except Exception:
    print(0)
" 2>/dev/null || echo 0
}

# A turn's own result text can mention rate limits without having hit one, so the failure has to
# come first: only a run that actually errored is a candidate.
is_rate_limited() {   # $1 = result json, $2 = stderr
  local failed
  failed=$("$PY" -c "
import json
try:
    d=json.load(open('$1'))
    print('1' if d.get('is_error') or d.get('subtype') != 'success' else '0')
except Exception:
    print('1')
" 2>/dev/null || echo 1)
  [ "$failed" = "1" ] || return 1
  grep -qiE 'usage limit|rate limit|limit reached|quota exceeded|429' "$1" "$2" 2>/dev/null
}

# Wait out a five-hour window rather than burning the stall detector on it. Polls in short
# chunks so the kill switch and the wall clock still win during a long wait.
sleep_for_quota() {
  local reset now wait slept
  reset=$(quota_reset_epoch); now=$(date +%s)
  if [ "$reset" -gt "$now" ]; then wait=$(( reset - now + 60 )); else wait=900; fi
  [ "$wait" -gt 7200 ] && wait=7200
  log "rate limited - waiting ${wait}s, until $(date -d "@$(( now + wait ))" '+%H:%M')"
  slept=0
  while [ "$slept" -lt "$wait" ]; do
    if [ -f "$STOPFILE" ]; then log "kill switch hit during quota wait"; return 1; fi
    if [ "$(date +%s)" -ge "$DEADLINE_EPOCH" ]; then log "wall clock hit during quota wait"; return 1; fi
    sleep 60; slept=$(( slept + 60 ))
  done
  return 0
}

journal() {
  # One line per turn, committed with the turn. This is the artifact a human
  # reads in the morning to reconstruct the night without opening any log.
  printf -- '- **turn %s** · `%s` · %s · %s\n' "$1" "$2" "$(date '+%H:%M')" "$3" >> "$JOURNAL"
}

# Never lose work across a branch switch. The supervisor runs outside Claude, so
# the repo's PreToolUse git guard does not apply to it -- but export/ and results/
# are gitignored, so a broad `add` cannot reach the proprietary data either way.
commit_stragglers() {
  if ! git diff --quiet HEAD 2>/dev/null || [ -n "$(git ls-files --others --exclude-standard)" ]; then
    git add -A >/dev/null 2>&1
    if git commit -q -m "loop: carry forward uncommitted work before turn $1" >/dev/null 2>&1; then
      log "committed stragglers on $(git rev-parse --abbrev-ref HEAD)"
    fi
  fi
}

# Returns 0 when the integration branch is safe to build on, 1 when a teammate
# has pushed and the branches have genuinely diverged. On divergence the loop
# does NOT auto-resolve: it keeps working on idea branches and leaves main alone.
sync_integration_branch() {
  [ "$PUSH" = "1" ] || return 0
  if ! git fetch -q "$REMOTE" "$INTEGRATION_BRANCH" 2>/dev/null; then
    log "WARN: fetch failed -- working offline this turn"
    return 0
  fi
  local local_ref remote_ref base
  local_ref=$(git rev-parse "$INTEGRATION_BRANCH")
  remote_ref=$(git rev-parse "$REMOTE/$INTEGRATION_BRANCH" 2>/dev/null) || return 0
  [ "$local_ref" = "$remote_ref" ] && return 0
  base=$(git merge-base "$INTEGRATION_BRANCH" "$REMOTE/$INTEGRATION_BRANCH")
  if [ "$base" = "$local_ref" ]; then
    log "$INTEGRATION_BRANCH is behind $REMOTE -- fast-forwarding"
    # `git branch -f` refuses to move a branch that is currently checked out, and the previous
    # turn usually leaves us standing on exactly this one. Pick the form that fits where we are.
    if [ "$(git rev-parse --abbrev-ref HEAD)" = "$INTEGRATION_BRANCH" ]; then
      git merge --ff-only "$REMOTE/$INTEGRATION_BRANCH" >/dev/null 2>&1 \
        || log "WARN: fast-forward of $INTEGRATION_BRANCH failed"
    else
      git branch -f "$INTEGRATION_BRANCH" "$REMOTE/$INTEGRATION_BRANCH" 2>/dev/null \
        || log "WARN: could not move $INTEGRATION_BRANCH"
    fi
    return 0
  fi
  if [ "$base" = "$remote_ref" ]; then
    return 0   # we are ahead; a merge turn will push
  fi
  log "DIVERGED: $INTEGRATION_BRANCH and $REMOTE/$INTEGRATION_BRANCH both moved. Not touching main."
  return 1
}

# The gate that has to hold before anything leaves this machine. The merge gate protects `main`;
# until this existed, nothing protected the branches -- and on the loop's first night a doc
# quoting three raw identifiers out of export/ reached origin exactly that way, because bootstrap
# pushed every branch before any check ran. The dataset is challenge-use-only: a push is
# redistribution, and it cannot be taken back.
data_safety_ok() {
  "$PY" -m unittest tests.test_data_safety > "$LOGDIR/data-safety.log" 2>&1
}

# Every branch goes to the remote every turn, so nothing the night produces exists only on this
# laptop -- including the branches whose ideas were rejected. Two exceptions: `main` is skipped
# while diverged, because pushing a diverged integration branch is how a shared repository gets
# damaged, and entire/* are the CLI's own checkpoint refs, which it pushes itself.
push_all_branches() {
  [ "$PUSH" = "1" ] || return 0

  if ! data_safety_ok; then
    log "DATA SAFETY RED -- pushing nothing this turn (loop/logs/data-safety.log)"
    journal "$TURN" "$TYPE" "push suppressed: tests.test_data_safety is red on the working tree"
    return 1
  fi

  local br local_sha remote_sha
  while read -r br; do
    case "$br" in
      entire/*) continue ;;
      "$INTEGRATION_BRANCH")
        [ "$DIVERGED" = "1" ] && continue ;;
    esac
    # Skip what the remote already has. Fewer pushes is not just faster: a branch the loop never
    # touched has no business being re-offered to a shared repo every twenty minutes.
    local_sha=$(git rev-parse "$br" 2>/dev/null) || continue
    remote_sha=$(git rev-parse "$REMOTE/$br" 2>/dev/null || echo "")
    [ "$local_sha" = "$remote_sha" ] && continue
    git push -q "$REMOTE" "$br" 2>/dev/null || log "WARN: push of $br failed"
  done < <(git for-each-ref --format='%(refname:short)' refs/heads)
}

stop_reason() {
  if [ -f "$STOPFILE" ]; then echo "kill switch $STOPFILE"; return 0; fi
  if [ "$(date +%s)" -ge "$DEADLINE_EPOCH" ]; then echo "wall clock ($LOOP_HOURS h)"; return 0; fi
  local turn spent
  turn=$(state_field turn 0)
  if [ "$turn" -ge "$LOOP_MAX_TURNS" ]; then echo "max turns ($LOOP_MAX_TURNS)"; return 0; fi
  spent=$(state_field cost_usd 0)
  if "$PY" -c "import sys;sys.exit(0 if float('$spent') >= float('$LOOP_BUDGET_USD') else 1)"; then
    echo "effort ceiling (notional \$$spent >= \$$LOOP_BUDGET_USD)"; return 0
  fi
  local week
  week=$(usage_pct seven_day)
  if [ "$week" -ge "$LOOP_MAX_SEVEN_DAY_PCT" ]; then
    echo "weekly quota ($week% >= $LOOP_MAX_SEVEN_DAY_PCT%) — leaving you the rest for the defense"
    return 0
  fi
  return 1
}

log "loop starting - deadline $(date -d "@$DEADLINE_EPOCH" '+%H:%M'), max $LOOP_MAX_TURNS turns, budget \$$LOOP_BUDGET_USD"
[ -f "$JOURNAL" ] || printf '# JOURNAL - one line per loop turn\n\n' > "$JOURNAL"

while true; do
  if reason=$(stop_reason); then
    log "stopping: $reason"
    journal "$(state_field turn 0)" "stop" "stopped: $reason"
    break
  fi

  TURN=$("${BACKLOG[@]}" turn --bump)
  commit_stragglers "$TURN"

  DIVERGED=0
  sync_integration_branch || DIVERGED=1
  if [ "$DIVERGED" = "0" ]; then
    git checkout -q "$INTEGRATION_BRANCH" 2>/dev/null || log "WARN: could not check out $INTEGRATION_BRANCH"
  fi

  TYPE=$("${BACKLOG[@]}" plan --deck-every "$DECK_EVERY" --review-every "$REVIEW_EVERY")
  # A diverged main makes merging unsafe; keep generating and evaluating instead.
  if [ "$DIVERGED" = "1" ] && [ "$TYPE" = "merge" ]; then
    log "turn $TURN: merge suppressed (main diverged) -> evaluate"
    TYPE="evaluate"
  fi
  "${BACKLOG[@]}" turn --type "$TYPE" >/dev/null

  case "$TYPE" in
    council)   TIMEOUT=$TIMEOUT_COUNCIL;   EFFORT=$EFFORT_COUNCIL;   ULTRA=$ULTRACODE_COUNCIL   ;;
    evaluate)  TIMEOUT=$TIMEOUT_EVALUATE;  EFFORT=$EFFORT_EVALUATE;  ULTRA=$ULTRACODE_EVALUATE  ;;
    implement) TIMEOUT=$TIMEOUT_IMPLEMENT; EFFORT=$EFFORT_IMPLEMENT; ULTRA=$ULTRACODE_IMPLEMENT ;;
    merge)     TIMEOUT=$TIMEOUT_MERGE;     EFFORT=$EFFORT_MERGE;     ULTRA=$ULTRACODE_MERGE     ;;
    deck)      TIMEOUT=$TIMEOUT_DECK;      EFFORT=$EFFORT_DECK;      ULTRA=$ULTRACODE_DECK      ;;
    review)    TIMEOUT=$TIMEOUT_REVIEW;    EFFORT=$EFFORT_REVIEW;    ULTRA=$ULTRACODE_REVIEW    ;;
    *)         log "unknown turn type '$TYPE' -- stopping"; break                               ;;
  esac

  EXTRA_ARGS=()
  if [ "$ULTRA" = "1" ]; then EXTRA_ARGS+=(--settings '{"ultracode":true}'); fi

  PROMPT_FILE="$ROOT/loop/prompts/$TYPE.md"
  if [ ! -f "$PROMPT_FILE" ]; then log "missing prompt $PROMPT_FILE -- stopping"; break; fi

  STEM="$LOGDIR/turn-$(printf '%03d' "$TURN")-$TYPE"
  log "turn $TURN: $TYPE (effort=$EFFORT$([ "$ULTRA" = "1" ] && printf ', ultracode'), timeout=${TIMEOUT}s) on $(git rev-parse --abbrev-ref HEAD) [5h $(usage_pct five_hour)%, 7d $(usage_pct seven_day)%]"

  # The prompt is the turn type's file plus a small live-state header, so a fresh
  # process knows the turn number and the queue without being told in context.
  {
    printf '# LOOP TURN %s - type: %s\n\n' "$TURN" "$TYPE"
    printf 'Team name: %s\nTeam members: %s\n\n' "${TEAM_NAME:-[FILL]}" "${TEAM_MEMBERS:-[FILL]}"
    printf 'Current backlog state:\n\n```json\n'
    "${BACKLOG[@]}" list --json
    printf '```\n\n'
    cat "$ROOT/loop/prompts/_common.md"
    printf '\n'
    cat "$PROMPT_FILE"
  } > "$STEM.prompt.md"

  # Hitting the five-hour window is expected on a subscription, not an error. Wait it out and
  # retry the same turn rather than letting three limited turns trip the stall detector and end
  # the night at 02:00. The turn counter does not advance across a retry.
  ATTEMPT=0
  while : ; do
    # INT first so the process can flush its result json; SIGKILL a minute later if it will not
    # go, otherwise one wedged turn holds the night open indefinitely.
    timeout --signal=INT --kill-after=60 "$TIMEOUT" \
      claude -p "$(cat "$STEM.prompt.md")" \
        --permission-mode bypassPermissions \
        --output-format json \
        --model "$MODEL" \
        --effort "$EFFORT" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
        < /dev/null > "$STEM.json" 2>"$STEM.err"
    RC=$?

    COST=$("$PY" -c "
import json
try:
    print(json.load(open('$STEM.json')).get('total_cost_usd') or 0.0)
except Exception:
    print(0.0)
" 2>/dev/null || echo 0.0)
    "${BACKLOG[@]}" cost --add "$COST" >/dev/null

    if [ "$RC" -ne 0 ] && [ "$ATTEMPT" -lt 3 ] && is_rate_limited "$STEM.json" "$STEM.err"; then
      ATTEMPT=$(( ATTEMPT + 1 ))
      journal "$TURN" "$TYPE" "rate limited, attempt $ATTEMPT — waiting for the window to reset"
      sleep_for_quota || break
      continue
    fi
    break
  done

  if [ "$RC" -eq 124 ] || [ "$RC" -eq 130 ]; then
    log "turn $TURN TIMED OUT after ${TIMEOUT}s"
    journal "$TURN" "$TYPE" "TIMED OUT after ${TIMEOUT}s (\$$COST)"
  elif [ "$RC" -ne 0 ]; then
    log "turn $TURN exited rc=$RC"
    journal "$TURN" "$TYPE" "exited rc=$RC (\$$COST) - see loop/logs/"
  else
    journal "$TURN" "$TYPE" "ok (\$$COST)"
  fi

  # Safety net: whatever the turn did or failed to do, the bookkeeping lands.
  "${BACKLOG[@]}" render >/dev/null
  commit_stragglers "$TURN"

  push_all_branches

  # Stall detector. Every failure mode that matters looks the same from outside: turns burn and
  # nothing moves -- a council that keeps proposing closed ideas, an evaluate turn that never
  # commits to a status, an implement turn that times out on the same idea. Rather than guard
  # each one, watch the two things that must change when work happens: the queue and the commits.
  FINGERPRINT="$("${BACKLOG[@]}" stats)|$(git rev-parse --all | md5sum)"
  if [ "$FINGERPRINT" = "${LAST_FINGERPRINT:-}" ]; then
    STALL=$(( ${STALL:-0} + 1 ))
    log "no progress this turn (stall $STALL/3)"
  else
    STALL=0
  fi
  LAST_FINGERPRINT="$FINGERPRINT"
  if [ "$STALL" -ge 3 ]; then
    log "stopping: three consecutive turns changed neither the queue nor any branch"
    journal "$TURN" "stop" "stopped: stalled — 3 turns with no queue or branch change"
    break
  fi

  log "turn $TURN done (\$$COST, total \$$(state_field cost_usd 0))"
done

log "loop finished."
"${BACKLOG[@]}" stats
