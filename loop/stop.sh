#!/usr/bin/env bash
# Ask the loop to stop after the turn that is currently running.
#
# This is deliberately cooperative rather than a kill: a turn that is halfway
# through a merge should be allowed to finish, otherwise the morning starts with
# a half-merged branch. Use --now only if you actually need the process dead.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

touch "$ROOT/.loop-stop"
echo "kill switch set: $ROOT/.loop-stop"
echo "the loop exits after the current turn finishes."

if [ "${1:-}" = "--now" ]; then
  pkill -f 'loop/run.sh' 2>/dev/null && echo "supervisor killed."
  pkill -f 'claude -p'   2>/dev/null && echo "running turn killed."
fi

echo "to restart later: rm $ROOT/.loop-stop && nohup loop/run.sh > loop/logs/supervisor.log 2>&1 &"
