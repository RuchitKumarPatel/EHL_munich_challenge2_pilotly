#!/usr/bin/env bash
# What the loop has done so far. Safe to run while it is running.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
PY="$ROOT/.venv/bin/python"

if pgrep -f 'loop/run.sh' > /dev/null; then
  echo "supervisor: RUNNING (pid $(pgrep -f 'loop/run.sh' | tr '\n' ' '))"
else
  echo "supervisor: not running"
fi
[ -f "$ROOT/.loop-stop" ] && echo "kill switch: SET (will stop after this turn)"

echo
"$PY" "$ROOT/loop/backlog.py" stats

echo
echo "--- branches ---"
git branch --format='%(refname:short)  %(objectname:short)  %(contents:subject)' | sed 's/^/  /'

echo
echo "--- last 10 journal lines ---"
tail -n 10 "$ROOT/loop/JOURNAL.md" 2>/dev/null | sed 's/^/  /'

echo
echo "--- last 5 supervisor lines ---"
tail -n 5 "$ROOT/loop/logs/supervisor.log" 2>/dev/null | sed 's/^/  /'
