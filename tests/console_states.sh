#!/bin/bash
# End-to-end check of router/console.html in a real browser.
#
# WHAT IT CHECKS
#     That every control actually wires through to the pipeline, and that the
#     BROWSER path lands on the same numbers results/frontier.csv holds. The
#     console is the only place where the pipeline's figures get re-plotted, so
#     "the chart agrees with the CSV" is a claim that needs a check, not a look.
#
# HOW, AND WHY NOT A DRIVER
#     chrome loads a URL, runs the page for --virtual-time-budget, dumps the DOM,
#     and we grep it. No CDP, no playwright: the console takes its whole initial
#     state from the query string, so a URL is a complete test fixture. That also
#     makes every case here something a person can open and look at.
#
# USAGE
#     make console            # in one terminal (or: python -m router.app)
#     bash tests/console_states.sh http://127.0.0.1:8765
#
# REQUIREMENTS
#     A chromium binary. Set CHROME=... if yours is elsewhere.
CHROME=${CHROME:-$(command -v chromium || command -v chromium-browser || \
        echo ~/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome)}
BASE=${1:-http://127.0.0.1:8765}

if [ ! -x "$CHROME" ]; then
  echo "no chromium at $CHROME — set CHROME=/path/to/chromium" >&2
  exit 2
fi
if ! curl -fsS -o /dev/null "$BASE/api/bootstrap"; then
  echo "no console answering at $BASE — start it with \`make console\`" >&2
  exit 2
fi
OUT=$(mktemp -d)
fails=0

dump() {  # dump <name> <query>
  timeout 90 "$CHROME" --headless --disable-gpu --no-sandbox --virtual-time-budget=25000 \
    --dump-dom "$BASE/$2" > "$OUT/$1.html" 2>/dev/null
}

want() {  # want <name> <label> <pattern>
  if grep -qF -- "$3" "$OUT/$1.html"; then
    echo "  [PASS] $2"
  else
    echo "  [FAIL] $2  (pattern not in DOM: $3)"
    fails=$((fails+1))
  fi
}

deny() {  # deny <name> <label> <pattern>
  if grep -qF -- "$3" "$OUT/$1.html"; then
    echo "  [FAIL] $2  (pattern SHOULD NOT be in DOM: $3)"
    fails=$((fails+1))
  else
    echo "  [PASS] $2"
  fi
}

echo "default state"
dump default ""
want default "chart drawn"                     "<svg"
want default "balanced metric on the axis"     "badness = 0.4·err_any"
want default "frontier table filled"           "alles zulässige → claude-fable-5"
want default "live demo rendered"              "Wahrscheinlichkeit je Modell"
want default "gate chain rendered"             "tau-Gate"
want default "caveats rendered"                "Der Export hat kein"

echo
# The whole point of this state: the browser path must land on the SAME numbers
# results/frontier.csv holds. n_boot=2000 is the published setting; at the
# interactive 400 the bounds move in the second decimal, which is correct
# arithmetic but not the figure the report quotes.
echo "?preset=repo_baseline&n_boot=2000 — must reproduce results/frontier.csv"
dump baseline "?preset=repo_baseline&n_boot=2000"
want baseline "axis switches to the err_any corner" "badness = 1·err_any"
want baseline "published logged cost"               "413&nbsp;\$"
want baseline "published all→fable cost"            "78&nbsp;\$"
want baseline "published gated-router cost"         "375&nbsp;\$"
want baseline "published all→fable bound"           "+6,65 pp"
want baseline "published gated-router bound"        "+0,76 pp"
want baseline "published rerouted count"            ">119<"

echo
echo "?lane=gpt&example=Slack — gate 4 finds no sign-stable target"
dump gpt "?lane=gpt&example=Slack"
want gpt "gpt arms in the distribution"  "gpt-5.6-"
deny gpt "no claude arm leaks in"        "claude-opus-5</div>"
want gpt "gate 4 refuses"                "kein Ziel"
want gpt "lane note explains the refusal" "bepreisen sich sol und terra"

echo
echo "?example=Bild — the vision gate blocks every reroute"
dump img "?example=Bild"
want img "vision gate blocks"     ">gesperrt<"
want img "image feature is rare"  "seltene"

echo
echo "?preset=severity&alpha=0.2 — knobs move together"
dump sev "?preset=severity&alpha=0.2"
want sev "severity metric on the axis" "0.55·trailing"
want sev "alpha reached the router"    "alpha=0.2"

echo
echo "DOM dumps kept in $OUT"
echo "failures: $fails"
echo
echo "Token counts are ESTIMATES; dollars are input-side only under an ASSUMED sheet."
exit $((fails > 0))
