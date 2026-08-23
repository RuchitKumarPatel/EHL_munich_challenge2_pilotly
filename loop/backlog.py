#!/usr/bin/env python3
"""State store for the overnight loop.

Single source of truth is loop/state/backlog.json plus loop/state/state.json.
Everything the loop mutates goes through this CLI so a turn can never leave the
state half written: every subcommand rewrites the whole file atomically.

The turn planner lives here too (`plan`), so the supervisor shell script never
has to reason about the queue -- it just runs what this prints.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "loop" / "state"
BACKLOG = STATE_DIR / "backlog.json"
STATE = STATE_DIR / "state.json"
IDEAS_MD = ROOT / "docs" / "IDEAS.md"

# Ids are ordered by `next_id`, so an id it cannot order is an id it can reissue.
_ID_RE = re.compile(r"^i\d+$")

# Status lifecycle. `blocked` is terminal for the night: a human unblocks it.
STATUSES = (
    "proposed",     # drafted, not yet judged
    "accepted",     # judged worth building
    "rejected",     # judged not worth building -- stays as a record
    "parked",       # worth building, but not tonight (too big / needs a human)
    "implemented",  # built on its branch, gates green, not yet on main
    "merged",       # on main
    "blocked",      # tried and failed, or main diverged -- needs a human
)
OPEN_STATUSES = ("proposed", "accepted", "implemented")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload) -> None:
    """Atomic write -- a killed turn must never truncate the state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class StateError(RuntimeError):
    """A state file exists but cannot be trusted. Never a default, never silent.

    Both files under loop/state/ are rewritten by every turn agent, which runs
    with bypassPermissions. The supervisor captures this CLI's stdout by command
    substitution and used to ignore the exit code, so a traceback here became an
    empty TURN and an empty TYPE and the loop stopped reporting "unknown turn
    type ''" -- the wrong cause for the right failure. Raising a named error and
    exiting EXIT_STATE_UNREADABLE lets run.sh print what actually happened.
    """


EXIT_STATE_UNREADABLE = 3   # distinct from argparse's 2 and from a plain failure


def _load_json(path: Path, what: str):
    """Parse `path`, or raise StateError naming the file and the position."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StateError(f"cannot read the {what} file {path}: {exc}") from None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise StateError(
            f"{path} is not valid JSON ({exc.msg} at line {exc.lineno} "
            f"column {exc.colno}). The {what} is left untouched; fix or restore "
            f"the file by hand -- this CLI will not guess at it."
        ) from None


def _check_backlog(items) -> list[dict]:
    """Every field the rest of this module indexes without a `.get`.

    Validated once, here, so no caller has to be defensive and no malformed item
    can reach `_by_status` (which indexed `i["status"]`), `next_id` (which
    swallowed a non-numeric id and could then hand out one already in use) or
    `render_ideas`.
    """
    if not isinstance(items, list):
        raise StateError(
            f"{BACKLOG} must hold a JSON list of ideas, found "
            f"{type(items).__name__}")
    seen = set()
    for pos, it in enumerate(items):
        where = f"{BACKLOG} item {pos}"
        if not isinstance(it, dict):
            raise StateError(f"{where} is a {type(it).__name__}, not an object")
        ident = it.get("id")
        if not isinstance(ident, str) or not _ID_RE.match(ident):
            raise StateError(
                f"{where} has id {ident!r}; every id must look like i0001. "
                f"An id this module cannot order is an id `add` could hand out "
                f"twice.")
        if ident in seen:
            raise StateError(f"{where}: duplicate id {ident}")
        seen.add(ident)
        status = it.get("status")
        if status not in STATUSES:
            raise StateError(
                f"{ident} has status {status!r}, which is not one of "
                f"{', '.join(STATUSES)}")
        if not isinstance(it.get("title"), str):
            raise StateError(f"{ident} has no title")
        if not isinstance(it.get("notes", []), list):
            raise StateError(f"{ident} has notes that are not a list")
    return items


def load_backlog() -> list[dict]:
    if not BACKLOG.exists():
        return []
    return _check_backlog(_load_json(BACKLOG, "backlog"))


def load_state() -> dict:
    # ABSENT IS NOT CORRUPT. The first turn of a night has no state file, so a
    # missing file defaults. A file that exists and does not parse must NOT --
    # defaulting it would set cost_usd to 0, and stop_reason reads exactly that
    # field to enforce the effort ceiling, so one corrupt file would buy the
    # loop an unlimited budget. See tests/test_backlog_state.py.
    if not STATE.exists():
        return {"turn": 0, "cost_usd": 0.0, "last_turn_type": None, "started": None}
    state = _load_json(STATE, "state")
    if not isinstance(state, dict):
        raise StateError(
            f"{STATE} must hold a JSON object, found {type(state).__name__}")
    return state


def save_backlog(items: list[dict]) -> None:
    _write_json(BACKLOG, items)


def save_state(state: dict) -> None:
    _write_json(STATE, state)


def _by_status(items: list[dict], status: str) -> list[dict]:
    return [i for i in items if i["status"] == status]


# ----------------------------------------------------------------- turn planner
def _cadence_due(turn: int, every: int) -> bool:
    """Is `turn` a multiple of the cadence `every`?

    Anything <= 0 means the cadence is OFF. Both cadences go through here so
    they disable identically: `deck_every` used to be divided into `turn` with
    no guard, so DECK_EVERY=0 raised ZeroDivisionError inside the command
    substitution that run.sh does not exit-check -- the supervisor got an empty
    TYPE and logged "unknown turn type" with the real cause buried in a log.
    Turn 0 is "before the first turn" and never satisfies a cadence.
    """
    return turn > 0 and every > 0 and turn % every == 0


def _deck_due(turn: int, deck_every: int, review_every: int) -> bool:
    """Is `turn` a deck turn, counting a deck deferred by a lost tie?

    `review` outranks `deck` on a turn that satisfies both cadences. Without the
    second clause here that would be a silent DROP rather than a deferral, and
    whenever `deck_every == review_every` every deck turn is a tie -- so the
    deck would never run. The deferred deck runs on the next turn instead,
    unless that turn is itself a review turn (the caller checks review first).
    """
    if _cadence_due(turn, deck_every):
        return True
    lost_a_tie_last_turn = (_cadence_due(turn - 1, deck_every)
                            and _cadence_due(turn - 1, review_every))
    return lost_a_tie_last_turn


def plan_turn(items: list[dict], turn: int, deck_every: int, review_every: int = 0) -> str:
    """Which turn type runs next. Pure function of the queue -- no model discretion.

    Order is load-bearing:
      * `review` comes first -- a quality and security pass that only runs when
        the queue happens to be empty is a pass that never runs on the night the
        loop is busiest, which is exactly the night it matters. It reads its own
        scope from `last_review_sha`, so a review turn with nothing new to look
        at costs almost nothing;
      * the deck check comes next so the slide set is never far out of date,
        whatever the queue is doing;
      * `merge` outranks `implement` so main keeps moving and the next idea
        branches off a current main (which keeps every merge a fast-forward);
      * `council` fires only when the queue is dry, because it is by far the
        most expensive turn type.

    REVIEW WINS A TIE AND THE DECK IS DEFERRED, NOT DROPPED. That is the fix for
    a measured bug rather than a preference. The deck check used to come first,
    so a turn matching both cadences went to `deck` and review waited for its
    next multiple -- which, whenever `deck_every` divides `review_every`, is
    also a deck turn. Over turns 1..40 that starved review completely for
    (5,10), (4,4), (5,5) and (3,6); the shipped (5,4) fired, but lost turn 20.
    A stale deck costs one turn and a skipped review costs the night, so the tie
    goes to review -- but simply swapping the two checks moves the starvation
    onto the deck instead (measured: with (4,4) and (5,5) the deck then never
    fires at all). `_deck_due` therefore also fires on the turn AFTER a lost
    tie, which bounds deck staleness at `deck_every + 1` turns for every pair.
    tests/test_backlog_plan.py pins both halves of that trade.
    """
    if _cadence_due(turn, review_every):
        return "review"
    if _deck_due(turn, deck_every, review_every):
        return "deck"
    if _by_status(items, "implemented"):
        return "merge"
    if _by_status(items, "accepted"):
        return "implement"
    if _by_status(items, "proposed"):
        return "evaluate"
    return "council"


def next_id(items: list[dict]) -> str:
    """One past the highest id present.

    No `try/continue` here any more: an id this cannot order used to be skipped,
    and the next id was then i0001 -- the id such a file is most likely to
    already hold. `_check_backlog` rejects the file instead, so by the time we
    get here every id matches `_ID_RE`.
    """
    n = max((int(i["id"][1:]) for i in items), default=0)
    return f"i{n + 1:04d}"


# -------------------------------------------------------------------- rendering
def render_ideas(items: list[dict], state: dict) -> str:
    lines = [
        "# IDEAS — the overnight loop's backlog",
        "",
        "Generated by `loop/backlog.py render`. Do not edit by hand; edit the JSON via the CLI.",
        "",
        f"Turn: **{state.get('turn', 0)}** · "
        f"spend so far: **${state.get('cost_usd', 0.0):.2f}** · "
        f"last turn type: **{state.get('last_turn_type') or '—'}**",
        "",
    ]
    for status in STATUSES:
        group = _by_status(items, status)
        if not group:
            continue
        lines.append(f"## {status} ({len(group)})")
        lines.append("")
        for it in group:
            branch = f" · `{it['branch']}`" if it.get("branch") else ""
            lines.append(f"### {it['id']} — {it['title']}{branch}")
            lines.append("")
            lines.append(f"*{it.get('kind', 'unspecified')}* — {it.get('why', '')}")
            lines.append("")
            for note in it.get("notes", []):
                lines.append(f"- turn {note['turn']}: {note['text']}")
            if it.get("notes"):
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# -------------------------------------------------------------------- subcommands
def cmd_plan(args) -> int:
    print(plan_turn(load_backlog(), load_state().get("turn", 0),
                    args.deck_every, args.review_every))
    return 0


def cmd_mark_review(args) -> int:
    """Record how far the last quality/security pass got, so the next one can
    scope itself to what landed since instead of re-reading the whole tree."""
    state = load_state()
    state["last_review_sha"] = args.sha
    state["last_review_turn"] = state.get("turn", 0)
    save_state(state)
    print(args.sha)
    return 0


def cmd_add(args) -> int:
    items = load_backlog()
    state = load_state()
    idea = {
        "id": next_id(items),
        "title": args.title,
        "kind": args.kind,
        "why": args.why,
        "status": "proposed",
        "branch": None,
        "created_turn": state.get("turn", 0),
        "notes": [],
    }
    items.append(idea)
    save_backlog(items)
    print(idea["id"])
    return 0


def cmd_set(args) -> int:
    items = load_backlog()
    turn = load_state().get("turn", 0)
    for it in items:
        if it["id"] != args.id:
            continue
        if args.status:
            it["status"] = args.status
        if args.branch:
            it["branch"] = args.branch
        if args.note:
            it.setdefault("notes", []).append({"turn": turn, "text": args.note})
        save_backlog(items)
        print(f"{it['id']} -> {it['status']}")
        return 0
    print(f"no such id: {args.id}")
    return 1


def cmd_list(args) -> int:
    items = load_backlog()
    if args.status:
        items = _by_status(items, args.status)
    if args.json:
        print(json.dumps(items, indent=2, ensure_ascii=False))
        return 0
    if not items:
        print("(empty)")
        return 0
    for it in items:
        print(f"{it['id']}  {it['status']:<12} {it.get('kind', ''):<14} {it['title']}")
    return 0


def cmd_render(args) -> int:
    items, state = load_backlog(), load_state()
    IDEAS_MD.parent.mkdir(parents=True, exist_ok=True)
    IDEAS_MD.write_text(render_ideas(items, state), encoding="utf-8")
    print(str(IDEAS_MD.relative_to(ROOT)))
    return 0


def cmd_turn(args) -> int:
    state = load_state()
    if args.bump:
        state["turn"] = state.get("turn", 0) + 1
    if args.type:
        state["last_turn_type"] = args.type
    if state.get("started") is None:
        state["started"] = _now()
    state["updated"] = _now()
    save_state(state)
    print(state["turn"])
    return 0


def cmd_cost(args) -> int:
    state = load_state()
    state["cost_usd"] = round(state.get("cost_usd", 0.0) + args.add, 4)
    save_state(state)
    print(f"{state['cost_usd']:.4f}")
    return 0


def cmd_stats(args) -> int:
    items, state = load_backlog(), load_state()
    counts = {s: len(_by_status(items, s)) for s in STATUSES if _by_status(items, s)}
    print(json.dumps({"turn": state.get("turn", 0),
                      "cost_usd": state.get("cost_usd", 0.0),
                      "open": sum(len(_by_status(items, s)) for s in OPEN_STATUSES),
                      "counts": counts}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("plan", help="print the turn type that should run next")
    sp.add_argument("--deck-every", type=int, default=5)
    sp.add_argument("--review-every", type=int, default=0,
                    help="0 disables the periodic quality/security review turn")
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("mark-review", help="record the sha the last review covered")
    sp.add_argument("--sha", required=True)
    sp.set_defaults(func=cmd_mark_review)

    sp = sub.add_parser("add", help="add a proposed idea")
    sp.add_argument("--title", required=True)
    sp.add_argument("--why", required=True)
    sp.add_argument("--kind", default="analysis",
                    choices=["analysis", "code", "presentation", "infra", "evaluation"])
    sp.set_defaults(func=cmd_add)

    sp = sub.add_parser("set", help="change an idea's status / branch, or append a note")
    sp.add_argument("--id", required=True)
    sp.add_argument("--status", choices=STATUSES)
    sp.add_argument("--branch")
    sp.add_argument("--note")
    sp.set_defaults(func=cmd_set)

    sp = sub.add_parser("list")
    sp.add_argument("--status", choices=STATUSES)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("render", help="rewrite docs/IDEAS.md from the JSON")
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser("turn", help="bump the turn counter / record the turn type")
    sp.add_argument("--bump", action="store_true")
    sp.add_argument("--type")
    sp.set_defaults(func=cmd_turn)

    sp = sub.add_parser("cost", help="accumulate spend")
    sp.add_argument("--add", type=float, required=True)
    sp.set_defaults(func=cmd_cost)

    sp = sub.add_parser("stats")
    sp.set_defaults(func=cmd_stats)

    args = p.parse_args()
    try:
        return args.func(args)
    except StateError as exc:
        # Stderr, and NOTHING on stdout: run.sh captures stdout as a scalar and
        # cannot tell an empty answer from an absent one. See ADR-019.
        print(f"backlog.py: {exc}", file=sys.stderr)
        return EXIT_STATE_UNREADABLE


if __name__ == "__main__":
    raise SystemExit(main())
