# Viktor Challenge — router pipeline.
#
# Every target runs a module of the `router` package with the project venv.
# `make all` runs the whole pipeline in dependency order; every other target is
# standalone so you can rerun one stage without the ones before it.
#
# Artifact contracts and the pinned acceptance numbers live in docs/CONTRACTS.md.
# Read that before changing any field name here or in router/.

PY      := .venv/bin/python
RESULTS := results

# Prerequisites of `all` must run left to right, never in parallel.
.NOTPARALLEL:
.DEFAULT_GOAL := all

.PHONY: all recon labels jobkey features costs model policy gates strata ope \
        family figs sweep report deck console console-check console-e2e test demo clean help dirs \
        tooling verify

## all: run the full pipeline in dependency order
#
# Order is load-bearing and is NOT the order the targets are declared in below:
#   * `policy` writes results/routes.jsonl, which `gates` reads, so gates runs
#     AFTER policy (an earlier revision of this file ran gates before model and
#     the gate could only ever report BLOCKED).
#   * `strata` and `family` each merge one section into results/estimates.json
#     and `report` refuses to run without the "family_contrast" section, so both
#     are part of `all`, not optional extras.
all: dirs recon labels jobkey features costs tooling model policy gates strata ope \
     family figs sweep report deck
	@echo ""
	@echo "pipeline complete -> $(RESULTS)/"

dirs:
	@mkdir -p $(RESULTS)

# ---------------------------------------------------------------- stage 1: facts
## recon: reconstruct turns and the cache split   -> results/recon.jsonl
recon: dirs
	$(PY) -m router.recon

## labels: tool-output outcomes and friction labels -> results/labels.jsonl
labels: dirs
	$(PY) -m router.labels

## jobkey: literal cron-path job key               -> results/jobkey.jsonl
jobkey: dirs
	$(PY) -m router.jobkey

# ------------------------------------------------- stage 2: pre-treatment design
## features: pre-treatment feature matrix -> results/features.npz + feature_manifest.json
features: dirs
	$(PY) -m router.features

## costs: cache-aware price model and effective multipliers
costs: dirs
	$(PY) -m router.costs

## tooling: tool-block interventions (omit/truncate/defer) -> results/tooling.json
#
# Reads the export directly for per-tool definitions (recon only carries the
# block TOTAL), plus recon.jsonl for the prefix split and jobkey.jsonl for the
# leak-free job-history policy. Runs after both.
tooling: dirs
	$(PY) -m router.tooling

# ---------------------------------------------------------------- stage 3: policy
## model: friction predictor fit on pre-treatment features only
model: dirs
	$(PY) -m router.model

## policy: five-gate chain + conformal tau       -> results/routes.jsonl
policy: dirs
	$(PY) -m router.policy

## gates: publication gate — six negative controls (reads routes.jsonl)
gates: dirs
	$(PY) -m router.gates

## verify: prose gate — every numeral in a user-facing artifact against claims.json
#
# NOT part of `all`. `all` rebuilds the claims table, and a deck half-way
# through an edit would wedge the pipeline for the wrong reason. The contract is
# enforced by tests/test_verify.py instead, so `make test` is the gate and this
# target is the readable report of what bound to what.
verify: dirs
	$(PY) -m router.verify

# ------------------------------------------------------------- stage 4: evidence
## strata: stratum table + support deficit       -> estimates.json["strata"]
strata: dirs
	$(PY) -m router.strata

## ope: off-policy evaluation — bounds, never point estimates
ope: dirs
	$(PY) -m router.ope

## family: cross-family bundle contrast          -> estimates.json["family_contrast"]
family: dirs
	$(PY) -m router.family_contrast

## figs: figures, including the cost-quality frontier
figs: dirs
	$(PY) -m router.figs

## sweep: the conformal knob swept -> results/sweep.json + sweep.png
#
# Re-runs the whole gate chain at eleven alphas and re-prices each result through
# figs.policy_point, so a swept point and a plotted point are the same
# computation. Runs AFTER figs (it recomputes the hull the chart draws) and
# BEFORE report (report refuses without results/sweep.json).
sweep: dirs
	$(PY) -m router.sweep

## report: metrics.json + claims.json + NUMBERS.md
report: dirs
	$(PY) -m router.report

## deck: regenerate presentation.html from claims.json -> presentation.html
#
# Runs LAST: every numeral on a slide is looked up by claims key through
# report._Q, so the deck cannot be built before claims.json exists, and it
# cannot outlive a claim that was renamed. Both figures are inlined as base64
# because results/ is gitignored and a relative src breaks for teammates.
deck: dirs
	$(PY) -m router.deck
	@echo ""
	$(PY) -m router.verify

# ------------------------------------------------------------- stage 5: console
# The console is not a pipeline stage: it writes no artifact and `all` does not
# depend on it. It READS results/ and refuses to start without it, so it always
# shows what the pipeline last produced rather than a cached copy of its own.

## console: serve the router console on http://127.0.0.1:8765 (run `make all` first)
console: dirs
	$(PY) -m router.app

## console-e2e: drive the console in a real browser (needs `make console` running)
console-e2e:
	bash tests/console_states.sh http://127.0.0.1:8765

## console-check: the console's acceptance checks, no port opened
console-check: dirs
	$(PY) -m router.outcome
	@echo ""
	$(PY) -m router.serve
	@echo ""
	$(PY) -m router.app --selftest

# ------------------------------------------------------------------ dev targets
## test: run the test suite (unittest; pytest if it is installed)
test:
	@if $(PY) -c "import pytest" >/dev/null 2>&1; then \
		$(PY) -m pytest -q; \
	else \
		$(PY) -m unittest discover -s tests -t . -v; \
	fi

## demo: run the pipeline end to end, then print the headline claims
demo: all
	@echo ""
	@echo "=== headline claims (every number below is keyed in $(RESULTS)/claims.json) ==="
	@$(PY) -c "import json,sys;\
p='$(RESULTS)/claims.json';\
d=json.load(open(p));\
[print('  %-42s %s' % (k, v)) for k, v in sorted(d.items())]" \
	  || echo "  (no $(RESULTS)/claims.json yet — run 'make report')"
	@echo ""
	@echo "Token counts are ESTIMATES (len(json.dumps(x))//4). The export has no usage field."

## clean: delete generated results/ only — never export/
clean:
	@test -n "$(RESULTS)" || { echo "refusing: RESULTS is empty"; exit 1; }
	@case "$(RESULTS)" in export|export/*|/*|.|..) \
		echo "refusing to remove '$(RESULTS)'"; exit 1;; esac
	rm -rf $(RESULTS)
	find router tests -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "removed $(RESULTS)/ — export/ untouched"

## help: list targets
help:
	@grep -hE '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'
