"""The sweep must not disturb the artifact the rest of the pipeline is pinned to.

`router.policy.route_all` writes `results/routes.jsonl` by default. `router.sweep`
calls it eleven times. The first version of the sweep left the default in place,
so every alpha overwrote the routing artifact and the last one won: `make report`
came back with `policy.rerouted.n = 497` (alpha=0.50) against an expected 119,
and refused. These tests exist so that cannot happen twice.

The textual guard is the durable half. A functional test only covers the call
sites that exist today; the guard covers the one someone adds next month.
"""
from __future__ import annotations

import ast
import hashlib
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_PY = REPO_ROOT / "router" / "sweep.py"
ROUTES = REPO_ROOT / "results" / "routes.jsonl"


class SweepNeverWritesTheRoutingArtifact(unittest.TestCase):
    def test_every_route_all_call_in_sweep_passes_write_false(self):
        """Textual guard: no `route_all(...)` in sweep.py may omit write=False."""
        tree = ast.parse(SWEEP_PY.read_text(encoding="utf-8"))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "route_all"
        ]
        self.assertTrue(calls, "no route_all call found — has sweep.py been renamed?")
        for call in calls:
            kw = {k.arg: k.value for k in call.keywords}
            self.assertIn(
                "write", kw,
                f"route_all at line {call.lineno} does not pass write=; it will "
                "overwrite results/routes.jsonl"
            )
            self.assertIsInstance(kw["write"], ast.Constant)
            self.assertIs(
                kw["write"].value, False,
                f"route_all at line {call.lineno} passes write={kw['write'].value!r}"
            )

    def test_route_all_with_write_false_leaves_the_file_alone(self):
        """Functional: the flag actually does what the guard assumes it does."""
        if not ROUTES.exists():
            self.skipTest("results/routes.jsonl not built — run `make policy`")
        from router import policy

        before = hashlib.sha256(ROUTES.read_bytes()).hexdigest()
        # An alpha far from the shipped one, so a write would be unmistakable.
        policy.route_all(alpha=0.50, write=False)
        after = hashlib.sha256(ROUTES.read_bytes()).hexdigest()
        self.assertEqual(
            before, after,
            "route_all(write=False) modified results/routes.jsonl"
        )


class SweepDocstringMatchesBehaviour(unittest.TestCase):
    """The module claims in prose that it never writes the artifact. Hold it to that."""

    def test_the_module_docstring_still_makes_the_claim(self):
        doc = ast.get_docstring(ast.parse(SWEEP_PY.read_text(encoding="utf-8"))) or ""
        self.assertTrue(
            re.search(r"never writes over|write=False|does not write", doc + SWEEP_PY.read_text(encoding="utf-8")),
            "sweep.py no longer states that it leaves routes.jsonl alone"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
