import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from method1.data.loader import load_trajectories


class SchemaTests(unittest.TestCase):
    def test_loader_groups_history_by_opening_context(self):
        record = {"model": "model-a", "input": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], "tools": []}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
            trajectories = load_trajectories(path)
        self.assertEqual(len(trajectories), 1)
        self.assertEqual(trajectories[0].n_calls, 2)

