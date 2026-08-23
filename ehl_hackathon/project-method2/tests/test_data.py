import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from method2.data import load_trajectories


class DataTests(unittest.TestCase):
    def test_schema_and_reconstruction(self):
        row = {"model": "m", "input": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], "tools": []}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.jsonl"
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            values = load_trajectories(path)
        self.assertEqual(len(values), 1)
        self.assertEqual(len(values[0].calls), 2)

