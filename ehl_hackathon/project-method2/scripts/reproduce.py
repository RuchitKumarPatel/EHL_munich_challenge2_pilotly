from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str, source: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), source], cwd=ROOT, check=True)


def main() -> None:
    source = sys.argv[1]
    run("train.py", source)
    run("evaluate.py", source)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_pareto.py")], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_method1.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
