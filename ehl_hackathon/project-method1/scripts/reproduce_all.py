from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(script: str, source: str) -> None:
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), source], cwd=ROOT, check=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(script)


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    run("prepare_data.py", source)
    run("build_trajectories.py", source)
    run("create_quality_labels.py", source)
    run("train_router.py", source)
    run("evaluate_router.py", source)
    run("generate_pareto.py", source)


if __name__ == "__main__":
    main()

