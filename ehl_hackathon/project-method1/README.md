# Project Method 1

Project Method 1 is an offline-first, cache-aware model router for Viktor agent trajectories. It reconstructs trajectories, extracts task and tool features, estimates composite outcomes, trains a difficulty-aware routing policy, evaluates it with matching and doubly robust estimates, and produces a cost-quality Pareto frontier.

## Quick start

```text
python scripts/reproduce_all.py path/to/export
python -m unittest discover -s tests -v
python demo/app.py path/to/export
```

The project accepts any directory containing `*.jsonl` files with exactly `model`, `input`, and `tools` fields. Dataset files stay outside the repository and are never copied into project artifacts.

All token counts and quality values are estimates or proxies unless a live evaluation adapter supplies verified outcomes.

The core pipeline, tests, and live demo use only the Python standard library. Install `requirements.txt` when you want PNG Pareto plots.

## Outputs

- `results/summary.json`
- `results/policy_metrics.json`
- `results/pareto.csv`
- `results/pareto.png` when matplotlib is installed
- `models/router/model.json`

## Live demo

Run `python demo/app.py path/to/export` and open `http://127.0.0.1:8765`. The demo is dependency-free and shows routing decisions, estimated costs, quality, uncertainty, and cache effects.
