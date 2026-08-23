# Project Method 2

Project Method 2 is a calibrated and conservative trajectory router. It uses model-specific nearest-neighbor evidence, conformal-style residual bounds, cache-aware costs, and an abstention policy that keeps the logged model whenever a cheaper route is not sufficiently supported.

## Run

```text
python scripts/reproduce.py path/to/export
python -m unittest discover -s tests -v
python demo/app.py path/to/export
```

`results/heldout_metrics.json` separates factual observed outcomes from counterfactual estimates. `results/pareto.csv` uses only calibrated lower-bound quality, so policy points are comparable.

