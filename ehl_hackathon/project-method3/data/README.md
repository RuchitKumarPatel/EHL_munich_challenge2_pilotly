# Data

No data is duplicated into this project. Point any script's `<source>` argument at:

- `../dataset1/export` — the enriched synthetic dataset built this session, with a
  companion `../dataset1/scenario_manifest.json` (ground truth labels + scenario
  tags) that `data/ground_truth.py` and `evaluation/segmentation.py` pick up
  automatically when `source` is that `export/` directory (or its parent).
- `../viktor-tumai-starter/viktor-tumai-starter/export` — the organizer's raw
  export. No manifest exists there; the pipeline falls back to the structural-proxy
  label and empty segmentation, by design.

Both are schema-identical JSONL (`{"model", "input", "tools"}` per line) — the same
`data/loader.py` and `data/schema.py` handle either.
