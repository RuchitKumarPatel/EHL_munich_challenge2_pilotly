# Models

Trained artifacts land here: `reward_model.json`, `propensity_model.json`,
`conformal_calibrators.json`, plus a `.metadata.json` next to the reward model
recording label provenance counts. Regenerate with `scripts/reproduce_all.py` or the
individual `scripts/train_*.py` / `scripts/calibrate_conformal.py`.

`models/organizer/` holds a run against the organizer's raw export, kept as a
compatibility demonstration (no ground truth, no scenario tags, smaller test set).
