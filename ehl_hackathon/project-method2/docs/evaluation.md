# Evaluation

Every policy has one comparable primary metric: model-specific calibrated quality lower bound. Factual observed proxy outcomes are reported separately and only for trajectories where the selected model equals the logged model.

The project does not claim that unsupported counterfactual predictions are observed outcomes. Insufficient support triggers a logged-model fallback.

The comparison CSV re-prices Method 1's observed static Opus-5 behavior on Method 2's frozen test split and evaluates both methods with Method 2's calibrated lower-bound metric. This avoids mixing cost units or outcome types.
