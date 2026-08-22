# Evaluation Protocol

Splits are created at trajectory level using a stable hash of the reconstructed opening context. Requests from one trajectory never appear in multiple partitions.

The primary report includes cost, quality proxy, cacheable tokens, uncached tokens, switches, uncertainty, and policy-level Pareto dominance.

Logged routes use the reconstructed observed proxy outcome. Alternative routes use the selected router's model-based quality prediction and are marked as counterfactual predictions. They are not treated as observed outcomes.

The off-policy report includes a direct estimate, a propensity-weighted correction, support counts, a propensity floor, and effective sample size. Logged selection probabilities are not observed in the export, so propensities are estimated and results are not presented as randomized causal estimates.
