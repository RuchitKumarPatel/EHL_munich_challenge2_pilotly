# Methodology

The router makes a trajectory-level model decision. This preserves shared prompt prefixes and avoids an unpriced cache reset from a mid-trajectory switch.

The feature layer combines text, tool, trajectory, and prefix-overlap features. The current offline router uses a lightweight regularized linear quality predictor with model-specific profiles and a strength prior. The interface is designed so a stronger embedding or pairwise preference model can replace it without changing evaluation code.

The policy optimizes predicted quality against cache-aware cost and exposes uncertainty. The evaluation reports logged, cheapest, strongest, and proposed policies.

Quality values generated without final outputs are labeled `proxy`. They combine tool-output completion, termination, error recovery, and repetition signals. A live judge adapter may be added only with calibration data.

