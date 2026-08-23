# Methodology

Method 2 estimates each model's trajectory outcome from nearby observed trajectories that used that model. It combines the local residual bound, local outcome variance, feature distance, and effective neighbor support into a conservative lower bound.

The router selects a cheaper alternative only if its lower bound remains within the allowed loss of the logged model's lower bound and it has sufficient model-specific support. Otherwise it abstains to the logged model.

The model routes whole trajectories to preserve cache reuse. Costs use item-level shared-prefix overlap and are estimates.

