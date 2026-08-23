# Literature

What this project borrows from, and what it deliberately doesn't attempt. Full
context for each of these is in the design conversation that preceded this
implementation; this is the reference list with a one-line note per entry on how it
maps onto this codebase.

- **RouteLLM** — Ong et al., "Learning to Route LLMs with Preference Data" (ICLR
  2025, arXiv:2406.18665). Binary win-predictor between a strong and weak model,
  fit on preference data, transfers across model pairs. `CascadeRouter`'s
  cheapest-that-clears-the-floor logic is this idea generalized to N candidate
  models with a calibrated (not raw-predicted) floor.

- **FrugalGPT** — Chen, Zaharia, Zou, arXiv:2305.05176. Three cost-reduction
  strategies (prompt adaptation, LLM approximation, cascade); cascade sequentially
  escalates on an unreliable response. This project cannot do the *literal* version
  (no live model access to observe a real cheap-model response before deciding to
  escalate) — `CascadeRouter` is a single-shot *predicted* cascade instead; see
  `docs/methodology.md` §4.

- **Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey**,
  arXiv:2603.04445. Six-paradigm taxonomy (difficulty-aware, preference-alignment,
  clustering, RL, uncertainty-quantification, cascading) this project's design sits
  inside (difficulty-aware + uncertainty-quantification + cascading).

- **UCCI** — arXiv:2605.18796, and **CP-Router** — arXiv:2505.19970. Calibration-first
  routing (isotonic regression / conformal prediction) beats uncalibrated confidence
  thresholds and per-workload threshold tuning. Direct motivation for replacing
  method2's uncorrected residual-percentile heuristic with proper split-conformal
  calibration (`quality/conformal.py`).

- **Doubly Robust Policy Evaluation** — Dudík, Langford, Li (ICML 2011) and
  follow-on work on the SWITCH estimator (adaptive DM/IPS blending). Standard
  contextual-bandit OPE; this project keeps method1's DR structure but replaces its
  context-free class-frequency propensity with a fitted one
  (`propensity/logging_policy.py`) — see `docs/methodology.md` §5. The SWITCH
  estimator's adaptive blending is not implemented; plain DR plus a reported direct
  estimate and support diagnostics is the current scope.

- **"The Replay Gap: Static Evaluation of Model Switching in LLM Agents Scores the
  Wrong World"**, arXiv:2608.08239. Branching-rollout experiments on SWE-bench
  agents show a mid-trajectory model swap rewrites 61–94% of subsequent actions
  relative to a same-model control — static replay evaluation of that swap is
  measuring a different trajectory than the swap would actually produce. This is
  the specific finding behind the decision to route once per trajectory rather than
  per step; see `docs/methodology.md` §2 and `docs/limitations.md`.

- **MTRouter** — ACL 2026, arXiv:2604.23530. Turn-level routing via joint
  history-model embeddings, evaluated on *live* environments (ScienceWorld, HLE),
  not static replay — consistent with, not a counterexample to, the Replay Gap
  finding: per-step routing works when you can evaluate it live, which this project
  cannot.

- Cross-model KV-cache transfer literature (e.g. arXiv:2608.03893) — confirms a
  model swap forces a real prefill-reprocessing cost in production systems, which
  motivated keeping (and, this session, actually fixing) the switch-penalty term in
  `pricing/cost_model.py`, though the specific token count remains an
  undocumented-by-measurement placeholder (`docs/limitations.md`).
