# Recode Guide (remaining work)

## Context recap
- Goal: recode project starting from `agent_comparison.py` into a cleaner, minimal structure under `recode/`.
- Keep: Markov + Magneto agents, forced block, Chernoff bounds, Laplace + low-rank, multiprocessing, tqdm, NumPy-only (no fallback).
- Keep all plots but reorganize layout; provide flags to toggle plot sections.
- Wrapper: Quarto report (`.qmd`) to explain algorithms/equations and reference code.

## Target folder structure
- `recode/design_optimizer/`
  - `base.py` (NumPy-only helpers)
  - `laplace.py` (Laplace core + low-rank + multiprocessing)
- `recode/models/`
  - `h_actions.py` (Function + h(t) models + jacobian)
- `recode/design/`
  - `actions.py` (Markov + Magneto generation)
  - `measurement.py` (measurement times + bonuses + forced sampling)
  - `build.py` (build_design_v5 + fixed times)
- `recode/plotting/`
  - `summary.py` (plot_summary reorganized + flags)
- `recode/runner/`
  - `agent_comparison.py` (pipeline + CLI + parallel run)
- `recode/report/`
  - `agent_comparison.qmd` (Quarto wrapper)

## Remaining work checklist

### 1) `recode/models/h_actions.py`
- [x] Imports and basic types started.
- [x] Minimal `Function` class (name, parameters, sim_priors, _evaluator, eval(inputs)).
- [ ] Implement helpers:
  - `_snap_times`, `_snap_times_or_default`.
  - `_eval_event_weighted` (NumPy only).
  - `_eval_action_avg` (NumPy only).
- [ ] Implement `build_h_action_function(...)`:
  - kernels: `event_weighted`, `action_avg`.
  - updates: `continuous`, `event`, `action`.
  - observation: `identity`, `sigmoid`.
  - analytic Jacobian (NumPy only) + precomputation.

### 2) `recode/design_optimizer/base.py`
- [ ] Port `numerical_jacobian` (NumPy only).
- [ ] Port `logdet_psd` (NumPy only).
- [ ] Keep minimal helpers used by Laplace/Chernoff.

### 3) `recode/design_optimizer/laplace.py`
- [ ] Port Laplace prior predictive:
  - `_prior_mean_and_var` (using `sim_priors`).
  - `_laplace_prior_predictive_gaussian` (NumPy only).
- [ ] Parallel Laplace for models using `ProcessPoolExecutor` + `cloudpickle`.
- [ ] Chernoff metrics now live in `recode/design_optimizer/chernoff.py` (pairwise C, bounds, maximin).
- [ ] Public API: `laplace_predictive(...)` (mu, Vy, optional lowrank).

### 4) `recode/design/measurement.py`
- [ ] `sample_measurement_times` (Eff then Rew then uniform; return sources option).
- [ ] `sample_bonuses`.
- [ ] `_sample_forced_meas_times` (segment by type sequence).

### 5) `recode/design/actions.py`
- [ ] `markov_actions_with_types`.
- [ ] Magneto agent:
  - `_magneto_softmax`.
  - `_event_weighted_h`.
  - `magneto_like_actions_with_value_learning`.
- [ ] `_normalize_type_sequence` helper.

### 6) `recode/design/build.py`
- [ ] `_build_fixed_times`.
- [ ] `build_design_v5` (includes A_typed, K_types, t_meas, t_all).

### 7) `recode/plotting/summary.py`
- [ ] Rebuild `plot_summary` from `test_multi_model.py` but reorganized.
- [ ] Add flags to enable/disable sections:
  - heatmap U/C
  - heatmap Chernoff error bound
  - h(t) curves + measured points
  - Eff/Rew scatter + t
  - histograms (gaps, Eff/Rew, Δt violin)
  - action type distribution

### 8) `recode/runner/agent_comparison.py`
- [ ] Port `run_agent` pipeline:
  - build models, build actions/design, Laplace/Chernoff score.
- [ ] CLI args + parallel task execution.
- [ ] tqdm global progress tracking.

### 9) `recode/report/agent_comparison.qmd`
- [ ] Explain algorithms + equations (h(t), Laplace, Chernoff).
- [ ] Import and run the pipeline.
- [ ] Reference code sections (short snippets or citations).

## Notes / decisions already made
- NumPy only (no pure-Python fallback).
- Keep multiprocessing and tqdm.
- Keep all plots but reorganize them.
- Reduce model generator files to one (`recode/models/h_actions.py`).
- Reduce design generator to one set of files under `recode/design/`.
- Action and typed-action were merged in your current draft; typed detection will be based on tuple shape.

## Current step
- `recode/models/h_actions.py`: implement helper functions and `build_h_action_function`.

