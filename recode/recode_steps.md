# Recode plan (agent_comparison)

## Target structure (based on existing recode/ subfolders)
- recode/models/
  - Core domain types: Event, Action, ActionType
  - Agent simulators (Markov, Magneto)
  - h(t) model functions + observation
- recode/design/
  - Time grid helpers (t_fixed)
  - Bonus + measurement sampling
  - Design builder (Eff_/Rew_/Bonus_/A/A_typed/t/t_meas)
- recode/design_optimizer/
  - Laplace/JSD approximation
  - Chernoff matrix + maximin score
- recode/plotting/
  - Summaries (heatmaps, violin, etc.)
- recode/runner/
  - CLI, orchestration, parallel runs

## Step-by-step plan
1) models/h_actions.py (first priority)
   - Formalize the minimal data structures used everywhere (Event, Action, ActionType, Design)
   - Define the interface for a model function: inputs (design dict) -> output vector y
   - Define observation options (identity/sigmoid) and how params are carried
   - Define kernels and update modes (event_weighted/action_avg + continuous/event/action)
   - Define an h(t) evaluation API that mirrors current behavior from functions/build_h_action_function
   - Decide how you store parameters and priors (needed by Laplace)

   Status: IN PROGRESS
   - Done: basic Function dataclass (name, parameters, sim_priors, _evaluator, eval(inputs))
   - Done: minimal eval extracts t/Eff_/Rew_ and calls _evaluator
   - Pending: formal Design type + observation + kernels/update modes

2) models/agents_markov.py and models/agents_magneto.py
   - Implement action generation logic (Markov and Magneto) with the same timing rules
   - Include forced block behavior and type sequence handling
   - Return consistent action lists + agent params/state

3) design/time_grid.py
   - Build fixed time grids and interval helpers (t_min/t_max/t_step, guard)

4) design/measurements.py
   - Bonus sampling
   - Measurement time sampling (free and forced)

5) design/build_design.py
   - Assemble design dict (t, t_meas, Eff_, Rew_, Bonus_, A, A_typed, K_types)

6) design_optimizer/laplace.py
   - Port Laplace predictive code and lowrank option
   - Keep interface compatible with model objects

7) design_optimizer/chernoff.py
   - Pairwise Chernoff matrix computation
   - Maximin score

8) runner/run_agent.py
   - Orchestrate full pipeline per agent/seed
   - Support light-results and include_models flags

9) runner/cli.py
   - Recreate CLI args and multi-run loop
   - Handle multiprocessing + progress

10) plotting/
   - Heatmaps and P(correct|i) distribution plots
   - Summary plot for last run per agent

## Design contracts to keep
- Design dict keys expected by model evaluation and plotting:
  - t, t_meas, Eff_, Rew_, Bonus_, A, A_typed, K_types
- Model object contract for Laplace:
  - parameters dict
  - sim_priors dict (Normal with mu/sigma)
  - eval(design_inputs) -> list[float]
  - optional jacobian(design_inputs, params)

## Notes
- Keep behavior parity first, then prune unused paths.
- Separate deterministic logic (design build) from RNG logic (sampling) to ease testing.

## next_steps
- Corriger l'evaluator actuel:
  - Ne pas appeler `snap_times` avec `marks=None` (pour `continuous`, utiliser directement `t`).
  - Ajouter un masque `dt >= 0` pour ignorer les evenements futurs (sinon `gamma**0` compte a tort).
  - Multiplier par les valeurs d'event (`eff_vals`, `rew_vals`) dans les sommes.
  - Remplacer `Warning(...)` par `raise ValueError(...)` ou `warnings.warn(...)`.
  - S'assurer de retourner le resultat (appel a `h(t)` ou logique inline).
- Coller le kernel `action_avg` suivant (version vectorisee, conforme a ta logique):

```python
            elif kernel == "action_avg":
                # K_types
                if K_types is None:
                    K_types_local = 1 + max((k for k, _, _ in A), default=-1)
                else:
                    K_types_local = int(K_types) if K_types > 0 else 1
                K_types_local = max(1, K_types_local)

                # regrouper par type
                eff_vals_by_type = [[] for _ in range(K_types_local)]
                eff_times_by_type = [[] for _ in range(K_types_local)]
                rew_vals_by_type = [[] for _ in range(K_types_local)]
                rew_times_by_type = [[] for _ in range(K_types_local)]

                for k, (e, te), (r, tr) in A:
                    idx = min(max(int(k), 0), K_types_local - 1)
                    eff_vals_by_type[idx].append(float(e))
                    eff_times_by_type[idx].append(float(te))
                    rew_vals_by_type[idx].append(float(r))
                    rew_times_by_type[idx].append(float(tr))

                t_arr = np.asarray(t, dtype=float)
                h_sum = np.zeros_like(t_arr, dtype=float)

                for k in range(K_types_local):
                    # choix des t_k selon update
                    if update == "continuous":
                        t_k = t_arr
                    elif update == "event":
                        marks = eff_times_by_type[k] + rew_times_by_type[k]
                        t_k = np.asarray(snap_times(t, marks), dtype=float)
                    elif update == "action":
                        marks = rew_times_by_type[k]
                        if marks:
                            t_k = np.asarray(snap_times(t, marks), dtype=float)
                        else:
                            t_k = np.full_like(t_arr, np.min(t_arr) - 1.0, dtype=float)
                    else:
                        raise ValueError("ERREUR Mauvais type d'update")

                    # contributions effort
                    if eff_times_by_type[k]:
                        eff_times_k = np.asarray(eff_times_by_type[k], dtype=float)
                        eff_vals_k = np.asarray(eff_vals_by_type[k], dtype=float)
                        dt_e = t_k[:, None] - eff_times_k[None, :]
                        mask_e = dt_e >= 0
                        dt_e = np.where(mask_e, dt_e, 0.0)
                        eff_contrib = (eff_vals_k[None, :] * (gamma ** dt_e) * mask_e).sum(axis=1)
                    else:
                        eff_contrib = 0.0

                    # contributions reward
                    if rew_times_by_type[k]:
                        rew_times_k = np.asarray(rew_times_by_type[k], dtype=float)
                        rew_vals_k = np.asarray(rew_vals_by_type[k], dtype=float)
                        dt_r = t_k[:, None] - rew_times_k[None, :]
                        mask_r = dt_r >= 0
                        dt_r = np.where(mask_r, dt_r, 0.0)
                        rew_contrib = (rew_vals_k[None, :] * (gamma ** dt_r) * mask_r).sum(axis=1)
                    else:
                        rew_contrib = 0.0

                    h_sum += (W_eff * eff_contrib) + (W_rew * rew_contrib)

                return h_sum / float(K_types_local)
```
