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
