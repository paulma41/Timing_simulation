# Chat summary (Codex tutoring)

## Goal
Recode the behavior of `agent_comparison.py` inside `recode/`, without Codex writing core code (tutor only). You are manually recoding; Codex provides guidance and small snippets for reference.

## Current progress
- `recode/recode_steps.md` updated with overall plan. Step 1 marked in progress; basic `Function` dataclass done.
- Active file: `recode/models/h_actions.py`.

## h_actions.py status
- You already added:
  - `Function` dataclass with `name`, `parameters`, `sim_priors`, `_evaluator`, optional `input/output`, and `eval(inputs)` that extracts `t`, `Eff_`, `Rew_` and calls `_evaluator`.
  - Types: `Number`, `Event = (value, time)`, `Action = (type_idx, Event, Event)`.
  - `ActionPair = (Event, Event)`.
  - `Design` as `TypedDict` (should NOT be decorated with `@dataclass`). Contains keys: `t`, `t_meas`, `Eff_`, `Rew_`, `Bonus_`, `A`, `A_typed`, `K_types`, and optional `meas_times`, `meas_sources`.
  - Aliases: `Kernel = Literal["event_weighted", "action_avg"]`, `UpdateMode = Literal["continuous", "event", "action"]`, `Observation = Literal["identity", "sigmoid"]`.
  - Stub `validate_model_spec(kernel, update, observation)`.

## Fix needed
- Remove `@dataclass` decorator from `Design` (TypedDict must not be a dataclass).

## Factory skeleton to implement (already provided)
- Add `build_h_action_function(...)` after `validate_model_spec`.
- In the skeleton:
  - set default params: `gamma`, `W_eff`, `W_rew`, `obs_temp`, `obs_bias`.
  - merge overrides if `params_init` is passed (only then update).
  - build `sim_priors` dict with Normal priors.
  - define `_evaluator(t, Eff_, Rew_, p)` returning list of floats (placeholder for now).
  - return `Function(...)`.

## Conceptual notes clarified
- Bonus are not actions: they are reward events `(value, time)`, added to `Rew_` and `Bonus_` in `build_design_v5`.
- `_evaluator` does not use action type unless you explicitly pass typed actions; the original model uses only `Eff_` and `Rew_` by default.
- For kernel `action_avg`, the *snapping* can depend on type **if** `A_typed` is provided; otherwise it falls back to global snapping.

## NumPy helper for "last t_r <= t_measure"
- Safe version:
  - sort `Rew_times`
  - `idx = np.searchsorted(Rew_times, t_measure, side="right") - 1`
  - if `idx < 0` -> none

## Next steps (after factory skeleton)
- Replace `_evaluator` placeholder with real logic from `functions/h_actions.py`:
  - implement snapping, event_weighted and action_avg kernels
  - apply observation (identity/sigmoid)
- Later: port Laplace, Chernoff, runner/CLI, plotting.

## Files referenced from original code
- `functions/function_spec.py`: original `Function` with type/range checks.
- `functions/h_actions.py`: reference implementation of h(t) kernel logic.
- `test_multi_model.py`: `_build_fixed_times`, `build_six_models`, `plot_summary`.
- `test_multi_model_bo_v5.py`: `sample_bonuses`, `sample_measurement_times`, `build_design_v5`, `maximin_chernoff`.
- `design_optimizer/laplace_jsd.py`: Laplace + Chernoff helpers.
