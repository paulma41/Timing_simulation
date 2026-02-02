# next_steps

- [ ] Dans `_evaluator`, appeler vraiment `h(t)` et appliquer l'observation (identity/sigmoid) avant de `return` (actuellement il y a un placeholder qui renvoie des zeros).
- [ ] Pour `action_avg`, utiliser les temps d'actions (ex. `Action_times` = temps de recompense par action) dans `_get_marks(update, ...)` quand `update == "action"`.
- [ ] Verifier que `K_types` est bien fourni quand `kernel == "action_avg"`, sinon lever une erreur claire.
- [ ] Garder `inputs.get(...)` si tu veux des valeurs par defaut; sinon passer en acces direct `inputs["A_typed"]` pour detecter vite les oublis.

```python
# action_avg (a coller dans _evaluator, en suivant ta logique)
elif kernel == "action_avg":
    if K_types is None or K_types <= 0:
        raise ValueError("K_types >= 1 doit etre fourni pour kernel 'action_avg'")

    eff_vals_by_type = {k: [] for k in range(K_types)}
    rew_vals_by_type = {k: [] for k in range(K_types)}
    eff_times_by_type = {k: [] for k in range(K_types)}
    rew_times_by_type = {k: [] for k in range(K_types)}

    for type_idx, (e_val, t_e), (r_val, t_r) in A_typed:
        eff_vals_by_type[type_idx].append(e_val)
        eff_times_by_type[type_idx].append(t_e)
        rew_vals_by_type[type_idx].append(r_val)
        rew_times_by_type[type_idx].append(t_r)

    t_arr = np.asarray(t, dtype=float)
    h_by_type = np.zeros((t_arr.size, K_types), dtype=float)

    for type_idx in range(K_types):
        Eff_times = np.sort(np.asarray(eff_times_by_type[type_idx], dtype=float))
        Rew_times = np.sort(np.asarray(rew_times_by_type[type_idx], dtype=float))
        Action_times = Rew_times  # temps d'action = t_r dans ta definition

        marks = _get_marks(update, Eff_times, Rew_times, Action_times)
        if marks is None:
            t_k = t_arr
        else:
            t_k = np.asarray(snap_times(t, marks), dtype=float)

        eff_vals = np.asarray(eff_vals_by_type[type_idx], dtype=float)
        rew_vals = np.asarray(rew_vals_by_type[type_idx], dtype=float)

        eff_contrib = _compute_contrib(t_k, eff_vals, Eff_times, gamma)
        rew_contrib = _compute_contrib(t_k, rew_vals, Rew_times, gamma)

        h_by_type[:, type_idx] = W_eff * eff_contrib + W_rew * rew_contrib

    h = (1.0 / K_types) * np.sum(h_by_type, axis=1)
    return h.tolist()
```
