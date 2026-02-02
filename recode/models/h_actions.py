from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Literal
from typing import TypedDict, NotRequired
import warnings
import math
import numpy as np

Number = Union[int, float]
Event  = Tuple[float, float]          # (value, time)
Action = Tuple[int, Event, Event]     # (type_idx, (e_val, t_e), (r_val, t_r))

@dataclass
class Function:
    """
    Modèle h(t) générique.
    - parameters: paramètres numériques
    - sim_priors: priors (Normal) pour Laplace
    - _evaluator(t, Eff_, Rew_, params) -> list[float]
        - t: list[float] temps d’évaluation (souvent design["t"] ou design["t_meas"])
        - Eff_, Rew_: listes d’événements (val, time)
        - params: dict des paramètres du modèle

    """
    name: str
    parameters: Dict[str, float]
    sim_priors: Dict[str, Any]
    _evaluator: Callable[[Any, Any, Any, Any, Any, Dict[str, float]], Any] = field(repr=False)
    _jacobian: Optional[Callable[[Dict[str, Any], Dict[str, float]], List[List[float]]]] = field(default=None, repr=False)
    # Optionnel (si tu veux des specs d’inputs/sorties plus tard)
    input: Optional[Dict[str, Dict[str, Any]]] = None
    output: Optional[Dict[str, Dict[str, Any]]] = None

    def eval(self, inputs: Dict[str, Any]) -> Any:
        if not isinstance(inputs, dict):
            raise TypeError("inputs doit être un dict.")
        # On garde les clés "t", "Eff_", "Rew_" pour rester compatible
        t    = inputs.get("t")
        Eff_ = inputs.get("Eff_", []) # NOTE: les .get est une méthode de base des dicts. get(a,b) = get a sinon b.
        Rew_ = inputs.get("Rew_", [])
        A_typed    = inputs.get("A_typed", [])
        K_types = inputs.get("K_types") # NOTE: Nombre de types d'actions
        return self._evaluator(t, Eff_, Rew_, A_typed, K_types, self.parameters)

    def jacobian(self, inputs: Dict[str, Any], params: Dict[str, float]) -> List[List[float]]:
        if self._jacobian is None:
            raise AttributeError("jacobian not implemented")
        return self._jacobian(inputs, params)
    
ActionPair = Tuple[Event, Event]  # ((e_val, t_e), (r_val, t_r))

class Design(TypedDict):
    t: List[float]
    t_meas: List[float]
    Eff_: List[Event]
    Rew_: List[Event]
    Bonus_: List[Event]
    A_typed: List[Action]   # inclut type_idx
    K_types: int
    # Optionnels (présents quand tu demandes les sources de mesures)
    meas_times: NotRequired[List[float]]
    meas_sources: NotRequired[List[str]]

Kernel = Literal["event_weighted", "action_avg"]
UpdateMode = Literal["continuous", "event", "action"]
Observation = Literal["identity", "sigmoid"]

def validate_model_spec(kernel: Kernel, update: UpdateMode, observation: Observation) -> None:
    """Placeholder pour valider les choix de modèle (à compléter plus tard)."""
    return

def build_h_action_function(    *,
    kernel: Kernel,
    update: UpdateMode,
    observation: Observation = "identity",
    params_init: Optional[Dict[str, float]] = None,
) -> Function:
    """
    Fabrique un modèle h(t) prêt à être utilisé par Laplace.
    """
    validate_model_spec(kernel, update, observation)

    # Paramètres par défaut
    params: Dict[str, float] = {
        "gamma": 0.95,
        "W_eff": -1.0,
        "W_rew": 1.0,
        "obs_temp": 1.0,
        "obs_bias": 0.0,
    }
    if params_init:
        params.update(params_init)
    # Priors (Normal) pour Laplace
    sim_priors: Dict[str, Any] = {
        # Aligned with legacy priors from functions/h_actions.py (where applicable)
        "gamma": {"dist": "Normal", "mu": 0.3315, "sigma": 0.2},
        "W_eff": {"dist": "Normal", "mu": -0.0051, "sigma": 0.008},
        "W_rew": {"dist": "Normal", "mu": 0.2536, "sigma": 0.05},
        "obs_temp": {"dist": "Normal", "mu": 1.0, "sigma": 0.0},
        "obs_bias": {"dist": "Normal", "mu": 0.0, "sigma": 0.0},
    }
    def _evaluator(t: List[float], Eff_: List[Event], Rew_: List[Event], A_typed: List[Action], K_types: Optional[int], p: Dict[str, float]) -> List[float]:
        def _sigmoid_array(arr: np.ndarray, temp: float, bias: float) -> np.ndarray:
            x = temp * arr + bias
            x = np.clip(x, -60.0, 60.0)
            return 1.0 / (1.0 + np.exp(-x))
        def _apply_observation_vals(vals: np.ndarray, obs_temp: float, obs_bias: float) -> np.ndarray:
            if observation != "sigmoid":
                return vals
            return _sigmoid_array(vals, obs_temp, obs_bias)
        """
        TODO: implémenter h(t) selon kernel/update
        puis appliquer observation (identity/sigmoid).
        """
        def snap_times(
            t: Sequence[float],
            marks: Sequence[float],
            *,
            default_value: Optional[float] = None,
        ) -> List[float]:  # Warning: marks doit être sorted
            if marks is None:
                return list(t)
            marks_arr = np.asarray(marks, dtype=float)
            if marks_arr.size == 0:
                return [float(default_value) for _ in t] if default_value is not None else list(t)
            t_arr = np.asarray(t, dtype=float)
            marks_sorted = np.sort(np.asarray(marks, dtype=float))

            # idx = index du dernier mark <= t_i (ou -1 si aucun)
            idx = np.searchsorted(marks_sorted, t_arr, side="right") - 1

            snapped = t_arr.copy()
            mask = idx >= 0
            snapped[mask] = marks_sorted[idx[mask]]
            if default_value is not None:
                snapped[~mask] = float(default_value)
            return snapped.tolist()
            
        def get_decay(Events, t_snapped):
            aux_times = t_snapped - np.asarray([t for _,t in Events])
            aux_times = aux_times * (aux_times >= 0)
            return aux_times
        def _compute_contrib(t_k: np.ndarray, vals: np.ndarray, times: np.ndarray, gamma: float) -> np.ndarray:
            if times.size == 0:
                return np.zeros_like(t_k, dtype=float)
            diff = t_k[:, None] - times[None, :]
            mask = diff >= 0.0
            diff = np.where(mask, diff, 0.0)
            return (vals[None, :] * (gamma ** diff) * mask).sum(axis=1)
        def _get_marks(update: UpdateMode, Eff_times: np.ndarray, Rew_times: np.ndarray, Action_times: np.ndarray) -> Optional[np.ndarray]:
            if update == "continuous":
                return None
            if update == "action":
                return Action_times
            if update == "event":
                if Eff_times.size == 0 and Rew_times.size == 0:
                    return np.array([], dtype=float)
                return np.sort(np.concatenate([Eff_times, Rew_times]))
            raise ValueError("ERREUR Mauvais type d'update")

        def h(t):
            gamma = p["gamma"]
            W_eff = p["W_eff"]
            W_rew = p["W_rew"]
            obs_temp = p.get("obs_temp", 1.0)
            obs_bias = p.get("obs_bias", 0.0)
            h_output = 0
            Eff_times = np.sort(np.asarray([t for _,t in Eff_]))
            Rew_times = np.sort(np.asarray([t for _,t in Rew_]))
            if Eff_times.size == 0 and Rew_times.size == 0:
                Events_times = np.array([], dtype=float)
            else:
                Events_times = np.sort(np.concatenate([Eff_times, Rew_times]))

            Action_times = np.sort(np.asarray([t_r for _, (_, _), (_, t_r) in A_typed], dtype=float))

            if kernel == "event_weighted":
                marks = _get_marks(update, Eff_times, Rew_times, Action_times)
                if marks is None:
                    t_arr = np.asarray(t, dtype=float)
                else:
                    default_value = 0.0 if update == "action" else None
                    t_arr = np.asarray(snap_times(t, marks, default_value=default_value), dtype=float)

                eff_vals = np.asarray([v for v, _ in Eff_], dtype=float)
                rew_vals = np.asarray([v for v, _ in Rew_], dtype=float)

                eff_contrib = _compute_contrib(t_arr, eff_vals, Eff_times, gamma)
                rew_contrib = _compute_contrib(t_arr, rew_vals, Rew_times, gamma)

                h = W_eff * eff_contrib + W_rew * rew_contrib
                h = _apply_observation_vals(h, float(obs_temp), float(obs_bias))
                return h.tolist()

            elif kernel == "action_avg":
                if K_types is None or K_types <= 0:
                    raise ValueError("K_types >= 1 doit être fourni pour kernel 'action_avg'")
                eff_vals_by_type = {k: [] for k in range(K_types)}
                rew_vals_by_type = {k: [] for k in range(K_types)}
                eff_times_by_type = {k: [] for k in range(K_types)}
                rew_times_by_type = {k: [] for k in range(K_types)}
                for type_idx, (e_val, t_e), (r_val, t_r) in A_typed:
                    eff_vals_by_type[type_idx].append(e_val)
                    eff_times_by_type[type_idx].append(t_e)
                    rew_vals_by_type[type_idx].append(r_val)
                    rew_times_by_type[type_idx].append(t_r)
                t_arr_k = np.zeros((len(t),K_types), dtype=float)
                t_arr = np.asarray(t, dtype=float)
                h = np.zeros((len(t),K_types), dtype=float)
                for type_idx in range(K_types):
                    Eff_times = np.sort(np.asarray(eff_times_by_type[type_idx]))
                    Rew_times = np.sort(np.asarray(rew_times_by_type[type_idx]))
                    if Eff_times.size == 0 and Rew_times.size == 0:
                        warnings.warn("No events for this type; using empty marks.")
                        Events_times = np.array([], dtype=float)
                    else:
                        Events_times = np.sort(np.concatenate([Eff_times, Rew_times]))

                    marks = _get_marks(update, Eff_times, Rew_times, Rew_times)  # action uses reward times here
                    eff_vals = np.asarray(eff_vals_by_type[type_idx], dtype=float)
                    rew_vals = np.asarray(rew_vals_by_type[type_idx], dtype=float)
                    if marks is None:
                        t_arr_k[:, type_idx] = t_arr
                    else:
                        default_value = 0.0 if update == "action" else None
                        t_arr_k[:, type_idx] = np.asarray(snap_times(t, marks, default_value=default_value), dtype=float)
                    t_k = t_arr_k[:, type_idx]
                    eff_contrib = _compute_contrib(t_k, eff_vals, Eff_times, gamma)
                    rew_contrib = _compute_contrib(t_k, rew_vals, Rew_times, gamma)

                    h[:,type_idx] = W_eff * eff_contrib + W_rew * rew_contrib
                h_avg = (1.0 / K_types) * np.sum(h, axis=1)
                h_avg = _apply_observation_vals(h_avg, float(obs_temp), float(obs_bias))
                return h_avg.tolist()
                    

        if not t:
            return []
        return h(t)

    def _jacobian(inputs: Dict[str, Any], params: Dict[str, float]) -> List[List[float]]:
        t_in = inputs.get("t", [])
        if not t_in:
            return []
        t_arr = np.asarray(t_in, dtype=float).reshape(-1)
        Eff_in: List[Event] = list(inputs.get("Eff_", []))
        Rew_in: List[Event] = list(inputs.get("Rew_", []))
        A_typed_in: List[Action] = list(inputs.get("A_typed", []))
        K_types_in = inputs.get("K_types")

        gamma = float(params.get("gamma", 0.95))
        W_eff = float(params.get("W_eff", -1.0))
        W_rew = float(params.get("W_rew", 1.0))
        obs_temp = float(params.get("obs_temp", 1.0))
        obs_bias = float(params.get("obs_bias", 0.0))

        def _sigmoid_array(arr: np.ndarray, temp: float, bias: float) -> np.ndarray:
            x = temp * arr + bias
            x = np.clip(x, -60.0, 60.0)
            return 1.0 / (1.0 + np.exp(-x))

        def _compute_contrib_and_dgamma(t_k: np.ndarray, vals: np.ndarray, times: np.ndarray, gamma_val: float) -> Tuple[np.ndarray, np.ndarray]:
            if times.size == 0:
                return np.zeros_like(t_k, dtype=float), np.zeros_like(t_k, dtype=float)
            diff = t_k[:, None] - times[None, :]
            mask = diff >= 0.0
            diff = np.where(mask, diff, 0.0)
            if gamma_val <= 0.0:
                gdt = (diff == 0.0).astype(float) * mask
                dgamma = np.zeros_like(gdt, dtype=float)
            else:
                gdt = (gamma_val ** diff) * mask
                dgamma = gdt * (diff / gamma_val)
            contrib = (vals[None, :] * gdt).sum(axis=1)
            dgamma_sum = (vals[None, :] * dgamma).sum(axis=1)
            return contrib, dgamma_sum

        def _get_marks(update_mode: UpdateMode, Eff_times: np.ndarray, Rew_times: np.ndarray, Action_times: np.ndarray) -> Optional[np.ndarray]:
            if update_mode == "continuous":
                return None
            if update_mode == "action":
                return Action_times
            if update_mode == "event":
                if Eff_times.size == 0 and Rew_times.size == 0:
                    return np.array([], dtype=float)
                return np.sort(np.concatenate([Eff_times, Rew_times]))
            raise ValueError("ERREUR Mauvais type d'update")

        def _snap_times(t_seq: Sequence[float], marks: Sequence[float]) -> List[float]:
            if marks is None:
                return list(t_seq)
            marks_arr = np.asarray(marks, dtype=float)
            if marks_arr.size == 0:
                return list(t_seq)
            t_local = np.asarray(t_seq, dtype=float)
            marks_sorted = np.sort(marks_arr)
            idx = np.searchsorted(marks_sorted, t_local, side="right") - 1
            snapped = t_local.copy()
            mask = idx >= 0
            snapped[mask] = marks_sorted[idx[mask]]
            return snapped.tolist()

        param_names = list(params.keys())
        n_params = len(param_names)
        J = np.zeros((t_arr.size, n_params), dtype=float)

        def _pidx(name: str) -> int:
            try:
                return param_names.index(name)
            except ValueError:
                return -1

        idx_gamma = _pidx("gamma")
        idx_weff = _pidx("W_eff")
        idx_wrew = _pidx("W_rew")
        idx_ot = _pidx("obs_temp")
        idx_ob = _pidx("obs_bias")

        Eff_times = np.sort(np.asarray([t for _, t in Eff_in], dtype=float))
        Rew_times = np.sort(np.asarray([t for _, t in Rew_in], dtype=float))
        Action_times = np.sort(np.asarray([t_r for _, (_, _), (_, t_r) in A_typed_in], dtype=float))

        if kernel == "event_weighted":
            marks = _get_marks(update, Eff_times, Rew_times, Action_times)
            t_eval = t_arr if marks is None else np.asarray(_snap_times(t_arr, marks), dtype=float)
            eff_vals = np.asarray([v for v, _ in Eff_in], dtype=float)
            rew_vals = np.asarray([v for v, _ in Rew_in], dtype=float)
            eff_contrib, eff_dgamma = _compute_contrib_and_dgamma(t_eval, eff_vals, Eff_times, gamma)
            rew_contrib, rew_dgamma = _compute_contrib_and_dgamma(t_eval, rew_vals, Rew_times, gamma)
            h_vals = W_eff * eff_contrib + W_rew * rew_contrib
            if idx_weff >= 0:
                J[:, idx_weff] = eff_contrib
            if idx_wrew >= 0:
                J[:, idx_wrew] = rew_contrib
            if idx_gamma >= 0:
                J[:, idx_gamma] = W_eff * eff_dgamma + W_rew * rew_dgamma
        else:
            if K_types_in is None or int(K_types_in) <= 0:
                raise ValueError("K_types >= 1 doit être fourni pour kernel 'action_avg'")
            K_types_val = int(K_types_in)
            eff_vals_by_type = {k: [] for k in range(K_types_val)}
            rew_vals_by_type = {k: [] for k in range(K_types_val)}
            eff_times_by_type = {k: [] for k in range(K_types_val)}
            rew_times_by_type = {k: [] for k in range(K_types_val)}
            for type_idx, (e_val, t_e), (r_val, t_r) in A_typed_in:
                eff_vals_by_type[type_idx].append(e_val)
                eff_times_by_type[type_idx].append(t_e)
                rew_vals_by_type[type_idx].append(r_val)
                rew_times_by_type[type_idx].append(t_r)

            total_eff = np.zeros_like(t_arr, dtype=float)
            total_rew = np.zeros_like(t_arr, dtype=float)
            total_dgamma_eff = np.zeros_like(t_arr, dtype=float)
            total_dgamma_rew = np.zeros_like(t_arr, dtype=float)

            for type_idx in range(K_types_val):
                Eff_times_k = np.sort(np.asarray(eff_times_by_type[type_idx], dtype=float))
                Rew_times_k = np.sort(np.asarray(rew_times_by_type[type_idx], dtype=float))
                Action_times_k = Rew_times_k
                marks = _get_marks(update, Eff_times_k, Rew_times_k, Action_times_k)
                t_k = t_arr if marks is None else np.asarray(_snap_times(t_arr, marks), dtype=float)
                eff_vals = np.asarray(eff_vals_by_type[type_idx], dtype=float)
                rew_vals = np.asarray(rew_vals_by_type[type_idx], dtype=float)
                eff_contrib, eff_dgamma = _compute_contrib_and_dgamma(t_k, eff_vals, Eff_times_k, gamma)
                rew_contrib, rew_dgamma = _compute_contrib_and_dgamma(t_k, rew_vals, Rew_times_k, gamma)
                total_eff += eff_contrib
                total_rew += rew_contrib
                total_dgamma_eff += eff_dgamma
                total_dgamma_rew += rew_dgamma

            scale = 1.0 / float(K_types_val)
            h_vals = scale * (W_eff * total_eff + W_rew * total_rew)
            if idx_weff >= 0:
                J[:, idx_weff] = scale * total_eff
            if idx_wrew >= 0:
                J[:, idx_wrew] = scale * total_rew
            if idx_gamma >= 0:
                J[:, idx_gamma] = scale * (W_eff * total_dgamma_eff + W_rew * total_dgamma_rew)

        if observation == "sigmoid":
            s = _sigmoid_array(h_vals, obs_temp, obs_bias)
            scale = obs_temp * s * (1.0 - s)
            if idx_gamma >= 0:
                J[:, idx_gamma] = J[:, idx_gamma] * scale
            if idx_weff >= 0:
                J[:, idx_weff] = J[:, idx_weff] * scale
            if idx_wrew >= 0:
                J[:, idx_wrew] = J[:, idx_wrew] * scale
            if idx_ot >= 0:
                J[:, idx_ot] = h_vals * (s * (1.0 - s))
            if idx_ob >= 0:
                J[:, idx_ob] = s * (1.0 - s)

        return J

    name = f"h_action[{kernel}|{update}|{observation}]"
    return Function(
        name=name,
        parameters=params,
        sim_priors=sim_priors,
        _evaluator=_evaluator,
        _jacobian=_jacobian,
    )
