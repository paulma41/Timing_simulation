from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Literal
from typing import TypedDict, NotRequired
import warnings
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
    _evaluator: Callable[[Any, Any, Any, Dict[str, float]], Any] = field(repr=False)
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
        "gamma": {"dist": "Normal", "mu": params["gamma"], "sigma": 0.0},
        "W_eff": {"dist": "Normal", "mu": params["W_eff"], "sigma": 0.0},
        "W_rew": {"dist": "Normal", "mu": params["W_rew"], "sigma": 0.0},
        "obs_temp": {"dist": "Normal", "mu": params["obs_temp"], "sigma": 0.0},
        "obs_bias": {"dist": "Normal", "mu": params["obs_bias"], "sigma": 0.0},
    }
    def _evaluator(t: List[float], Eff_: List[Event], Rew_: List[Event], A_typed: List[Action], K_types: Optional[int], p: Dict[str, float]) -> List[float]:
        """
        TODO: implémenter h(t) selon kernel/update
        puis appliquer observation (identity/sigmoid).
        """
        def snap_times(t: Sequence[float], marks: Sequence[float]) -> List[float]: #Warning: marks doit être sorted
            if marks is None:
                return list(t)
            marks_arr = np.asarray(marks, dtype=float)
            if marks_arr.size == 0:
                return list(t)
            t_arr = np.asarray(t, dtype=float)
            marks_sorted = np.sort(np.asarray(marks, dtype=float))

            # idx = index du dernier mark <= t_i (ou -1 si aucun)
            idx = np.searchsorted(marks_sorted, t_arr, side="right") - 1

            snapped = t_arr.copy()
            mask = idx >= 0
            snapped[mask] = marks_sorted[idx[mask]]
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
                    t_arr = np.asarray(snap_times(t, marks), dtype=float)

                eff_vals = np.asarray([v for v, _ in Eff_], dtype=float)
                rew_vals = np.asarray([v for v, _ in Rew_], dtype=float)

                eff_contrib = _compute_contrib(t_arr, eff_vals, Eff_times, gamma)
                rew_contrib = _compute_contrib(t_arr, rew_vals, Rew_times, gamma)

                h = W_eff * eff_contrib + W_rew * rew_contrib
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
                        t_arr_k[:, type_idx] = np.asarray(snap_times(t, marks), dtype=float)
                    t_k = t_arr_k[:, type_idx]
                    eff_contrib = _compute_contrib(t_k, eff_vals, Eff_times, gamma)
                    rew_contrib = _compute_contrib(t_k, rew_vals, Rew_times, gamma)

                    h[:,type_idx] = W_eff * eff_contrib + W_rew * rew_contrib
                return ((1/K_types) * np.sum(h,axis = 1)).tolist()
                    

        if not t:
            return []
        # placeholder minimal
        return [0.0 for _ in t]

    name = f"h_action[{kernel}|{update}|{observation}]"
    return Function(
        name=name,
        parameters=params,
        sim_priors=sim_priors,
        _evaluator=_evaluator,
    )