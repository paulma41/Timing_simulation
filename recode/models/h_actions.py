from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Literal
from typing import TypedDict, NotRequired

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
        A    = inputs.get("A", [])
        K_types = inputs.get("K_types") # NOTE: Nombre de types d'actions
        return self._evaluator(t, Eff_, Rew_, A, K_types, self.parameters)
    
ActionPair = Tuple[Event, Event]  # ((e_val, t_e), (r_val, t_r))

class Design(TypedDict):
    t: List[float]
    t_meas: List[float]
    Eff_: List[Event]
    Rew_: List[Event]
    Bonus_: List[Event]
    A: List[ActionPair]
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
    def _evaluator(t: List[float], Eff_: List[Event], Rew_: List[Event], A: List[Action], K_types: Optional[int], p: Dict[str, float]) -> List[float]:
        """
        TODO: implémenter h(t) selon kernel/update
        puis appliquer observation (identity/sigmoid).
        """
        def snap_times(t: Sequence[float], marks: Sequence[float]) -> List[float]: #Warning: marks doit être sorted
            if not marks:
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
        def h(t):
            gamma = p["gamma"]
            W_eff = p["W_eff"]
            W_rew = p["W_rew"]
            h_output = 0
            Eff_times = np.sort(np.asarray([t for _,t in Eff_]))
            Rew_times = np.sort(np.asarray([t for _,t in Rew_]))
            Events_times = np.sort(np.concatenate([Eff_times,Rew_times]))
            Action_times = np.sort(np.asarray([t_r for _, (_, _), (_, t_r) in A], dtype=float))

            if kernel == "event_weighted":
                if update == "continuous":
                    marks = None
                elif update == "action":
                    marks = Action_times
                elif update == "event":
                    marks = Events_times
                else:
                    Warning("ERREUR Mauvais type d'update")
                t_arr = np.array(snap_times(t,marks))
                eff_diff = t_arr[:, None] - Eff_times[None, :] # NOTE: ajout des None pour ajouter un axe et vectoriser, résultat n*m
                eff_diff = np.maximum(eff_diff, 0.0)
                rew_diff = t_arr[:,None] - Rew_times[None,:]
                rew_diff = np.maximum(rew_diff, 0.0)
                return W_eff*np.sum(gamma**eff_diff, axis = 1) + W_rew*np.sum(gamma**rew_diff, axis = 1)
            elif kernel == "action_avg":

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