from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Literal
from typing import TypedDict, NotRequired

import numpy as np

Number = Union[int, float]
Event = Tuple[float, float]          # (value, time)
Action= Tuple[int, Event, Event]  # (type_idx, (e_val, t_e), (r_val, t_r))

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
        t = inputs.get("t")
        Eff_ = inputs.get("Eff_", [])
        Rew_ = inputs.get("Rew_", [])
        return self._evaluator(t, Eff_, Rew_, self.parameters)
    
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
    def _evaluator(t: List[float], Eff_: List[Event], Rew_: List[Event], p: Dict[str, float]) -> List[float]:
        """
        TODO: implémenter h(t) selon kernel/update
        puis appliquer observation (identity/sigmoid).
        """
        def snap_time(t_measure: float, update_mode: UpdateMode):
            if update_mode == "continuous":
                return t_measure
            Rew_times = np.asarray([t for _, t in Rew_])
            if Rew_times.size == 0:
                return None
            Rew_times.sort()
            if update_mode == "action":
                idx = np.searchsorted(Rew_times, t_measure, side="right") - 1
                if idx < 0:
                    return None  # aucun t_r <= t_measure
                return Rew_times[idx]
            Eff_times = np.asarray([t for _, t in Eff_])
            Events_times = np.sort(np.concatenate(Rew_times,Eff_times))
            if update_mode == "event":
                idx = np.searchsorted(Events_times, t_measure, side="right") - 1
                if idx < 0:
                    return None  # aucun t_r <= t_measure
                return Events_times[idx]
        if kernel == "event_weighted":
            h = t: 
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