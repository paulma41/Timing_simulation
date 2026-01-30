from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Union, get_args, get_origin

try:  # Optional numpy for acceleration
    import numpy as np
    HAVE_NUMPY = True
except Exception:  # pragma: no cover
    HAVE_NUMPY = False

from .function_spec import Function


Number = Union[int, float]
ArrayLike = Union[Sequence[Number], Number]


def _h_eval(t: ArrayLike, S1: Sequence[Number], S2: Sequence[Number], params: Dict[str, float]) -> Union[float, List[float]]:
    """
    Implémentation de:
        h(t) = C + B t
             + sum_{tau in S1, tau < t} A1 * exp(-lambda1 * (t - tau))
             + sum_{tau in S2, tau < t} A2 * exp(-lambda2 * (t - tau))

    - t: scalaire ou séquence de temps
    - S1, S2: séquences d'événements (temps croissants ou non)
    - params: dict avec clés {'C','B','A1','lambda1','A2','lambda2'}
    """
    C = float(params["C"])  # constante
    B = float(params["B"])  # pente
    A1 = float(params["A1"])  # amplitude S1
    l1 = float(params["lambda1"])  # décroissance S1 (>0)
    A2 = float(params["A2"])  # amplitude S2
    l2 = float(params["lambda2"])  # décroissance S2 (>0)
    # Robustesse: forcer des lambdas non négatives pour éviter des explosions numériques
    if l1 < 0:
        l1 = 0.0
    if l2 < 0:
        l2 = 0.0

    def eval_at_time(ti: float) -> float:
        s1 = 0.0
        if S1:
            for tau in S1:
                if tau < ti:
                    dt = ti - tau
                    # clamp exponent for safety in pure-Python path
                    x = -l1 * dt
                    if x < -745.0:
                        e = 0.0
                    elif x > 709.0:
                        # extremely large -> treat as +inf but clip to a large number to avoid OverflowError
                        e = float("inf")
                    else:
                        e = math.exp(x)
                    s1 += A1 * e
        s2 = 0.0
        if S2:
            for tau in S2:
                if tau < ti:
                    dt = ti - tau
                    x = -l2 * dt
                    if x < -745.0:
                        e = 0.0
                    elif x > 709.0:
                        e = float("inf")
                    else:
                        e = math.exp(x)
                    s2 += A2 * e
        return C + B * ti + s1 + s2

    # Si t est scalaire
    if isinstance(t, (int, float)):
        return float(eval_at_time(float(t)))

    # Sinon on suppose que t est itérable
    if HAVE_NUMPY:
        try:
            t_arr = np.asarray(t, dtype=float).reshape(-1)
        except Exception as exc:
            raise TypeError("'t' doit être convertible en array de réels.") from exc
        s1_vec = 0.0
        if S1:
            s1_arr = np.asarray(S1, dtype=float).reshape(-1)
            if s1_arr.size > 0:
                mask1 = (s1_arr[None, :] < t_arr[:, None])
                dt1 = np.maximum(t_arr[:, None] - s1_arr[None, :], 0.0)
                # clamp exponent to avoid overflow/underflow
                x1 = -l1 * dt1
                x1 = np.clip(x1, -745.0, 709.0)
                s1_contrib = A1 * np.exp(x1) * mask1
                s1_vec = s1_contrib.sum(axis=1)
        s2_vec = 0.0
        if S2:
            s2_arr = np.asarray(S2, dtype=float).reshape(-1)
            if s2_arr.size > 0:
                mask2 = (s2_arr[None, :] < t_arr[:, None])
                dt2 = np.maximum(t_arr[:, None] - s2_arr[None, :], 0.0)
                x2 = -l2 * dt2
                x2 = np.clip(x2, -745.0, 709.0)
                s2_contrib = A2 * np.exp(x2) * mask2
                s2_vec = s2_contrib.sum(axis=1)
        h = C + B * t_arr + s1_vec + s2_vec
        return [float(x) for x in h.tolist()]
    else:
        try:
            return [float(eval_at_time(float(ti))) for ti in t]  # type: ignore[arg-type]
        except TypeError as exc:  # t n'est pas itérable
            raise TypeError("'t' doit être un nombre réel ou une séquence de réels.") from exc


def build_h_function(parameters: Dict[str, float] | None = None) -> Function:
    """
    Fabrique un objet Function pour h(t) avec paramètres par défaut et priors.

    Paramètres par défaut:
        C=0.0, B=0.0, A1=1.0, lambda1=1.0, A2=1.0, lambda2=1.0

    Priors (indicatifs pour simulation):
        - C ~ Normal(0, 1)
        - B ~ Normal(0, 1)
        - A1, A2 ~ HalfNormal(1)
        - lambda1, lambda2 ~ Gamma(2, 1) [shape, rate]
    """
    default_parameters: Dict[str, float] = {
        "C": 0.0,
        "B": 0.0,
        "A1": 1.0,
        "lambda1": 1.0,
        "A2": 1.0,
        "lambda2": 1.0,
    }

    params = {**default_parameters, **(parameters or {})}

    sim_priors: Dict[str, Dict[str, Any]] = {
        "C": {"dist": "Normal", "mu": 0.0, "sigma": 1.0},
        "B": {"dist": "Normal", "mu": 0.0, "sigma": 1.0},
        "A1": {"dist": "HalfNormal", "sigma": 1.0},
        "lambda1": {"dist": "Gamma", "alpha": 2.0, "beta": 1.0},  # beta = rate
        "A2": {"dist": "HalfNormal", "sigma": 1.0},
        "lambda2": {"dist": "Gamma", "alpha": 2.0, "beta": 1.0},
    }

    from typing import Sequence as TSequence, Union as TUnion

    # Spécifications d'entrées avec types Python et ranges optionnels
    input_spec: Dict[str, Dict[str, Any]] = {
        "t": {
            "desc": "temps d'évaluation t (scalaire ou séquence)",
            "py_type": TUnion[float, TSequence[float]],
            "required": True,
            # Exemple de range (désactivé par défaut):
            # "range": {"min": 0.0, "max": None, "include_min": True, "include_max": True, "elementwise": True},
        },
        "S1": {
            "desc": "séquence d'instants d'événements pour S1",
            "py_type": TSequence[float],
            "required": True,
            # "range": {"min": 0.0, "max": None, "include_min": True, "include_max": True, "elementwise": True},
        },
        "S2": {
            "desc": "séquence d'instants d'événements pour S2",
            "py_type": TSequence[float],
            "required": True,
            # "range": {"min": 0.0, "max": None, "include_min": True, "include_max": True, "elementwise": True},
        },
    }

    # Spécifications de sortie
    output_spec: Dict[str, Dict[str, Any]] = {
        "h": {
            "desc": "valeur de h(t)",
            "py_type": Union[float, List[float]],
        }
    }

    return Function(
        name="h_linear_exp",
        parameters=params,
        input=input_spec,
        output=output_spec,
        sim_priors=sim_priors,
        _evaluator=_h_eval,
    )


# Exemple d'usage (à titre indicatif):
# from functions import build_h_function
# f = build_h_function({"C": 0.5, "B": 0.1, "A1": 2.0, "lambda1": 0.8, "A2": 1.5, "lambda2": 1.2})
# val = f.eval({"t": 3.0, "S1": [0.4, 1.2, 2.1], "S2": [0.7, 1.8]})
# vals = f.eval({"t": [0.0, 0.5, 1.0, 2.0, 3.0], "S1": [0.4, 1.2, 2.1], "S2": [0.7, 1.8]})
