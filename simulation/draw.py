from __future__ import annotations

import math
import random
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union

from functions import Function


RngLike = Union[random.Random, int]


def _ensure_rng(rng: Optional[RngLike]) -> random.Random:
    if rng is None:
        return random.Random()
    if isinstance(rng, random.Random):
        return rng
    # assume int seed
    return random.Random(rng)


def _sample_normal(rng: random.Random, prior: Mapping[str, Any]) -> float:
    mu = float(prior.get("mu", 0.0))
    sigma = float(prior.get("sigma", 1.0))
    if sigma < 0:
        raise ValueError("Normal prior: sigma must be >= 0")
    return rng.gauss(mu, sigma)


def _sample_halfnormal(rng: random.Random, prior: Mapping[str, Any]) -> float:
    sigma = float(prior.get("sigma", 1.0))
    if sigma < 0:
        raise ValueError("HalfNormal prior: sigma must be >= 0")
    return abs(rng.gauss(0.0, sigma))


def _sample_gamma(rng: random.Random, prior: Mapping[str, Any]) -> float:
    # Accept various aliases for parameters
    alpha = prior.get("alpha")
    if alpha is None:
        alpha = prior.get("k", prior.get("shape", None))
    if alpha is None:
        raise ValueError("Gamma prior requires 'alpha'/'shape'/'k'")
    alpha = float(alpha)
    if alpha <= 0:
        raise ValueError("Gamma prior: alpha/shape must be > 0")

    # Support either rate (beta) or scale (theta/scale)
    rate = prior.get("beta", prior.get("rate", None))
    scale = prior.get("theta", prior.get("scale", None))
    if scale is None and rate is None:
        # default scale 1.0
        scale = 1.0
    if scale is None and rate is not None:
        rate = float(rate)
        if rate <= 0:
            raise ValueError("Gamma prior: rate/beta must be > 0")
        scale = 1.0 / rate
    scale = float(scale)
    if scale <= 0:
        raise ValueError("Gamma prior: scale/theta must be > 0")

    # random.gammavariate uses shape alpha and scale (theta)
    return rng.gammavariate(alpha, scale)


def draw_param(
    func: Function,
    rng: Optional[RngLike] = None,
    *,
    set_on_function: bool = True,
    override_priors: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, float]:
    """
    Échantillonne un dictionnaire de paramètres selon les priors de `func`.

    - func: instance de `functions.Function` (ex: build_h_function(...))
    - rng: None (nouveau RNG), int (seed) ou instance random.Random
    - set_on_function: si True, met à jour func.parameters
    - override_priors: dictionnaire pour remplacer/compléter func.sim_priors

    Priors supportés (dist, insensible à la casse):
      - Normal: mu, sigma
      - HalfNormal: sigma
      - Gamma: alpha (ou shape/k) et beta (rate) OU theta/scale

    Retourne le dict complet des paramètres après échantillonnage (copie).
    """
    rng_ = _ensure_rng(rng)
    priors: Mapping[str, Mapping[str, Any]] = func.sim_priors
    if override_priors:
        # shallow merge
        priors = {**priors, **override_priors}  # type: ignore[assignment]

    new_params: Dict[str, float] = dict(func.parameters)

    for name, prior in priors.items():
        dist = str(prior.get("dist", "")).strip().lower()
        if dist == "normal":
            new_params[name] = float(_sample_normal(rng_, prior))
        elif dist == "halfnormal":
            new_params[name] = float(_sample_halfnormal(rng_, prior))
        elif dist == "gamma":
            new_params[name] = float(_sample_gamma(rng_, prior))
        else:
            raise ValueError(f"Unsupported prior distribution for '{name}': {prior.get('dist')}")

    if set_on_function:
        func.parameters.update(new_params)

    return new_params


# Exemple rapide:
# from functions import build_h_function
# from simulation import draw_param
# f = build_h_function()
# sampled = draw_param(f, rng=123)
# val = f.eval({"t": 1.5, "S1": [0.1, 1.0], "S2": [0.4]})

