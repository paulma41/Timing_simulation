from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import cloudpickle

from .base import numerical_jacobian
from recode.models.h_actions import Function

def _prior_mean_and_var(func: Function) -> tuple[np.ndarray, np.ndarray, list[str]]:
    param_names = list(func.parameters.keys())
    N_params = len(param_names)
    mu = np.zeros((N_params), dtype=float)
    var = np.zeros((N_params), dtype=float)
    sim_priors = func.sim_priors
    for i_param,name in enumerate(param_names):
        mu[i_param] = sim_priors[name]["mu"]
        var[i_param] = sim_priors[name]["sigma"] **2
    return mu, var, param_names

def _laplace_prior_predictive_gaussian(
    func: Function,
    design_inputs: Mapping[str, Any],
    sigma_noise: float,
    eps: float = 1e-5,
    return_lowrank: bool = False,
    compute_full: bool = True,
) -> tuple[list[float], Optional[list[list[float]]], Optional[dict[str, Any]]]:
    
    mu, var, names = _prior_mean_and_var(func)
    Sigma = np.diag(var)
    params = dict(zip(names, mu))
    func.parameters.update(params)
    mu_y = np.asarray(func.eval(design_inputs), dtype=float).reshape(-1)
    n = mu_y.size
    I = np.identity(n, dtype=float)
    try:
        J = func.jacobian(design_inputs, params)
    except:
        J = numerical_jacobian(func, design_inputs, params, eps=eps)
    sigma2 = float(sigma_noise**2)
    R = sigma2 * I
    if compute_full:
        Vy = R + J @ Sigma @ J.T
    else:
        Vy = None
    if return_lowrank:
        sqrt_var = np.sqrt(np.diag(Sigma))
        A = J * sqrt_var[None, :]
        lowrank =  {"A": A.tolist(), "sigma2": sigma2}
    else:
        lowrank = None
    return mu_y, Vy, lowrank
    
def laplace_predictive(
    models: Sequence[Function],
    design_inputs: Mapping[str, Any],
    *,
    sigma: float = 1.0,
    eps: float = 1e-5,
    n_jobs: int = 1,
    return_lowrank: bool = False,
    compute_full: bool = True,
) -> dict[str, Any]:
    output = {"mu": [], "Vy": [], "lowrank": []}
    if n_jobs <= 1:
        for model in models:
            mu_y, Vy, lowrank = _laplace_prior_predictive_gaussian(
                model,
                design_inputs,
                sigma,
                eps,
                return_lowrank,
                compute_full,
            )
            output["mu"].append(mu_y)
            output["Vy"].append(Vy)
            output["lowrank"].append(lowrank)
    if n_jobs > 1:
        models_blob = cloudpickle.dumps(models)
        # pool.map / executor.map sur idx
        import multiprocessing as mp
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.starmap(
                _laplace_worker,
                [
                    (models_blob, i, design_inputs, sigma, eps, return_lowrank, compute_full)
                    for i in range(len(models))
                ],
            )
        results.sort(key=lambda x: x[0])
        for _, mu_y, Vy, lowrank in results:
            output["mu"].append(mu_y)
            output["Vy"].append(Vy)
            output["lowrank"].append(lowrank)
    return output

def _laplace_worker(
    models_blob: bytes,
    idx: int,
    design_inputs: Mapping[str, Any],
    sigma: float,
    eps: float,
    return_lowrank: bool,
    compute_full: bool,
) -> tuple[int, list[float], Optional[list[list[float]]], Optional[dict[str, Any]]]:
    models: list[Function] = cloudpickle.loads(models_blob)
    model = models[idx]
    mu_y, Vy, lowrank = _laplace_prior_predictive_gaussian(
        model, design_inputs, sigma, eps, return_lowrank, compute_full
    )
    return idx, mu_y, Vy, lowrank
