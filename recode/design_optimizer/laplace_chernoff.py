"""
Laplace–Chernoff risk (PLOS Comp Biol 2011).

Implements Eq. (Laplace–Chernoff risk) from the paper summary:

    b_LC(u) = H(p(m))
             + 1/2 * ( sum_m p(m) log |Q~_m(u)|
                       - log | sum_m p(m) (Δg_m Δg_m^T + Q~_m(u)) | )

where:
  - g_m(μ_m, u) is the predictive mean for model m,
  - Δg_m = g_m(μ_m, u) - sum_{m'} p(m') g_{m'}(μ_{m'}, u),
  - Q~_m(u) = Q_m + J_m R_m J_m^T (effective predictive covariance),
  - H(p(m)) = -sum_m p(m) log p(m).

In this code:
  - g_m(μ_m, u) is the output of func.eval(design_inputs) with prior-mean params,
  - Q~_m(u) is approximated by the Laplace predictive covariance Vy_m.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .base import logdet_psd
from .laplace import laplace_predictive
from recode.models.h_actions import Function


def laplace_chernoff_risk(
    models: Sequence[Function],
    design_inputs: Mapping[str, Any],
    *,
    sigma: float = 1.0,
    eps: float = 1e-5,
    model_priors: Sequence[float] | None = None,
) -> float:
    """
    Compute the Laplace–Chernoff risk for model selection.

    """

    # --- predictive moments (Laplace)
    pred = laplace_predictive(
        models,
        design_inputs,
        sigma=sigma,
        eps=eps,
        n_jobs=1,
        return_lowrank=False,
    )
    mu_list = [np.asarray(m, dtype=float).reshape(-1) for m in pred["mu"]]
    Vy_list = [np.asarray(v, dtype=float) for v in pred["Vy"]]

    # --- model priors
    if model_priors is None:
        model_priors = [1.0 / len(models) for _ in models]
    priors = np.asarray(model_priors, dtype=float)
    priors = priors / priors.sum()

    # --- H(p(m))
    eps_p = 1e-12
    H = -float(np.sum(priors * np.log(np.clip(priors, eps_p, 1.0))))

    # --- mixture mean of predictive means
    mu_bar = np.zeros_like(mu_list[0], dtype=float)
    for p, mu in zip(priors, mu_list):
        mu_bar += p * mu

    # --- term: sum_m p(m) log |Q~_m|
    sum_logdet = 0.0
    for p, S in zip(priors, Vy_list):
        sum_logdet += p * logdet_psd(S.tolist())

    # --- term: log | sum_m p(m) (Δg_m Δg_m^T + Q~_m) |
    mix = np.zeros_like(Vy_list[0], dtype=float)
    for p, mu, S in zip(priors, mu_list, Vy_list):
        d = (mu - mu_bar).reshape(-1, 1)
        mix += p * (d @ d.T + S)
    logdet_mix = logdet_psd(mix.tolist())

    return float(H + 0.5 * (sum_logdet - logdet_mix))
