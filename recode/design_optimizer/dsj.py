# TO CHECK

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .base import logdet_psd


def jsd_bound_pair(
    mu0: Sequence[float],
    cov0: Sequence[Sequence[float]],
    mu1: Sequence[float],
    cov1: Sequence[Sequence[float]],
) -> float:
    """
    Jensen-Shannon bound b for the 2-model (uniform priors) case, base-2.
    b = 1 - DJS, where DJS = Hy - sH (see VBA_JensenShannon convention).
    """
    mu0_arr = np.asarray(mu0, dtype=float).reshape(-1)
    mu1_arr = np.asarray(mu1, dtype=float).reshape(-1)
    S0 = np.asarray(cov0, dtype=float)
    S1 = np.asarray(cov1, dtype=float)

    ld0 = logdet_psd(S0.tolist())
    ld1 = logdet_psd(S1.tolist())
    if not (math.isfinite(ld0) and math.isfinite(ld1)):
        return float("nan")

    logdet2_0 = ld0 / math.log(2.0)
    logdet2_1 = ld1 / math.log(2.0)
    sH = 0.25 * (logdet2_0 + logdet2_1)

    mu_bar = 0.5 * (mu0_arr + mu1_arr)
    d0 = (mu0_arr - mu_bar).reshape(-1, 1)
    d1 = (mu1_arr - mu_bar).reshape(-1, 1)
    mix = 0.5 * (d0 @ d0.T + S0) + 0.5 * (d1 @ d1.T + S1)
    ld_mix = logdet_psd(mix.tolist())
    if not math.isfinite(ld_mix):
        return float("nan")

    Hy = 0.5 * (ld_mix / math.log(2.0))
    DJS = Hy - sH
    return max(float("-inf"), 1.0 - DJS)
