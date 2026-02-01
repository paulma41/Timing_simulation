from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import cloudpickle

from .base import numerical_jacobian
from recode.models.h_actions import Function

def _prior_mean_and_var(func: Function) -> tuple[list[float], list[float], list[str]]:
    ...
def _laplace_prior_predictive_gaussian(
    func: Function,
    design_inputs: Mapping[str, Any],
    sigma_noise: float,
    eps: float = 1e-5,
    return_lowrank: bool = False,
) -> tuple[list[float], list[list[float]], Optional[dict[str, Any]]]:
    ...
def laplace_predictive(
    models: Sequence[Function],
    design_inputs: Mapping[str, Any],
    *,
    sigma: float = 1.0,
    eps: float = 1e-5,
    n_jobs: int = 1,
    return_lowrank: bool = False,
) -> dict[str, Any]:
    ...
def _laplace_worker(
    models_blob: bytes,
    idx: int,
    design_inputs: Mapping[str, Any],
    sigma: float,
    eps: float,
    return_lowrank: bool,
) -> tuple[int, list[float], list[list[float]], Optional[dict[str, Any]]]:
    ...
