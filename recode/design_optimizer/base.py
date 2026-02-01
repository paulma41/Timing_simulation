import numpy as np
from typing import Any, Dict, List, Mapping


def numerical_jacobian(
    f: Any,
    inputs: Dict[str, Any],
    params: Mapping[str, float],
    eps: float = 1e-5,
) -> List[List[float]]:
    """
    Central finite-difference Jacobian.
    f.eval(inputs) -> float or list[float]
    params: mapping of parameter name -> value
    """
    y0 = f.eval(inputs)
    y_list = y0 if isinstance(y0, list) else [float(y0)]

    pnames = list(params.keys())
    J = np.zeros((len(y_list), len(pnames)), dtype=float)
    
    for j, pname in enumerate(pnames):
        params_plus = dict(params)
        p0 = float(params[pname])
        delta = eps * max(1.0, abs(p0))
        params_plus[pname] = p0 + delta
        old_params = dict(f.parameters)
        try:
            f.parameters.update(params_plus)
            y_plus = f.eval(inputs)
        finally:
            f.parameters.update(old_params)
        y_plus_list = y_plus if isinstance(y_plus, list) else [float(y_plus)]

        params_minus = dict(params)
        params_minus[pname] = p0 - delta
        old_params = dict(f.parameters)
        try:
            f.parameters.update(params_minus)
            y_minus = f.eval(inputs)
        finally:
            f.parameters.update(old_params)
        y_minus_list = y_minus if isinstance(y_minus, list) else [float(y_minus)]

        J[:, j] = (np.asarray(y_plus_list) - np.asarray(y_minus_list)) / (2.0 * delta)

    return J.tolist()


def logdet_psd(M: List[List[float]], ridge: float = 1e-12) -> float:
    """
    Log-determinant for PSD matrices using slogdet with diagonal ridge.
    Returns -inf if non-positive definite after ridge.
    """
    A = np.asarray(M, dtype=float)
    if ridge and ridge > 0.0:
        A = A + ridge * np.eye(A.shape[0], dtype=float)
    sign, ld = np.linalg.slogdet(A)
    if sign <= 0.0:
        return float("-inf")
    return float(ld)
