from __future__ import annotations

import math
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .base import logdet_psd


def _normalize_lowrank_entry(entry: Any, n_dim: int) -> Optional[Tuple[np.ndarray, float]]:
    """
    Normalize a low-rank covariance representation.

    Accepts:
      - (A, sigma2) tuple
      - {"A": A, "sigma2": sigma2} mapping
    where covariance ~= A A^T + sigma2 I.
    """
    if entry is None:
        return None
    if isinstance(entry, tuple) and len(entry) == 2:
        A, sigma2 = entry
    elif isinstance(entry, Mapping):
        A = entry.get("A")
        sigma2 = entry.get("sigma2")
    else:
        return None
    if A is None or sigma2 is None:
        return None
    A_arr = np.asarray(A, dtype=float)
    if A_arr.size == 0:
        A_arr = np.zeros((n_dim, 0), dtype=float)
    elif A_arr.ndim == 1:
        if n_dim <= 0 or (A_arr.size % n_dim) != 0:
            return None
        A_arr = A_arr.reshape(n_dim, -1)
    if A_arr.shape[0] != n_dim:
        return None
    try:
        sigma2_f = float(sigma2)
    except Exception:
        return None
    return A_arr, sigma2_f


def _lowrank_logdet(A: np.ndarray, sigma2: float) -> float:
    if sigma2 <= 0.0:
        return float("nan")
    n = A.shape[0]
    r = A.shape[1]
    if r == 0:
        return n * math.log(sigma2)
    AtA = A.T @ A
    M = np.eye(r, dtype=float) + (1.0 / sigma2) * AtA
    try:
        L = np.linalg.cholesky(M)
        ld = 2.0 * np.log(np.diag(L)).sum()
    except Exception:
        sign, ld = np.linalg.slogdet(M)
        if sign <= 0.0:
            return float("nan")
    return n * math.log(sigma2) + float(ld)


def _lowrank_solve(A: np.ndarray, sigma2: float, b: np.ndarray) -> Optional[np.ndarray]:
    if sigma2 <= 0.0:
        return None
    r = A.shape[1]
    if r == 0:
        return b / sigma2
    At = A.T
    AtA = At @ A
    M = np.eye(r, dtype=float) + (1.0 / sigma2) * AtA
    try:
        rhs = (1.0 / sigma2) * (At @ b)
        z = np.linalg.solve(M, rhs)
    except Exception:
        return None
    return (1.0 / sigma2) * (b - A @ z)


def _chernoff_gaussian_pair_lowrank(
    mu1: Sequence[float],
    mu2: Sequence[float],
    lr0: Tuple[np.ndarray, float],
    lr1: Tuple[np.ndarray, float],
    n_grid: int = 21,
) -> float:
    if n_grid <= 0:
        return float("nan")

    A0, sigma2_0 = lr0
    A1, sigma2_1 = lr1
    if sigma2_0 <= 0.0 or sigma2_1 <= 0.0:
        return float("nan")

    m0 = np.asarray(mu1, dtype=float).reshape(-1, 1)
    m1 = np.asarray(mu2, dtype=float).reshape(-1, 1)
    diff = m1 - m0
    n_dim = diff.shape[0]

    ld0 = _lowrank_logdet(A0, sigma2_0)
    ld1 = _lowrank_logdet(A1, sigma2_1)
    if not (math.isfinite(ld0) and math.isfinite(ld1)):
        return float("nan")

    best = float("-inf")
    for k in range(n_grid):
        s = float(k + 1) / float(n_grid + 1)
        sigma2_s = (1.0 - s) * sigma2_0 + s * sigma2_1
        if sigma2_s <= 0.0:
            continue

        w0 = math.sqrt(1.0 - s)
        w1 = math.sqrt(s)
        if A0.shape[1] == 0 and A1.shape[1] == 0:
            ld_s = n_dim * math.log(sigma2_s)
            q = float(diff.T @ diff) / sigma2_s
        else:
            if A0.shape[1] == 0:
                A_s = A1 * w1
            elif A1.shape[1] == 0:
                A_s = A0 * w0
            else:
                A_s = np.hstack((A0 * w0, A1 * w1))
            ld_s = _lowrank_logdet(A_s, sigma2_s)
            v = _lowrank_solve(A_s, sigma2_s, diff)
            if v is None or not math.isfinite(ld_s):
                continue
            q = float(diff.T @ v)

        term_quad = s * (1.0 - s) * q
        term_log = ld_s - ((1.0 - s) * ld0 + s * ld1)
        C_s = 0.5 * (term_quad + term_log)
        if math.isfinite(C_s) and C_s > best:
            best = C_s

    if best == float("-inf"):
        return float("nan")
    return max(0.0, best)


def _chernoff_gaussian_pair(
    mu1: Sequence[float],
    cov1: Sequence[Sequence[float]],
    mu2: Sequence[float],
    cov2: Sequence[Sequence[float]],
    n_grid: int = 21,
    lowrank0: Optional[Any] = None,
    lowrank1: Optional[Any] = None,
) -> float:
    """
    Chernoff information between two Gaussian predictive densities.

    C_s = 0.5 * [ s(1-s) (mu2-mu1)^T Sigma_s^{-1} (mu2-mu1)
                  + log det(Sigma_s) - (1-s) log det(Sigma_0) - s log det(Sigma_1) ]
    Sigma_s = (1-s) Sigma_0 + s Sigma_1
    """
    if n_grid <= 0:
        return float("nan")

    if lowrank0 is not None or lowrank1 is not None:
        lr0 = _normalize_lowrank_entry(lowrank0, len(mu1))
        lr1 = _normalize_lowrank_entry(lowrank1, len(mu2))
        if lr0 is not None and lr1 is not None:
            c_lr = _chernoff_gaussian_pair_lowrank(mu1, mu2, lr0, lr1, n_grid=n_grid)
            if math.isfinite(c_lr):
                return c_lr

    m0 = np.asarray(mu1, dtype=float).reshape(-1, 1)
    m1 = np.asarray(mu2, dtype=float).reshape(-1, 1)
    S0 = np.asarray(cov1, dtype=float)
    S1 = np.asarray(cov2, dtype=float)

    try:
        ld0 = logdet_psd(S0.tolist())
        ld1 = logdet_psd(S1.tolist())
    except Exception:
        return float("nan")

    diff = m1 - m0
    best = float("-inf")

    for k in range(n_grid):
        s = float(k + 1) / float(n_grid + 1)  # avoid 0 and 1
        try:
            Ss = (1.0 - s) * S0 + s * S1
            ld_s = logdet_psd(Ss.tolist())
            if not (math.isfinite(ld_s) and math.isfinite(ld0) and math.isfinite(ld1)):
                continue
            v = np.linalg.solve(Ss, diff)
            q = float(diff.T @ v)
        except Exception:
            continue
        term_quad = s * (1.0 - s) * q
        term_log = ld_s - ((1.0 - s) * ld0 + s * ld1)
        C_s = 0.5 * (term_quad + term_log)
        if math.isfinite(C_s) and C_s > best:
            best = C_s

    if best == float("-inf"):
        return float("nan")
    return max(0.0, best)


def _pairwise_chernoff_matrix(
    mu_list: Sequence[Sequence[float]],
    cov_list: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    n_grid: int = 21,
    cov_lowrank: Optional[Sequence[Any]] = None,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Build pairwise Chernoff information matrix C[i][j] and the associated
    upper bound on misclassification probability:
      P_err(i vs j) <= 0.5 * exp(-C[i][j])
    """
    m = len(mu_list)
    C = [[0.0 for _ in range(m)] for _ in range(m)]
    P = [[0.0 for _ in range(m)] for _ in range(m)]

    lowrank_arrs: Optional[List[Optional[Tuple[np.ndarray, float]]]] = None
    if cov_lowrank is not None and len(cov_lowrank) == m:
        lowrank_arrs = []
        for idx in range(m):
            lowrank_arrs.append(_normalize_lowrank_entry(cov_lowrank[idx], len(mu_list[idx])))

    for i in range(m):
        for j in range(i + 1, m):
            lr0 = lowrank_arrs[i] if lowrank_arrs is not None else None
            lr1 = lowrank_arrs[j] if lowrank_arrs is not None else None
            if cov_list is None:
                if lr0 is None or lr1 is None:
                    raise ValueError("cov_list is None and low-rank factors are missing")
                c_ij = _chernoff_gaussian_pair_lowrank(mu_list[i], mu_list[j], lr0, lr1, n_grid=n_grid)
            else:
                c_ij = _chernoff_gaussian_pair(
                    mu_list[i],
                    cov_list[i],
                    mu_list[j],
                    cov_list[j],
                    n_grid=n_grid,
                    lowrank0=lr0,
                    lowrank1=lr1,
                )
            C[i][j] = c_ij
            C[j][i] = c_ij
            P_val = 0.5 * math.exp(-c_ij) if math.isfinite(c_ij) else float("nan")
            P[i][j] = P_val
            P[j][i] = P_val
    return C, P


def maximin_chernoff(C: Sequence[Sequence[float]]) -> float:
    """
    Maximin score: for each row, take the minimum off-diagonal Chernoff,
    then return the minimum across rows.
    """
    m = len(C)
    row_min: List[float] = []
    for i in range(m):
        vals = [C[i][j] for j in range(m) if j != i and math.isfinite(C[i][j])]
        row_min.append(min(vals) if vals else float("-inf"))
    return min(row_min) if row_min else float("-inf")
