from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Optional, Callable

try:
    import numpy as np  # type: ignore
    HAVE_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    HAVE_NUMPY = False

try:
    import cloudpickle  # type: ignore
    HAVE_CLOUDPICKLE = True
except Exception:  # pragma: no cover
    cloudpickle = None  # type: ignore
    HAVE_CLOUDPICKLE = False

from functions import Function
from .base import numerical_jacobian, logdet_psd


def _prior_mean_and_var(func: Function) -> Tuple[List[float], List[float], List[str]]:
    """
    Extract per-parameter prior mean/variance from func.sim_priors.
    Falls back to current parameter value with zero variance if no prior.
    """
    means: List[float] = []
    vars_: List[float] = []
    names: List[str] = []

    for name, val in func.parameters.items():
        names.append(name)
        prior = func.sim_priors.get(name) if isinstance(func.sim_priors, Mapping) else None
        if prior is not None and str(prior.get("dist", "")).strip().lower() == "normal":
            mu = float(prior.get("mu", val))
            sigma = float(prior.get("sigma", 0.0))
            means.append(mu)
            vars_.append(max(0.0, sigma * sigma))
        else:
            means.append(float(val))
            vars_.append(0.0)

    return means, vars_, names


def _laplace_prior_predictive_gaussian(
    func: Function,
    design_inputs: Mapping[str, Any],
    sigma_noise: float,
    eps: float = 1e-5,
    use_analytic_jac: bool = True,
    return_lowrank: bool = False,
) -> Tuple[List[float], List[List[float]], Optional[Dict[str, Any]]]:
    """
    Laplace approximation of the prior predictive p(y|M) by linearizing
    the deterministic model output around the prior mean of parameters.

    y(theta) ≈ y(mu_theta) + J (theta - mu_theta),
    theta ~ N(mu_theta, Sigma_theta)  (diagonal here),
    so y ~ N(muy, Vy) with
      muy = y(mu_theta),
      Vy  = J Sigma_theta J^T + sigma_noise^2 I.

    Returns (muy, Vy, lowrank). lowrank is a dict with keys 'A' and 'sigma2'
    when return_lowrank is True, otherwise None.
    """
    prior_means, prior_vars, names = _prior_mean_and_var(func)
    theta_mean: Dict[str, float] = {name: prior_means[i] for i, name in enumerate(names)}

    # Evaluate model at prior mean
    old_params = dict(func.parameters)
    try:
        func.parameters.update(theta_mean)
        y0 = func.eval(dict(design_inputs))
        y_list: List[float] = y0 if isinstance(y0, list) else [float(y0)]

        # Jacobienne analytique si disponible
        if use_analytic_jac and hasattr(func, "jacobian"):
            try:
                J = func.jacobian(dict(design_inputs), dict(func.parameters))  # type: ignore[attr-defined]
            except Exception as exc:
                func_name = getattr(func, "name", "<unnamed>")
                warnings.warn(
                    f"Analytic jacobian failed for {func_name}; falling back to numerical_jacobian: {exc}",
                    RuntimeWarning,
                )
                J = numerical_jacobian(func, dict(design_inputs), theta_mean, eps=eps)
        else:
            J = numerical_jacobian(func, dict(design_inputs), theta_mean, eps=eps)
    finally:
        func.parameters.update(old_params)

    n_y = len(y_list)
    p = len(names)

    # Build covariance Vy = J diag(var) J^T + sigma^2 I
    if HAVE_NUMPY:
        Jm = np.asarray(J, dtype=float)
        var_arr = np.asarray(prior_vars, dtype=float)
        if p > 0:
            # Scale columns by sqrt(var)
            sqrt_var = np.sqrt(np.maximum(var_arr, 0.0))
            A = Jm * sqrt_var[None, :]
            Vy = A @ A.T
        else:
            A = np.zeros((n_y, 0), dtype=float)
            Vy = np.zeros((n_y, n_y), dtype=float)
        sigma2 = float(sigma_noise) ** 2
        if sigma_noise > 0.0:
            Vy = Vy + sigma2 * np.eye(n_y, dtype=float)
        lowrank = None
        if return_lowrank:
            lowrank = {"A": A.tolist(), "sigma2": sigma2}
        return y_list, Vy.tolist(), lowrank

    # Fallback pure Python
    Vy_py: List[List[float]] = [[0.0 for _ in range(n_y)] for _ in range(n_y)]
    for i in range(n_y):
        for k in range(n_y):
            s = 0.0
            for j in range(p):
                vj = prior_vars[j]
                if vj == 0.0:
                    continue
                s += J[i][j] * J[k][j] * vj
            if i == k and sigma_noise > 0.0:
                s += float(sigma_noise) ** 2
            Vy_py[i][k] = s

    return y_list, Vy_py, None


def _jensen_shannon_gaussians(
    mus: Sequence[Sequence[float]],
    covs: Sequence[Sequence[Sequence[float]]],
) -> Tuple[float, float]:
    """
    Jensen-Shannon divergence between Gaussian predictive densities,
    following VBA_JensenShannon for sources.type == 0 (gaussian).

    Returns (DJS, b) where b is the associated bound term Hp - DJS
    in base 2 (same convention as VBA_JensenShannon).
    """
    n = len(mus)
    if n == 0:
        return float("nan"), float("nan")
    p = len(mus[0])
    if any(len(m) != p for m in mus):
        raise ValueError("All mean vectors must have same length")
    if any(len(C) != p or any(len(row) != p for row in C) for C in covs):
        raise ValueError("All covariance matrices must be p x p")

    # Uniform model weights
    w = [1.0 / float(n)] * n

    # Helper: log2(det(.)) using logdet_psd (natural log)
    def _logdet2(M: List[List[float]]) -> float:
        ld = logdet_psd(M)
        if not math.isfinite(ld):
            return float("-inf")
        return ld / math.log(2.0)

    # Sum of weighted entropies (up to additive constants)
    sH = 0.0
    for i in range(n):
        logdet_Qi = _logdet2(covs[i])  # base 2
        sH += 0.5 * w[i] * logdet_Qi

    # Mixture mean
    muy_mix: List[float] = [0.0 for _ in range(p)]
    for i in range(n):
        mi = mus[i]
        wi = w[i]
        for d in range(p):
            muy_mix[d] += wi * float(mi[d])

    # Mixture covariance Vy = sum_i w_i (Q_i + (mu_i - muy)(mu_i - muy)^T)
    if HAVE_NUMPY:
        mus_arr = [np.asarray(m, dtype=float).reshape(p, 1) for m in mus]
        muy = np.asarray(muy_mix, dtype=float).reshape(p, 1)
        Vy = np.zeros((p, p), dtype=float)
        for i in range(n):
            diff = mus_arr[i] - muy
            Qi = np.asarray(covs[i], dtype=float)
            Vy += w[i] * (Qi + diff @ diff.T)
        Vy_list = Vy.tolist()
    else:
        Vy_list = [[0.0 for _ in range(p)] for _ in range(p)]
        for i in range(n):
            wi = w[i]
            mi = mus[i]
            Qi = covs[i]
            for a in range(p):
                for b in range(p):
                    diff_a = float(mi[a]) - muy_mix[a]
                    diff_b = float(mi[b]) - muy_mix[b]
                    Vy_list[a][b] += wi * (Qi[a][b] + diff_a * diff_b)

    Hy = 0.5 * _logdet2(Vy_list)
    DJS = Hy - sH

    # Bound term Hp - DJS (Hp: entropy of weights, here uniform)
    Hp = math.log2(float(n)) if n > 0 else 0.0
    b = max(-math.inf, Hp - DJS)

    return DJS, b


def laplace_jsd_for_design(
    models: Sequence[Function],
    design_inputs: Mapping[str, Any],
    *,
    sigma: float = 1.0,
    eps: float = 1e-5,
    n_jobs: int = 1,
    compute_jsd: bool = True,
    progress_cb: Optional[Callable[[int], None]] = None,
    return_lowrank: bool = False,
) -> Dict[str, Any]:
    """
    Laplace/Jensen-Shannon design efficiency for model comparison.

    For each model, approximates the prior predictive p(y|M_k) by a
    Gaussian N(muy_k, Vy_k) using a Laplace (delta method) expansion,
    then computes the Jensen-Shannon divergence between these densities
    as in VBA_designEfficiency (flag='models').

    Returns a dict with:
      - 'DJS': Jensen-Shannon divergence (base-2, up to constants)
      - 'b': bound term Hp - DJS as in VBA_JensenShannon
      - 'mu_y': list of mean vectors per model
      - 'Vy': list of covariance matrices per model
      - 'Vy_lowrank' (optional): list of low-rank cov factors when requested
    If compute_jsd is False, only mu_y and Vy are computed; DJS/b/U_pair/B_pair
    are returned as None/empty to skip extra work.
    """
    if not models:
        raise ValueError("'models' must be a non-empty sequence")

    mu_y_list: List[List[float]] = []
    Vy_list: List[List[List[float]]] = []
    lowrank_list: Optional[List[Optional[Dict[str, Any]]]] = [] if return_lowrank else None

    jobs = max(1, int(n_jobs)) if n_jobs is not None else 1

    if jobs == 1:
        for f in models:
            mu_y, Vy, lowrank = _laplace_prior_predictive_gaussian(
                f,
                design_inputs,
                sigma_noise=float(sigma),
                eps=eps,
                return_lowrank=return_lowrank,
            )
            mu_y_list.append(mu_y)
            Vy_list.append(Vy)
            if return_lowrank and lowrank_list is not None:
                lowrank_list.append(lowrank)
            if progress_cb is not None:
                progress_cb(1)
    else:
        if not HAVE_CLOUDPICKLE or cloudpickle is None:  # pragma: no cover
            raise RuntimeError("cloudpickle est requis pour n_jobs > 1 (Laplace-JSD).")
        try:
            from concurrent.futures import ProcessPoolExecutor
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("concurrent.futures est requis pour n_jobs > 1 (Laplace-JSD).") from exc
        models_blob = cloudpickle.dumps(list(models))
        with ProcessPoolExecutor(max_workers=min(jobs, len(models))) as ex:
            futs = [
                ex.submit(_laplace_jsd_worker, models_blob, i, design_inputs, float(sigma), eps, return_lowrank)
                for i in range(len(models))
            ]
            tmp: Dict[int, Tuple[List[float], List[List[float]], Optional[Dict[str, Any]]]] = {}
            for fut in futs:
                idx, mu_y, Vy, lowrank = fut.result()
                tmp[idx] = (mu_y, Vy, lowrank)
                if progress_cb is not None:
                    progress_cb(1)
            for i in range(len(models)):
                mu_y, Vy, lowrank = tmp[i]
                mu_y_list.append(mu_y)
                Vy_list.append(Vy)
                if return_lowrank and lowrank_list is not None:
                    lowrank_list.append(lowrank)

    if not compute_jsd:
        res = {"DJS": None, "b": None, "mu_y": mu_y_list, "Vy": Vy_list, "U_pair": [], "B_pair": []}
        if return_lowrank:
            res["Vy_lowrank"] = lowrank_list
        return res

    DJS, b = _jensen_shannon_gaussians(mu_y_list, Vy_list)
    U_pair, B_pair = _pairwise_jsd_matrix(mu_y_list, Vy_list)
    res = {"DJS": DJS, "b": b, "mu_y": mu_y_list, "Vy": Vy_list, "U_pair": U_pair, "B_pair": B_pair}
    if return_lowrank:
        res["Vy_lowrank"] = lowrank_list
    return res


def _laplace_jsd_worker(
    models_blob: bytes,
    idx: int,
    design_inputs: Mapping[str, Any],
    sigma: float,
    eps: float,
    return_lowrank: bool = False,
) -> Tuple[int, List[float], List[List[float]], Optional[Dict[str, Any]]]:
    """
    Worker picklable pour paralléliser laplace_jsd_for_design.
    """
    if not HAVE_CLOUDPICKLE or cloudpickle is None:  # pragma: no cover
        raise RuntimeError("cloudpickle requis pour le parallélisme Laplace-JSD")
    models_local: Sequence[Function] = cloudpickle.loads(models_blob)
    mu_y, Vy, lowrank = _laplace_prior_predictive_gaussian(
        models_local[idx],
        design_inputs,
        sigma_noise=float(sigma),
        eps=eps,
        return_lowrank=return_lowrank,
    )
    return idx, mu_y, Vy, lowrank


def _pairwise_jsd_matrix(mu_list: Sequence[Sequence[float]], cov_list: Sequence[Sequence[Sequence[float]]]) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Build the pairwise Jensen-Shannon matrix U[i][j] and a legacy bound
    matrix B[i][j] = Hp - DJS (Hp=1, base2) for 2-model JSD.
    This B matrix is kept for backward compatibility but is not used in
    the newer Chernoff-based bounds.
    """
    m = len(mu_list)
    U = [[0.0 for _ in range(m)] for _ in range(m)]
    B = [[0.0 for _ in range(m)] for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            djs, _ = _jensen_shannon_gaussians([mu_list[i], mu_list[j]], [cov_list[i], cov_list[j]])
            U[i][j] = djs
            B[i][j] = max(-math.inf, 1.0 - djs)  # Hp=1 (base2) for 2 models
    return U, B


def _normalize_lowrank_entry(entry: Any, n_dim: int) -> Optional[Tuple["np.ndarray", float]]:
    if not HAVE_NUMPY or np is None:
        return None
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


def _lowrank_logdet(A: "np.ndarray", sigma2: float) -> float:
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


def _lowrank_solve(A: "np.ndarray", sigma2: float, b: "np.ndarray") -> Optional["np.ndarray"]:
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
    lr0: Tuple["np.ndarray", float],
    lr1: Tuple["np.ndarray", float],
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
    Approximate Chernoff information C(P,Q) between two Gaussian predictive
    densities N(mu1, cov1) and N(mu2, cov2) by a grid search over s in (0,1).

    For Gaussian models, the Chernoff exponent at s is:
      C_s = 0.5 * [ s(1-s) (mu2-mu1)^T Sigma_s^{-1} (mu2-mu1)
                    + log det(Sigma_s) - (1-s) log det(Sigma_0) - s log det(Sigma_1) ]
      where Sigma_s = (1-s) Sigma_0 + s Sigma_1.

    We return max_s C_s over a uniform grid in (0,1).
    Values are computed in natural log units, so an error bound is
      P_err <= 0.5 * exp(-C)  (priors égales).
    """
    if n_grid <= 0:
        return float("nan")
    if not HAVE_NUMPY or np is None:
        # Chernoff approx requiert numpy pour la linalg; si indisponible, on renvoie NaN.
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
        s = float(k + 1) / float(n_grid + 1)  # évite 0 et 1
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
    cov_list: Sequence[Sequence[Sequence[float]]],
    n_grid: int = 21,
    cov_lowrank: Optional[Sequence[Any]] = None,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Build pairwise Chernoff information matrix C[i][j] and the associated
    upper bound on misclassification probability:
      P_err(i vs j) <= 0.5 * exp(-C[i][j])
    computed from Laplace Gaussian summaries.
    If cov_lowrank is provided, it is used to accelerate logdet/solve.
    """
    m = len(mu_list)
    C = [[0.0 for _ in range(m)] for _ in range(m)]
    P = [[0.0 for _ in range(m)] for _ in range(m)]
    lowrank_arrs: Optional[List[Optional[Tuple["np.ndarray", float]]]] = None
    if cov_lowrank is not None and len(cov_lowrank) == m and HAVE_NUMPY and np is not None:
        lowrank_arrs = []
        for idx in range(m):
            lowrank_arrs.append(_normalize_lowrank_entry(cov_lowrank[idx], len(mu_list[idx])))
    for i in range(m):
        for j in range(i + 1, m):
            lr0 = lowrank_arrs[i] if lowrank_arrs is not None else None
            lr1 = lowrank_arrs[j] if lowrank_arrs is not None else None
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


def laplace_jsd_matrix_for_design(
    models: Sequence[Function],
    design_inputs: Mapping[str, Any],
    *,
    sigma: float = 1.0,
    eps: float = 1e-5,
    n_jobs: int = 1,
) -> Dict[str, Any]:
    """
    Convenience wrapper returning the full pairwise JSD matrix for one design.
    """
    lap = laplace_jsd_for_design(models, design_inputs, sigma=float(sigma), eps=eps, n_jobs=n_jobs)
    mu_list = lap.get("mu_y", [])
    cov_list = lap.get("Vy", [])
    if not mu_list or not cov_list:
        return {"DJS": lap.get("DJS"), "b": lap.get("b"), "U": [], "mu_y": mu_list, "Vy": cov_list}
    U_mat, B_mat = _pairwise_jsd_matrix(mu_list, cov_list)
    return {
        "DJS": lap.get("DJS"),
        "b": lap.get("b"),
        "U": U_mat,
        "B": B_mat,
        "mu_y": mu_list,
        "Vy": cov_list,
    }


def _aggregate_pairwise(U: Sequence[Sequence[float]], objective: str) -> float:
    """
    Aggregate a pairwise matrix into a scalar score.
    objective: 'maximin' (default), 'avgmin', or 'mean' (average off-diagonal).
    """
    if not U:
        return float("nan")
    m = len(U)
    row_min: List[float] = []
    off_diag: List[float] = []
    for i in range(m):
        vals = [float(U[i][j]) for j in range(m) if j != i and math.isfinite(float(U[i][j]))]
        if vals:
            row_min.append(min(vals))
            off_diag.extend(vals)
    if objective == "maximin":
        return min(row_min) if row_min else float("nan")
    if objective == "avgmin":
        return sum(row_min) / len(row_min) if row_min else float("nan")
    if objective == "mean":
        return sum(off_diag) / len(off_diag) if off_diag else float("nan")
    raise ValueError(f"Unknown objective '{objective}' (expected 'maximin', 'avgmin', or 'mean').")


def laplace_jsd_separability(
    models: Sequence[Function],
    designs: Sequence[Mapping[str, Any]],
    *,
    optimizer: str = "laplace-jsd",
    sigma: float = 1.0,
    eps: float = 1e-5,
    n_jobs: int = 1,
    objective: str = "maximin",
) -> Dict[str, Any]:
    """
    Offline separability analysis over a list of candidate designs.

    For each design, compute the Laplace/Jensen-Shannon pairwise matrix (optimizer='laplace-jsd')
    and aggregate it into a scalar score using the chosen objective.
    """
    if optimizer != "laplace-jsd":
        raise ValueError("Only optimizer='laplace-jsd' is supported for Laplace separability.")
    best_idx: Optional[int] = None
    best_score = float("-inf")
    per_design: List[Dict[str, Any]] = []
    for idx, d in enumerate(designs):
        res = laplace_jsd_matrix_for_design(models, d, sigma=sigma, eps=eps, n_jobs=n_jobs)
        U = res.get("U", [])
        score = _aggregate_pairwise(U, objective)
        per_design.append({"index": idx, "design": d, "score": score, "U": U, "details": res})
        if math.isfinite(score) and (best_idx is None or score > best_score):
            best_idx = idx
            best_score = score
    return {
        "optimizer": optimizer,
        "objective": objective,
        "per_design": per_design,
        "best_index": best_idx,
        "best_score": best_score if best_idx is not None else float("nan"),
        "best_design": designs[best_idx] if best_idx is not None else None,
    }


__all__ = ["laplace_jsd_for_design", "laplace_jsd_matrix_for_design", "laplace_jsd_separability"]
