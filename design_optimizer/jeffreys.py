from __future__ import annotations

import math
import random
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Mapping, Optional, Union, List, Sequence, Tuple

try:
    import numpy as np  # type: ignore
    HAVE_NUMPY = True
except Exception:  # pragma: no cover
    HAVE_NUMPY = False

try:
    import cloudpickle  # type: ignore
    HAVE_CLOUDPICKLE = True
except Exception:  # pragma: no cover
    cloudpickle = None  # type: ignore
    HAVE_CLOUDPICKLE = False

from functions import Function
from simulation import draw_param
from .base import ensure_rng, normalize_fixed_input, generate_candidate_inputs

_PARALLEL_MODELS: Optional[List[Function]] = None


def _loglik_gaussian(y: List[float], mu: List[float], sigma: float) -> float:
    n = len(y)
    if n != len(mu):
        raise ValueError("y and mu must have same length")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if HAVE_NUMPY:
        yy = np.asarray(y, dtype=float)
        mm = np.asarray(mu, dtype=float)
        r = yy - mm
        s2 = float(sigma) ** 2
        return float(-0.5 * (n * math.log(2.0 * math.pi * s2) + (r @ r) / s2))
    else:
        s2 = float(sigma) ** 2
        rss = 0.0
        for i in range(n):
            d = y[i] - mu[i]
            rss += d * d
        return -0.5 * (n * math.log(2.0 * math.pi * s2) + rss / s2)


def _logmeanexp(vals: List[float]) -> float:
    if not vals:
        return float('-inf')
    m = max(vals)
    if m == float('-inf'):
        return float('-inf')
    s = sum(math.exp(v - m) for v in vals)
    return m + math.log(s / len(vals))


def _compute_bayes_row(
    idx: int,
    models: Sequence[Function],
    design_inputs: Mapping[str, Any],
    sigma: float,
    K_outer: int,
    K_inner: int,
    rng_seed: Optional[Union[int, random.Random]],
    *,
    resample_params: bool = True,
) -> Tuple[int, List[float]]:
    """
    Monte-Carlo estimation d'une ligne de la matrice de Jeffreys.

    Retourne (idx, row_values) pour faciliter l'utilisation dans un pool.
    """
    rng_local = ensure_rng(rng_seed)
    Mi = models[idx]
    m = len(models)
    row_sum = [0.0 for _ in range(m)]
    outer_samples = max(1, int(K_outer))
    inner_samples = max(1, int(K_inner))

    if not resample_params:
        # On suppose que les params ont été fixés en amont ; on ne resample pas ici.
        pass

    for _ in range(outer_samples):
        if resample_params:
            draw_param(Mi, rng=rng_local, set_on_function=True)
        y_mu = Mi.eval(dict(design_inputs))
        y_mu_list = y_mu if isinstance(y_mu, list) else [float(y_mu)]
        if HAVE_NUMPY:
            noise = (np.random.randn(len(y_mu_list)) * float(sigma)).tolist()
        else:
            noise = [rng_local.gauss(0.0, float(sigma)) for _ in y_mu_list]
        y_obs = [m_ + e for m_, e in zip(y_mu_list, noise)]

        lp_vals: List[float] = []
        for Mj in models:
            log_pred_list: List[float] = []
            for _ in range(inner_samples):
                if resample_params:
                    draw_param(Mj, rng=rng_local, set_on_function=True)
                mu = Mj.eval(dict(design_inputs))
                mu_list = mu if isinstance(mu, list) else [float(mu)]
                log_pred_list.append(_loglik_gaussian(y_obs, mu_list, sigma))
            lp_vals.append(_logmeanexp(log_pred_list))

        lp_i = lp_vals[idx]
        for j in range(m):
            row_sum[j] += lp_i - lp_vals[j]

    denom = float(outer_samples)
    row = [val / denom for val in row_sum]
    return idx, row


def _compute_bayes_row_adaptive(
    idx: int,
    models: Sequence[Function],
    design_inputs: Mapping[str, Any],
    sigma: float,
    outer_min: int,
    outer_max: int,
    inner_min: int,
    inner_max: int,
    tol_rel: float,
    tol_abs: float,
    rng_seed: Optional[Union[int, random.Random]],
    *,
    resample_params: bool = True,
) -> Tuple[int, List[float]]:
    """
    Variante adaptative par ligne : on augmente outer (et potentiellement inner)
    jusqu'à stabiliser les estimations de U[i][j] selon une tolérance.
    """
    rng_local = ensure_rng(rng_seed)
    Mi = models[idx]
    m = len(models)
    mean: List[float] = [0.0 for _ in range(m)]
    m2: List[float] = [0.0 for _ in range(m)]

    outer_samples = 0
    inner_samples = max(1, int(inner_min))
    outer_cap = max(outer_min, outer_max, 1)
    inner_cap = max(inner_min, inner_max, 1)

    def _should_stop() -> bool:
        if outer_samples < max(outer_min, 1):
            return False
        if outer_samples <= 1:
            return False
        max_mean = max(abs(v) for v in mean) if mean else 0.0
        threshold_global = tol_abs + tol_rel * max_mean
        max_se = 0.0
        for j in range(m):
            var = m2[j] / (outer_samples - 1) if outer_samples > 1 else float("inf")
            se = math.sqrt(max(var, 0.0)) / math.sqrt(outer_samples)
            max_se = max(max_se, se)
        return max_se <= threshold_global

    if not resample_params:
        # Pas de resampling si flag désactivé
        pass

    while outer_samples < outer_cap:
        if resample_params:
            draw_param(Mi, rng=rng_local, set_on_function=True)
        y_mu = Mi.eval(dict(design_inputs))
        y_mu_list = y_mu if isinstance(y_mu, list) else [float(y_mu)]
        if HAVE_NUMPY:
            noise = (np.random.randn(len(y_mu_list)) * float(sigma)).tolist()
        else:
            noise = [rng_local.gauss(0.0, float(sigma)) for _ in y_mu_list]
        y_obs = [m_ + e for m_, e in zip(y_mu_list, noise)]

        lp_vals: List[float] = []
        for Mj in models:
            log_pred_list: List[float] = []
            for _ in range(inner_samples):
                if resample_params:
                    draw_param(Mj, rng=rng_local, set_on_function=True)
                mu = Mj.eval(dict(design_inputs))
                mu_list = mu if isinstance(mu, list) else [float(mu)]
                log_pred_list.append(_loglik_gaussian(y_obs, mu_list, sigma))
            lp_vals.append(_logmeanexp(log_pred_list))

        lp_i = lp_vals[idx]
        outer_samples += 1
        for j in range(m):
            x = lp_i - lp_vals[j]
            delta = x - mean[j]
            mean[j] += delta / outer_samples
            m2[j] += delta * (x - mean[j])

        if _should_stop():
            break
        if inner_samples < inner_cap and outer_samples >= max(2, outer_min // 2):
            inner_samples = min(inner_samples * 2, inner_cap)

    return idx, mean


def _parallel_models_initializer(models_blob: bytes) -> None:
    """Initializer for worker processes to hydrate shared models."""
    global _PARALLEL_MODELS
    if not HAVE_CLOUDPICKLE or cloudpickle is None:  # pragma: no cover
        raise RuntimeError("cloudpickle is required for process-based parallelism.")
    _PARALLEL_MODELS = cloudpickle.loads(models_blob)


def _compute_bayes_row_parallel(
    idx: int,
    design_inputs: Mapping[str, Any],
    sigma: float,
    K_outer: int,
    K_inner: int,
    rng_seed: int,
    adaptive: bool,
    outer_min: int,
    outer_max: int,
    inner_min: int,
    inner_max: int,
    tol_rel: float,
    tol_abs: float,
    resample_params: bool,
) -> Tuple[int, List[float]]:
    if _PARALLEL_MODELS is None:
        raise RuntimeError("Parallel models not initialized in worker process.")
    if adaptive:
        return _compute_bayes_row_adaptive(
            idx,
            _PARALLEL_MODELS,
            design_inputs,
            sigma,
            outer_min,
            outer_max,
            inner_min,
            inner_max,
            tol_rel,
            tol_abs,
            rng_seed,
            resample_params=resample_params,
        )
    return _compute_bayes_row(
        idx,
        _PARALLEL_MODELS,
        design_inputs,
        sigma,
        K_outer,
        K_inner,
        rng_seed,
        resample_params=resample_params,
    )


def expected_log_bayes_factor_for_design(
    f_true: Function,
    f_alt: Function,
    design_inputs: Mapping[str, Any],
    *,
    sigma: float = 1.0,
    K_outer: int = 50,
    K_inner: int = 50,
    rng: Optional[Union[int, random.Random]] = None,
) -> float:
    """
    Jeffreys utility: E_{theta_true,y ~ M_true}[ log p(y|M_true) - log p(y|M_alt) ]
    Approximée par Monte-Carlo.

    - f_true, f_alt: objets Function pour les deux modèles.
    - design_inputs: design fixé (dict t,S1,S2...).
    - sigma: écart-type du bruit d'observation (Gaussien, iid).
    - K_outer: nb d'échantillons pour theta_true et y.
    - K_inner: nb d'échantillons de paramètres par modèle pour approx. prédictive.
    """
    rng_ = ensure_rng(rng)

    # snap current params to restore later
    true_params_backup = dict(f_true.parameters)
    alt_params_backup = dict(f_alt.parameters)
    utility_vals: List[float] = []

    for _ in range(int(K_outer)):
        # Sample true parameters and generate synthetic observation
        draw_param(f_true, rng=rng_, set_on_function=True)
        y_mu = f_true.eval(dict(design_inputs))
        y_mu_list = y_mu if isinstance(y_mu, list) else [float(y_mu)]

        # Simulate noise
        if HAVE_NUMPY:
            noise = (np.random.randn(len(y_mu_list)) * float(sigma)).tolist()
        else:
            noise = [rng_.gauss(0.0, float(sigma)) for _ in y_mu_list]
        y_obs = [m + e for m, e in zip(y_mu_list, noise)]

        # Approximate predictive log-likelihood under each model via MC over params
        log_pred_true: List[float] = []
        log_pred_alt: List[float] = []

        for _ in range(int(K_inner)):
            # True model param sample
            draw_param(f_true, rng=rng_, set_on_function=True)
            mu_t = f_true.eval(dict(design_inputs))
            mu_t_list = mu_t if isinstance(mu_t, list) else [float(mu_t)]
            log_pred_true.append(_loglik_gaussian(y_obs, mu_t_list, sigma))

            # Alt model param sample
            draw_param(f_alt, rng=rng_, set_on_function=True)
            mu_a = f_alt.eval(dict(design_inputs))
            mu_a_list = mu_a if isinstance(mu_a, list) else [float(mu_a)]
            log_pred_alt.append(_loglik_gaussian(y_obs, mu_a_list, sigma))

        # Log predictive densities (log of mean likelihood)
        lp_true = _logmeanexp(log_pred_true)
        lp_alt = _logmeanexp(log_pred_alt)
        utility_vals.append(lp_true - lp_alt)

    # restore params
    f_true.parameters.update(true_params_backup)
    f_alt.parameters.update(alt_params_backup)

    # Return average utility
    if HAVE_NUMPY:
        return float(np.mean(np.asarray(utility_vals, dtype=float)))
    return sum(utility_vals) / max(1, len(utility_vals))


def jeffreys_optimizer(
    f_true: Function,
    f_alt: Function,
    N: int,
    fixed_input: Optional[Mapping[str, Any]] = None,
    *,
    sigma: float = 1.0,
    K_outer: int = 30,
    K_inner: int = 30,
    budget: int = 100,
    rng: Optional[Union[int, random.Random]] = None,
) -> Dict[str, Any]:
    """
    Recherche aléatoire de design qui maximise l'utilité de Jeffreys (log-BF attendu).

    Retourne un dict: { 'inputs': design, 'utility': score, 'meta': {...} }
    """
    rng_ = ensure_rng(rng)
    norm = normalize_fixed_input(f_true, N, fixed_input)

    best_u = float('-inf')
    best_design: Optional[Dict[str, Any]] = None
    samples: List[float] = []

    for _ in range(int(budget)):
        cand = generate_candidate_inputs(f_true, N, norm, rng_)
        # Validate against f_true input spec
        try:
            _ = f_true.eval(cand)
            _ = f_alt.eval(cand)
        except Exception:
            continue
        u = expected_log_bayes_factor_for_design(
            f_true, f_alt, cand, sigma=sigma, K_outer=K_outer, K_inner=K_inner, rng=rng_
        )
        samples.append(u)
        if u > best_u:
            best_u = u
            best_design = cand

    if best_design is None:
        raise RuntimeError("Aucun design valide trouvé pour l'utilité de Jeffreys.")

    return {
        'inputs': best_design,
        'utility': best_u,
        'meta': {
            'evaluations': budget,
            'samples_mean': (float(np.mean(samples)) if HAVE_NUMPY and samples else (sum(samples)/len(samples) if samples else float('nan'))),
            'samples_std': (float(np.std(samples)) if HAVE_NUMPY and samples else float('nan')),
            'sigma': sigma,
            'K_outer': K_outer,
            'K_inner': K_inner,
        }
    }


def expected_log_bayes_factor_matrix_for_design(
    models: Sequence[Function],
    design_inputs: Mapping[str, Any],
    *,
    sigma: float = 1.0,
    K_outer: int = 30,
    K_inner: int = 30,
    rng: Optional[Union[int, random.Random]] = None,
    n_jobs: int = 1,
    progress: bool = False,
    adaptive: bool = False,
    K_outer_min: Optional[int] = None,
    K_outer_max: Optional[int] = None,
    K_inner_min: Optional[int] = None,
    K_inner_max: Optional[int] = None,
    tol_rel: float = 0.05,
    tol_abs: float = 0.05,
    resample_params: bool = True,
) -> List[List[float]]:
    """
    Calcule la matrice U où U[i][j] = E_{θ_i,y~M_i}[ log p(y|M_i) - log p(y|M_j) ]
    via Monte-Carlo (Jeffreys multi-modèles) pour un design donné.

    Paramètres supplémentaires:
      - n_jobs: nombre de workers (processus) utilisés pour paralléliser les lignes.
      - progress: si True, affiche une barre de progression au fil des lignes.
    """
    rng_ = ensure_rng(rng)
    m = len(models)
    if m == 0:
        return []
    U = [[0.0 for _ in range(m)] for _ in range(m)]

    # Backups for parameters to restore after sampling
    backups = [dict(mod.parameters) for mod in models]

    pbar = None
    if progress and m > 0:
        try:
            from tqdm import tqdm  # type: ignore

            pbar = tqdm(total=m, desc="Jeffreys U rows", leave=False)
        except Exception:
            pbar = None  # pragma: no cover

    try:
        jobs = max(1, int(n_jobs)) if n_jobs is not None else 1
        jobs = min(jobs, os.cpu_count() or jobs)
        outer_min = int(K_outer_min if K_outer_min is not None else K_outer)
        outer_max = int(K_outer_max if K_outer_max is not None else K_outer)
        inner_min = int(K_inner_min if K_inner_min is not None else K_inner)
        inner_max = int(K_inner_max if K_inner_max is not None else K_inner)

        if not resample_params:
            for mod in models:
                draw_param(mod, rng=rng_, set_on_function=True)

        if jobs == 1:
            for i in range(m):
                if adaptive:
                    _, row = _compute_bayes_row_adaptive(
                        i,
                        models,
                        design_inputs,
                        sigma,
                        outer_min,
                        outer_max,
                        inner_min,
                        inner_max,
                        float(tol_rel),
                        float(tol_abs),
                        rng_,
                        resample_params=resample_params,
                    )
                else:
                    _, row = _compute_bayes_row(
                        i, models, design_inputs, sigma, K_outer, K_inner, rng_, resample_params=resample_params
                    )
                U[i] = row
                if pbar is not None:
                    pbar.update(1)
        else:
            worker_count = min(jobs, m)
            if not HAVE_CLOUDPICKLE or cloudpickle is None:
                raise RuntimeError("cloudpickle est requis pour n_jobs > 1 (parallélisme multi-processus).")
            models_blob = cloudpickle.dumps(models)
            seeds = [rng_.randint(0, 2**31 - 1) for _ in range(m)]
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_parallel_models_initializer,
                initargs=(models_blob,),
            ) as executor:
                futures = [
                    executor.submit(
                        _compute_bayes_row_parallel,
                        idx,
                        design_inputs,
                        sigma,
                        K_outer,
                        K_inner,
                        seeds[idx],
                        bool(adaptive),
                        outer_min,
                        outer_max,
                        inner_min,
                        inner_max,
                        float(tol_rel),
                        float(tol_abs),
                        resample_params,
                    )
                    for idx in range(m)
                ]
                for fut in as_completed(futures):
                    idx, row = fut.result()
                    U[idx] = row
                    if pbar is not None:
                        pbar.update(1)

    finally:
        # Restore parameters
        for mod, p in zip(models, backups):
            mod.parameters.update(p)
        if pbar is not None:
            pbar.close()

    return U


def jeffreys_optimizer_multi(
    models: Sequence[Function],
    N: int,
    fixed_input: Optional[Mapping[str, Any]] = None,
    *,
    sigma: float = 1.0,
    K_outer: int = 20,
    K_inner: int = 20,
    budget: int = 60,
    rng: Optional[Union[int, random.Random]] = None,
    objective: str = "maximin",  # 'maximin' or 'avgmin'
    progress: bool = False,
    verbose: int = 0,
) -> Dict[str, Any]:
    """
    Optimisation de design pour discriminer plusieurs modèles simultanément.

    - objective='maximin': maximise min_{i} min_{j≠i} U[i][j]
    - objective='avgmin': maximise moyenne_i min_{j≠i} U[i][j]
    Retourne { 'inputs': design, 'utility': score, 'U': matrice, 'meta': {...} }.
    """
    if not models or len(models) < 2:
        raise ValueError("'models' doit contenir au moins deux fonctions")
    rng_ = ensure_rng(rng)
    # Use input spec of the first model (assumed compatible)
    norm = normalize_fixed_input(models[0], N, fixed_input)

    best_score = float('-inf')
    best_design: Optional[Dict[str, Any]] = None
    best_U: Optional[List[List[float]]] = None

    iterator = range(int(budget))
    _tqdm = None
    if progress:
        try:
            from tqdm import tqdm as _tqdm  # type: ignore
            iterator = _tqdm(iterator, desc="Jeffreys opt", total=int(budget))
        except Exception:
            _tqdm = None

    for it in iterator:
        cand = generate_candidate_inputs(models[0], N, norm, rng_)
        # Validate candidates on all models
        try:
            for mod in models:
                _ = mod.eval(cand)
        except Exception:
            continue

        U = expected_log_bayes_factor_matrix_for_design(
            models, cand, sigma=sigma, K_outer=K_outer, K_inner=K_inner, rng=rng_
        )

        # Score aggregation
        m = len(models)
        row_min = []
        for i in range(m):
            vals = [U[i][j] for j in range(m) if j != i]
            row_min.append(min(vals) if vals else float('-inf'))
        if objective == 'avgmin':
            score = sum(row_min) / len(row_min)
        else:  # default maximin
            score = min(row_min)

        if verbose >= 2:
            print(f"[iter {it}] score={score:.3f}")
        if score > best_score:
            best_score = score
            best_design = cand
            best_U = U

    if best_design is None or best_U is None:
        raise RuntimeError("Aucun design valide trouvé pour l'utilité de Jeffreys (multi-modèles).")

    return {
        'inputs': best_design,
        'utility': best_score,
        'U': best_U,
        'meta': {
            'evaluations': budget,
            'sigma': sigma,
            'K_outer': K_outer,
            'K_inner': K_inner,
            'objective': objective,
        }
    }


def jeffreys_optimizer_multi_bo(
    models: Sequence[Function],
    N_t: int,
    *,
    t_min: float = 0.0,
    t_max: float = 600.0,
    min_gap: float = 5.0,
    fixed_input: Optional[Mapping[str, Any]] = None,
    sigma: float = 1.0,
    K_outer: int = 20,
    K_inner: int = 20,
    n_calls: int = 40,
    n_initial_points: int = 10,
    rng: Optional[Union[int, random.Random]] = None,
    objective: str = "maximin",  # 'maximin' or 'avgmin'
    progress: bool = True,
) -> Dict[str, Any]:
    """Optimisation multi-modèles par Bayesian Optimization (scikit-optimize).

    On paramétrise le design t par un vecteur u in [0,1]^N_t,
    puis on applique une projection tri + espacement min (min_gap) sur [t_min, t_max].
    """
    if not models or len(models) < 2:
        raise ValueError("'models' doit contenir au moins deux fonctions")

    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scikit-optimize (skopt) est requis pour jeffreys_optimizer_multi_bo") from exc

    if progress:
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:  # pragma: no cover
            tqdm = None  # type: ignore
    else:
        tqdm = None  # type: ignore

    rng_ = ensure_rng(rng)

    # fixed_input peut contenir Eff_, Rew_, A, etc. On le garde tel quel.
    # On ne normalise pas via normalize_fixed_input ici; on gère seulement 't'.

    def _project_u_to_t(u: Sequence[float]) -> List[float]:
        # u in [0,1]^N_t -> t sorted in [t_min, t_max] with spacing >= min_gap
        if len(u) != N_t:
            raise ValueError("u doit avoir une longueur N_t")
        scaled = [float(t_min + (t_max - t_min) * max(0.0, min(1.0, ui))) for ui in u]
        scaled.sort()
        t_vals: List[float] = []
        last: Optional[float] = None
        for v in scaled:
            if last is None or v - last >= min_gap:
                t_vals.append(v)
                last = v
        # Si trop peu de points, on essaie de compléter en répartissant uniformément
        if len(t_vals) < max(1, N_t // 2):
            step = max(min_gap, (t_max - t_min) / max(1, N_t))
            t_vals = [t_min + i * step for i in range(N_t) if t_min + i * step <= t_max]
        return t_vals

    # Prépare les inputs fixes hors t
    base_inputs: Dict[str, Any] = {}
    if fixed_input:
        for k, v in fixed_input.items():
            # Pour 't', on ignore v ici (sera remplacé par projection)
            if k == 't':
                continue
            # fixed_input peut être sous forme ('fixed', value) ou dict
            if isinstance(v, tuple) and len(v) >= 2 and v[0] == 'fixed':
                base_inputs[k] = v[1]
            elif isinstance(v, dict) and v.get('mode') == 'fixed':
                base_inputs[k] = v.get('value')
            else:
                # pour la BO, on n'autorise que des valeurs fixes hors t
                pass

    def _score_for_t(t_vals: List[float]) -> float:
        # Construit un design et évalue le critère multi-modèles
        cand_inputs = dict(base_inputs)
        cand_inputs['t'] = t_vals
        # validation
        try:
            for mod in models:
                _ = mod.eval(cand_inputs)
        except Exception:
            # design invalide -> très mauvais score
            return float('inf')

        U = expected_log_bayes_factor_matrix_for_design(
            models, cand_inputs, sigma=sigma, K_outer=K_outer, K_inner=K_inner, rng=rng_
        )
        m = len(models)
        row_min = []
        for i in range(m):
            vals = [U[i][j] for j in range(m) if j != i]
            row_min.append(min(vals) if vals else float('-inf'))
        if objective == 'avgmin':
            score = sum(row_min) / len(row_min)
        else:
            score = min(row_min)
        # gp_minimize minimise, on renvoie -score
        return -score

    def objective_func(u: List[float]) -> float:
        t_vals = _project_u_to_t(u)
        return _score_for_t(t_vals)

    space = [Real(0.0, 1.0, name=f"u{i}") for i in range(N_t)]

    n_calls = int(n_calls)
    n_initial_points = int(n_initial_points)

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=n_calls, desc="Jeffreys BO", leave=True)

        def _callback(_res) -> None:
            if pbar is not None:
                pbar.update(1)
    else:
        _callback = None  # type: ignore

    res = gp_minimize(
        objective_func,
        space,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=rng_.randint(0, 2**31 - 1),
        callback=None if _callback is None else [_callback],
    )

    if pbar is not None:
        pbar.close()

    best_u = res.x
    best_t = _project_u_to_t(best_u)
    best_inputs = dict(base_inputs)
    best_inputs['t'] = best_t
    # Recalculer U et score pour le design optimal
    U_best = expected_log_bayes_factor_matrix_for_design(
        models, best_inputs, sigma=sigma, K_outer=K_outer, K_inner=K_inner, rng=rng_,
    )
    m = len(models)
    row_min = []
    for i in range(m):
        vals = [U_best[i][j] for j in range(m) if j != i]
        row_min.append(min(vals) if vals else float('-inf'))
    if objective == 'avgmin':
        best_score = sum(row_min) / len(row_min)
    else:
        best_score = min(row_min)

    return {
        'inputs': best_inputs,
        'utility': best_score,
        'U': U_best,
        'meta': {
            'n_calls': n_calls,
            'n_initial_points': n_initial_points,
            'sigma': sigma,
            'K_outer': K_outer,
            'K_inner': K_inner,
            'objective': objective,
        },
    }
