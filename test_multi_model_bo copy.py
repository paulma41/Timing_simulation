from __future__ import annotations

import argparse
import math
import os
import random
from typing import Any, Dict, List, Sequence, Tuple

from functions import build_h_action_function
from design_optimizer import expected_log_bayes_factor_matrix_for_design
from test_multi_model import plot_summary


def make_demo_pools() -> Tuple[List[float], List[float]]:
    Eff_pool = [-1.5, -1.0, -0.5, 0.]
    Rew_pool = [2., 1.5, 1.0, 0.5, 0.]
    return Eff_pool, Rew_pool


def build_six_models():
    models = []
    for kernel in ["event_weighted", "action_avg"]:
        for update in ["continuous", "event", "action"]:
            f = build_h_action_function(kernel=kernel, update=update)
            models.append(f)
    return models


def sample_times_with_gap(
    rng: random.Random,
    n: int,
    t_min: float,
    t_max: float,
    min_gap: float,
) -> List[float]:
    if n <= 0:
        return []
    vals = sorted(rng.uniform(t_min, t_max) for _ in range(n))
    spaced: List[float] = []
    last = None
    for v in vals:
        if last is None or v - last >= min_gap:
            spaced.append(v)
            last = v
    if len(spaced) >= max(1, n // 2):
        return spaced
    step = max(min_gap, (t_max - t_min) / max(1, n))
    return [t_min + i * step for i in range(n) if t_min + i * step <= t_max]


def decode_design_from_vector(
    x: List[float],
    N_t: int,
    N_eff: int,
    Eff_pool: Sequence[float],
    Rew_pool: Sequence[float],
    t_min: float,
    t_max: float,
    min_gap_t: float,
    delta_er: float,
) -> Dict[str, Any]:
    """
    Décodage du vecteur BO x en design complet (t, Eff_, Rew_, A).
    Ordre des dimensions:
      - t_latent: N_t réels
      - Eff_t_raw: N_eff réels
      - Rew_t_raw: N_eff réels
      - Eff_idx: N_eff entiers
      - Rew_idx: N_eff entiers
    """
    idx = 0
    t_latent = list(x[idx : idx + N_t])
    idx += N_t
    Eff_t_raw = list(x[idx : idx + N_eff])
    idx += N_eff
    Rew_t_raw = list(x[idx : idx + N_eff])
    idx += N_eff
    Eff_idx = [int(round(v)) for v in x[idx : idx + N_eff]]
    idx += N_eff
    Rew_idx = [int(round(v)) for v in x[idx : idx + N_eff]]

    # Projeter t en utilisant les valeurs de t_latent (tri + espacement min)
    t_candidates = sorted(max(t_min, min(t_max, v)) for v in t_latent)
    t_proj: List[float] = []
    last = None
    for v in t_candidates:
        if last is None or v - last >= min_gap_t:
            t_proj.append(v)
            last = v
    if len(t_proj) < max(1, len(t_latent) // 2):
        step = max(min_gap_t, (t_max - t_min) / max(1, len(t_latent)))
        t_proj = [t_min + i * step for i in range(len(t_latent)) if t_min + i * step <= t_max]

    # Projeter temps Eff_/Rew_ avec contraintes
    Eff_t: List[float] = []
    Rew_t: List[float] = []
    for k in range(N_eff):
        te = max(t_min, min(t_max - delta_er, Eff_t_raw[k]))
        tr = max(te + delta_er, min(t_max, Rew_t_raw[k]))
        Eff_t.append(te)
        Rew_t.append(tr)

    # Indices -> valeurs (avec modulo au cas où)
    neff_pool = len(Eff_pool)
    nrew_pool = len(Rew_pool)
    Eff_val = [Eff_pool[Eff_idx[k] % neff_pool] for k in range(N_eff)]
    Rew_val = [Rew_pool[Rew_idx[k] % nrew_pool] for k in range(N_eff)]

    Eff_ = list(zip(Eff_val, Eff_t))
    Rew_ = list(zip(Rew_val, Rew_t))
    A = list(zip(Eff_, Rew_))

    return {
        "t": t_proj,
        "Eff_": Eff_,
        "Rew_": Rew_,
        "A": A,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-model BO design optimization (Jeffreys)")
    parser.add_argument("--N_t", type=int, default=11, help="number of measurement times")
    parser.add_argument("--n-calls", type=int, default=100, help="number of BO evaluations")
    parser.add_argument("--n-init", type=int, default=10, help="number of BO initial random evaluations")
    parser.add_argument("--sigma", type=float, default=0.1, help="obs noise stddev")
    parser.add_argument("--outer", type=int, default=20, help="K_outer Monte Carlo samples")
    parser.add_argument("--inner", type=int, default=20, help="K_inner Monte Carlo samples per model")
    parser.add_argument("--jobs", type=int, default=16, help="number of parallel workers for Jeffreys utility (<=0 => auto)")
    parser.add_argument("--adaptive-utility", action="store_true", help="use adaptive per-row MC sampling")
    parser.add_argument("--outer-min", type=int, default=None, help="minimum K_outer for adaptive utility (default: auto)")
    parser.add_argument("--inner-min", type=int, default=None, help="minimum K_inner for adaptive utility (default: auto)")
    parser.add_argument("--tol-rel", type=float, default=0.05, help="relative tolerance for adaptive utility SE stop")
    parser.add_argument("--tol-abs", type=float, default=0.05, help="absolute tolerance for adaptive utility SE stop")
    parser.add_argument(
        "--utility-progress",
        action="store_true",
        help="affiche un tqdm pour chaque évaluation de la matrice U (même en parallèle)",
    )
    args = parser.parse_args()

    print("[info] Building models…")
    models = build_six_models()
    Eff_pool, Rew_pool = make_demo_pools()
    N_eff = 10

    bad_objective = 1e12  # large finite penalty for invalid evaluations

    parallel_jobs = args.jobs if args.jobs is not None else 1
    if parallel_jobs <= 0:
        parallel_jobs = os.cpu_count() or 1
    parallel_jobs = min(parallel_jobs, len(models))
    print(f"[info] Parallel workers for Jeffreys utility: {parallel_jobs}")
    show_utility_progress = bool(args.utility_progress)
    if show_utility_progress:
        print("[info] Utility progress bar enabled")

    # BO space: t_latent, Eff_t_raw, Rew_t_raw, Eff_idx, Rew_idx
    t_min, t_max = 0.0, 600.0
    min_gap_t = 5.0
    delta_er = 5.0

    adaptive_outer_min = args.outer_min if args.outer_min is not None else max(5, args.outer // 4)
    adaptive_outer_max = max(args.outer, adaptive_outer_min)
    adaptive_inner_min = args.inner_min if args.inner_min is not None else max(5, args.inner // 2)
    adaptive_inner_max = max(args.inner, adaptive_inner_min)

    try:
        from skopt import gp_minimize
        from skopt.space import Real, Integer
    except Exception as exc:
        raise RuntimeError("scikit-optimize (skopt) est requis pour test_multi_model_bo") from exc

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None  # type: ignore

    space = []
    # t_latent
    for i in range(args.N_t):
        space.append(Real(t_min, t_max, name=f"t_{i}"))
    # Eff_t_raw
    for k in range(N_eff):
        space.append(Real(t_min, t_max, name=f"te_{k}"))
    # Rew_t_raw
    for k in range(N_eff):
        space.append(Real(t_min, t_max, name=f"tr_{k}"))
    # Eff_idx
    for k in range(N_eff):
        space.append(Integer(0, len(Eff_pool) - 1, name=f"ie_{k}"))
    # Rew_idx
    for k in range(N_eff):
        space.append(Integer(0, len(Rew_pool) - 1, name=f"ir_{k}"))

    rng = random.Random(123)
    objective_cache: Dict[Tuple[float, ...], float] = {}

    def objective(x: List[float]) -> float:
        key = tuple(round(float(v), 6) for v in x)
        if key in objective_cache:
            return objective_cache[key]
        design_inputs = decode_design_from_vector(
            x,
            N_t=args.N_t,
            N_eff=N_eff,
            Eff_pool=Eff_pool,
            Rew_pool=Rew_pool,
            t_min=t_min,
            t_max=t_max,
            min_gap_t=min_gap_t,
            delta_er=delta_er,
        )
        try:
            U = expected_log_bayes_factor_matrix_for_design(
                models,
                design_inputs,
                sigma=args.sigma,
                K_outer=args.outer,
                K_inner=args.inner,
                rng=rng,
                n_jobs=parallel_jobs,
                progress=show_utility_progress,
                adaptive=args.adaptive_utility,
                K_outer_min=adaptive_outer_min,
                K_outer_max=adaptive_outer_max,
                K_inner_min=adaptive_inner_min,
                K_inner_max=adaptive_inner_max,
                tol_rel=args.tol_rel,
                tol_abs=args.tol_abs,
            )
        except Exception:
            return bad_objective

        m = len(models)
        row_min: List[float] = []
        for i in range(m):
            vals = [U[i][j] for j in range(m) if j != i]
            row_min.append(min(vals) if vals else float("-inf"))
        score = min(row_min)
        if not math.isfinite(score):
            objective_cache[key] = bad_objective
            return bad_objective
        # gp_minimize minimise, on renvoie -score
        objective_cache[key] = -score
        return -score

    n_calls = max(1, args.n_calls)
    n_init = max(1, min(args.n_init, n_calls))

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=n_calls, desc="Jeffreys BO", leave=True)

        def _callback(_res) -> None:
            if pbar is not None:
                pbar.update(1)
    else:
        _callback = None  # type: ignore

    initial_point_generator = "sobol"
    try:
        from skopt.sampler import Sobol  # type: ignore

        initial_point_generator = Sobol()
    except Exception:
        try:
            initial_point_generator = "lhs"
        except Exception:
            initial_point_generator = "random"

    print(f"[info] Running BO with N_t={args.N_t}, N_eff={N_eff}, n_calls={n_calls}, n_init={n_init}")
    res = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_initial_points=n_init,
        initial_point_generator=initial_point_generator,
        random_state=rng.randint(0, 2**31 - 1),
        callback=None if _callback is None else [_callback],
    )

    if pbar is not None:
        pbar.close()

    best_x = res.x
    best_design = decode_design_from_vector(
        best_x,
        N_t=args.N_t,
        N_eff=N_eff,
        Eff_pool=Eff_pool,
        Rew_pool=Rew_pool,
        t_min=t_min,
        t_max=t_max,
        min_gap_t=min_gap_t,
        delta_er=delta_er,
    )
    U_best = expected_log_bayes_factor_matrix_for_design(
        models,
        best_design,
        sigma=args.sigma,
        K_outer=args.outer,
        K_inner=args.inner,
        rng=rng,
        n_jobs=parallel_jobs,
        progress=show_utility_progress,
        adaptive=args.adaptive_utility,
        K_outer_min=adaptive_outer_min,
        K_outer_max=adaptive_outer_max,
        K_inner_min=adaptive_inner_min,
        K_inner_max=adaptive_inner_max,
        tol_rel=args.tol_rel,
        tol_abs=args.tol_abs,
    )
    m = len(models)
    row_min: List[float] = []
    for i in range(m):
        vals = [U_best[i][j] for j in range(m) if j != i]
        row_min.append(min(vals) if vals else float("-inf"))
    best_score = min(row_min)

    print("[result] Best utility (maximin):", best_score)
    print("[result] Best t:", best_design["t"])
    print("[result] Eff_:", best_design["Eff_"])
    print("[result] Rew_:", best_design["Rew_"])
    print("[result] U matrix:")
    for row in U_best:
        print([round(x, 3) for x in row])

    # Plot summary using the same helper as random search
    try:
        plot_summary(U_best, best_design, models, title="BO maximin")
    except Exception as exc:
        print(f"[diag] unable to plot summary: {exc}")


if __name__ == "__main__":
    main()
