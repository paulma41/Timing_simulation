import argparse
import math
import os
import random
from typing import Any, Dict, List, Sequence, Tuple

from functions import build_h_action_function
from design_optimizer import expected_log_bayes_factor_matrix_for_design
from test_multi_model import plot_summary


def make_demo_pools() -> Tuple[List[float], List[float]]:
    Eff_pool = [-1.5, -1.0, -0.5, 0.0]
    Rew_pool = [1.5, 1.0, 0.5, 0.0]
    return Eff_pool, Rew_pool


def build_six_models():
    models = []
    for kernel in ["event_weighted", "action_avg"]:
        for update in ["continuous", "event", "action"]:
            f = build_h_action_function(kernel=kernel, update=update)
            models.append(f)
    return models


def _build_fixed_times(t_min: float, t_max: float, step: float) -> List[float]:
    """Construit les temps fixes t_min, t_min+step, ..., t_max (inclus)."""
    if step <= 0:
        return [t_min, t_max]
    vals: List[float] = []
    v = t_min
    while v < t_max:
        vals.append(v)
        v += step
    if not vals or vals[-1] < t_max:
        vals.append(t_max)
    return vals


def decode_design_from_vector(
    x: List[float],
    *,
    t_min: float,
    t_max: float,
    min_gap_t: float,
    delta_er: float,
    Eff_pool: Sequence[float],
    Rew_pool: Sequence[float],
    t_step: float,
    rng: random.Random,
) -> Dict[str, Any]:
    """
    D�code le vecteur BO x en design complet en respectant la structure :
      - t fixes tous les t_step (inclus t_min et t_max)
      - un t optimis� par intervalle (entre deux t fixes successifs)
      - un seul couple Eff/Rew par intervalle, al�atoire mais couvrant toutes les combinaisons.
    """
    t_fixed = _build_fixed_times(t_min, t_max, t_step)
    n_intervals = max(0, len(t_fixed) - 1)
    if len(x) != n_intervals:
        raise ValueError(f"Expected {n_intervals} t values, got {len(x)}")

    # Contraintes par intervalle pour chaque t optimis�
    t_opt: List[float] = []
    for i in range(n_intervals):
        low = t_fixed[i] + min_gap_t
        high = t_fixed[i + 1] - min_gap_t
        if high <= low:
            t_val = (t_fixed[i] + t_fixed[i + 1]) / 2.0
        else:
            t_val = max(low, min(high, x[i]))
        t_opt.append(t_val)

    # Tous les t de mesure
    t_all = sorted(set(t_fixed + t_opt))

    # Construit une liste de couples Eff/Rew couvrant toutes les combinaisons
    combos = [(e, r) for e in Eff_pool for r in Rew_pool]
    rng.shuffle(combos)

    Eff_: List[Tuple[float, float]] = []
    Rew_: List[Tuple[float, float]] = []
    A: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    for i in range(n_intervals):
        start = t_fixed[i]
        end = t_fixed[i + 1]
        if end - start <= delta_er:
            continue
        eff_val, rew_val = combos[i % len(combos)]
        eff_t = rng.uniform(start, end - delta_er)
        rew_t = rng.uniform(eff_t + delta_er, end)
        Eff_.append((eff_val, eff_t))
        Rew_.append((rew_val, rew_t))
        A.append(((eff_val, eff_t), (rew_val, rew_t)))

    return {"t": t_all, "Eff_": Eff_, "Rew_": Rew_, "A": A}


def main():
    parser = argparse.ArgumentParser(description="Multi-model BO design optimization (Jeffreys)")
    parser.add_argument("--optimizer", choices=["bo", "cma"], default="bo", help="optimizer type (bo=gp_minimize, cma=CMA-ES)")
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
    parser.add_argument("--t-step", type=float, default=30.0, help="fixed step between measurement times")
    parser.add_argument("--cma-popsize", type=int, default=16, help="population size for CMA-ES")
    parser.add_argument("--cma-maxiter", type=int, default=50, help="max iterations for CMA-ES")
    parser.add_argument("--cma-sigma-factor", type=float, default=0.2, help="initial sigma factor (relative to interval width) for CMA-ES")
    parser.add_argument(
        "--utility-progress",
        action="store_true",
        help="affiche un tqdm pour chaque �valuation de la matrice U (m�me en parall�le)",
    )
    args = parser.parse_args()

    print("[info] Building models.")
    models = build_six_models()
    Eff_pool, Rew_pool = make_demo_pools()

    bad_objective = 1e12  # large finite penalty for invalid evaluations

    parallel_jobs = args.jobs if args.jobs is not None else 1
    if parallel_jobs <= 0:
        parallel_jobs = os.cpu_count() or 1
    parallel_jobs = min(parallel_jobs, len(models))
    print(f"[info] Parallel workers for Jeffreys utility: {parallel_jobs}")
    show_utility_progress = bool(args.utility_progress)
    if show_utility_progress:
        print("[info] Utility progress bar enabled")

    # BO space: un t optimis� par intervalle de taille t_step
    t_min, t_max = 0.0, 600.0
    min_gap_t = 5.0
    delta_er = 5.0
    t_step = float(args.t_step)
    t_fixed = _build_fixed_times(t_min, t_max, t_step)
    n_intervals = max(0, len(t_fixed) - 1)

    adaptive_outer_min = args.outer_min if args.outer_min is not None else max(5, args.outer // 4)
    adaptive_outer_max = max(args.outer, adaptive_outer_min)
    adaptive_inner_min = args.inner_min if args.inner_min is not None else max(5, args.inner // 2)
    adaptive_inner_max = max(args.inner, adaptive_inner_min)

    try:
        from skopt import gp_minimize
        from skopt.space import Real
        from skopt.sampler import Sobol  # type: ignore
    except Exception as exc:
        raise RuntimeError("scikit-optimize (skopt) est requis pour test_multi_model_bo") from exc

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None  # type: ignore

    space = []
    for i in range(n_intervals):
        low = t_fixed[i] + min_gap_t
        high = t_fixed[i + 1] - min_gap_t
        if high <= low:
            high = low
        space.append(Real(low, high, name=f"t_opt_{i}"))

    rng_main = random.Random(123)
    objective_cache: Dict[Tuple[float, ...], float] = {}

    def objective(x: List[float]) -> float:
        key = tuple(round(float(v), 6) for v in x)
        if key in objective_cache:
            return objective_cache[key]
        rng_design = random.Random(hash(key) & 0x7FFFFFFF)
        design_inputs = decode_design_from_vector(
            x,
            t_min=t_min,
            t_max=t_max,
            min_gap_t=min_gap_t,
            delta_er=delta_er,
            Eff_pool=Eff_pool,
            Rew_pool=Rew_pool,
            t_step=t_step,
            rng=rng_design,
        )
        try:
            U = expected_log_bayes_factor_matrix_for_design(
                models,
                design_inputs,
                sigma=args.sigma,
                K_outer=args.outer,
                K_inner=args.inner,
                rng=rng_main,
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

    if args.optimizer == "bo":
        initial_point_generator = "sobol"
        try:
            initial_point_generator = Sobol()
        except Exception:
            try:
                initial_point_generator = "lhs"
            except Exception:
                initial_point_generator = "random"

        print(f"[info] Running BO with {n_intervals} optimised t (step={t_step}), n_calls={n_calls}, n_init={n_init}")
        res = gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            n_initial_points=n_init,
            initial_point_generator=initial_point_generator,
            random_state=rng_main.randint(0, 2**31 - 1),
            callback=None if _callback is None else [_callback],
        )
        best_x = res.x
    else:
        try:
            import cma  # type: ignore
        except Exception as exc:
            raise RuntimeError("cma (pycma) est requis pour --optimizer cma") from exc

        sob = Sobol()
        sob_points = sob.generate([(b.low, b.high) for b in space], max(1, n_init))
        sob_scores = [(objective(p), p) for p in sob_points]
        sob_scores.sort(key=lambda t: t[0])
        x0 = sob_scores[0][1]
        interval_widths = [b.high - b.low for b in space]
        sigma0 = (sum(interval_widths) / max(1, len(interval_widths))) * float(args.cma_sigma_factor)
        bounds_cma = [[b.low for b in space], [b.high for b in space]]
        print(f"[info] Running CMA-ES with dim={len(space)}, popsize={args.cma_popsize}, maxiter={args.cma_maxiter}")
        es = cma.CMAEvolutionStrategy(x0, sigma0, {"bounds": bounds_cma, "popsize": int(args.cma_popsize), "maxiter": int(args.cma_maxiter)})
        while not es.stop():
            X = es.ask()
            es.tell(X, [objective(x) for x in X])
            if pbar is not None:
                pbar.update(len(X))
            es.disp()
        best_x = es.result.xbest

    if pbar is not None:
        pbar.close()

    best_design = decode_design_from_vector(
        best_x,
        t_min=t_min,
        t_max=t_max,
        min_gap_t=min_gap_t,
        delta_er=delta_er,
        Eff_pool=Eff_pool,
        Rew_pool=Rew_pool,
        t_step=t_step,
        rng=random.Random(0),
    )
    U_best = expected_log_bayes_factor_matrix_for_design(
        models,
        best_design,
        sigma=args.sigma,
        K_outer=args.outer,
        K_inner=args.inner,
        rng=rng_main,
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

    try:
        plot_summary(U_best, best_design, models, title="BO maximin")
    except Exception as exc:
        print(f"[diag] unable to plot summary: {exc}")


if __name__ == "__main__":
    main()
