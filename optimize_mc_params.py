import argparse
import os
import random
from typing import Any, Dict, List, Sequence, Tuple

import math

from design_optimizer import expected_log_bayes_factor_matrix_for_design
from test_multi_model import (
    build_six_models,
    make_demo_pools,
    decode_design_structured,
    _build_fixed_times,
)


def parse_list_floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip() != ""]


def parse_list_ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def diff_U(Ua: List[List[float]], Ub: List[List[float]]) -> Tuple[float, float]:
    """Return (max_abs_diff, mean_abs_diff) between two matrices."""
    if len(Ua) != len(Ub):
        return float("inf"), float("inf")
    m = len(Ua)
    max_d = 0.0
    sum_d = 0.0
    cnt = 0
    for i in range(m):
        if len(Ua[i]) != len(Ub[i]):
            return float("inf"), float("inf")
        for j in range(len(Ua[i])):
            d = abs(Ua[i][j] - Ub[i][j])
            max_d = max(max_d, d)
            sum_d += d
            cnt += 1
    mean_d = sum_d / max(1, cnt)
    return max_d, mean_d


def generate_random_design(
    rng: random.Random,
    t_min: float,
    t_max: float,
    min_gap_t: float,
    delta_er: float,
    t_step: float,
    Eff_pool: Sequence[float],
    Rew_pool: Sequence[float],
) -> Dict[str, Any]:
    t_fixed = _build_fixed_times(t_min, t_max, t_step)
    n_intervals = len(t_fixed) - 1
    x = [rng.uniform(t_fixed[i] + min_gap_t, t_fixed[i + 1] - min_gap_t) for i in range(n_intervals)]
    return decode_design_structured(
        x,
        t_min=t_min,
        t_max=t_max,
        min_gap_t=min_gap_t,
        delta_er=delta_er,
        Eff_pool=Eff_pool,
        Rew_pool=Rew_pool,
        t_step=t_step,
        rng=rng,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search for MC parameters (outer/inner/tolerances)")
    parser.add_argument("--designs", type=int, default=5, help="number of random designs to test")
    parser.add_argument("--outer-grid", type=str, default="20,40,80,120", help="comma-separated outer values")
    parser.add_argument("--inner-grid", type=str, default="10,20,40,60", help="comma-separated inner values")
    parser.add_argument("--tol-rel-grid", type=str, default="0.1,0.05,0.02", help="comma-separated tol_rel values (adaptive only)")
    parser.add_argument("--tol-abs-grid", type=str, default="0.1,0.05,0.02", help="comma-separated tol_abs values (adaptive only)")
    parser.add_argument("--baseline-outer", type=int, default=200, help="outer for reference U")
    parser.add_argument("--baseline-inner", type=int, default=200, help="inner for reference U")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["adaptive", "nonadaptive", "both"],
        default="both",
        help="which search to run (adaptive, nonadaptive, or both)",
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed")
    parser.add_argument("--t-step", type=float, default=30.0, help="fixed step between measurement times")
    parser.add_argument("--min-gap", type=float, default=5.0, help="minimum gap between times")
    parser.add_argument("--delta-er", type=float, default=5.0, help="minimum Eff/Rew separation")
    parser.add_argument("--max-error", type=float, default=None, help="optional threshold on max_abs_diff to filter candidates")
    parser.add_argument("--no-progress", action="store_true", help="disable progress bars in utility")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    models = build_six_models()
    Eff_pool, Rew_pool = make_demo_pools()

    t_min, t_max = 0.0, 600.0
    min_gap_t = args.min_gap
    delta_er = args.delta_er
    t_step = args.t_step

    outer_grid = parse_list_ints(args.outer_grid)
    inner_grid = parse_list_ints(args.inner_grid)
    tol_rel_grid = parse_list_floats(args.tol_rel_grid)
    tol_abs_grid = parse_list_floats(args.tol_abs_grid)

    # Generate random designs once
    designs: List[Dict[str, Any]] = [
        generate_random_design(
            rng,
            t_min=t_min,
            t_max=t_max,
            min_gap_t=min_gap_t,
            delta_er=delta_er,
            t_step=t_step,
            Eff_pool=Eff_pool,
            Rew_pool=Rew_pool,
        )
        for _ in range(max(1, args.designs))
    ]

    # Reference matrices at high budget
    ref_mats: List[List[List[float]]] = []
    for d in designs:
        U_ref = expected_log_bayes_factor_matrix_for_design(
            models,
            d,
            sigma=0.1,
            K_outer=args.baseline_outer,
            K_inner=args.baseline_inner,
            rng=rng,
            n_jobs=min(len(models), os.cpu_count() or 1),
            progress=not args.no_progress,
        )
        ref_mats.append(U_ref)

    modes = []
    if args.mode in ("nonadaptive", "both"):
        modes.append(False)
    if args.mode in ("adaptive", "both"):
        modes.append(True)

    for adaptive_flag in modes:
        results = []
        best = None
        for outer in outer_grid:
            for inner in inner_grid:
                if adaptive_flag:
                    for tol_rel in tol_rel_grid:
                        for tol_abs in tol_abs_grid:
                            cfg = (outer, inner, tol_rel, tol_abs, True)
                            total_cost = 0
                            max_errors = []
                            mean_errors = []
                            for d, U_ref in zip(designs, ref_mats):
                                U = expected_log_bayes_factor_matrix_for_design(
                                    models,
                                    d,
                                    sigma=0.1,
                                    K_outer=outer,
                                    K_inner=inner,
                                    rng=rng,
                                    n_jobs=min(len(models), os.cpu_count() or 1),
                                    progress=False,
                                    adaptive=True,
                                    K_outer_min=max(5, outer // 4),
                                    K_outer_max=outer,
                                    K_inner_min=max(5, inner // 2),
                                    K_inner_max=inner,
                                    tol_rel=tol_rel,
                                    tol_abs=tol_abs,
                                )
                                md, mnd = diff_U(U, U_ref)
                                max_errors.append(md)
                                mean_errors.append(mnd)
                                total_cost += outer * inner
                            avg_max_err = sum(max_errors) / len(max_errors)
                            avg_mean_err = sum(mean_errors) / len(mean_errors)
                            results.append((avg_max_err, total_cost, cfg, avg_mean_err))
                else:
                    cfg = (outer, inner, None, None, False)
                    total_cost = 0
                    max_errors = []
                    mean_errors = []
                    for d, U_ref in zip(designs, ref_mats):
                        U = expected_log_bayes_factor_matrix_for_design(
                            models,
                            d,
                            sigma=0.1,
                            K_outer=outer,
                            K_inner=inner,
                            rng=rng,
                            n_jobs=min(len(models), os.cpu_count() or 1),
                            progress=False,
                        )
                        md, mnd = diff_U(U, U_ref)
                        max_errors.append(md)
                        mean_errors.append(mnd)
                        total_cost += outer * inner
                    avg_max_err = sum(max_errors) / len(max_errors)
                    avg_mean_err = sum(mean_errors) / len(mean_errors)
                    results.append((avg_max_err, total_cost, cfg, avg_mean_err))

        filtered = []
        for rec in results:
            if args.max_error is not None and rec[0] > args.max_error:
                continue
            filtered.append(rec)
        if filtered:
            results = filtered

        results.sort(key=lambda r: (r[0], r[1]))
        if results:
            best = results[0]

        header = "adaptive" if adaptive_flag else "non-adaptive"
        print(f"\n=== MC param search results ({header}) ===")
        for avg_max_err, cost, cfg, avg_mean_err in results[:10]:
            outer, inner, tol_rel, tol_abs, adaptive = cfg
            print(
                f"outer={outer}, inner={inner}, adaptive={adaptive}, "
                f"tol_rel={tol_rel}, tol_abs={tol_abs}, "
                f"avg_max_err={avg_max_err:.3g}, avg_mean_err={avg_mean_err:.3g}, cost≈{cost}"
            )
        if best:
            avg_max_err, cost, cfg, avg_mean_err = best
            outer, inner, tol_rel, tol_abs, adaptive = cfg
            print("\nBest config:")
            print(
                f"outer={outer}, inner={inner}, adaptive={adaptive}, "
                f"tol_rel={tol_rel}, tol_abs={tol_abs}, "
                f"avg_max_err={avg_max_err:.3g}, avg_mean_err={avg_mean_err:.3g}, cost≈{cost}"
            )


if __name__ == "__main__":
    main()
