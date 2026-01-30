from __future__ import annotations

import argparse
import math
import random
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm  # type: ignore
    HAVE_TQDM = True
except Exception:  # pragma: no cover
    HAVE_TQDM = False

from test_multi_model import _build_fixed_times
from test_multi_model_bo import make_demo_pools, build_six_models, decode_design_actions_v4
from design_optimizer.laplace_jsd import (
    laplace_jsd_for_design,
    _pairwise_chernoff_matrix,
    _jensen_shannon_gaussians,
)
from skopt import gp_minimize
from skopt.space import Real, Integer


def _random_design_v4(
    rng: random.Random,
    *,
    N_t: int,
    Eff_pool: Sequence[float],
    Rew_pool: Sequence[float],
    t_min: float,
    t_max: float,
    t_step: float,
    min_gap_t: float,
    delta_er: float,
    delta_max: float,
    guard_after_t: float,
) -> Dict[str, Any]:
    t_fixed = _build_fixed_times(t_min, t_max, t_step)
    n_intervals = max(0, len(t_fixed) - 1)
    x: List[float] = []
    # t optimisés
    for _ in range(N_t):
        x.append(rng.uniform(t_min, t_max))
    # eff_idx / rew_idx (3 types)
    for _ in range(3):
        x.append(rng.randrange(len(Eff_pool)))
    for _ in range(3):
        x.append(rng.randrange(len(Rew_pool)))
    # bonus_idx (2)
    for _ in range(2):
        x.append(rng.randrange(len(Rew_pool)))
    # delta (3 types)
    max_delta = min(delta_max, max(delta_er, t_step))
    for _ in range(3):
        x.append(rng.uniform(delta_er, max_delta))
    # action_choice par intervalle (0..3)
    for _ in range(n_intervals):
        x.append(rng.randrange(0, 4))
    # bonus_u (2 par intervalle)
    for _ in range(n_intervals * 2):
        x.append(rng.random())

    rng_design = random.Random(rng.randint(0, 2**31 - 1))
    return decode_design_actions_v4(
        x,
        N_t=N_t,
        Eff_pool=Eff_pool,
        Rew_pool=Rew_pool,
        t_min=t_min,
        t_max=t_max,
        t_step=t_step,
        min_gap_t=min_gap_t,
        delta_er=delta_er,
        delta_max=delta_max,
        guard_after_t=guard_after_t,
        max_bonus_total=0,
        bonus_intervals=None,
        rng=rng_design,
    )


def _bo_design_for_pair(
    models: Sequence[Any],
    pair: Tuple[int, int],
    tau: float,
    *,
    Eff_pool: Sequence[float],
    Rew_pool: Sequence[float],
    N_t: int,
    t_step: float,
    min_gap_t: float,
    delta_er: float,
    delta_max: float,
    guard_after_t: float,
    sigma: float,
    n_calls: int,
    rng: random.Random,
) -> Tuple[Dict[str, Any], float, float]:
    """BO léger (skopt) pour maximiser Chernoff sur une paire, retourne design et meilleurs Chernoff/JSD."""
    t_min = 0.0
    t_max = float(tau)
    t_fixed = _build_fixed_times(t_min, t_max, t_step)
    n_intervals = max(0, len(t_fixed) - 1)

    space = []
    for _ in range(N_t):
        space.append(Real(t_min, t_max))
    for _ in range(3):
        space.append(Integer(0, len(Eff_pool) - 1))
    for _ in range(3):
        space.append(Integer(0, len(Rew_pool) - 1))
    for _ in range(2):
        space.append(Integer(0, len(Rew_pool) - 1))
    max_delta = min(delta_max, max(delta_er, t_step))
    for _ in range(3):
        space.append(Real(delta_er, max_delta))
    for _ in range(n_intervals):
        space.append(Integer(0, 3))  # action choice
    for _ in range(n_intervals * 2):
        space.append(Real(0.0, 1.0))

    def objective(x: List[float]) -> float:
        design = decode_design_actions_v4(
            x,
            N_t=N_t,
            Eff_pool=Eff_pool,
            Rew_pool=Rew_pool,
            t_min=t_min,
            t_max=t_max,
            t_step=t_step,
            min_gap_t=min_gap_t,
            delta_er=delta_er,
            delta_max=delta_max,
            guard_after_t=guard_after_t,
            max_bonus_total=0,
            bonus_intervals=None,
            rng=random.Random(rng.randint(0, 2**31 - 1)),
        )
        lap = laplace_jsd_for_design([models[pair[0]], models[pair[1]]], design, sigma=sigma, n_jobs=1)
        mu = lap.get("mu_y", [])
        Vy = lap.get("Vy", [])
        if not mu or not Vy:
            return 1e6
        C_mat, _P = _pairwise_chernoff_matrix(mu, Vy)
        c_val = C_mat[0][1]
        if not math.isfinite(c_val):
            return 1e6
        return -c_val  # maximise Chernoff

    res = gp_minimize(
        objective,
        space,
        n_calls=max(1, n_calls),
        n_initial_points=max(1, min(5, n_calls)),
        random_state=rng.randint(0, 2**31 - 1),
    )
    best_x = res.x
    best_design = decode_design_actions_v4(
        best_x,
        N_t=N_t,
        Eff_pool=Eff_pool,
        Rew_pool=Rew_pool,
        t_min=t_min,
        t_max=t_max,
        t_step=t_step,
        min_gap_t=min_gap_t,
        delta_er=delta_er,
        delta_max=delta_max,
        guard_after_t=guard_after_t,
        max_bonus_total=0,
        bonus_intervals=None,
        rng=random.Random(rng.randint(0, 2**31 - 1)),
    )
    lap_best = laplace_jsd_for_design([models[pair[0]], models[pair[1]]], best_design, sigma=sigma, n_jobs=1)
    mu_b = lap_best.get("mu_y", [])
    Vy_b = lap_best.get("Vy", [])
    if mu_b and Vy_b:
        C_mat, _P = _pairwise_chernoff_matrix(mu_b, Vy_b)
        jsd_b, _ = _jensen_shannon_gaussians([mu_b[0], mu_b[1]], [Vy_b[0], Vy_b[1]])
        return best_design, float(C_mat[0][1]), float(jsd_b)
    return best_design, float("nan"), float("nan")


def allocation_curves(
    models: Sequence[Any],
    *,
    T_max: float,
    t_step: float = 30.0,
    n_realizations: int = 5,
    n_candidates: int = 10,
    N_t: int = 5,
    sigma: float = 0.1,
    delta_er: float = 1.0,
    delta_max: float = 10.0,
    guard_after_t: float = 2.0,
    seed: int = 1234,
    n_jobs_outer: int = 1,
    show_progress: bool = True,
) -> Dict[str, Any]:
    Eff_pool, Rew_pool = make_demo_pools()
    tau_vals = np.arange(t_step, T_max + 1e-6, t_step)
    pairs = [(i, j) for i in range(len(models)) for j in range(i + 1, len(models))]

    rng = random.Random(seed)

    cher_mean = {p: [] for p in pairs}
    cher_std = {p: [] for p in pairs}
    jsd_mean = {p: [] for p in pairs}
    jsd_std = {p: [] for p in pairs}

    tasks = [(tau, pair) for tau in tau_vals for pair in pairs]
    outer_jobs = max(1, n_jobs_outer)
    pbar = tqdm(total=len(tasks), desc="pairs x tau", leave=True) if (show_progress and HAVE_TQDM) else None

    def _eval_task(tau: float, pair: Tuple[int, int]) -> Tuple[float, Tuple[int, int], float, float]:
        cher_samples: List[float] = []
        jsd_samples: List[float] = []
        for _ in range(max(1, n_realizations)):
            _rng_local = random.Random(rng.randint(0, 2**31 - 1))
            best_design, best_c, best_jsd = _bo_design_for_pair(
                models,
                pair,
                tau,
                Eff_pool=Eff_pool,
                Rew_pool=Rew_pool,
                N_t=N_t,
                t_step=t_step,
                min_gap_t=5.0,
                delta_er=delta_er,
                delta_max=delta_max,
                guard_after_t=guard_after_t,
                sigma=sigma,
                n_calls=n_candidates,
                rng=_rng_local,
            )
            if math.isfinite(best_c):
                cher_samples.append(best_c)
            if math.isfinite(best_jsd):
                jsd_samples.append(best_jsd)
        cher_m = float(np.mean(cher_samples)) if cher_samples else float("nan")
        cher_s = float(np.std(cher_samples)) if cher_samples else float("nan")
        jsd_m = float(np.mean(jsd_samples)) if jsd_samples else float("nan")
        jsd_s = float(np.std(jsd_samples)) if jsd_samples else float("nan")
        return tau, pair, cher_m, cher_s, jsd_m, jsd_s

    if outer_jobs == 1:
        for tau, pair in tasks:
            tau, pair, cm, cs, jm, js = _eval_task(tau, pair)
            cher_mean[pair].append(cm)
            cher_std[pair].append(cs)
            jsd_mean[pair].append(jm)
            jsd_std[pair].append(js)
            if pbar is not None:
                pbar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=outer_jobs) as ex:
            futs = [ex.submit(_eval_task, tau, pair) for tau, pair in tasks]
            for fut in as_completed(futs):
                tau, pair, cm, cs, jm, js = fut.result()
                idx_tau = int((tau - tau_vals[0]) / t_step)
                cher_mean[pair].append(cm)
                cher_std[pair].append(cs)
                jsd_mean[pair].append(jm)
                jsd_std[pair].append(js)
                if pbar is not None:
                    pbar.update(1)
    if pbar is not None:
        pbar.close()

    # Plot with checkboxes
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    ax_c, ax_j = axes
    lines = {}
    bands = {}
    tau_plot = tau_vals.tolist()
    for pair in pairs:
        label = f"{pair[0]}-{pair[1]}"
        cm = np.asarray(cher_mean[pair], dtype=float)
        cs = np.asarray(cher_std[pair], dtype=float)
        jm = np.asarray(jsd_mean[pair], dtype=float)
        js = np.asarray(jsd_std[pair], dtype=float)
        # Ensure same length as tau_plot (in case of threading order issues)
        if len(cm) != len(tau_plot):
            # pad with nan
            cm = np.full_like(tau_plot, np.nan, dtype=float)
            cs = np.full_like(tau_plot, np.nan, dtype=float)
            jm = np.full_like(tau_plot, np.nan, dtype=float)
            js = np.full_like(tau_plot, np.nan, dtype=float)
        l_c, = ax_c.plot(tau_plot, cm, label=label)
        b_c = ax_c.fill_between(tau_plot, cm - cs, cm + cs, alpha=0.2)
        l_j, = ax_j.plot(tau_plot, jm, label=label)
        b_j = ax_j.fill_between(tau_plot, jm - js, jm + js, alpha=0.2)
        lines[label] = (l_c, l_j)
        bands[label] = (b_c, b_j)

    ax_c.set_title("Chernoff vs allocation")
    ax_c.set_ylabel("Chernoff")
    ax_c.grid(True, linestyle=":", alpha=0.3)
    ax_j.set_title("Laplace-JSD vs allocation")
    ax_j.set_xlabel("tau (s)")
    ax_j.set_ylabel("JSD")
    ax_j.grid(True, linestyle=":", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 0.8, 1])
    rax = plt.axes([0.82, 0.3, 0.15, 0.6])
    labels = [f"{i}-{j}" for i, j in pairs]
    visibility = [True] * len(labels)
    check = CheckButtons(rax, labels, visibility)

    def toggle(label: str) -> None:
        lc, lj = lines[label]
        bc, bj = bands[label]
        vis = not lc.get_visible()
        lc.set_visible(vis)
        lj.set_visible(vis)
        bc.set_visible(vis)
        bj.set_visible(vis)
        plt.draw()

    check.on_clicked(toggle)
    plt.show()

    return {
        "pairs": pairs,
        "tau": tau_vals.tolist(),
        "cher_mean": cher_mean,
        "cher_std": cher_std,
        "jsd_mean": jsd_mean,
        "jsd_std": jsd_std,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Allocation curves for Chernoff and Laplace-JSD.")
    parser.add_argument("--T-max", type=float, default=900.0, help="Budget total (s).")
    parser.add_argument("--t-step", type=float, default=60.0, help="Pas de tau (s).")
    parser.add_argument("--n-realizations", type=int, default=5, help="Nombre de réalisations par tau.")
    parser.add_argument("--n-candidates", type=int, default=20, help="Nombre d'évaluations BO (n_calls) par réalisation.")
    parser.add_argument("--N-t", type=int, default=0, help="Nombre de temps optimisés v4.")
    parser.add_argument("--sigma", type=float, default=0.1, help="Bruit observation pour Laplace-JSD.")
    parser.add_argument("--seed", type=int, default=1234, help="Graine RNG.")
    parser.add_argument("--n-jobs-outer", type=int, default=5, help="Parallélisme sur paires x tau.")
    args = parser.parse_args()

    models = build_six_models()
    allocation_curves(
        models,
        T_max=args.T_max,
        t_step=args.t_step,
        n_realizations=args.n_realizations,
        n_candidates=args.n_candidates,
        N_t=args.N_t,
        sigma=args.sigma,
        seed=args.seed,
        n_jobs_outer=args.n_jobs_outer,
    )


if __name__ == "__main__":
    main()
