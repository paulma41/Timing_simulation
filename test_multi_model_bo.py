from __future__ import annotations

import argparse
import math
import os
import random
import pickle
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
from functions import build_h_action_function
from design_optimizer import expected_log_bayes_factor_matrix_for_design
from design_optimizer.laplace_jsd import laplace_jsd_for_design, _jensen_shannon_gaussians, _pairwise_chernoff_matrix
from test_multi_model import plot_summary, _build_fixed_times


def make_demo_pools() -> Tuple[List[float], List[float]]:
    Eff_pool = -1*np.linspace(0.1,1.5,10)
    Rew_pool = np.linspace(0.5,2,10)
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


def decode_design_from_vector_v2(
    x: List[float],
    N_t: int,
    N_eff: int,
    Eff_pool: Sequence[float],
    Rew_pool: Sequence[float],
    t_min: float,
    t_max: float,
    min_gap_t: float,
    delta_er: float,
    rng: random.Random,
) -> Dict[str, Any]:
    """
    Variante: temps fixes (0 puis +60), N_t temps optimisables,
    Eff/Rew indices optimisés, instants Eff/Rew tirés aléatoirement en alternance.
    Ordre de x: t_optim (N_t), Eff_idx (N_eff), Rew_idx (N_eff).
    """
    idx = 0
    t_optim = list(x[idx : idx + N_t])
    idx += N_t
    Eff_idx = [int(round(v)) for v in x[idx : idx + N_eff]]
    idx += N_eff
    Rew_idx = [int(round(v)) for v in x[idx : idx + N_eff]]

    # Temps fixes : 0 puis tous les 60
    fixed_t: List[float] = []
    current = t_min
    while current <= t_max:
        fixed_t.append(current)
        current += 60.0

    # Temps optimisés bornés à [t_min, t_max]
    t_candidates = fixed_t + [max(t_min, min(t_max, v)) for v in t_optim]
    t_candidates.sort()

    # Appliquer min_gap
    t_proj: List[float] = []
    last = None
    for v in t_candidates:
        if last is None or v - last >= min_gap_t:
            t_proj.append(v)
            last = v
    if len(t_proj) < len(fixed_t):
        t_proj = fixed_t  # fallback

    # Tirage aléatoire des instants Eff/Rew (alternés E, R, E, R)
    times_events = sorted(rng.uniform(t_min, t_max) for _ in range(2 * N_eff))
    Eff_t: List[float] = []
    Rew_t: List[float] = []
    for k in range(N_eff):
        e_t = times_events[2 * k]
        r_t = times_events[2 * k + 1]
        if r_t - e_t < delta_er:
            r_t = min(t_max, e_t + delta_er)
        Eff_t.append(e_t)
        Rew_t.append(r_t)

    # Indices -> valeurs (avec modulo)
    neff_pool = len(Eff_pool)
    nrew_pool = len(Rew_pool)
    Eff_val = [Eff_pool[Eff_idx[k] % neff_pool] for k in range(N_eff)]
    Rew_val = [Rew_pool[Rew_idx[k] % nrew_pool] for k in range(N_eff)]

    Eff_ = list(zip(Eff_val, Eff_t))
    Rew_ = list(zip(Rew_val, Rew_t))
    A = list(zip(Eff_, Rew_))

    return {"t": t_proj, "Eff_": Eff_, "Rew_": Rew_, "A": A}


def decode_design_actions_v4(
    x: List[float],
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
    max_bonus_total: int | None = None,
    bonus_intervals: Sequence[int] | None = None,
    rng: random.Random,
) -> Dict[str, Any]:
    """
    Version orientée actions types (3 max) :
      - 3 types d'actions globaux (eff/reward + delta)
      - 0 ou 1 action par intervalle de 30s (ou t_step), choisie parmi ces 3 types
      - t_rew = t_eff + delta_k, tirés aléatoirement dans l'intervalle

    x layout:
      [t_opt (N_t),
       eff_idx (3),
       rew_idx (3),
       bonus_idx (2),
       delta (3),
       action_choice per interval (len(t_fixed)-1),
       bonus_u per interval (2 * len(t_fixed)-2)]
    action_choice: 0 => aucune, 1..3 => type d'action (modulo 3, +1)
    """
    idx = 0
    # Temps optimisés additionnels
    t_opt = list(x[idx : idx + N_t])
    idx += N_t

    eff_idx = [int(round(v)) for v in x[idx : idx + 3]]
    idx += 3
    rew_idx = [int(round(v)) for v in x[idx : idx + 3]]
    idx += 3
    bonus_idx = [int(round(v)) for v in x[idx : idx + 2]]
    idx += 2
    delta_vals = [max(delta_er, min(float(v), delta_max)) for v in x[idx : idx + 3]]
    idx += 3

    t_fixed = _build_fixed_times(t_min, t_max, t_step)
    n_intervals = max(0, len(t_fixed) - 1)
    action_choices = [int(round(v)) for v in x[idx : idx + n_intervals]]
    idx += n_intervals
    bonus_per_interval = 2
    bonus_u = [float(v) for v in x[idx : idx + n_intervals * bonus_per_interval]]

    # Temps de mesure : fixes + optimisés (bornage + min_gap)
    t_candidates = t_fixed + [max(t_min, min(t_max, v)) for v in t_opt]
    t_candidates.sort()
    t_proj: List[float] = []
    last = None
    for v in t_candidates:
        if last is None or v - last >= min_gap_t:
            t_proj.append(v)
            last = v
    if len(t_proj) < len(t_fixed):
        t_proj = t_fixed  # fallback si min_gap trop strict

    Eff_: List[Tuple[float, float]] = []
    Rew_: List[Tuple[float, float]] = []
    Bonus_: List[Tuple[float, float]] = []

    neff = len(Eff_pool)
    nrew = len(Rew_pool)
    k_types = 3  # normalisation dans action_avg

    # Actions Eff/Rew (au plus une par intervalle)
    for m in range(n_intervals):
        choice_raw = action_choices[m] if m < len(action_choices) else 0
        choice = choice_raw % 4  # 0 = none, 1..3 = type
        if choice == 0:
            continue
        k = choice - 1
        delta_k = delta_vals[k]
        start = t_fixed[m] + guard_after_t
        end = t_fixed[m + 1] - guard_after_t
        if end - start <= delta_k:
            continue  # interval trop court
        t_eff = rng.uniform(start, end - delta_k)
        t_rew = t_eff + delta_k
        eff_val = Eff_pool[eff_idx[k] % neff]
        rew_val = Rew_pool[rew_idx[k] % nrew]
        Eff_.append((eff_val, t_eff))
        Rew_.append((rew_val, t_rew))

    # Bonus: si bonus_intervals est fourni, on place un bonus dans chaque intervalle indiqué,
    # sinon on conserve l'ancien comportement (max 2 bonus globaux).
    if bonus_intervals is not None:
        valid_intervals = [m for m in bonus_intervals if 0 <= m < n_intervals]
        for b_idx, m in enumerate(valid_intervals):
            start_bonus = t_fixed[m] + guard_after_t
            end_bonus = t_fixed[m + 1] - guard_after_t
            if end_bonus <= start_bonus:
                continue
            u = bonus_u[b_idx] if b_idx < len(bonus_u) else 0.0
            u_clamped = max(0.0, min(1.0, u))
            t_bonus = start_bonus + u_clamped * (end_bonus - start_bonus)
            bonus_val = Rew_pool[bonus_idx[b_idx % len(bonus_idx)] % nrew] if bonus_idx else Rew_pool[0 % nrew]
            bonus_evt = (bonus_val, t_bonus)
            Bonus_.append(bonus_evt)
            Rew_.append(bonus_evt)
    else:
        bonus_used = 0
        max_bonus_total = 2
        for m in range(n_intervals):
            # Seuls les intervalles où une action a été placée sont éligibles
            # (on cherche un reward dans Rew_ dans l'intervalle correspondant).
            if bonus_used >= max_bonus_total:
                break
            start_interval = t_fixed[m]
            end_interval = t_fixed[m + 1]
            # Cherche un reward dans l'intervalle pour ancrer le bonus après
            rewards_m = [r_t for (_, r_t) in Rew_ if start_interval <= r_t <= end_interval]
            if not rewards_m:
                continue
            t_rew_anchor = max(rewards_m)
            start_bonus = t_rew_anchor + guard_after_t
            end_bonus = end_interval - guard_after_t
            if end_bonus <= start_bonus:
                continue
            for b in range(bonus_per_interval):
                if bonus_used >= max_bonus_total:
                    break
                idx_u = m * bonus_per_interval + b
                u = bonus_u[idx_u] if idx_u < len(bonus_u) else 0.0
                u_clamped = max(0.0, min(1.0, u))
                t_bonus = start_bonus + u_clamped * (end_bonus - start_bonus)
                bonus_val = Rew_pool[bonus_idx[b % len(bonus_idx)] % nrew] if bonus_idx else Rew_pool[0 % nrew]
                bonus_evt = (bonus_val, t_bonus)
                Bonus_.append(bonus_evt)
                Rew_.append(bonus_evt)
                bonus_used += 1

    A = list(zip(Eff_, Rew_))
    Rew_sorted = sorted(Rew_, key=lambda p: p[1])
    return {"t": t_proj, "Eff_": Eff_, "Rew_": Rew_sorted, "Bonus_": Bonus_, "A": A, "K_types": k_types}


def plot_utility_convergence(history: List[List[List[float]]], models: Sequence[Any]) -> None:
    """
    Trace U[i,j] en fonction des ťvaluations (une sous-figure par couple i,j).
    """
    if not history:
        print("[diag] no utility history to plot.")
        return
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"[diag] unable to plot convergence: {exc}")
        return

    n_iter = len(history)
    m = len(history[0])
    fig, axes = plt.subplots(m, m, figsize=(3 * m, 3 * m), sharex=True)
    if m == 1:
        axes = [[axes]]  # type: ignore
    for i in range(m):
        for j in range(m):
            ax = axes[i][j]
            series = np.array([history[k][i][j] for k in range(n_iter)], dtype=float)
            ax.plot(range(1, n_iter + 1), series, linewidth=1.0)
            ax.axhline(y=0.0, color="k", linestyle=":", linewidth=0.6)
            if i == m - 1:
                ax.set_xlabel("iteration")
            if j == 0:
                ax.set_ylabel(f"U[{i},{j}]")
            ax.set_title(f"{i}→{j}", fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.3)
    fig.suptitle("Evolution de U[i,j] au fil des ťvaluations")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Multi-model BO design optimization (Jeffreys / Laplace-JS)")
    parser.add_argument("--N_t", type=int, default=5, help="number of optimised measurement times")
    parser.add_argument("--n-calls", type=int, default=150, help="number of BO evaluations")
    parser.add_argument("--n-init", type=int, default=50, help="number of BO initial random evaluations")
    parser.add_argument("--sigma", type=float, default=0.1, help="obs noise stddev")
    parser.add_argument("--outer", type=int, default=20, help="K_outer Monte Carlo samples")
    parser.add_argument("--inner", type=int, default=20, help="K_inner Monte Carlo samples per model")
    parser.add_argument("--jobs", type=int, default=16, help="number of parallel workers for Jeffreys utility (<=0 => auto)")
    parser.add_argument("--adaptive-utility", action="store_true", help="use adaptive per-row MC sampling")
    parser.add_argument("--outer-min", type=int, default=None, help="minimum K_outer for adaptive utility (default: auto)")
    parser.add_argument("--inner-min", type=int, default=None, help="minimum K_inner for adaptive utility (default: auto)")
    parser.add_argument("--tol-rel", type=float, default=0.02, help="relative tolerance for adaptive utility SE stop")
    parser.add_argument("--tol-abs", type=float, default=0.02, help="absolute tolerance for adaptive utility SE stop")
    parser.add_argument(
        "--criterion",
        choices=["jeffreys", "laplace-jsd"],
        default="jeffreys",
        help="choix de l'utilité de design: 'jeffreys' (MC log-BF cible) ou 'laplace-jsd' (Jensen-Shannon Laplace)",
    )
    parser.add_argument(
        "--single-eval",
        action="store_true",
        help="saute le BO et calcule une seule matrice U sur un design tire au hasard (diag convergence inner/outer)",
    )
    parser.add_argument(
        "--utility-progress",
        action="store_true",
        help="affiche un tqdm pour chaque évaluation de la matrice U (même en parallèle)",
    )
    parser.add_argument(
        "--plot-convergence",
        action="store_true",
        help="trace l'evolution de chaque U[i,j] au fil des evaluations (ouvre une fenetre matplotlib)",
    )
    parser.add_argument(
        "--design-samples",
        type=int,
        default=1,
        help="nombre de tirages Eff/Rew par design, on moyenne le score/utility sur ces tirages",
    )
    parser.add_argument(
        "--max-bonus",
        type=int,
        default=2,
        help="nombre maximal global de bonus dans le design v4 (0 => aucun bonus)",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=600.0,
        help="temps maximum du design (t_max), en secondes (par défaut 600).",
    )
    parser.add_argument(
        "--check-covariance",
        action="store_true",
        help="enregistre les U[i,j] et trace une heatmap de corrélation sur les évaluations",
    )
    args = parser.parse_args()

    print("[info] Building models…")
    models = build_six_models()
    Eff_pool, Rew_pool = make_demo_pools()
    # v4: 3 types d'actions globales
    N_eff_types = 3
    N_eff = N_eff_types  # compat interne pour anciens usages

    bad_objective = 1e12  # large finite penalty for invalid evaluations

    parallel_jobs = args.jobs if args.jobs is not None else 1
    if parallel_jobs <= 0:
        parallel_jobs = os.cpu_count() or 1
    parallel_jobs = min(parallel_jobs, len(models))
    print(f"[info] Parallel workers for Jeffreys utility: {parallel_jobs}")
    show_utility_progress = bool(args.utility_progress)
    if show_utility_progress:
        print("[info] Utility progress bar enabled")
    track_convergence = bool(args.plot_convergence)
    track_covariance = bool(args.check_covariance)
    max_bonus_cli = int(args.max_bonus) if hasattr(args, "max_bonus") else 2
    utility_history: List[List[List[float]]] = []
    objective_cache: Dict[Tuple[float, ...], float] = {}
    cache_file = ".bo_cache_jeffreys.pkl" if args.criterion == "jeffreys" else ".bo_cache_laplace.pkl"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as fh:
                objective_cache = pickle.load(fh)
                print(f"[info] Loaded cache {cache_file} with {len(objective_cache)} entries.")
        except Exception as exc:
            print(f"[warn] Unable to load cache {cache_file}: {exc}")

    # BO space (v4): t_opt, eff_idx(3), rew_idx(3), bonus_idx(2), delta(3), action_choices, bonus_u (2 par intervalle)
    t_min = 0.0
    t_max = float(getattr(args, "t_max", 600.0))
    min_gap_t = 5.0
    delta_er = 5.0
    t_step = 30.0
    delta_max = 10.0
    guard_after_t = 2.0
    t_fixed = _build_fixed_times(t_min, t_max, t_step)
    n_intervals = max(0, len(t_fixed) - 1)

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
    # t_opt (N_t)
    for i in range(args.N_t):
        space.append(Real(t_min, t_max, name=f"t_{i}"))
    # eff_idx / rew_idx (3 types)
    for k in range(N_eff_types):
        space.append(Integer(0, len(Eff_pool) - 1, name=f"ie_{k}"))
    for k in range(N_eff_types):
        space.append(Integer(0, len(Rew_pool) - 1, name=f"ir_{k}"))
    # bonus reward indices (2 bonus)
    for k in range(2):
        space.append(Integer(0, len(Rew_pool) - 1, name=f"ib_{k}"))
    # delta par type
    for k in range(N_eff_types):
        space.append(Real(delta_er, delta_max, name=f"delta_{k}"))
    # action choices par intervalle (0 = none, 1..3 = type)
    for m in range(n_intervals):
        space.append(Integer(0, N_eff_types, name=f"a_{m}"))
    # bonus placements (2 par intervalle, 0..1 pour interpolation entre Rew et prochain Eff)
    for m in range(n_intervals):
        space.append(Real(0.0, 1.0, name=f"b1_{m}"))
        space.append(Real(0.0, 1.0, name=f"b2_{m}"))

    rng = random.Random(123)
    m_models = len(models)
    i_target = rng.randrange(m_models)
    j_choices = [j for j in range(m_models) if j != i_target]
    j_target = rng.choice(j_choices)
    if args.criterion == "jeffreys":
        print(f"[info] Optimisation ciblée sur U[{i_target},{j_target}]")
    else:
        print("[info] Optimisation Laplace-JS multi-modèles (Jensen-Shannon)")
    utility_history: List[List[List[float]]] = []
    objective_cache: Dict[Tuple[float, ...], float] = {}
    cache_file = ".bo_cache_jeffreys.pkl" if args.criterion == "jeffreys" else ".bo_cache_laplace.pkl"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as fh:
                objective_cache = pickle.load(fh)
                print(f"[info] Loaded cache {cache_file} with {len(objective_cache)} entries.")
        except Exception as exc:
            print(f"[warn] Unable to load cache {cache_file}: {exc}")

    if args.single_eval:
        print("[info] Single-eval mode: une seule évaluation du critère de design")
        x0: List[float] = []
        for _ in range(args.N_t):
            x0.append(rng.uniform(t_min, t_max))
        for _ in range(N_eff_types):
            x0.append(rng.randrange(len(Eff_pool)))
        for _ in range(N_eff_types):
            x0.append(rng.randrange(len(Rew_pool)))
        for _ in range(2):
            x0.append(rng.randrange(len(Rew_pool)))
        for _ in range(N_eff_types):
            x0.append(rng.uniform(delta_er, delta_max))
        for _ in range(n_intervals):
            x0.append(rng.randrange(0, N_eff_types + 1))
        for _ in range(n_intervals * 2):
            x0.append(rng.random())

        design_inputs = decode_design_actions_v4(
            x0,
            N_t=args.N_t,
            Eff_pool=Eff_pool,
            Rew_pool=Rew_pool,
            t_min=t_min,
            t_max=t_max,
            t_step=t_step,
            min_gap_t=min_gap_t,
            delta_er=delta_er,
            delta_max=delta_max,
            guard_after_t=guard_after_t,
            bonus_intervals=[] if max_bonus_cli <= 0 else None,
            rng=rng,
        )
        if args.criterion == "jeffreys":
            U_single = expected_log_bayes_factor_matrix_for_design(
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
                resample_params=False,
            )
            print(f"[result] U[{i_target},{j_target}] (single eval): {U_single[i_target][j_target]}")
            print("[result] t:", design_inputs["t"])
            print("[result] Eff_:", design_inputs["Eff_"])
            print("[result] Rew_:", design_inputs["Rew_"])
            print("[result] U matrix (single eval):")
            for row in U_single:
                print([round(x, 3) for x in row])
            if track_convergence:
                plot_utility_convergence([U_single], models)
        else:
            lap = laplace_jsd_for_design(models, design_inputs, sigma=args.sigma)
            print(f"[result] Laplace-JS DJS (single eval): {lap['DJS']}")
            print(f"[result] Bound term b (Hp-DJS): {lap['b']}")
            print("[result] t:", design_inputs["t"])
            print("[result] Eff_:", design_inputs["Eff_"])
            print("[result] Rew_:", design_inputs["Rew_"])
            mu_list = lap.get("mu_y", [])
            Vy_list = lap.get("Vy", [])
            if (track_convergence or track_covariance) and mu_list and Vy_list:
                m = len(mu_list)
                U_lap = [[0.0 for _ in range(m)] for _ in range(m)]
                for i in range(m):
                    for j in range(m):
                        if i == j:
                            continue
                        djs, _ = _jensen_shannon_gaussians([mu_list[i], mu_list[j]], [Vy_list[i], Vy_list[j]])
                        U_lap[i][j] = djs
                print("[result] Pairwise JSD matrix (single eval):")
                for row in U_lap:
                    print([round(x, 3) for x in row])
                if track_convergence:
                    plot_utility_convergence([U_lap], models)
                if track_covariance:
                    try:
                        import matplotlib.pyplot as plt  # type: ignore
                        import numpy as np  # type: ignore
                        corr = np.corrcoef(np.array([[uij for row in U_lap for uij in row]]), rowvar=False)
                        fig, ax = plt.subplots(figsize=(5, 4))
                        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="corr(U[i,j])")
                        ax.set_title("Correlation of U[i,j] (single eval)")
                        plt.tight_layout()
                        plt.show()
                    except Exception as exc:
                        print(f"[warn] Unable to plot covariance (single eval): {exc}")
        return

    # Mode covariance: n_calls designs aléatoires, pas de BO
    if args.check_covariance:
        print(f"[info] Check-covariance mode: {args.n_calls} évaluations aléatoires sans BO")
        for _ in range(max(1, args.n_calls)):
            x_rand: List[float] = []
            for _ in range(args.N_t):
                x_rand.append(rng.uniform(t_min, t_max))
            for _ in range(N_eff_types):
                x_rand.append(rng.randrange(len(Eff_pool)))
            for _ in range(N_eff_types):
                x_rand.append(rng.randrange(len(Rew_pool)))
            for _ in range(2):
                x_rand.append(rng.randrange(len(Rew_pool)))
            for _ in range(N_eff_types):
                x_rand.append(rng.uniform(delta_er, delta_max))
            for _ in range(n_intervals):
                x_rand.append(rng.randrange(0, N_eff_types + 1))
            for _ in range(n_intervals * 2):
                x_rand.append(rng.random())
            rng_base = random.Random(hash(tuple(round(float(v), 6) for v in x_rand)) & 0x7FFFFFFF)
            U_acc: List[List[float]] = []
            score_acc = 0.0
            for s in range(max(1, args.design_samples)):
                rng_design = random.Random(rng_base.randint(0, 2**31 - 1))
                design_inputs = decode_design_actions_v4(
                    x_rand,
                    N_t=args.N_t,
                    Eff_pool=Eff_pool,
                    Rew_pool=Rew_pool,
                    t_min=t_min,
                    t_max=t_max,
                    t_step=t_step,
                    min_gap_t=min_gap_t,
                    delta_er=delta_er,
                    delta_max=delta_max,
                    guard_after_t=guard_after_t,
                    bonus_intervals=[] if max_bonus_cli <= 0 else None,
                    rng=rng_design,
                )
                if args.criterion == "jeffreys":
                    U_curr = expected_log_bayes_factor_matrix_for_design(
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
                        resample_params=False,
                    )
                    score_acc += U_curr[i_target][j_target]
                else:
                    lap = laplace_jsd_for_design(models, design_inputs, sigma=args.sigma)
                    score_acc += float(lap["DJS"])
                    mu_list = lap.get("mu_y", [])
                    Vy_list = lap.get("Vy", [])
                    if mu_list and Vy_list:
                        mloc = len(mu_list)
                        U_curr = [[0.0 for _ in range(mloc)] for _ in range(mloc)]
                        for i in range(mloc):
                            for j in range(mloc):
                                if i == j:
                                    continue
                                djs, _ = _jensen_shannon_gaussians([mu_list[i], mu_list[j]], [Vy_list[i], Vy_list[j]])
                                U_curr[i][j] = djs
                    else:
                        U_curr = [[0.0 for _ in range(len(models))] for _ in range(len(models))]
                if not U_acc:
                    U_acc = [[u for u in row] for row in U_curr]
                else:
                    for i in range(len(U_acc)):
                        for j in range(len(U_acc[i])):
                            U_acc[i][j] += U_curr[i][j]
            n_samp = max(1, args.design_samples)
            U_avg = [[u / n_samp for u in row] for row in U_acc] if U_acc else []
            utility_history.append(U_avg)
        # Plots
        if track_convergence and utility_history:
            plot_utility_convergence(utility_history, models)
        if track_covariance and utility_history:
            try:
                import numpy as np  # type: ignore
                import matplotlib.pyplot as plt  # type: ignore
                pair_keys = sorted({(i, j) for i in range(len(models)) for j in range(len(models)) if i != j})
                data = np.array([[row[i][j] for i, j in pair_keys] for row in utility_history], dtype=float)
                if data.shape[1] > 1:
                    corr = np.corrcoef(data, rowvar=False)
                    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(pair_keys)), max(4, 0.4 * len(pair_keys))))
                    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
                    ax.set_xticks(range(len(pair_keys)))
                    ax.set_yticks(range(len(pair_keys)))
                    ax.set_xticklabels([f"{i},{j}" for i, j in pair_keys], rotation=90, fontsize=6)
                    ax.set_yticklabels([f"{i},{j}" for i, j in pair_keys], fontsize=6)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="corr(U[i,j])")
                    ax.set_title("Correlation of U[i,j] over random evaluations")
                    plt.tight_layout()
                    plt.show()
            except Exception as exc:
                print(f"[warn] Unable to plot covariance: {exc}")
        return

    def objective(x: List[float]) -> float:
        key = tuple(round(float(v), 6) for v in x)
        if key in objective_cache:
            return objective_cache[key]
        rng_base = random.Random(hash(key) & 0x7FFFFFFF)
        score_acc = 0.0  # utilisé uniquement pour Jeffreys
        U_acc: List[List[float]] = []
        for s in range(max(1, args.design_samples)):
            rng_design = random.Random(rng_base.randint(0, 2**31 - 1))
            design_inputs = decode_design_actions_v4(
                x,
                N_t=args.N_t,
                Eff_pool=Eff_pool,
                Rew_pool=Rew_pool,
                t_min=t_min,
                t_max=t_max,
                t_step=t_step,
                min_gap_t=min_gap_t,
                delta_er=delta_er,
                delta_max=delta_max,
                guard_after_t=guard_after_t,
                max_bonus_total=max_bonus_cli if max_bonus_cli > 0 else 0,
                bonus_intervals=[] if max_bonus_cli <= 0 else None,
                rng=rng_design,
            )
            try:
                if args.criterion == "jeffreys":
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
                    score_acc += U[i_target][j_target]
                else:
                    # Laplace-JS : on utilise un critère maximin sur la matrice pairwise,
                    # pas le scalaire DJS global.
                    lap = laplace_jsd_for_design(models, design_inputs, sigma=args.sigma)
                    mu_list = lap.get("mu_y", [])
                    Vy_list = lap.get("Vy", [])
                    if mu_list and Vy_list:
                        mloc = len(mu_list)
                        U = [[0.0 for _ in range(mloc)] for _ in range(mloc)]
                        for i in range(mloc):
                            for j in range(mloc):
                                if i == j:
                                    continue
                                djs, _ = _jensen_shannon_gaussians([mu_list[i], mu_list[j]], [Vy_list[i], Vy_list[j]])
                                U[i][j] = djs
                    else:
                        U = [[0.0 for _ in range(len(models))] for _ in range(len(models))]
            except Exception:
                return bad_objective
            # Accumuler la matrice U pour construire un critère maximin (Laplace-JS)
            if not U_acc:
                U_acc = [[u for u in row] for row in U]
            else:
                for i in range(len(U_acc)):
                    for j in range(len(U_acc[i])):
                        U_acc[i][j] += U[i][j]
        n_samp = max(1, args.design_samples)
        U_avg = [[u / n_samp for u in row] for row in U_acc] if U_acc else []
        if track_convergence or track_covariance and U_avg:
            utility_history.append(U_avg)

        if args.criterion == "jeffreys":
            score = score_acc / n_samp if n_samp > 0 else bad_objective
        else:
            # Maximin sur la matrice pairwise Laplace-JS (U_avg)
            if not U_avg:
                score = bad_objective
            else:
                mloc = len(U_avg)
                row_min: List[float] = []
                for i in range(mloc):
                    vals = [U_avg[i][j] for j in range(mloc) if j != i]
                    row_min.append(min(vals) if vals else float("-inf"))
                score = min(row_min)

        if not math.isfinite(score):
            return bad_objective
        val = -score
        objective_cache[key] = val
        return val

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

    initial_point_generator = "random"
    try:
        initial_point_generator = "lhs"
    except Exception:
        initial_point_generator = "random"

    print(f"[info] Running BO v4 with N_t={args.N_t}, N_types={N_eff_types}, n_calls={n_calls}, n_init={n_init}")
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
    # Re-evaluate best_x with averaging over design samples
    def evaluate_design(x: List[float]):
        rng_base = random.Random(hash(tuple(round(float(v), 6) for v in x)) & 0x7FFFFFFF)
        score_acc = 0.0
        U_acc: List[List[float]] = []
        design_display = None
        for s in range(max(1, args.design_samples)):
            rng_design = random.Random(rng_base.randint(0, 2**31 - 1))
            design_inputs = decode_design_actions_v4(
                x,
                N_t=args.N_t,
                Eff_pool=Eff_pool,
                Rew_pool=Rew_pool,
                t_min=t_min,
                t_max=t_max,
                t_step=t_step,
                min_gap_t=min_gap_t,
                delta_er=delta_er,
                delta_max=delta_max,
                guard_after_t=guard_after_t,
                bonus_intervals=[] if max_bonus_cli <= 0 else None,
                rng=rng_design,
            )
            if design_display is None:
                design_display = design_inputs
            if args.criterion == "jeffreys":
                U_curr = expected_log_bayes_factor_matrix_for_design(
                    models,
                    design_inputs,
                    sigma=args.sigma,
                    K_outer=args.outer,
                    K_inner=args.inner,
                    rng=rng,
                    n_jobs=parallel_jobs,
                    progress=False,
                    adaptive=args.adaptive_utility,
                    K_outer_min=adaptive_outer_min,
                    K_outer_max=adaptive_outer_max,
                    K_inner_min=adaptive_inner_min,
                    K_inner_max=adaptive_inner_max,
                    tol_rel=args.tol_rel,
                    tol_abs=args.tol_abs,
                    resample_params=False,
                )
                score_acc += U_curr[i_target][j_target]
            else:
                lap_curr = laplace_jsd_for_design(models, design_inputs, sigma=args.sigma)
                score_acc += float(lap_curr["DJS"])
                mu_list = lap_curr.get("mu_y", [])
                Vy_list = lap_curr.get("Vy", [])
                if mu_list and Vy_list:
                    mloc = len(mu_list)
                    U_curr = [[0.0 for _ in range(mloc)] for _ in range(mloc)]
                    for i in range(mloc):
                        for j in range(mloc):
                            if i == j:
                                continue
                            djs, _ = _jensen_shannon_gaussians([mu_list[i], mu_list[j]], [Vy_list[i], Vy_list[j]])
                            U_curr[i][j] = djs
                else:
                    U_curr = [[0.0 for _ in range(len(models))] for _ in range(len(models))]
            if not U_acc:
                U_acc = [[u for u in row] for row in U_curr]
            else:
                for i in range(len(U_acc)):
                    for j in range(len(U_acc[i])):
                        U_acc[i][j] += U_curr[i][j]
        n_samp = max(1, args.design_samples)
        U_avg = [[u / n_samp for u in row] for row in U_acc] if U_acc else []
        score_avg = score_acc / n_samp
        return design_display or {}, U_avg, score_avg

    best_design, U_best, score_avg = evaluate_design(best_x)
    m = len(models)
    B_best = None
    if args.criterion == "jeffreys":
        target_score = score_avg
        print(f"[result] Best utility for target U[{i_target},{j_target}]:", target_score)
    else:
        print("[result] Best Laplace-JS DJS:", score_avg)
        # Chernoff error-bound matrix on best design
        lap_best = laplace_jsd_for_design(models, best_design, sigma=args.sigma)
        mu_best = lap_best.get("mu_y", [])
        Vy_best = lap_best.get("Vy", [])
        if mu_best and Vy_best:
            _, B_best = _pairwise_chernoff_matrix(mu_best, Vy_best)
        else:
            B_best = [[float("nan") for _ in range(m)] for _ in range(m)]
    print("[result] Best t:", best_design["t"])
    print("[result] Eff_:", best_design["Eff_"])
    print("[result] Rew_:", best_design["Rew_"])
    print("[result] U matrix:")
    for row in U_best:
        print([round(x, 3) for x in row])
    if B_best is not None:
        print("[result] Chernoff error-bound matrix (P_err <= 0.5 * exp(-C)):")
        for row in B_best:
            print([round(x, 3) for x in row])

    try:
        with open(cache_file, "wb") as fh:
            pickle.dump(objective_cache, fh)
        print(f"[info] Saved cache {cache_file} with {len(objective_cache)} entries.")
    except Exception as exc:
        print(f"[warn] Unable to save cache {cache_file}: {exc}")

    if track_convergence or track_covariance:
        if track_convergence:
            plot_utility_convergence(utility_history, models)
        if track_covariance and utility_history:
            try:
                import numpy as np  # type: ignore
                import matplotlib.pyplot as plt  # type: ignore
                pair_keys = sorted({(i, j) for i in range(len(models)) for j in range(len(models)) if i != j})
                data = np.array([[row[i][j] for i, j in pair_keys] for row in utility_history], dtype=float)
                if data.shape[1] > 1:
                    corr = np.corrcoef(data, rowvar=False)
                    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(pair_keys)), max(4, 0.4 * len(pair_keys))))
                    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
                    ax.set_xticks(range(len(pair_keys)))
                    ax.set_yticks(range(len(pair_keys)))
                    ax.set_xticklabels([f"{i},{j}" for i, j in pair_keys], rotation=90, fontsize=6)
                    ax.set_yticklabels([f"{i},{j}" for i, j in pair_keys], fontsize=6)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="corr(U[i,j])")
                    ax.set_title("Correlation of U[i,j] over evaluations")
                    plt.tight_layout()
                    plt.show()
            except Exception as exc:
                print(f"[warn] Unable to plot covariance heatmap: {exc}")

    # Plot summary using the same helper as random search
    try:
        plot_summary(
            U_best,
            best_design,
            models,
            title="BO maximin",
            b_matrix=B_best if args.criterion == "laplace-jsd" else None,
        )
    except Exception as exc:
        print(f"[diag] unable to plot summary: {exc}")


if __name__ == "__main__":
    main()
