from __future__ import annotations

import argparse
import math
import random
from typing import Any, Dict, List, Sequence, Tuple, Optional

from design_optimizer.laplace_jsd import laplace_jsd_for_design, _pairwise_chernoff_matrix
from test_multi_model_bo import make_demo_pools, _build_fixed_times
from test_multi_model import build_six_models, plot_summary


def choose_action_types(rng: random.Random, Eff_pool: Sequence[float], Rew_pool: Sequence[float], n_types: int = 3) -> List[Tuple[float, float]]:
    """Tire n_types couples (Eff, Rew) avec contrainte Eff < Rew."""
    types: List[Tuple[float, float]] = []
    trials = 0
    max_trials = 200 * max(1, n_types)
    while len(types) < n_types and trials < max_trials:
        e_val = rng.choice(Eff_pool)
        r_val = rng.choice(Rew_pool)
        trials += 1
        if e_val < r_val:
            types.append((e_val, r_val))
    if len(types) < n_types:
        raise RuntimeError("Impossible de tirer suffisamment de types d'actions Eff<Rew.")
    return types


def sample_actions_per_interval(
    rng: random.Random,
    t_fixed: Sequence[float],
    guard_after_t: float,
    delta_min: float = 0.1,
    delta_max: float = 5.0,
) -> List[Tuple[int, float, float, float, float]]:
    """
    Retourne une liste d'actions (type_idx, start, end, t_eff, t_rew) :
      - 1 action potentielle par intervalle t_step, avec proba 1/4 pour chaque type, 1/4 pour 'aucune'.
      - t_rew ~ U(start+guard, end-guard); t_eff sera recalculé plus tard (delta_k optimisé).
    """
    actions: List[Tuple[int, float, float, float, float]] = []
    n_intervals = max(0, len(t_fixed) - 1)
    for m in range(n_intervals):
        start = t_fixed[m] + guard_after_t
        end = t_fixed[m + 1] - guard_after_t
        if end <= start + delta_min:
            continue
        choice = rng.randrange(4)  # 0 = aucune, 1..3 = type
        if choice == 0:
            continue
        type_idx = choice - 1
        t_rew = rng.uniform(start + delta_min, end)
        # t_eff provisoire mis au bord gauche (pas 0) pour éviter un placeholder à 0
        actions.append((type_idx, start, end, start, t_rew))
    actions.sort(key=lambda x: x[4])  # tri par t_rew
    return actions


def sample_measurement_times(
    rng: random.Random,
    actions: Sequence[Tuple[int, float, float, float, float]],
    bonuses_t: Sequence[float],
    n_t: int,
    t_max: float,
    t_min: float = 0.0,
    return_sources: bool = False,
) -> List[float] | Tuple[List[float], List[str]]:
    """
    Tire au plus N_t//2 mesures après des Eff (t = t_e + 0.5 si delta > 0.5),
    et au plus N_t//2 mesures après des Rew (t = t_r + 0.5, bonus inclus).
    Une même action ne peut contribuer qu’à un seul t (soit après Eff, soit après Rew).
    Le reste est complété aléatoirement.
    Si return_sources=True, renvoie aussi une liste de sources ("between_eff_rew", "after_rew", "uniform").
    """
    pairs: List[Tuple[float, str]] = []
    half = max(1, n_t // 2)

    # Eff sélectionnés aléatoirement parmi ceux qui ont delta > 0.5
    eff_candidates = [(idx, t_eff, t_rew) for idx, (_, _, _, t_eff, t_rew) in enumerate(actions) if (t_rew - t_eff) > 0.5]
    rng.shuffle(eff_candidates)
    used_actions = set()
    for idx, t_eff, t_rew in eff_candidates:
        t_meas = max(t_min, min(t_max, t_eff + 0.5))
        if abs(t_meas - 0.5) < 1e-9:
            print(f"[dbg] meas from Eff idx={idx} te={t_eff:.3f} tr={t_rew:.3f} -> t_meas=0.5")
        pairs.append((t_meas, "between_eff_rew"))
        used_actions.add(idx)
        if len(pairs) >= half:
            break

    # Rew (incluant bonus)
    rew_candidates: List[Tuple[int | None, float]] = []
    for idx, (_, _, _, _, t_rew) in enumerate(actions):
        if idx in used_actions:
            continue
        rew_candidates.append((idx, t_rew))
    for tb in bonuses_t:
        rew_candidates.append((None, tb))

    rng.shuffle(rew_candidates)
    for idx_opt, t_rew in rew_candidates:
        t_meas = max(t_min, min(t_max, t_rew + 0.5))
        if abs(t_meas - 0.5) < 1e-9:
            src = "bonus" if idx_opt is None else f"Rew idx={idx_opt}"
            print(f"[dbg] meas from {src} tr={t_rew:.3f} -> t_meas=0.5")
        pairs.append((t_meas, "after_rew"))
        if len(pairs) >= 2 * half:  # half eff + half rew max
            break

    # Compléter si besoin
    while len(pairs) < n_t:
        r = rng.uniform(t_min, t_max)
        if abs(r - 0.5) < 1e-9:
            print(f"[dbg] meas from fallback uniform -> t_meas=0.5")
        pairs.append((r, "uniform"))
    pairs.sort(key=lambda x: x[0])
    times = [p[0] for p in pairs]
    if not return_sources:
        return times
    sources = [p[1] for p in pairs]
    return times, sources


def sample_bonuses(
    rng: random.Random,
    n_bonus: int,
    t_fixed: Sequence[float],
    guard_after_t: float,
    before_margin: float = 8.0,
) -> List[float]:
    bonuses: List[float] = []
    n_intervals = max(0, len(t_fixed) - 1)
    if n_intervals == 0 or n_bonus <= 0:
        return bonuses
    for _ in range(n_bonus):
        m = rng.randrange(0, n_intervals)
        start = t_fixed[m] + guard_after_t
        end = t_fixed[m + 1] - before_margin
        if end <= start:
            continue
        bonuses.append(rng.uniform(start, end))
    bonuses.sort()
    return bonuses


def build_design_v5(
    action_types: Sequence[Tuple[float, float]],
    actions_idx_times: Sequence[Tuple[int, float, float, float, float]],
    bonuses_t: Sequence[float],
    rew_bonus_vals: Sequence[float],
    t_fixed: Sequence[float],
    meas_times: Sequence[float],
    meas_sources: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    Eff_: List[Tuple[float, float]] = []
    Rew_: List[Tuple[float, float]] = []
    Bonus_: List[Tuple[float, float]] = []
    A: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    A_typed: List[Tuple[int, Tuple[float, float], Tuple[float, float]]] = []

    for type_idx, _start, _end, t_eff, t_rew in actions_idx_times:
        e_val, r_val = action_types[type_idx % len(action_types)]
        Eff_.append((e_val, t_eff))
        Rew_.append((r_val, t_rew))
        A.append(((e_val, t_eff), (r_val, t_rew)))
        A_typed.append((type_idx, (e_val, t_eff), (r_val, t_rew)))

    for idx, tb in enumerate(bonuses_t):
        valb = rew_bonus_vals[idx % len(rew_bonus_vals)]
        Bonus_.append((valb, tb))
        Rew_.append((valb, tb))

    # t_meas : temps de mesure (t_fixed + mesures explicites), triés/uniques
    t_meas = sorted(set(t_fixed) | set(meas_times))
    # t_all : tous les temps utilisés pour l'évaluation complète
    t_all = sorted(set(t_meas) | {t for _, t in Eff_} | {t for _, t in Rew_})
    Rew_sorted = sorted(Rew_, key=lambda p: p[1])

    design = {
        "t": t_all,
        "t_meas": t_meas,
        "Eff_": Eff_,
        "Rew_": Rew_sorted,
        "Bonus_": Bonus_,
        "A": A,
        "A_typed": A_typed,
        "K_types": max(1, len(action_types)),
    }
    if meas_sources is not None:
        design["meas_times"] = list(meas_times)
        design["meas_sources"] = list(meas_sources)
    return design


def maximin_chernoff(C: Sequence[Sequence[float]]) -> float:
    m = len(C)
    row_min: List[float] = []
    for i in range(m):
        vals = [C[i][j] for j in range(m) if j != i and math.isfinite(C[i][j])]
        row_min.append(min(vals) if vals else float("-inf"))
    return min(row_min) if row_min else float("-inf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v5 : 3 types d'actions fixes, action par intervalle (1/4 chacun), delta Eff->Rew optimisés."
    )
    parser.add_argument("--N_t", type=int, default=24, help="nombre de temps de mesure (pair recommandé)")
    parser.add_argument("--N_bonus", type=int, default=20, help="nombre de bonus tirés (avant t_fixed-8s)")
    parser.add_argument("--sigma", type=float, default=0.1, help="bruit observation pour Laplace-JSD")
    parser.add_argument("--t-step", type=float, default=30.0, help="pas entre temps fixes")
    parser.add_argument("--t-min", type=float, default=0.0, help="t_min")
    parser.add_argument("--t-max", type=float, default=7200.0, help="t_max")
    parser.add_argument("--seed", type=int, default=1203, help="graine RNG")
    parser.add_argument("--bo-calls", type=int, default=100, help="n_eval gp_minimize")
    parser.add_argument("--bo-init", type=int, default=15, help="n_init gp_minimize")
    parser.add_argument("--jobs", type=int, default=10, help="workers parallèles pour Laplace-JSD (<=0 => auto)")
    parser.add_argument(
        "--drop-action-avg",
        action="store_true",
        help="si activé, ignore les modèles kernel=action_avg (ne garde que les 3 event_weighted)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    models = build_six_models()
    if args.drop_action_avg:
        models = models[:3]  # ordre connu : 0..2 event_weighted, 3..5 action_avg
    Eff_pool, Rew_pool = make_demo_pools()

    t_min, t_max = float(args.t_min), float(args.t_max)
    t_fixed = _build_fixed_times(t_min, t_max, float(args.t_step))
    guard_after_t = 0.5

    action_types = choose_action_types(rng, Eff_pool, Rew_pool, n_types=3)
    actions_idx_times = sample_actions_per_interval(
        rng,
        t_fixed,
        guard_after_t=guard_after_t,
        delta_min=0.1,
        delta_max=5.0,
    )
    bonuses_t = sample_bonuses(rng, args.N_bonus, t_fixed, guard_after_t, before_margin=8.0)
    # Les temps de mesure seront tirés après l'optimisation (pas de placeholder ici)
    meas_times: List[float] = []

    try:
        from skopt import gp_minimize
        from skopt.space import Integer, Real
        from tqdm import tqdm  # type: ignore
    except Exception as exc:
        raise RuntimeError("skopt et tqdm sont requis pour la BO v5") from exc

    n_types = len(action_types)
    space = []
    for k in range(n_types):
        space.append(Integer(0, len(Eff_pool) - 1, name=f"ie_{k}"))
    for k in range(n_types):
        space.append(Integer(0, len(Rew_pool) - 1, name=f"ir_{k}"))
    for k in range(n_types):
        space.append(Real(0.1, 5.0, name=f"delta_{k}"))

    def objective(x: List[float]) -> float:
        eff_idx = [int(round(v)) for v in x[:n_types]]
        rew_idx = [int(round(v)) for v in x[n_types : 2 * n_types]]
        delta_vals = [float(v) for v in x[2 * n_types : 3 * n_types]]
        eff_vals = [Eff_pool[i] for i in eff_idx]
        rew_vals = [Rew_pool[i] for i in rew_idx]
        if any(e >= r for e, r in zip(eff_vals, rew_vals)):
            return 1e6
        actions_timed: List[Tuple[int, float, float, float, float]] = []
        for type_idx, start, end, _te, t_rew in actions_idx_times:
            delta = delta_vals[type_idx % n_types]
            t_eff = max(start, t_rew - delta)
            if t_eff >= t_rew or t_eff < start:
                return 1e6
            actions_timed.append((type_idx, start, end, t_eff, t_rew))
        # Pour l'objectif, on ne fige pas encore les temps de mesure : on utilise seulement t_fixed
        design = build_design_v5(
            action_types=[(e, r) for e, r in zip(eff_vals, rew_vals)],
            actions_idx_times=actions_timed,
            bonuses_t=bonuses_t,
            rew_bonus_vals=Rew_pool,
            t_fixed=t_fixed,
            meas_times=meas_times,
        )
        try:
            lap = laplace_jsd_for_design(models, design, sigma=args.sigma, n_jobs=jobs)
            mu = lap.get("mu_y", [])
            Vy = lap.get("Vy", [])
            if not mu or not Vy:
                return 1e6
            C_mat, _ = _pairwise_chernoff_matrix(mu, Vy)
            score = maximin_chernoff(C_mat)
            return -score  # gp_minimize minimise
        except Exception:
            return 1e6

    pbar = tqdm(total=args.bo_calls, desc="BO v5", leave=True)

    def _cb(_res) -> None:
        if pbar is not None:
            pbar.update(1)

    res = gp_minimize(
        objective,
        space,
        n_calls=max(1, args.bo_calls),
        n_initial_points=max(1, args.bo_init),
        callback=[_cb],
        random_state=rng.randint(0, 2**31 - 1),
    )
    if pbar is not None:
        pbar.close()

    best_x = res.x
    eff_idx = [int(round(v)) for v in best_x[:n_types]]
    rew_idx = [int(round(v)) for v in best_x[n_types : 2 * n_types]]
    delta_vals = [float(v) for v in best_x[2 * n_types : 3 * n_types]]
    eff_vals = [Eff_pool[i] for i in eff_idx]
    rew_vals = [Rew_pool[i] for i in rew_idx]
    actions_timed: List[Tuple[int, float, float, float, float]] = []
    for type_idx, start, end, _te, t_rew in actions_idx_times:
        delta = delta_vals[type_idx % n_types]
        t_eff = max(start, t_rew - delta)
        actions_timed.append((type_idx, start, end, t_eff, t_rew))
    # Tirage des temps de mesure uniquement après optimisation (aléatoire final)
    meas_times_best, meas_sources = sample_measurement_times(
        rng,
        actions_timed,
        bonuses_t,
        n_t=max(2, args.N_t),
        t_max=t_max,
        return_sources=True,
    )
    best_design = build_design_v5(
        action_types=[(e, r) for e, r in zip(eff_vals, rew_vals)],
        actions_idx_times=actions_timed,
        bonuses_t=bonuses_t,
        rew_bonus_vals=Rew_pool,
        t_fixed=t_fixed,
        meas_times=meas_times_best,
        meas_sources=meas_sources,
    )
    # auto-detect jobs
    jobs = args.jobs if args.jobs is not None else 1
    if jobs <= 0:
        jobs = max(1, len(models))

    lap_best = laplace_jsd_for_design(models, best_design, sigma=args.sigma, n_jobs=jobs)
    mu_best = lap_best.get("mu_y", [])
    Vy_best = lap_best.get("Vy", [])
    C_best, P_best = _pairwise_chernoff_matrix(mu_best, Vy_best) if mu_best and Vy_best else ([], [])
    score_best = maximin_chernoff(C_best)

    print(f"[result] Maximin Chernoff score (best): {score_best}")
    print("[result] t:", best_design["t"])
    print("[result] Eff_:", best_design["Eff_"])
    print("[result] Rew_:", best_design["Rew_"])
    print("[result] Bonus_:", best_design.get("Bonus_", []))
    print("[result] Chernoff matrix (C):")
    for row in C_best:
        print([round(x, 3) for x in row])
    print("[result] Chernoff error-bound matrix (P_err <= 0.5*exp(-C)):")
    for row in P_best:
        print([round(x, 5) for x in row])

    try:
        U_plot = lap_best.get("U_pair", [])
        plot_summary(
            U_plot if U_plot else C_best,
            best_design,
            models,
            title="v5 BO Laplace-JSD (Chernoff)",
            b_matrix=P_best,
        )
    except Exception as exc:
        print(f"[diag] unable to plot summary: {exc}")


if __name__ == "__main__":
    main()
