from __future__ import annotations

import argparse
import math
import random
from typing import Any, Dict, List, Sequence, Tuple

from design_optimizer.laplace_jsd import laplace_jsd_for_design, _pairwise_chernoff_matrix
from test_multi_model_bo import _build_fixed_times
from test_multi_model import build_six_models, plot_summary
from test_multi_model_bo_v5 import (
    build_design_v5,
    sample_measurement_times,
    maximin_chernoff,
)

# Valeurs fixes d'actions (identiques à agent_comparison)
ACTION_TYPES: List[Tuple[float, float]] = [
    (0.5, 1.0),    # type 0
    (0.25, 0.5),   # type 1
    (0.125, 0.25), # type 2
]
DELTAS: List[float] = [12.0, 6.0, 3.0]


def build_forced_actions(
    n_per_type: int,
    gaps: Sequence[float],
    t_max_forced: float,
    jitter: float = 1.0,
    seed: int = 0,
) -> List[Tuple[int, float, float, float, float]]:
    """
    Construit des actions forcées en 3 blocs séquentiels (type 0, type 1, type 2),
    avec n_per_type actions par bloc, espacées de gaps[k] entre rewards du type k.
    Jitter uniform ±jitter appliqué sur delta_k pour chaque action (déterministe via seed).
    Retourne actions_idx_times : (type_idx, start, end, t_eff, t_rew), triées par t_rew.
    """
    rng = random.Random(seed)
    # On génère des temps dans l'ordre t_e1 < t_r1 < t_e2 < t_r2 < ... (pas de superposition)
    actions: List[Tuple[int, float, float, float, float]] = []
    t_cur = 0.0
    for type_idx in range(3):
        gap = max(1.0, float(gaps[type_idx]))
        for i in range(n_per_type):
            t_eff = t_cur
            delta = DELTAS[type_idx] + rng.uniform(-jitter, jitter)
            delta = max(0.1, delta)
            t_rew = t_eff + delta
            if t_rew >= t_max_forced:
                break
            actions.append((type_idx, t_eff, t_rew, t_eff, t_rew))
            t_cur = t_rew + gap  # entrelacement obligatoire t_e, t_r, t_e, t_r ...
    # clamp
    actions = [a for a in actions if a[2] < t_max_forced and a[3] < a[4]]
    actions.sort(key=lambda x: x[4])
    return actions


def objective_forced(
    x: List[float],
    *,
    t_max_forced: float,
    models,
    sigma: float,
    n_jobs: int,
    jitter: float,
    seed: int,
) -> float:
    n_per_type = int(round(x[0]))
    gaps = [max(1.0, float(x[i])) for i in range(1, 4)]
    if n_per_type <= 0:
        return 1e6
    actions_idx_times = build_forced_actions(
        n_per_type=n_per_type,
        gaps=gaps,
        t_max_forced=t_max_forced,
        jitter=jitter,
        seed=seed,
    )
    t_fixed = _build_fixed_times(0.0, t_max_forced, t_max_forced)
    design = build_design_v5(
        action_types=ACTION_TYPES,
        actions_idx_times=actions_idx_times,
        bonuses_t=[],
        rew_bonus_vals=[r for _, r in ACTION_TYPES],
        t_fixed=t_fixed,
        meas_times=[],
    )
    try:
        lap = laplace_jsd_for_design(models, design, sigma=sigma, n_jobs=n_jobs)
        mu = lap.get("mu_y", [])
        Vy = lap.get("Vy", [])
        if not mu or not Vy:
            return 1e6
        C_mat, _ = _pairwise_chernoff_matrix(mu, Vy)
        score = maximin_chernoff(C_mat)
        return -score  # minimisation
    except Exception:
        return 1e6


def optimize_forced_block(
    *,
    t_max_forced: float = 1200.0,
    seed: int = 1203,
    sigma: float = 0.1,
    bo_calls: int = 40,
    bo_init: int = 10,
    n_jobs: int = 4,
    jitter: float = 1.0,
):
    rng = random.Random(seed)
    models_all = build_six_models()
    models = models_all[3:]  # focus sur modèles 3,4,5 (action_avg)
    try:
        from skopt import gp_minimize
        from skopt.space import Integer, Real
        from tqdm import tqdm  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("skopt et tqdm requis pour la BO du bloc forcé") from exc

    space = [
        Integer(1, 10, name="n_per_type"),
        Real(5.0, 200.0, name="gap_type0"),
        Real(5.0, 200.0, name="gap_type1"),
        Real(5.0, 200.0, name="gap_type2"),
    ]

    pbar = tqdm(total=bo_calls, desc="BO forced block", leave=True)

    def _cb(_res) -> None:
        pbar.update(1)

    res = gp_minimize(
        lambda x: objective_forced(
            x,
            models=models,
            sigma=sigma,
            n_jobs=max(1, n_jobs),
            t_max_forced=t_max_forced,
            jitter=jitter,
            seed=seed,
        ),
        space,
        n_calls=max(1, bo_calls),
        n_initial_points=max(1, bo_init),
        random_state=rng.randint(0, 2**31 - 1),
        callback=[_cb],
    )
    pbar.close()

    n_per_type = int(round(res.x[0]))
    gaps = [float(res.x[1]), float(res.x[2]), float(res.x[3])]
    actions_idx_times = build_forced_actions(
        n_per_type=n_per_type,
        gaps=gaps,
        t_max_forced=t_max_forced,
        jitter=jitter,
        seed=seed,
    )
    t_fixed = _build_fixed_times(0.0, t_max_forced, t_max_forced)
    design = build_design_v5(
        action_types=ACTION_TYPES,
        actions_idx_times=actions_idx_times,
        bonuses_t=[],
        rew_bonus_vals=[r for _, r in ACTION_TYPES],
        t_fixed=t_fixed,
        meas_times=[],  # mesures tirées après
    )
    # Temps de mesure tirés après optimisation (pas de bonus)
    meas_times = sample_measurement_times(
        rng, actions_idx_times, [], n_t=24, t_max=t_max_forced
    )
    design["t_meas"] = sorted(set(design.get("t_meas", [])) | set(meas_times))
    design["t"] = sorted(set(design.get("t", [])) | set(meas_times))

    lap_best = laplace_jsd_for_design(models, design, sigma=sigma, n_jobs=max(1, n_jobs))
    mu_best = lap_best.get("mu_y", [])
    Vy_best = lap_best.get("Vy", [])
    C_best, P_best = _pairwise_chernoff_matrix(mu_best, Vy_best) if mu_best and Vy_best else ([], [])
    score_best = maximin_chernoff(C_best) if C_best else float("nan")

    return {
        "score": score_best,
        "C": C_best,
        "P_err_bound": P_best,
        "design": design,
        "laplace": lap_best,
        "models": models,
        "action_types": ACTION_TYPES,
        "gaps": gaps,
        "n_per_type": n_per_type,
        "forced_actions": actions_idx_times,
        "laplace": lap_best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimisation BO d'un bloc forcé (t_max forcé)")
    parser.add_argument("--t-max-forced", type=float, default=1200.0, help="durée du bloc forcé")
    parser.add_argument("--seed", type=int, default=1203, help="seed RNG")
    parser.add_argument("--sigma", type=float, default=0.1, help="bruit observation")
    parser.add_argument("--bo-calls", type=int, default=40, help="n_calls gp_minimize")
    parser.add_argument("--bo-init", type=int, default=10, help="n_init gp_minimize")
    parser.add_argument("--jobs", type=int, default=4, help="n_jobs pour Laplace-JSD")
    parser.add_argument("--jitter", type=float, default=1.0, help="jitter +/- sur delta_k")
    args = parser.parse_args()

    res = optimize_forced_block(
        t_max_forced=float(args.t_max_forced),
        seed=int(args.seed),
        sigma=float(args.sigma),
        bo_calls=int(args.bo_calls),
        bo_init=int(args.bo_init),
        n_jobs=int(args.jobs),
        jitter=float(args.jitter),
    )
    print(f"[result] maximin Chernoff score: {res['score']}")
    print("[result] C matrix (rounded):")
    for row in res.get("C", []):
        print([round(x, 3) for x in row])
    print("[result] action_types:", res.get("action_types"))
    print("[result] gaps:", res.get("gaps"))
    print("[result] n_per_type:", res.get("n_per_type"))
    print("[result] forced actions:", res.get("forced_actions"))
    try:
        plot_summary(
            res.get("C", []),
            res.get("design", {}),
            build_six_models()[3:],  # modèles 3,4,5
            title=f"forced block t_max={args.t_max_forced}",
            b_matrix=res.get("P_err_bound"),
        )
    except Exception as exc:
        print(f"[warn] unable to plot summary: {exc}")


if __name__ == "__main__":
    main()
