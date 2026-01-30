from __future__ import annotations

import argparse
import random
from typing import Any, Dict, List

from compute_model_separability import compute_model_separability
from plot_model_separability import plot_model_separability
from test_multi_model import build_six_models
from test_multi_model_bo import make_demo_pools, decode_design_actions_v4
from test_multi_model import _build_fixed_times


def generate_demo_designs_v4(
    n: int,
    *,
    rng: random.Random,
    N_t: int,
    t_min: float,
    t_max: float,
    t_step: float,
    min_gap_t: float,
    delta_er: float,
    delta_max: float,
    guard_after_t: float,
) -> List[Dict[str, Any]]:
    Eff_pool, Rew_pool = make_demo_pools()
    designs: List[Dict[str, Any]] = []
    t_fixed = _build_fixed_times(t_min, t_max, t_step)
    n_intervals = max(0, len(t_fixed) - 1)
    max_delta = min(delta_max, max(delta_er, t_step))
    for _ in range(max(1, n)):
        x: List[float] = []
        for _ in range(N_t):
            x.append(rng.uniform(t_min, t_max))
        for _ in range(3):
            x.append(rng.randrange(len(Eff_pool)))
        for _ in range(3):
            x.append(rng.randrange(len(Rew_pool)))
        for _ in range(3):
            x.append(rng.uniform(delta_er, max_delta))
        for _ in range(n_intervals):
            x.append(rng.randrange(0, 4))
        rng_design = random.Random(rng.randint(0, 2**31 - 1))
        d = decode_design_actions_v4(
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
            rng=rng_design,
        )
        designs.append(d)
    return designs


def main() -> None:
    parser = argparse.ArgumentParser(description="Démo de séparabilité de modèles via Laplace-JSD.")
    parser.add_argument("--designs", type=int, default=8, help="Nombre de designs aléatoires.")
    parser.add_argument("--N-t", type=int, default=5, help="Nombre de temps optimisables.")
    parser.add_argument("--t-step", type=float, default=30.0, help="Pas entre temps fixes (définit les intervalles).")
    parser.add_argument("--sigma", type=float, default=0.1, help="Bruit d'observation pour Laplace-JSD.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallelisme Laplace-JSD.")
    parser.add_argument("--seed", type=int, default=42, help="Graine RNG.")
    parser.add_argument("--show-plot", action="store_true", help="Afficher heatmap et dendrogramme.")
    parser.add_argument("--progress", action="store_true", help="Afficher tqdm sur les évaluations pairwise.")
    parser.add_argument("--delta-er", type=float, default=1.0, help="Séparation minimale Eff/Rew.")
    parser.add_argument("--delta-max", type=float, default=10.0, help="Séparation maximale Eff/Rew.")
    parser.add_argument("--guard-after-t", type=float, default=2.0, help="Zone muette après t_fixed pour Eff/Rew.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    models = build_six_models()
    K = len(models)
    labels = [f"M{i}" for i in range(K)]

    designs = generate_demo_designs_v4(
        args.designs,
        rng=rng,
        N_t=args.N_t,
        t_min=0.0,
        t_max=600.0,
        t_step=args.t_step,
        min_gap_t=5.0,
        delta_er=args.delta_er,
        delta_max=args.delta_max,
        guard_after_t=args.guard_after_t,
    )

    print(f"[info] Calcul de la séparabilité sur {len(designs)} designs, {K} modèles.")
    if designs and "K_types" in designs[0]:
        print(f"[diag] first design K_types: {designs[0].get('K_types')}")
    result = compute_model_separability(
        models,
        designs,
        model_labels=labels,
        sigma=args.sigma,
        n_jobs=args.n_jobs,
        progress=args.progress,
    )

    E_max = result["E_max"]
    best_idx = result["best_design_idx"]
    print("[result] E_max (séparabilité max par paire) :")
    for i in range(K):
        row = ["{:.3f}".format(E_max[i, j]) for j in range(K)]
        print("  ", row)
    print("[result] Indices du meilleur design par paire :")
    for i in range(K):
        row = [int(best_idx[i, j]) for j in range(K)]
        print("  ", row)

    if args.show_plot:
        plot_model_separability(result, show=True)


if __name__ == "__main__":
    main()
