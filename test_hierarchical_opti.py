from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, List

from design_optimizer import laplace_jsd_separability
from test_multi_model import build_six_models
from test_multi_model_bo import make_demo_pools, decode_design_actions_v4
from test_multi_model import _build_fixed_times
from compute_model_separability import compute_model_separability
from plot_model_separability import plot_model_separability


def generate_random_designs_actions(
    n: int,
    *,
    rng: random.Random,
    N_t: int,
    Eff_pool,
    Rew_pool,
    t_min: float,
    t_max: float,
    t_step: float,
    min_gap_t: float,
    delta_er: float,
    delta_max: float,
    guard_after_t: float,
) -> List[Dict[str, Any]]:
    designs: List[Dict[str, Any]] = []
    t_fixed = _build_fixed_times(t_min, t_max, t_step)
    n_intervals = max(0, len(t_fixed) - 1)
    max_delta = min(delta_max, max(delta_er, t_step))
    for _ in range(max(1, n)):
        x: List[float] = []
        # t optimisés
        for _ in range(N_t):
            x.append(rng.uniform(t_min, t_max))
        # eff_idx / rew_idx (3 types)
        for _ in range(3):
            x.append(rng.randrange(len(Eff_pool)))
        for _ in range(3):
            x.append(rng.randrange(len(Rew_pool)))
        # delta (3 types)
        for _ in range(3):
            x.append(rng.uniform(delta_er, max_delta))
        # action_choice par intervalle (0..3)
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
    parser = argparse.ArgumentParser(description="Classement offline de designs (tirages aléatoires) via Laplace-JSD.")
    parser.add_argument("--designs", type=int, default=20, help="Nombre de designs aléatoires à générer.")
    parser.add_argument("--N-t", type=int, default=5, help="Nombre de temps de mesure optimisables.")
    parser.add_argument("--t-step", type=float, default=30.0, help="Pas entre temps fixes (définit les intervalles).")
    parser.add_argument("--sigma", type=float, default=0.1, help="Écart-type du bruit d'observation.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Nombre de workers parallèles (Laplace-JSD).",
    )
    parser.add_argument(
        "--objective",
        choices=["maximin", "avgmin", "mean"],
        default="maximin",
        help="Agrégation de la matrice pairwise JSD.",
    )
    parser.add_argument("--top", type=int, default=5, help="Nombre de meilleurs designs à afficher.")
    parser.add_argument("--seed", type=int, default=123, help="Graine RNG pour la génération des designs.")
    parser.add_argument("--t-min", type=float, default=0.0, help="Temps minimum.")
    parser.add_argument("--t-max", type=float, default=600.0, help="Temps maximum.")
    parser.add_argument("--min-gap-t", type=float, default=5.0, help="Écart minimal entre temps de mesure.")
    parser.add_argument("--delta-er", type=float, default=1.0, help="Séparation minimale Eff/Rew.")
    parser.add_argument("--delta-max", type=float, default=10.0, help="Séparation maximale Eff/Rew.")
    parser.add_argument("--guard-after-t", type=float, default=2.0, help="Zone muette après chaque t_fixed pour Eff/Rew.")
    parser.add_argument("--plot", action="store_true", help="Affiche heatmap + dendrogramme (6 feuilles).")
    parser.add_argument("--progress", action="store_true", help="Barre de progression sur les paires/designs.")
    args = parser.parse_args()

    print("[info] Chargement des modèles.")
    models = build_six_models()
    Eff_pool, Rew_pool = make_demo_pools()
    rng = random.Random(args.seed)

    print(f"[info] Génération de {args.designs} designs aléatoires (seed={args.seed}) avec actions_v4.")
    designs = generate_random_designs_actions(
        args.designs,
        rng=rng,
        N_t=args.N_t,
        Eff_pool=Eff_pool,
        Rew_pool=Rew_pool,
        t_min=args.t_min,
        t_max=args.t_max,
        t_step=args.t_step,
        min_gap_t=args.min_gap_t,
        delta_er=args.delta_er,
        delta_max=args.delta_max,
        guard_after_t=args.guard_after_t,
    )

    res = laplace_jsd_separability(
        models,
        designs,
        optimizer="laplace-jsd",
        sigma=args.sigma,
        n_jobs=args.n_jobs,
        objective=args.objective,
    )

    per_design = res.get("per_design", [])
    per_design_sorted = sorted(per_design, key=lambda d: d.get("score", float("-inf")), reverse=True)
    top_k = min(max(1, args.top), len(per_design_sorted))

    print(f"[result] Objective={args.objective}, optimizer=laplace-jsd, top {top_k}/{len(per_design_sorted)} designs:")
    for rank, entry in enumerate(per_design_sorted[:top_k], start=1):
        idx = entry.get("index")
        score = entry.get("score")
        d = entry.get("design", {})
        t = d.get("t", None)
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "nan"
        print(f"  #{rank} design[{idx}] score={score_str}")
        if t is not None:
            print(f"    t={t}")
        if "Eff_" in d and "Rew_" in d:
            print(f"    Eff_={d.get('Eff_')}")
            print(f"    Rew_={d.get('Rew_')}")

    best_idx = res.get("best_index")
    best_score = res.get("best_score")
    if best_idx is not None:
        print(f"[best] index={best_idx}, score={best_score}, objective={args.objective}")
    if designs and "K_types" in designs[0]:
        print(f"[diag] first design K_types: {designs[0].get('K_types')}")

    # Pairwise separability + dendrogramme (toujours 6 feuilles)
    result_separability = compute_model_separability(
        models,
        designs,
        sigma=args.sigma,
        n_jobs=args.n_jobs,
        progress=args.progress,
    )
    print("[info] Pairwise separability computed (compute_model_separability).")
    if args.plot:
        try:
            plot_model_separability(result_separability, show=True)
        except Exception as exc:
            print(f"[diag] unable to plot separability: {exc}")


if __name__ == "__main__":
    main()
