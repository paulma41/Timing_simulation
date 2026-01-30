from __future__ import annotations

import argparse
import os
import pickle
import random
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from design_optimizer.laplace_jsd import laplace_jsd_for_design, _pairwise_chernoff_matrix
from test_multi_model_bo import build_six_models
from test_hierarchical_opti import generate_random_designs_actions


def _score_design(models: Sequence[Any], design: Dict[str, Any], sigma: float) -> Tuple[float, List[List[float]]]:
    """Score maximin sur la matrice de Chernoff pour un design donn�. Retourne (score, C_matrix)."""
    lap = laplace_jsd_for_design(models, design, sigma=sigma, n_jobs=1)
    mu = lap.get("mu_y", [])
    Vy = lap.get("Vy", [])
    if not mu or not Vy:
        return float("-inf"), []
    C_mat, _P = _pairwise_chernoff_matrix(mu, Vy)
    m = len(C_mat)
    row_min: List[float] = []
    for i in range(m):
        vals = [C_mat[i][j] for j in range(m) if j != i]
        row_min.append(min(vals) if vals else float("-inf"))
    return (min(row_min) if row_min else float("-inf")), C_mat


def main() -> None:
    parser = argparse.ArgumentParser(description="R�-�valuation des meilleurs candidats du cache avec davantage de designs.")
    parser.add_argument("--cache-file", type=str, default="stimuli_range_cache.pkl", help="Cache pickle existant.")
    parser.add_argument("--top-n", type=int, default=5, help="Nombre de candidats � r�-�valuer.")
    parser.add_argument("--designs-per-range", type=int, default=50, help="Nombre de designs al�atoires pour la r�-�valuation.")
    parser.add_argument("--sigma", type=float, default=0.1, help="Bruit d'observation pour Chernoff.")
    parser.add_argument("--t-max", type=float, default=600.0, help="t_max pour g�n�ration des designs.")
    parser.add_argument("--t-step", type=float, default=30.0, help="Pas entre temps fixes.")
    parser.add_argument("--N-t", type=int, default=5, help="Nombre de temps optimis�s (v4).")
    parser.add_argument("--min-gap-t", type=float, default=5.0, help="Ecart minimal entre mesures.")
    parser.add_argument("--guard-after-t", type=float, default=2.0, help="P�riode sans �v�nement apr�s t_fixed.")
    parser.add_argument("--seed", type=int, default=42, help="Graine RNG.")
    args = parser.parse_args()

    if not os.path.exists(args.cache_file):
        print(f"[err] Cache introuvable: {args.cache_file}")
        return

    with open(args.cache_file, "rb") as f:
        cache = pickle.load(f)
    if not isinstance(cache, dict) or not cache:
        print("[err] Cache vide ou invalide.")
        return

    candidates: List[Dict[str, Any]] = list(cache.values())
    candidates_sorted = sorted(candidates, key=lambda d: d.get("score", float("-inf")), reverse=True)
    top_n = max(1, min(args.top_n, len(candidates_sorted)))
    selected = candidates_sorted[:top_n]

    rng = random.Random(args.seed)
    models = build_six_models()

    print(f"[info] R�-�valuation des {top_n} meilleurs candidats sur {args.designs_per_range} designs chacun.")

    for idx, cand in enumerate(selected, 1):
        Eff_pool = cand["Eff_pool"]
        Rew_pool = cand["Rew_pool"]
        delta_min = cand["delta_min"]
        delta_max = cand["delta_max"]
        orig_score = cand.get("score", float("-inf"))

        scores: List[float] = []
        C_accum: List[List[float]] = []
        designs = generate_random_designs_actions(
            args.designs_per_range,
            rng=random.Random(args.seed + idx),
            N_t=args.N_t,
            Eff_pool=Eff_pool,
            Rew_pool=Rew_pool,
            t_min=0.0,
            t_max=args.t_max,
            t_step=args.t_step,
            min_gap_t=args.min_gap_t,
            delta_er=delta_min,
            delta_max=delta_max,
            guard_after_t=args.guard_after_t,
        )
        for d in designs:
            sc, C_mat = _score_design(models, d, sigma=args.sigma)
            scores.append(sc)
            if C_mat:
                if not C_accum:
                    C_accum = [[c for c in row] for row in C_mat]
                else:
                    for i in range(len(C_accum)):
                        for j in range(len(C_accum[i])):
                            C_accum[i][j] += C_mat[i][j]
        new_score = float(np.nanmean(scores)) if scores else float("-inf")
        C_mean = None
        if C_accum:
            nC = args.designs_per_range
            C_mean = [[c / nC for c in row] for row in C_accum]

        print(f"[cand {idx}] orig={orig_score:.4g}  new={new_score:.4g}")
        if C_mean:
            arr = np.array(C_mean)
            print(f"           Chernoff min/max (off-diag): {arr[~np.eye(arr.shape[0],dtype=bool)].min():.4g} / {arr[~np.eye(arr.shape[0],dtype=bool)].max():.4g}")


if __name__ == "__main__":
    main()
