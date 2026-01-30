from __future__ import annotations

import argparse
import itertools
import math
import random
from typing import Any, Dict, List, Sequence, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from design_optimizer.laplace_jsd import laplace_jsd_for_design, _pairwise_chernoff_matrix
from test_multi_model_bo import build_six_models
from test_hierarchical_opti import generate_random_designs_actions

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:  # pragma: no cover
    HAVE_MPL = False

try:
    from tqdm import tqdm  # type: ignore
    HAVE_TQDM = True
except Exception:  # pragma: no cover
    HAVE_TQDM = False

def _score_design(models: Sequence[Any], design: Dict[str, Any], sigma: float) -> List[List[float]]:
    """Retourne la matrice de Chernoff pour un design (ou [] en cas d'échec)."""
    lap = laplace_jsd_for_design(models, design, sigma=sigma, n_jobs=1)
    mu = lap.get("mu_y", [])
    Vy = lap.get("Vy", [])
    if not mu or not Vy:
        return []
    C_mat, _P = _pairwise_chernoff_matrix(mu, Vy)
    return C_mat


def _chernoff_at_t(models: Sequence[Any], design: Dict[str, Any], sigma: float, t_meas: List[float]) -> List[List[float]]:
    """Calcule la matrice de Chernoff sur les temps de mesure t_meas en remplaçant les t_m dans le design."""
    design_override = dict(design)
    design_override["t_m"] = list(t_meas)
    lap = laplace_jsd_for_design(models, design_override, sigma=sigma, n_jobs=1)
    mu = lap.get("mu_y", [])
    Vy = lap.get("Vy", [])
    if not mu or not Vy:
        return []
    C_mat, _P = _pairwise_chernoff_matrix(mu, Vy)
    return C_mat


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimisation des temps de mesure (N_t=2) sur des designs aléatoires.")
    parser.add_argument("--n-designs", type=int, default=20, help="Nombre de designs aléatoires (communs à toutes les paires).")
    parser.add_argument("--sigma", type=float, default=0.1, help="Bruit d'observation.")
    parser.add_argument("--t-max", type=float, default=50.0, help="Durée des designs (secondes).")
    parser.add_argument("--t-step", type=float, default=30.0, help="Pas entre temps fixes (s'il y en avait).")
    parser.add_argument("--min-gap-t", type=float, default=0.0, help="Ecart minimal entre les deux mesures.")
    parser.add_argument("--seed", type=int, default=12223, help="Graine RNG.")
    parser.add_argument("--n-jobs", type=int, default=6, help="Parallélisme pour l'évaluation des paires.")
    parser.add_argument("--grid-step-coarse", type=float, default=3.0, help="Pas de grille (secondes) pour la recherche grossière.")
    parser.add_argument("--grid-step-fine", type=float, default=1.0, help="Pas de grille (secondes) pour le raffinement local.")
    parser.add_argument("--fine-radius", type=float, default=3.0, help="Rayon (secondes) autour du meilleur grossier pour la recherche fine.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    models = build_six_models()
    K = len(models)

    # Génère des designs aléatoires de 60s, une seule action, pas de mesures fixes (N_t=0)
    designs: List[Dict[str, Any]] = generate_random_designs_actions(
        args.n_designs,
        rng=rng,
        N_t=0,
        Eff_pool=[-1.0],  # amplitudes simples
        Rew_pool=[1.0],
        t_min=0.0,
        t_max=args.t_max,
        t_step=args.t_step,
        min_gap_t=5.0,
        delta_er=5.0,
        delta_max=10.0,
        guard_after_t=2.0,
    )

    # Prépare une grille de temps candidates (coarse + fine)
    t_grid_coarse = np.arange(0.0, args.t_max + 1e-9, args.grid_step_coarse)

    def best_two_times_for_pair(i: int, j: int) -> Tuple[float, float, float]:
        """Retourne (t1, t2, score) maximisant la somme des Chernoff sur les designs, avec gap >= min_gap_t."""
        best = (-math.inf, 0.0, 0.0)

        # Étape grossière
        for t1 in t_grid_coarse:
            t2_candidates = t_grid_coarse[t_grid_coarse - t1 >= args.min_gap_t]
            if t2_candidates.size == 0:
                continue
            for t2 in t2_candidates:
                c_vals = []
                for d in designs:
                    C = _chernoff_at_t(models, d, sigma=args.sigma, t_meas=[float(t1), float(t2)])
                    if C:
                        c_vals.append(C[i][j])
                if not c_vals:
                    continue
                score = float(np.nanmean(c_vals))
                if score > best[0]:
                    best = (score, float(t1), float(t2))

        # Raffinement local autour du meilleur trouvé (si on a un point de départ)
        if math.isfinite(best[0]):
            t1_center, t2_center = best[1], best[2]
            # Grille fine autour des centres
            t1_fine = np.arange(
                max(0.0, t1_center - args.fine_radius),
                min(args.t_max, t1_center + args.fine_radius) + 1e-9,
                args.grid_step_fine,
            )
            for t1 in t1_fine:
                t2_candidates = np.arange(
                    max(t1 + args.min_gap_t, t2_center - args.fine_radius),
                    min(args.t_max, t2_center + args.fine_radius) + 1e-9,
                    args.grid_step_fine,
                )
                if t2_candidates.size == 0:
                    continue
                for t2 in t2_candidates:
                    c_vals = []
                    for d in designs:
                        C = _chernoff_at_t(models, d, sigma=args.sigma, t_meas=[float(t1), float(t2)])
                        if C:
                            c_vals.append(C[i][j])
                    if not c_vals:
                        continue
                    score = float(np.nanmean(c_vals))
                    if score > best[0]:
                        best = (score, float(t1), float(t2))

        return best[1], best[2], best[0]

    best_times: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
    pair_list = list(itertools.combinations(range(K), 2))
    pbar = tqdm(total=len(pair_list), desc="pairs") if HAVE_TQDM else None

    if args.n_jobs and args.n_jobs > 1:
        with ThreadPoolExecutor(max_workers=args.n_jobs) as ex:
            futs = {ex.submit(best_two_times_for_pair, i, j): (i, j) for (i, j) in pair_list}
            for fut in as_completed(futs):
                i, j = futs[fut]
                t1, t2, val = fut.result()
                best_times[(i, j)] = (t1, t2, val)
                print(f"[pair {i},{j}] t1={t1:.2f}s t2={t2:.2f}s  score={val:.4g}")
                if pbar is not None:
                    pbar.update(1)
    else:
        for (i, j) in pair_list:
            t1, t2, val = best_two_times_for_pair(i, j)
            best_times[(i, j)] = (t1, t2, val)
            print(f"[pair {i},{j}] t1={t1:.2f}s t2={t2:.2f}s  score={val:.4g}")
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    # Plot des t1/t2 optimaux pour chaque paire
    if HAVE_MPL and best_times:
        fig, ax = plt.subplots(figsize=(6, 6))
        for (i, j), (t1, t2, val) in best_times.items():
            ax.plot([t1, t2], [i, j], marker="o", linestyle="-", label=f"{i},{j} (score={val:.3g})")
        ax.set_xlabel("Temps de mesure (s)")
        ax.set_ylabel("Paires (i,j)")
        ax.set_title("Temps de mesure optimaux par paire (gap >= min_gap_t)")
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
