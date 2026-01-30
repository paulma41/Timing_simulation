from __future__ import annotations

import argparse
import math
import os
import pickle
import random
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:  # pragma: no cover
    HAVE_MPL = False

from concurrent.futures import ThreadPoolExecutor, as_completed

from design_optimizer.laplace_jsd import laplace_jsd_for_design, _pairwise_chernoff_matrix
from pair_reweighting import iterative_reweight_pairs
from test_hierarchical_opti import generate_random_designs_actions
from test_multi_model import _build_fixed_times, build_six_models
from test_multi_model_bo import make_demo_pools, decode_design_actions_v4

try:
    from tqdm import tqdm  # type: ignore

    HAVE_TQDM = True
except Exception:  # pragma: no cover
    HAVE_TQDM = False


def build_profiles(E_all: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Construit les profils par paire (i,j) : E_all[i,j,:], transformés en -log(C),
    où C est l'information de Chernoff pour chaque design candidat.
    """
    if E_all.ndim != 3:
        raise ValueError("E_all doit être de dimension (K, K, D)")
    K, _, D = E_all.shape
    pairs: List[Tuple[int, int]] = []
    profs: List[np.ndarray] = []
    eps = 1e-12
    for i in range(K):
        for j in range(i + 1, K):
            v = np.array(E_all[i, j, :], dtype=float)
            v = np.where(np.isfinite(v) & (v > 0), v, eps)
            profs.append(-np.log(v))
            pairs.append((i, j))
    profiles = np.vstack(profs) if profs else np.zeros((0, D), dtype=float)
    return pairs, profiles


def pick_representatives(
    pairs: List[Tuple[int, int]],
    profiles: np.ndarray,
    labels: np.ndarray,
) -> List[Tuple[int, Tuple[int, int]]]:
    """
    Pour chaque cluster de paires, choisit la paire la plus proche du centroïde
    (distance euclidienne) comme représentante.
    """
    reps: List[Tuple[int, Tuple[int, int]]] = []
    unique_labels = sorted(set(int(l) for l in labels))
    for c in unique_labels:
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        sub = profiles[idx, :]
        centroid = sub.mean(axis=0)
        dists = np.linalg.norm(sub - centroid[None, :], axis=1)
        k = int(idx[np.argmin(dists)])
        reps.append((c, pairs[k]))
    return reps


def plot_dendro(Z: np.ndarray, labels: Sequence[str], show: bool = True) -> None:
    if not HAVE_MPL:
        return
    plt.figure(figsize=(8, 4))
    dendrogram(Z, labels=list(labels), leaf_rotation=90)
    plt.tight_layout()
    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimisation de design par repondération adaptative des paires (Chernoff)."
    )
    parser.add_argument("--designs", type=int, default=100, help="Nombre de designs aléatoires à générer.")
    parser.add_argument("--N-t", type=int, default=0, help="Nombre de temps de mesure optimisables (v4).")
    parser.add_argument("--t-step", type=float, default=30.0, help="Pas entre temps fixes (définit les intervalles).")
    parser.add_argument("--sigma", type=float, default=0.1, help="Bruit d'observation pour Laplace-JSD.")
    parser.add_argument("--n-jobs", type=int, default=10, help="Workers parallèles pour Laplace-JSD.")
    parser.add_argument("--seed", type=int, default=155, help="Graine RNG.")
    parser.add_argument("--min-gap-t", type=float, default=5.0, help="Écart minimal entre temps de mesure.")
    parser.add_argument("--delta-er", type=float, default=1.0, help="Séparation minimale Eff/Rew.")
    parser.add_argument("--delta-max", type=float, default=10.0, help="Séparation maximale Eff/Rew.")
    parser.add_argument("--guard-after-t", type=float, default=2.0, help="Zone muette après t_fixed pour Eff/Rew.")
    parser.add_argument("--n-clusters", type=int, default=4, help="Nombre de clusters de paires.")
    parser.add_argument("--T-max", type=float, default=5400.0, help="Budget total (en secondes) à répartir.")
    parser.add_argument("--show-plot", action="store_true", help="Affiche le dendrogramme des paires.")
    parser.add_argument(
        "--progress", action="store_true", default=True, help="Affiche tqdm pendant le calcul de séparabilité."
    )
    parser.add_argument(
        "--outer-jobs",
        type=int,
        default=3,
        help="Parallélisme sur les designs pour le profil initial; 0 = auto simple.",
    )
    parser.add_argument(
        "--max-real",
        type=int,
        default=0,
        help="Nombre maximal de réalisations adaptatives (iter de repondération).",
    )
    parser.add_argument(
        "--cher-eps",
        type=float,
        default=0.05,
        help="Tolérance d'uniformité sur les bornes Chernoff cumulées (hors diagonale).",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=20,
        help="Nombre de candidats random v4 par paire et par réalisation.",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default=".smart_clustering_cache.pkl",
        help="Fichier pickle pour persister les meilleurs designs (x, score, poids).",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=50,
        help="Nombre maximal de designs à conserver dans le cache.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    models = build_six_models()
    Eff_pool, Rew_pool = make_demo_pools()

    print(f"[info] Génération de {args.designs} designs aléatoires (seed={args.seed}).")
    designs = generate_random_designs_actions(
        args.designs,
        rng=rng,
        N_t=args.N_t,
        Eff_pool=Eff_pool,
        Rew_pool=Rew_pool,
        t_min=0.0,
        t_max=600.0,
        t_step=args.t_step,
        min_gap_t=args.min_gap_t,
        delta_er=args.delta_er,
        delta_max=args.delta_max,
        guard_after_t=args.guard_after_t,
    )

    # Profil initial : Chernoff C(i,j; d) pour chaque design candidat
    print("[info] Calcul de la séparabilité (Chernoff) sur tous les designs candidats.")
    cpu = os.cpu_count() or 4
    inner_jobs = max(1, min(args.n_jobs, max(1, cpu // 2)))
    outer_jobs = args.outer_jobs if args.outer_jobs > 0 else max(1, cpu // 2)
    outer_jobs = max(1, min(outer_jobs, len(designs)))
    K = len(models)
    D = len(designs)
    E_all = np.full((K, K, D), np.nan, dtype=float)

    def _compute_one(d_idx: int) -> Tuple[int, np.ndarray]:
        design = designs[d_idx]
        lap = laplace_jsd_for_design(models, design, sigma=args.sigma, n_jobs=inner_jobs)
        mu = lap.get("mu_y", [])
        Vy = lap.get("Vy", [])
        if not mu or not Vy:
            return d_idx, np.full((K, K), float("nan"))
        C_mat, _P = _pairwise_chernoff_matrix(mu, Vy)
        return d_idx, np.asarray(C_mat, dtype=float)

    if outer_jobs == 1:
        iterator = range(D)
        if args.progress and HAVE_TQDM:
            iterator = tqdm(iterator, desc="designs", total=D)
        for d_idx in iterator:
            idx, C_mat = _compute_one(d_idx)
            E_all[:, :, idx] = C_mat
    else:
        iterator = range(D)
        pbar = tqdm(total=D, desc="designs", leave=True) if args.progress and HAVE_TQDM else None
        with ThreadPoolExecutor(max_workers=outer_jobs) as ex:
            futs = {ex.submit(_compute_one, d_idx): d_idx for d_idx in iterator}
            for fut in as_completed(futs):
                idx, C_mat = fut.result()
                E_all[:, :, idx] = C_mat
                if pbar is not None:
                    pbar.update(1)
        if pbar is not None:
            pbar.close()

    pairs, profiles = build_profiles(E_all)
    n_pairs = len(pairs)
    if n_pairs == 0:
        print("[warn] Pas de profils construits (trop peu de modèles ?).")
        return

    print(f"[info] {n_pairs} paires, profil de dimension {profiles.shape[1]} (designs).")
    Z = linkage(profiles, method="ward")
    labels = fcluster(Z, args.n_clusters, criterion="maxclust")
    reps = pick_representatives(pairs, profiles, labels)

    print("[result] Paires représentatives par cluster:")
    for c, p in reps:
        print(f"  cluster {c}: {p[0]}-{p[1]}")

    pair_to_cluster: Dict[Tuple[int, int], int] = {pair: int(lbl) for pair, lbl in zip(pairs, labels)}
    print("[result] Affectation de cluster par paire:")
    for pair, lbl in pair_to_cluster.items():
        print(f"  {pair[0]}-{pair[1]} -> {lbl}")

    # Corrélation entre profils (paires x designs) pour le repondérage
    Corr = np.corrcoef(profiles) if n_pairs > 1 else np.zeros((n_pairs, n_pairs), dtype=float)

    # Information de Chernoff cumulée C_cum[i,j] sur toutes les réalisations
    C_cum: Dict[Tuple[int, int], float] = {pair: 0.0 for pair in pairs}

    t_step = args.t_step
    delta_er = args.delta_er
    delta_max = args.delta_max
    guard_after_t = args.guard_after_t
    N_t = args.N_t
    T_max = float(args.T_max)
    rng_pair_global = random.Random(args.seed + 12345)

    # Cache persistant des meilleurs designs (x, score, poids)
    cache_path = os.path.abspath(args.cache_file)
    cache_size = max(1, int(args.cache_size))
    design_cache: List[Dict[str, Any]] = []
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                design_cache = pickle.load(fh)
            print(f"[info] Loaded design cache {cache_path} with {len(design_cache)} entries.")
        except Exception as exc:
            print(f"[warn] Unable to load cache {cache_path}: {exc}")

    last_pair_opt_results: List[Dict[str, Any]] = []

    for r in range(1, max(1, args.max_real) + 1):
        print(f"\n[info] Réalisation adaptative #{r}")

        # Construire S_init à partir du déficit de Chernoff (pairs peu informées)
        S_init = np.zeros(n_pairs, dtype=float)
        if r == 1:
            # Difficulté initiale: S = -mean Chernoff(i,j; d)
            for idx, (i, j) in enumerate(pairs):
                vals = np.asarray(E_all[i, j, :], dtype=float)
                vals = np.where(np.isfinite(vals), vals, np.nan)
                if np.all(np.isnan(vals)):
                    S_init[idx] = 0.0
                else:
                    mean_C = float(np.nanmean(vals))
                    S_init[idx] = -mean_C
            finite_mask = np.isfinite(S_init)
            if np.any(finite_mask):
                min_val = float(np.min(S_init[finite_mask]))
                S_init = S_init - min_val + 1e-9
            else:
                S_init[:] = 1.0
        else:
            # Déficit basé sur l'information cumulée : plus C_cum est petit,
            # plus la paire doit recevoir de poids.
            C_vals = np.array([C_cum[p] for p in pairs], dtype=float)
            finite_mask = np.isfinite(C_vals)
            if np.any(finite_mask):
                C_max = float(np.max(C_vals[finite_mask]))
                # S_init ∝ (C_max - C_cum) => plus C_cum est bas, plus S_init est grand.
                S_init = C_max - C_vals
                finite_mask2 = np.isfinite(S_init)
                if np.any(finite_mask2):
                    min_val = float(np.min(S_init[finite_mask2]))
                    S_init = S_init - min_val + 1e-9
                else:
                    S_init[:] = 1.0
            else:
                S_init[:] = 1.0

        # Repondération itérative avec la corrélation
        order0 = np.argsort(-S_init)
        S_sorted = S_init[order0].tolist()
        pairs_sorted = [pairs[k] for k in order0]
        Corr_sorted = Corr[order0][:, order0]
        pairs_rw, S_rw = iterative_reweight_pairs(pairs_sorted, S_sorted, Corr_sorted)

        print("[result] Paires repondérées (S_final normalisé) pour cette réalisation:")
        for p, w in zip(pairs_rw, S_rw):
            print(f"  {p[0]}-{p[1]}: {w:.4f}")

        # Allocation de budget T_max en fonction de S_final (arrondi au multiple de 30s)
        alloc_raw = T_max * np.asarray(S_rw, dtype=float)
        alloc = np.where(alloc_raw > 0, np.ceil(alloc_raw / 30.0) * 30.0, 0.0).astype(int)
        print(f"[result] Allocation de budget par paire (T_max={T_max}, arrondi au multiple de 30s):")
        for (p, a) in zip(pairs_rw, alloc):
            print(f"  {p[0]}-{p[1]}: {a}")

        # Tirage du nombre de bonus par paire (Poisson, 1 bonus / 300s en moyenne) et mapping aux intervalles
        print("[result] Allocation des bonus par paire (Poisson, 1/300s en moyenne):")
        bonus_info: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for (p, a) in zip(pairs_rw, alloc):
            t_max_pair = float(a)
            if t_max_pair <= 0.0:
                bonus_info[p] = {"t_max": 0.0, "n_bonus": 0, "intervals": []}
                print(f"  {p[0]}-{p[1]}: t_max=0, n_bonus=0, intervals=[]")
                continue
            lam = t_max_pair / 300.0
            n_bonus = int(np.random.poisson(lam))
            n_intervals = max(1, int(t_max_pair // t_step))
            intervals: List[int] = []
            for i_intervals in range(n_bonus):
                t_b = random.uniform(i_intervals * 300, (i_intervals + 1) * 300)
                m = int(min(max(0, int(t_b // t_step)), n_intervals - 1))
                intervals.append(m)
            bonus_info[p] = {"t_max": t_max_pair, "n_bonus": n_bonus, "intervals": intervals}
            print(f"  {p[0]}-{p[1]}: t_max={int(t_max_pair)}, n_bonus={n_bonus}, intervals={intervals}")

        if args.show_plot and r == 1:
            pair_labels = [f"{i}-{j}" for i, j in pairs]
            plot_dendro(Z, pair_labels, show=True)

        # Optimisation v4 par paire (random search léger) pour cette réalisation
        print("[info] Optimisation v4 par paire avec budgets et bonus tirés (réalisation unique).")
        rng_pair = random.Random(rng_pair_global.randint(0, 2**31 - 1))
        pair_opt_results: List[Dict[str, Any]] = []

        for (i, j), info in bonus_info.items():
            t_max_pair = info["t_max"]
            if t_max_pair <= 0.0:
                continue
            intervals = info["intervals"]
            t_min_pair = 0.0
            t_max_pair = float(t_max_pair)
            t_fixed_pair = _build_fixed_times(t_min_pair, t_max_pair, t_step)
            n_intervals = max(0, len(t_fixed_pair) - 1)

            best_score = float("-inf")
            best_design = None
            for _ in range(max(1, args.n_candidates)):
                x: List[float] = []
                # t optimisés
                for _ in range(N_t):
                    x.append(rng_pair.uniform(t_min_pair, t_max_pair))
                # eff_idx / rew_idx (3 types)
                for _ in range(3):
                    x.append(rng_pair.randrange(len(Eff_pool)))
                for _ in range(3):
                    x.append(rng_pair.randrange(len(Rew_pool)))
                # bonus_idx (2)
                for _ in range(2):
                    x.append(rng_pair.randrange(len(Rew_pool)))
                # delta (3 types)
                max_delta = min(delta_max, max(delta_er, t_step))
                for _ in range(3):
                    x.append(rng_pair.uniform(delta_er, max_delta))
                # action_choice par intervalle (0..3)
                for _ in range(n_intervals):
                    x.append(rng_pair.randrange(0, 4))
                # bonus_u (2 par intervalle)
                for _ in range(n_intervals * 2):
                    x.append(rng_pair.random())

                rng_design = random.Random(rng_pair.randint(0, 2**31 - 1))
                design_pair = decode_design_actions_v4(
                    x,
                    N_t=N_t,
                    Eff_pool=Eff_pool,
                    Rew_pool=Rew_pool,
                    t_min=t_min_pair,
                    t_max=t_max_pair,
                    t_step=t_step,
                    min_gap_t=args.min_gap_t,
                    delta_er=delta_er,
                    delta_max=delta_max,
                    guard_after_t=guard_after_t,
                    bonus_intervals=[],
                    rng=rng_design,
                )
                # Chernoff pour la paire (i,j)
                lap_pair = laplace_jsd_for_design([models[i], models[j]], design_pair, sigma=args.sigma, n_jobs=args.n_jobs)
                mu_pair = lap_pair.get("mu_y", [])
                Vy_pair = lap_pair.get("Vy", [])
                if not mu_pair or not Vy_pair:
                    continue
                C_pair, _P_pair = _pairwise_chernoff_matrix(mu_pair, Vy_pair)
                c_val = float(C_pair[0][1])
                if not math.isfinite(c_val):
                    continue
                # Mise à jour du meilleur pour cette paire dans cette réalisation
                if c_val > best_score:
                    best_score = c_val
                    best_design = design_pair
                # Mise à jour du cache global (x, score, poids local ≈ S_rw pour cette paire)
                entry = {
                    "pair": (i, j),
                    "x": x,
                    "score": c_val,
                    "weight": float(S_rw[pairs_rw.index((i, j))]) if (i, j) in pairs_rw else 0.0,
                }
                design_cache.append(entry)
                # Garder seulement les cache_size meilleurs designs (score décroissant)
                design_cache.sort(key=lambda e: float(e.get("score", float("-inf"))), reverse=True)
                if len(design_cache) > cache_size:
                    design_cache = design_cache[:cache_size]

            if best_design is not None and math.isfinite(best_score) and best_score > 0.0:
                print(f"[pair-opt] ({i},{j}) best Chernoff={best_score:.4f}, t_max={int(t_max_pair)}")
                pair_opt_results.append({"pair": (i, j), "design": best_design, "chernoff": best_score})
                C_cum[(i, j)] = C_cum.get((i, j), 0.0) + best_score

        last_pair_opt_results = pair_opt_results

        # Vérifier l'uniformité des bornes Chernoff cumulées
        B_vals: List[float] = []
        for (i, j) in pairs:
            if i == j:
                continue
            C_ij = C_cum.get((i, j), 0.0)
            if C_ij > 0.0 and math.isfinite(C_ij):
                B_ij = 0.5 * math.exp(-C_ij)
                B_vals.append(B_ij)
        if B_vals:
            B_min = min(B_vals)
            B_max = max(B_vals)
            diff = B_max - B_min
            print(f"[cher] Bornes cumulées hors diagonale: min={B_min:.4g}, max={B_max:.4g}, diff={diff:.4g}")
            if diff <= args.cher_eps:
                print(f"[info] Critère d'uniformité atteint (ε={args.cher_eps}); arrêt après {r} réalisations.")
                break
        else:
            print("[cher] Impossible de calculer des bornes Chernoff cumulées (pas de C_cum finies).")
            break

    # Summary plot pour le dernier design combiné
    if last_pair_opt_results:
        try:
            from test_cluster_opti import build_combined_design  # type: ignore
            from test_multi_model import plot_summary  # type: ignore

            combined_design = build_combined_design(
                last_pair_opt_results,
                Eff_pool,
                Rew_pool,
                t_min=0.0,
            )
            lap_comb = laplace_jsd_for_design(models, combined_design, sigma=args.sigma, n_jobs=args.n_jobs)
            mu_c = lap_comb.get("mu_y", [])
            Vy_c = lap_comb.get("Vy", [])
            if mu_c and Vy_c:
                C_comb, P_comb = _pairwise_chernoff_matrix(mu_c, Vy_c)
                plot_summary(
                    C_comb,
                    combined_design,
                    models,
                    title="Smart clustering combined design (Chernoff, dernière réalisation)",
                    b_matrix=P_comb,
                )
        except Exception as exc:
            print(f"[diag] unable to plot combined summary: {exc}")
    # Sauvegarde du cache des designs
    try:
        with open(cache_path, "wb") as fh:
            pickle.dump(design_cache, fh)
        print(f"[info] Saved design cache to {cache_path} ({len(design_cache)} entries, top {cache_size}).")
    except Exception as exc:
        print(f"[warn] Unable to save design cache {cache_path}: {exc}")


if __name__ == "__main__":
    main()
