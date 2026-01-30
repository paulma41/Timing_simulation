from __future__ import annotations

import argparse
import math
import os
import pickle
import random
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from design_optimizer.laplace_jsd import laplace_jsd_for_design, _pairwise_chernoff_matrix
from test_multi_model_bo import build_six_models, decode_design_actions_v4
from test_hierarchical_opti import generate_random_designs_actions
from test_multi_model import _build_fixed_times

from concurrent.futures import ThreadPoolExecutor, as_completed

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


def _score_design(models: Sequence[Any], design: Dict[str, Any], sigma: float) -> Tuple[float, List[List[float]]]:
    """Score maximin sur la matrice de Chernoff pour un design donné. Retourne (score, C_matrix)."""
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


def random_pool(rng: random.Random, size: int, lo: float, hi: float) -> List[float]:
    vals = [rng.uniform(lo, hi) for _ in range(size)]
    vals.sort()
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Random search on stimulus ranges (Eff_pool, Rew_pool, delta).")
    parser.add_argument("--candidates", type=int, default=1, help="Number of range candidates to try.")
    parser.add_argument("--designs-per-range", type=int, default=20, help="Random designs per candidate range.")
    parser.add_argument("--pool-size", type=int, default=3, help="Number of values in Eff_pool/Rew_pool.")
    parser.add_argument("--sigma", type=float, default=0.1, help="Observation noise for Chernoff evaluation.")
    parser.add_argument("--t-max", type=float, default=600.0, help="t_max for generated designs.")
    parser.add_argument("--t-step", type=float, default=30.0, help="Step between fixed times.")
    parser.add_argument("--N-t", type=int, default=0, help="Number of optimised t in v4.")
    parser.add_argument("--min-gap-t", type=float, default=5.0, help="Min gap between measurement times.")
    parser.add_argument("--guard-after-t", type=float, default=2.0, help="Guard after t_fixed for events.")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed.")
    parser.add_argument("--n-jobs", type=int, default=6, help="Parallélisme sur les candidats (threads).")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="stimuli_range_cache.pkl",
        help="Fichier pickle pour mettre en cache les évaluations de ranges.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    models = build_six_models()

    best_score = float("-inf")
    best_conf: Dict[str, Any] = {}
    candidates_data: List[Dict[str, Any]] = []
    cache: Dict[Tuple[Tuple[float, ...], Tuple[float, ...], float, float], Dict[str, Any]] = {}

    # Chargement du cache disque si disponible
    if hasattr(args, "cache_file") and args.cache_file and os.path.exists(args.cache_file):
        try:
            with open(args.cache_file, "rb") as f:
                loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    cache.update(loaded)
            print(f"[cache] Loaded {len(cache)} entrées depuis {args.cache_file}")
        except Exception as e:  # pragma: no cover
            print(f"[cache] Impossible de charger {args.cache_file}: {e}")

    def eval_candidate(cand_idx: int) -> Dict[str, Any]:
        rng_local = random.Random(args.seed + cand_idx)
        Eff_pool = random_pool(rng_local, args.pool_size, -2.0, -0.1)
        Rew_pool = random_pool(rng_local, args.pool_size, 0.1, 2.0)
        delta_min = rng_local.uniform(0.5, 5.0)
        delta_max = rng_local.uniform(delta_min, 10.0)

        key = (
            tuple(round(v, 6) for v in Eff_pool),
            tuple(round(v, 6) for v in Rew_pool),
            round(delta_min, 6),
            round(delta_max, 6),
        )
        if key in cache:
            return cache[key]

        scores: List[float] = []
        C_accum: List[List[float]] = []
        designs = generate_random_designs_actions(
            args.designs_per_range,
            rng=rng_local,
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
        cand_score = float(np.nanmean(scores)) if scores else float("-inf")
        C_mean = None
        if C_accum:
            nC = args.designs_per_range
            C_mean = [[c / nC for c in row] for row in C_accum]
        res = {
            "Eff_pool": Eff_pool,
            "Rew_pool": Rew_pool,
            "delta_min": delta_min,
            "delta_max": delta_max,
            "score": cand_score,
            "C_mean": C_mean,
        }
        cache[key] = res
        return res

    cand_range = range(max(1, args.candidates))
    pbar = tqdm(total=len(cand_range), desc="candidates", leave=True) if HAVE_TQDM else None
    n_jobs = max(1, args.n_jobs)
    results: List[Dict[str, Any]] = []
    if n_jobs == 1:
        for c in cand_range:
            res = eval_candidate(c)
            results.append(res)
            if pbar is not None:
                pbar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futs = {ex.submit(eval_candidate, c): c for c in cand_range}
            for fut in as_completed(futs):
                res = fut.result()
                results.append(res)
                if pbar is not None:
                    pbar.update(1)
    if pbar is not None:
        pbar.close()

    # Sauvegarde du cache sur disque
    if hasattr(args, "cache_file") and args.cache_file:
        try:
            with open(args.cache_file, "wb") as f:
                pickle.dump(cache, f)
            print(f"[cache] Sauvegardé {len(cache)} entrées dans {args.cache_file}" )
        except Exception as e:  # pragma: no cover
            print(f"[cache] Impossible de sauvegarder le cache dans {args.cache_file}: {e}")

    for res in results:
        candidates_data.append(res)
        if res["score"] > best_score:
            best_score = res["score"]
            best_conf = {
                "Eff_pool": res["Eff_pool"],
                "Rew_pool": res["Rew_pool"],
                "delta_min": res["delta_min"],
                "delta_max": res["delta_max"],
            }

    print("[best] score:", best_score)
    print("[best] Eff_pool:", best_conf.get("Eff_pool"))
    print("[best] Rew_pool:", best_conf.get("Rew_pool"))
    print("[best] delta range:", best_conf.get("delta_min"), best_conf.get("delta_max"))

    # Analyse de la variabilit� par paire (Chernoff)
    matrices = [r["C_mean"] for r in candidates_data if r.get("C_mean")]
    if matrices:
        m = len(matrices[0])
        pair_min = np.full((m, m), np.inf)
        pair_max = np.full((m, m), -np.inf)
        for C in matrices:
            arr = np.array(C)
            pair_min = np.minimum(pair_min, arr)
            pair_max = np.maximum(pair_max, arr)
        zero_pairs = []
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                if np.isfinite(pair_min[i, j]) and np.isfinite(pair_max[i, j]) and pair_max[i, j] - pair_min[i, j] == 0:
                    zero_pairs.append((i, j, pair_min[i, j]))
        if zero_pairs:
            print("[info] Paires sans variation de Chernoff (min=max) :")
            for i, j, val in zero_pairs:
                print(f"    pair ({i},{j}) -> {val}")
        else:
            print("[info] Toutes les paires ont une variation non nulle de Chernoff.")

    # Profils de pools (line plots) pour le top ~2 % uniquement
    if HAVE_MPL and candidates_data:
        scores_all = [c["score"] for c in candidates_data]
        if scores_all:
            thresh = np.nanpercentile(scores_all, 90.0)
            top10 = [c for c in candidates_data if c["score"] >= thresh]
            if top10:
                scores_top = [c["score"] for c in top10]
                best_thresh = np.nanpercentile(scores_top, 98.0)
                top2 = [c for c in top10 if c["score"] >= best_thresh]
                if top2:
                    s_top2 = [c["score"] for c in top2]
                    s_min_top = float(np.nanmin(s_top2))
                    s_max_top = float(np.nanmax(s_top2))
                    def _alpha(score: float) -> float:
                        if not math.isfinite(s_min_top) or not math.isfinite(s_max_top) or s_max_top == s_min_top:
                            return 1.0
                        t = (score - s_min_top) / (s_max_top - s_min_top)
                        return max(0.1, min(1.0, t))

                    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
                    ax_eff, ax_rew = axes
                    x_idx = list(range(1, args.pool_size + 1))

                    for c in top2:
                        sc = c["score"]
                        alpha = _alpha(sc)
                        eff = c["Eff_pool"]
                        rew = c["Rew_pool"]
                        ax_eff.plot(x_idx, eff, color="black", alpha=alpha, linewidth=1)
                        ax_rew.plot(x_idx, rew, color="black", alpha=alpha, linewidth=1)

                    ax_eff.set_title("Profils Eff_pool (top ~2 %, noir, alpha proportionnel au score)")
                    ax_eff.set_ylabel("Effort")
                    ax_eff.set_xticks(x_idx)

                    ax_rew.set_title("Profils Rew_pool (top ~2 %, noir, alpha proportionnel au score)")
                    ax_rew.set_xlabel("Index dans le pool")
                    ax_rew.set_ylabel("Reward")
                    ax_rew.set_xticks(x_idx)

                    plt.tight_layout()
                    plt.show()

    # Profils delta_k (approximation à partir de delta_min/delta_max) pour le top ~2 % uniquement
    if HAVE_MPL and candidates_data:
        scores_all = [c["score"] for c in candidates_data]
        if scores_all:
            thresh = np.nanpercentile(scores_all, 90.0)
            top10 = [c for c in candidates_data if c["score"] >= thresh]
            if top10:
                scores_top = [c["score"] for c in top10]
                best_thresh = np.nanpercentile(scores_top, 98.0)
                top2 = [c for c in top10 if c["score"] >= best_thresh]
                if top2:
                    s_top2 = [c["score"] for c in top2]
                    s_min_top = float(np.nanmin(s_top2))
                    s_max_top = float(np.nanmax(s_top2))
                    def _alpha(score: float) -> float:
                        if not math.isfinite(s_min_top) or not math.isfinite(s_max_top) or s_max_top == s_min_top:
                            return 1.0
                        t = (score - s_min_top) / (s_max_top - s_min_top)
                        return max(0.1, min(1.0, t))

                    fig, ax_delta = plt.subplots(1, 1, figsize=(6, 4))
                    x_idx = [1, 2, 3]

                    for c in top2:
                        sc = c["score"]
                        alpha = _alpha(sc)
                        dmin = c["delta_min"]
                        dmax = c["delta_max"]
                        delta_profile = [dmin, 0.5 * (dmin + dmax), dmax]
                        ax_delta.plot(x_idx, delta_profile, color="black", alpha=alpha, linewidth=1)

                    ax_delta.set_title("Profils delta_k (top ~2 %, noir, alpha proportionnel au score)")
                    ax_delta.set_xlabel("Index k")
                    ax_delta.set_ylabel("delta")
                    ax_delta.set_xticks(x_idx)

                    plt.tight_layout()
                    plt.show()

    print("min/max scores_all:", min(scores_all), max(scores_all))
if __name__ == "__main__":
    main()
