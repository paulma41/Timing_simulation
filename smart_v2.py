from __future__ import annotations

import argparse
import math
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from design_optimizer.laplace_jsd import laplace_jsd_for_design, _pairwise_chernoff_matrix
from test_hierarchical_opti import generate_random_designs_actions
from test_multi_model import build_six_models, _build_fixed_times
from test_multi_model_bo import make_demo_pools, decode_design_actions_v4
from test_cluster_opti import build_combined_design  # type: ignore

try:
    from tqdm import tqdm  # type: ignore

    HAVE_TQDM = True
except Exception:  # pragma: no cover
    HAVE_TQDM = False


def _compute_chernoff_matrix(models: Sequence[Any], design: Dict[str, Any], sigma: float, n_jobs: int) -> np.ndarray:
    lap = laplace_jsd_for_design(models, design, sigma=sigma, n_jobs=n_jobs)
    mu = lap.get("mu_y", [])
    Vy = lap.get("Vy", [])
    if not mu or not Vy:
        return np.full((len(models), len(models)), float("nan"))
    C_mat, _P = _pairwise_chernoff_matrix(mu, Vy)
    return np.asarray(C_mat, dtype=float)


def _choose_worst_pair_from_C(C: np.ndarray) -> Tuple[int, int]:
    """Retourne (i,j) avec i<j et C[i,j] minimal (pire séparabilité actuelle)."""
    K = C.shape[0]
    best_val = float("inf")
    best_pair = (0, 1)
    for i in range(K):
        for j in range(i + 1, K):
            val = float(C[i, j])
            if not math.isfinite(val):
                continue
            if val < best_val:
                best_val = val
                best_pair = (i, j)
    return best_pair

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart v2: construction séquentielle du design par blocs de t_max=30, ciblés sur la pire paire."
    )
    parser.add_argument("--designs-offline", type=int, default=50, help="Nombre de designs aléatoires pour le profil a priori.")
    parser.add_argument("--N-t", type=int, default=0, help="Nombre de temps de mesure optimisables par bloc (v4).")
    parser.add_argument("--t-step", type=float, default=30.0, help="Pas entre temps fixes dans un bloc.")
    parser.add_argument("--sigma", type=float, default=0.1, help="Bruit d'observation pour Laplace-JSD.")
    parser.add_argument("--n-jobs", type=int, default=3, help="Workers parallèles pour Laplace-JSD.")
    parser.add_argument("--seed", type=int, default=123, help="Graine RNG.")
    parser.add_argument("--min-gap-t", type=float, default=5.0, help="Écart minimal entre temps de mesure.")
    parser.add_argument("--delta-er", type=float, default=1.0, help="Séparation minimale Eff/Rew.")
    parser.add_argument("--delta-max", type=float, default=10.0, help="Séparation maximale Eff/Rew.")
    parser.add_argument("--guard-after-t", type=float, default=2.0, help="Zone muette après t_fixed pour Eff/Rew.")
    parser.add_argument("--block-duration", type=float, default=30.0, help="Durée d'un bloc (t_max pour v4).")
    parser.add_argument("--n-blocks", type=int, default=5, help="Nombre total de blocs souhaités.")
    parser.add_argument(
        "--cache-file",
        type=str,
        default=".smart_v2_cache.pkl",
        help="Cache pickle pour reprendre un design existant (combined_design, blocs, C_matrix).",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=40,
        help="Nombre d'évaluations v4 (appel Chernoff) par bloc et par paire (utilisé comme n_calls pour BO).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Affiche tqdm pendant les calculs.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    models = build_six_models()
    Eff_pool, Rew_pool = make_demo_pools()
    K = len(models)
    pairs_all = [(i, j) for i in range(K) for j in range(i + 1, K)]

    # Charger un design combiné existant si cache présent
    cache_path = os.path.abspath(args.cache_file)
    combined_design: Dict[str, Any] = {}
    block_results: List[Dict[str, Any]] = []
    C_curr = np.zeros((K, K), dtype=float)
    blocks_done = 0

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                cache = pickle.load(fh)
            combined_design = cache.get("combined_design", {}) or {}
            block_results = cache.get("blocks", []) or []
            blocks_done = int(cache.get("blocks_done", len(block_results)))
            print(f"[info] Loaded cache {cache_path}: {blocks_done} blocs déjà construits.")
            if combined_design:
                C_curr = _compute_chernoff_matrix(models, combined_design, sigma=args.sigma, n_jobs=args.n_jobs)
            else:
                C_curr = np.zeros((K, K), dtype=float)
        except Exception as exc:
            print(f"[warn] Unable to load cache {cache_path}: {exc}")
            combined_design = {}
            block_results = []
            C_curr = np.zeros((K, K), dtype=float)
            blocks_done = 0

    # Profil a priori uniquement si pas de design existant
    worst_pair_prior = None
    if not combined_design and args.designs_offline > 0:
        print(f"[info] Profil a priori sur {args.designs_offline} designs aléatoires.")
        designs_off = generate_random_designs_actions(
            args.designs_offline,
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
        C_off_max = np.full((K, K), -math.inf, dtype=float)

        def _offline_one(d_idx: int) -> Tuple[int, np.ndarray]:
            design = designs_off[d_idx]
            # n_jobs=1 ici pour éviter de doubler le parallélisme (on parallélise au niveau des designs)
            C_mat = _compute_chernoff_matrix(models, design, sigma=args.sigma, n_jobs=1)
            return d_idx, C_mat

        outer_jobs = max(1, min(args.n_jobs, len(designs_off)))
        if outer_jobs == 1:
            iterator = range(len(designs_off))
            if args.progress and HAVE_TQDM:
                iterator = tqdm(iterator, desc="offline designs", total=len(designs_off))
            for d_idx in iterator:
                _, C_mat = _offline_one(d_idx)
                if not np.isfinite(C_mat).any():
                    continue
                for i in range(K):
                    for j in range(i + 1, K):
                        val = float(C_mat[i, j])
                        if math.isfinite(val) and val > C_off_max[i, j]:
                            C_off_max[i, j] = val
                            C_off_max[j, i] = val
        else:
            iterator = range(len(designs_off))
            pbar = tqdm(total=len(designs_off), desc="offline designs", leave=True) if args.progress and HAVE_TQDM else None
            with ThreadPoolExecutor(max_workers=outer_jobs) as ex:
                futs = {ex.submit(_offline_one, d_idx): d_idx for d_idx in iterator}
                for fut in as_completed(futs):
                    _, C_mat = fut.result()
                    if not np.isfinite(C_mat).any():
                        if pbar is not None:
                            pbar.update(1)
                        continue
                    for i in range(K):
                        for j in range(i + 1, K):
                            val = float(C_mat[i, j])
                            if math.isfinite(val) and val > C_off_max[i, j]:
                                C_off_max[i, j] = val
                                C_off_max[j, i] = val
                    if pbar is not None:
                        pbar.update(1)
            if pbar is not None:
                pbar.close()
        # Pire paire a priori = plus petit max Chernoff
        best_val = float("inf")
        best_pair = (0, 1)
        for i, j in pairs_all:
            val = float(C_off_max[i, j])
            if not math.isfinite(val):
                continue
            if val < best_val:
                best_val = val
                best_pair = (i, j)
        worst_pair_prior = best_pair
        print(f"[info] Pire paire a priori (sur offline): {worst_pair_prior} (max Chernoff={best_val:.4g})")

    # Déterminer le nombre de blocs restants à construire
    n_blocks_target = max(0, int(args.n_blocks))
    if blocks_done >= n_blocks_target:
        print(f"[info] Aucun bloc supplémentaire demandé (blocs existants={blocks_done}, cible={n_blocks_target}).")
    else:
        print(f"[info] Construction de {n_blocks_target - blocks_done} blocs supplémentaires (sur cible {n_blocks_target}).")

    # Construction séquentielle bloc par bloc
    t_block = float(args.block_duration)
    for b in range(blocks_done, n_blocks_target):
        if combined_design:
            # Pire paire actuelle selon Chernoff sur le design combiné
            worst_i, worst_j = _choose_worst_pair_from_C(C_curr)
            print(f"[info] Bloc #{b+1}: pire paire actuelle = ({worst_i},{worst_j}), Chernoff={C_curr[worst_i,worst_j]:.4g}")
        else:
            # Premier bloc : utiliser la pire paire a priori si dispo, sinon random
            if worst_pair_prior is not None:
                worst_i, worst_j = worst_pair_prior
            else:
                worst_i, worst_j = random.choice(pairs_all)
            print(f"[info] Bloc #{b+1}: paire initiale choisie = ({worst_i},{worst_j})")

        # Optimiser un bloc v4 pour cette paire dans [0, t_block] via BO (skopt.gp_minimize)
        N_t = args.N_t
        delta_er = args.delta_er
        delta_max = args.delta_max
        guard_after_t = args.guard_after_t
        t_min_block = 0.0
        t_max_block = t_block

        # Préparer un RNG local pour ce bloc
        rng_block = random.Random(rng.randint(0, 2**31 - 1))

        # Espace v4 pour ce bloc
        t_fixed_block = _build_fixed_times(t_min_block, t_max_block, args.t_step)
        n_intervals_block = max(0, len(t_fixed_block) - 1)
        max_delta_block = min(delta_max, max(delta_er, args.t_step))

        try:
            from skopt import gp_minimize  # type: ignore
            from skopt.space import Real, Integer  # type: ignore
        except Exception as exc:
            raise RuntimeError("scikit-optimize (skopt) est requis pour smart_v2 (BO par bloc).") from exc

        space: List[Any] = []
        # t_opt
        for _ in range(N_t):
            space.append(Real(t_min_block, t_max_block, name="t_opt"))
        # eff_idx / rew_idx
        for _ in range(3):
            space.append(Integer(0, len(Eff_pool) - 1, name="ie"))
        for _ in range(3):
            space.append(Integer(0, len(Rew_pool) - 1, name="ir"))
        # bonus_idx
        for _ in range(2):
            space.append(Integer(0, len(Rew_pool) - 1, name="ib"))
        # delta
        for _ in range(3):
            space.append(Real(delta_er, max_delta_block, name="delta"))
        # action choices
        for _ in range(n_intervals_block):
            space.append(Integer(0, 3, name="act"))
        # bonus_u
        for _ in range(n_intervals_block * 2):
            space.append(Real(0.0, 1.0, name="bu"))

        def objective(x_vec: List[float]) -> float:
            """
            Objectif BO pour le bloc courant : maximise la Chernoff pour la paire
            (worst_i,worst_j) sur le design global (blocs existants + ce bloc).
            """
            rng_design = random.Random(rng_block.randint(0, 2**31 - 1))
            design_block = decode_design_actions_v4(
                x_vec,
                N_t=N_t,
                Eff_pool=Eff_pool,
                Rew_pool=Rew_pool,
                t_min=t_min_block,
                t_max=t_max_block,
                t_step=args.t_step,
                min_gap_t=args.min_gap_t,
                delta_er=delta_er,
                delta_max=delta_max,
                guard_after_t=guard_after_t,
                rng=rng_design,
            )
            try:
                # Construire un design global candidat = blocs existants + ce bloc
                blocks_candidate: List[Dict[str, Any]] = []
                for idx_block, br in enumerate(block_results + [{"pair": (worst_i, worst_j), "design": design_block}]):
                    d_b = dict(br)
                    design_b = d_b.get("design", {})
                    # On ne retire pas t_fixed ici pour rester cohérent avec build_combined_design
                    d_b["design"] = design_b
                    blocks_candidate.append(d_b)
                design_global = build_combined_design(
                    blocks_candidate,
                    Eff_pool,
                    Rew_pool,
                    t_min=0.0,
                    gap_between_blocks=0.0,
                )
                lap_global = laplace_jsd_for_design(
                    models,
                    design_global,
                    sigma=args.sigma,
                    n_jobs=args.n_jobs,
                )
                mu_g = lap_global.get("mu_y", [])
                Vy_g = lap_global.get("Vy", [])
                if not mu_g or not Vy_g:
                    return 1e6
                C_global, _P_global = _pairwise_chernoff_matrix(mu_g, Vy_g)
                c_val = float(C_global[worst_i][worst_j])
                if not math.isfinite(c_val):
                    return 1e6
                # On minimise, donc on renvoie -Chernoff globale pour la paire cible
                return -c_val
            except Exception:
                return 1e6

        n_calls = max(5, int(args.n_candidates))
        n_init = max(3, min(10, n_calls // 2))
        res_block = gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            n_initial_points=n_init,
            random_state=rng_block.randint(0, 2**31 - 1),
        )

        best_x = res_block.x
        rng_design_final = random.Random(rng_block.randint(0, 2**31 - 1))
        best_design_block = decode_design_actions_v4(
            best_x,
            N_t=N_t,
            Eff_pool=Eff_pool,
            Rew_pool=Rew_pool,
            t_min=t_min_block,
            t_max=t_max_block,
            t_step=args.t_step,
            min_gap_t=args.min_gap_t,
            delta_er=delta_er,
            delta_max=delta_max,
            guard_after_t=guard_after_t,
            rng=rng_design_final,
        )
        # Evaluer le Chernoff du bloc optimisé
        lap_pair_best = laplace_jsd_for_design(
            [models[worst_i], models[worst_j]],
            best_design_block,
            sigma=args.sigma,
            n_jobs=args.n_jobs,
        )
        mu_pair_best = lap_pair_best.get("mu_y", [])
        Vy_pair_best = lap_pair_best.get("Vy", [])
        best_score = float("nan")
        if mu_pair_best and Vy_pair_best:
            C_pair_best, _ = _pairwise_chernoff_matrix(mu_pair_best, Vy_pair_best)
            best_score = float(C_pair_best[0][1])

        if best_design_block is None:
            print(f"[warn] Aucun bloc valide trouvé pour la paire ({worst_i},{worst_j}) au bloc #{b+1}.")
            continue

        print(f"[block] Bloc #{b+1}: paire ({worst_i},{worst_j}), Chernoff bloc={best_score:.4f}")

        # Ajouter ce bloc à la liste et recombiner le design global
        block_results.append({"pair": (worst_i, worst_j), "design": best_design_block})
        combined_design = build_combined_design(
            block_results,
            Eff_pool,
            Rew_pool,
            t_min=0.0,
            gap_between_blocks=0.0,
        )
        C_curr = _compute_chernoff_matrix(models, combined_design, sigma=args.sigma, n_jobs=args.n_jobs)

        # Afficher la pire paire actuelle après ajout de ce bloc
        wi, wj = _choose_worst_pair_from_C(C_curr)
        print(f"[block] Après bloc #{b+1}: pire paire = ({wi},{wj}), Chernoff={C_curr[wi,wj]:.4g}")

    # Sauvegarder le cache (design combiné, blocs, C_matrix)
    try:
        cache = {
            "combined_design": combined_design,
            "blocks": block_results,
            "C_matrix": C_curr.tolist(),
            "blocks_done": len(block_results),
        }
        with open(cache_path, "wb") as fh:
            pickle.dump(cache, fh)
        print(f"[info] Saved smart_v2 cache to {cache_path} (blocs={len(block_results)}).")
    except Exception as exc:
        print(f"[warn] Unable to save smart_v2 cache {cache_path}: {exc}")

    # Affichage d'un résumé final si souhaité (on peut utiliser plot_summary si nécessaire)
    try:
        from test_multi_model import plot_summary  # type: ignore

        if combined_design and C_curr.size:
            lap_final = laplace_jsd_for_design(models, combined_design, sigma=args.sigma, n_jobs=args.n_jobs)
            mu_final = lap_final.get("mu_y", [])
            Vy_final = lap_final.get("Vy", [])
            if mu_final and Vy_final:
                _, P_comb = _pairwise_chernoff_matrix(mu_final, Vy_final)
                plot_summary(
                    C_curr.tolist(),
                    combined_design,
                    models,
                    title="Smart v2 combined design (Chernoff)",
                    b_matrix=P_comb,
                )
    except Exception as exc:
        print(f"[diag] unable to plot final summary: {exc}")


if __name__ == "__main__":
    main()
