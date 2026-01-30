from __future__ import annotations

import argparse
import random
from typing import Any, Dict, List, Sequence, Tuple

from scipy.cluster.hierarchy import fcluster
from skopt import gp_minimize
from skopt.space import Real, Integer

from design_optimizer.laplace_jsd import (
    laplace_jsd_for_design,
    _jensen_shannon_gaussians,
    _pairwise_chernoff_matrix,
)
from compute_model_separability import compute_model_separability
from test_multi_model import plot_summary, _build_fixed_times
from test_multi_model_bo import make_demo_pools, build_six_models, decode_design_actions_v4
from test_hierarchical_opti import generate_random_designs_actions

try:
    import cloudpickle  # type: ignore
    HAVE_CLOUDPICKLE = True
except Exception:
    cloudpickle = None  # type: ignore
    HAVE_CLOUDPICKLE = False


def _build_cross_pairs(node_dict: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    Construit la liste des paires (i,j) croisées entre le sous-ensemble gauche
    et le sous-ensemble droit d'un nœud de linkage, en indices locaux.
    """
    members = node_dict["members"]
    left_members = node_dict["left"]
    right_members = node_dict["right"]
    local_index = {g: idx for idx, g in enumerate(members)}
    pairs: List[Tuple[int, int]] = []
    for a in left_members:
        for b in right_members:
            if a in local_index and b in local_index:
                pairs.append((local_index[a], local_index[b]))
    return pairs


def _node_worker(payload: Any) -> Dict[str, Any]:
    """
    Worker de haut niveau pour paralléliser l'optimisation par nœud
    (doit être défini au niveau module pour être pickleable).
    """
    if not HAVE_CLOUDPICKLE or cloudpickle is None:
        raise RuntimeError("cloudpickle requis pour le parallélisme de test_cluster_opti")
    (
        rank,
        node_dict,
        models_blob_local,
        Eff_pool_local,
        Rew_pool_local,
        args_local,
        seed_local,
    ) = payload
    models_local = cloudpickle.loads(models_blob_local)
    members = node_dict["members"]
    models_node = [models_local[i] for i in members]
    cross_pairs = _build_cross_pairs(node_dict)
    rng_local = random.Random(seed_local)
    res_node = optimize_cluster(
        models_node,
        Eff_pool_local,
        Rew_pool_local,
        N_t=args_local.N_t,
        t_min=args_local.t_min,
        t_max=args_local.t_max,
        t_step=args_local.t_step,
        min_gap_t=args_local.min_gap_t,
        delta_er=args_local.delta_er,
        delta_max=args_local.delta_max,
        guard_after_t=args_local.guard_after_t,
        sigma=args_local.sigma,
        n_calls=args_local.n_calls,
        n_init=args_local.n_init,
        n_jobs=1,
        rng=rng_local,
        title_prefix=f"Node {node_dict['node_id']} (models {members})",
        cross_pairs=cross_pairs,
    )
    res_node["node_id"] = node_dict["node_id"]
    res_node["members"] = members
    res_node["height"] = node_dict["height"]
    res_node["_rank"] = rank
    return res_node

def optimize_cluster(
    models: Sequence[Any],
    Eff_pool: Sequence[float],
    Rew_pool: Sequence[float],
    *,
    N_t: int,
    t_min: float,
    t_max: float,
    t_step: float,
    min_gap_t: float,
    delta_er: float,
    delta_max: float,
    guard_after_t: float,
    sigma: float,
    n_calls: int,
    n_init: int,
    n_jobs: int,
    rng: random.Random,
    title_prefix: str,
    cross_pairs: Sequence[Tuple[int, int]] | None = None,
) -> Dict[str, Any]:
    """Optimisation BO (Laplace-JSD) sur un sous-ensemble de modèles."""
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None  # type: ignore

    space = []
    # Temps optimisés
    for i in range(N_t):
        space.append(Real(t_min, t_max, name=f"t_{i}"))
    # 3 types d'actions : eff_idx, rew_idx
    for k in range(3):
        space.append(Integer(0, len(Eff_pool) - 1, name=f"ie_{k}"))
    for k in range(3):
        space.append(Integer(0, len(Rew_pool) - 1, name=f"ir_{k}"))
    # bonus reward indices (2 bonus)
    for k in range(2):
        space.append(Integer(0, len(Rew_pool) - 1, name=f"ib_{k}"))
    # deltas
    max_delta = min(delta_max, max(delta_er, t_step))
    for k in range(3):
        space.append(Real(delta_er, max_delta, name=f"delta_{k}"))
    # choix d'action par intervalle (0..3)
    t_fixed = _build_fixed_times(t_min, t_max, t_step)
    n_intervals = max(0, len(t_fixed) - 1)
    for m in range(n_intervals):
        space.append(Integer(0, 3, name=f"act_{m}"))
    # bonus placements (2 par intervalle)
    for m in range(n_intervals):
        space.append(Real(0.0, 1.0, name=f"b1_{m}"))
        space.append(Real(0.0, 1.0, name=f"b2_{m}"))

    n_calls = max(1, n_calls)
    n_init = max(1, min(n_init, n_calls))

    def objective(x: List[float]) -> float:
        design_inputs = decode_design_actions_v4(
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
            rng=rng,
        )
        try:
            lap = laplace_jsd_for_design(models, design_inputs, sigma=sigma, n_jobs=n_jobs)
            score_matrix = None
            score = float(lap.get("DJS", float("nan")))
            if cross_pairs:
                mu_list = lap.get("mu_y", [])
                Vy_list = lap.get("Vy", [])
                mloc = len(models)
                if mu_list and Vy_list and len(mu_list) == mloc and len(Vy_list) == mloc:
                    score_matrix = [[0.0 for _ in range(mloc)] for _ in range(mloc)]
                    for ia, ib in cross_pairs:
                        djs_ij, _ = _jensen_shannon_gaussians(
                            [mu_list[ia], mu_list[ib]], [Vy_list[ia], Vy_list[ib]]
                        )
                        score_matrix[ia][ib] = djs_ij
                    # min des paires croisées A×B
                    vals = [score_matrix[ia][ib] for ia, ib in cross_pairs]
                    score = min(vals) if vals else float("nan")
        except Exception:
            return 1e12
        if not (score == score):
            return 1e12
        return -score  # gp_minimize minimise

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=n_calls, desc=f"{title_prefix} BO", leave=True)

        def _callback(_res) -> None:
            if pbar is not None:
                pbar.update(1)
    else:
        _callback = None  # type: ignore

    res = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_initial_points=n_init,
        random_state=rng.randint(0, 2**31 - 1),
        callback=None if pbar is None else [_callback],
    )

    if pbar is not None:
        pbar.close()

    best_x = res.x
    best_design = decode_design_actions_v4(
        best_x,
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
        rng=rng,
    )
    lap_best = laplace_jsd_for_design(models, best_design, sigma=sigma, n_jobs=n_jobs)
    if "K_types" in best_design:
        print(f"[diag] best_design K_types: {best_design.get('K_types')}")
    U_best = lap_best.get("U_pair")
    mu_list = lap_best.get("mu_y", [])
    Vy_list = lap_best.get("Vy", [])
    if U_best is None and mu_list and Vy_list:
        m = len(models)
        U_best = [[0.0 for _ in range(m)] for _ in range(m)]
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                djs_ij, _ = _jensen_shannon_gaussians(
                    [mu_list[i], mu_list[j]], [Vy_list[i], Vy_list[j]]
                )
                U_best[i][j] = djs_ij
    # Chernoff-based error bounds, only on best design
    if mu_list and Vy_list:
        _, B_best = _pairwise_chernoff_matrix(mu_list, Vy_list)
    else:
        m = len(models)
        B_best = [[float("nan") for _ in range(m)] for _ in range(m)]

    # Plot désactivé ici pour éviter des pauses multiples; on garde seulement le plot final combiné.
    # try:
    #     plot_summary(U_best, best_design, models, title=title_prefix)
    # except Exception as exc:
    #     print(f"[diag] unable to plot cluster/node summary '{title_prefix}': {exc}")

    return {"design": best_design, "U": U_best, "B": B_best}


def build_combined_design(
    cluster_results: List[Dict[str, Any]],
    Eff_pool: Sequence[float],
    Rew_pool: Sequence[float],
    *,
    t_min: float,
    gap_between_blocks: float = 5.0,
) -> Dict[str, Any]:
    combined_t: List[float] = []
    combined_Eff_: List[Any] = []
    combined_Rew_: List[Any] = []
    combined_A: List[Any] = []
    combined_Bonus: List[Any] = []

    current_offset = t_min
    for block in cluster_results:
        d = block["design"]
        t_block = d.get("t", []) or []
        Eff_block = d.get("Eff_", []) or []
        Rew_block = d.get("Rew_", []) or []
        A_block = d.get("A", []) or []
        Bonus_block = d.get("Bonus_", []) or []

        min_t_block = min(t_block) if t_block else 0.0
        shift = current_offset - min_t_block

        t_shifted = [float(t) + shift for t in t_block]
        Eff_shifted = [(val, float(t) + shift) for (val, t) in Eff_block]
        Rew_shifted = [(val, float(t) + shift) for (val, t) in Rew_block]
        A_shifted = [
            ((e_val, float(e_t) + shift), (r_val, float(r_t) + shift))
            for ((e_val, e_t), (r_val, r_t)) in A_block
        ]

        combined_t.extend(t_shifted)
        combined_Eff_.extend(Eff_shifted)
        combined_Rew_.extend(Rew_shifted)
        combined_A.extend(A_shifted)
        bonus_shifted = [(val, float(t) + shift) for (val, t) in Bonus_block]
        combined_Bonus.extend(bonus_shifted)

        if combined_t:
            current_offset = max(combined_t) + gap_between_blocks

    combined_t_sorted = sorted(combined_t)
    return {"t": combined_t_sorted, "Eff_": combined_Eff_, "Rew_": combined_Rew_, "Bonus_": combined_Bonus, "A": combined_A}


def extract_nodes_from_linkage(Z: Sequence[Sequence[float]], K: int) -> List[Dict[str, Any]]:
    """Retourne la liste des nœuds internes avec membres gauche/droite, triée du plus petit au plus grand."""
    clusters: Dict[int, List[int]] = {i: [i] for i in range(K)}
    nodes: List[Dict[str, Any]] = []
    for r, row in enumerate(Z):
        left, right, height, _ = row
        left = int(left)
        right = int(right)
        members = []
        left_members = clusters.get(left, [left])
        right_members = clusters.get(right, [right])
        members += left_members
        members += right_members
        node_id = K + r
        clusters[node_id] = members
        if len(members) > 1:
            nodes.append(
                {
                    "node_id": node_id,
                    "members": members,
                    "left": left_members,
                    "right": right_members,
                    "height": float(height),
                }
            )
    nodes_sorted = sorted(nodes, key=lambda t: (len(t["members"]), t["height"]))
    return nodes_sorted


def main() -> None:
    parser = argparse.ArgumentParser(description="Node-wise BO design optimisation (Laplace-JSD).")
    parser.add_argument(
        "--designs",
        type=int,
        default=20,
        help="Nombre de designs aléatoires pour la séparabilité offline.",
    )
    parser.add_argument("--N-t", type=int, default=5, help="Nombre de temps de mesure optimisables.")
    parser.add_argument("--t-step", type=float, default=30.0, help="Pas entre temps fixes (définit les intervalles).")
    parser.add_argument("--sigma", type=float, default=0.1, help="Écart-type du bruit d'observation.")
    parser.add_argument("--n-jobs", type=int, default=6, help="Nombre de workers parallèles pour Laplace-JSD.")
    parser.add_argument("--n-calls", type=int, default=60, help="n_calls pour BO par nœud.")
    parser.add_argument("--n-init", type=int, default=16, help="n_init pour BO par nœud.")
    parser.add_argument("--seed", type=int, default=123, help="Graine RNG.")
    parser.add_argument("--t-min", type=float, default=0.0, help="Temps minimum.")
    parser.add_argument("--t-max", type=float, default=600.0, help="Temps maximum.")
    parser.add_argument("--min-gap-t", type=float, default=5.0, help="Écart minimal entre temps de mesure.")
    parser.add_argument("--delta-er", type=float, default=1.0, help="Séparation minimale Eff/Rew.")
    parser.add_argument("--delta-max", type=float, default=10.0, help="Séparation maximale Eff/Rew.")
    parser.add_argument("--guard-after-t", type=float, default=2.0, help="Zone muette après t_fixed pour Eff/Rew.")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Barre de progression pour compute_model_separability.",
    )
    parser.add_argument(
        "--linkage",
        type=str,
        default="average",
        help="Méthode de linkage scipy (average, complete, ward, ...).",
    )
    parser.add_argument(
        "--node-workers",
        type=int,
        default=None,
        help="Workers pour paralléliser les nœuds (None -> utilise n-jobs).",
    )
    args = parser.parse_args()

    print("[info] Building models…")
    all_models = build_six_models()
    Eff_pool, Rew_pool = make_demo_pools()
    rng = random.Random(args.seed)

    print(f"[info] Generating {args.designs} random designs (actions_v4) for separability (seed={args.seed}).")
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

    print("[info] Computing model separability and linkage.")
    sep_result = compute_model_separability(
        all_models,
        designs,
        sigma=args.sigma,
        n_jobs=args.n_jobs,
        progress=args.progress,
        linkage_method=args.linkage,
    )
    Z = sep_result["Z"]
    K = len(all_models)
    nodes = extract_nodes_from_linkage(Z, K)
    print(f"[info] Optimising {len(nodes)} nodes (size>1), from leaves to root.")

    node_results: List[Dict[str, Any]] = []
    max_workers = args.node_workers if args.node_workers is not None else args.n_jobs
    max_workers = max(1, min(max_workers, len(nodes)))

    if HAVE_CLOUDPICKLE and cloudpickle is not None and max_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Prepare immutable payloads
        models_blob = cloudpickle.dumps(list(all_models))
        tasks = []
        for rank, node in enumerate(nodes, start=1):
            tasks.append(
                (
                    rank,
                    node,
                    models_blob,
                    Eff_pool,
                    Rew_pool,
                    args,
                    args.seed + rank,
                )
            )

        print(f"[info] Optimising nodes in parallel with {max_workers} workers.")
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = [exe.submit(_node_worker, task) for task in tasks]
            for fut in as_completed(futures):
                node_results.append(fut.result())
        # Restore order (ascending rank)
        node_results = sorted(node_results, key=lambda d: d.get("_rank", 0))
        for d in node_results:
            d.pop("_rank", None)
    else:
        for rank, node in enumerate(nodes, start=1):
            node_id = node["node_id"]
            members = node["members"]
            height = node["height"]
            print(f"[info] Optimising node {node_id} (#{rank}, size={len(members)}, height={height:.3f}): models {members}")
            models_node = [all_models[i] for i in members]
            cross_pairs = _build_cross_pairs(node)
            res_node = optimize_cluster(
                models_node,
                Eff_pool,
                Rew_pool,
                N_t=args.N_t,
                t_min=args.t_min,
                t_max=args.t_max,
                t_step=args.t_step,
                min_gap_t=args.min_gap_t,
                delta_er=args.delta_er,
                delta_max=args.delta_max,
                guard_after_t=args.guard_after_t,
                sigma=args.sigma,
                n_calls=args.n_calls,
                n_init=args.n_init,
                n_jobs=min(args.n_jobs, len(models_node)),
                rng=rng,
                title_prefix=f"Node {node_id} (models {members})",
                cross_pairs=cross_pairs,
            )
            res_node["node_id"] = node_id
            res_node["members"] = members
            res_node["height"] = height
            node_results.append(res_node)

    if node_results:
        print("[info] Building combined design from all nodes (ascending size).")
        combined_design = build_combined_design(
            node_results,
            Eff_pool,
            Rew_pool,
            t_min=args.t_min,
        )
        lap_comb = laplace_jsd_for_design(all_models, combined_design, sigma=args.sigma, n_jobs=args.n_jobs)
        if "K_types" in combined_design:
            print(f"[diag] combined_design K_types: {combined_design.get('K_types')}")
        U_combined = lap_comb.get("U_pair")
        mu_list_c = lap_comb.get("mu_y", [])
        Vy_list_c = lap_comb.get("Vy", [])
        if U_combined is None and mu_list_c and Vy_list_c:
            m_all = len(all_models)
            U_combined = [[0.0 for _ in range(m_all)] for _ in range(m_all)]
            for i in range(m_all):
                for j in range(m_all):
                    if i == j:
                        continue
                    djs_ij, _ = _jensen_shannon_gaussians(
                        [mu_list_c[i], mu_list_c[j]], [Vy_list_c[i], Vy_list_c[j]]
                    )
                    U_combined[i][j] = djs_ij
        if mu_list_c and Vy_list_c:
            _, B_combined = _pairwise_chernoff_matrix(mu_list_c, Vy_list_c)
        else:
            m_all = len(all_models)
            B_combined = [[float("nan") for _ in range(m_all)] for _ in range(m_all)]

        print("[result] Combined design t:", combined_design["t"])
        print("[result] Combined Eff_:", combined_design["Eff_"])
        print("[result] Combined Rew_:", combined_design["Rew_"])
        print("[result] Combined U matrix (Laplace-JSD):")
        for row in U_combined:
            print([round(x, 3) for x in row])
        try:
            plot_summary(
                U_combined,
                combined_design,
                all_models,
                title="Combined node design (Laplace-JSD)",
                b_matrix=B_combined,
            )
        except Exception as exc:
            print(f"[diag] unable to plot combined design summary: {exc}")

        if B_combined:
            print("[result] Combined Chernoff error-bound matrix (P_err <= 0.5 * exp(-C)):]")
            for row in B_combined:
                print([round(x, 3) for x in row])
            # Weights per pair (i<j) normalised to sum to 1 over finite bounds
            import math as _math
            weights: List[Tuple[Tuple[int, int], float]] = []
            total_w = 0.0
            for i in range(len(B_combined)):
                for j in range(i + 1, len(B_combined)):
                    val = float(B_combined[i][j])
                    if _math.isfinite(val) and val > 0.0:
                        total_w += val
                        weights.append(((i, j), val))
            if total_w > 0.0 and weights:
                print("[result] Normalised pair weights w[i,j] ∝ B[i,j] (i<j), sum=1:")
                for (i, j), val in weights:
                    print(f"  ({i},{j}): {val/total_w:.3g}")


if __name__ == "__main__":
    main()
