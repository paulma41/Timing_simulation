from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from design_optimizer.laplace_jsd import laplace_jsd_for_design, _pairwise_chernoff_matrix

try:
    from tqdm import tqdm  # type: ignore

    HAVE_TQDM = True
except Exception:  # pragma: no cover
    HAVE_TQDM = False


def compute_design_efficiency(
    models: Sequence[Any],
    design: Mapping[str, Any],
    *,
    optimizer: str = "laplace-jsd",
    sigma: float = 1.0,
    n_jobs: int = 1,
    eps: float = 1e-5,
) -> float:
    """
    Mesure scalaire de séparabilité (Laplace-JSD) pour une paire de modèles
    et un design donné. Utilisée comme score E[i,j,d] dans l'analyse offline.
    """
    if optimizer != "laplace-jsd":
        raise ValueError("compute_design_efficiency ne supporte que optimizer='laplace-jsd'")
    res = laplace_jsd_for_design(models, design, sigma=float(sigma), eps=eps, n_jobs=int(max(1, n_jobs)))
    return float(res.get("DJS", float("nan")))


def compute_model_separability(
    models: Sequence[Any],
    designs: Sequence[Mapping[str, Any]],
    model_labels: Optional[Sequence[str]] = None,
    *,
    sigma: float = 1.0,
    n_jobs: int = 1,
    eps: float = 1e-5,
    linkage_method: str = "average",
    progress: bool = False,
) -> Dict[str, Any]:
    """
    Analyse offline de séparabilité pairwise entre modèles sur un ensemble
    de designs candidats, en utilisant Laplace-JSD pour le score et la
    borne de Chernoff pour construire le dendrogramme.

    Pour chaque paire (i,j) et chaque design d :
      - on calcule un score de séparabilité scalaire E[i,j,d] = JSD_ij(d),
      - on retient E_max[i,j] = max_d E[i,j,d] et d*_ij = argmax_d E[i,j,d].

    Pour construire la matrice de distance D utilisée par linkage :
      - pour chaque (i,j), on prend le design optimal d*_ij,
      - on calcule la borne de Chernoff B_ij (P_err(i vs j) <= 0.5 * exp(-C_ij)),
      - on définit D[i,j] = 1 - B_ij (plus la borne est faible, plus la distance est grande).
    """
    K = len(models)
    D = len(designs)
    if K < 2:
        raise ValueError("Il faut au moins deux modèles.")
    if D < 1:
        raise ValueError("Il faut au moins un design candidat.")

    labels = list(model_labels) if model_labels is not None else [f"M{i}" for i in range(K)]
    if len(labels) != K:
        raise ValueError("model_labels doit avoir la même longueur que models.")

    E_all = np.full((K, K, D), np.nan, dtype=float)
    E_max = np.full((K, K), -math.inf, dtype=float)
    best_design_idx = np.full((K, K), -1, dtype=int)
    B_ch = np.full((K, K), np.nan, dtype=float)  # Chernoff bound sur le meilleur design

    total_eval = (K * (K - 1) // 2) * D
    pbar = tqdm(total=total_eval, desc="Laplace-JSD pairs", leave=True) if (progress and HAVE_TQDM) else None

    for i in range(K):
        for j in range(i + 1, K):
            best_val = -math.inf
            best_d = -1
            best_B_ij = math.nan
            for d, design in enumerate(designs):
                # Score de séparabilité (JSD)
                try:
                    lap = laplace_jsd_for_design(
                        [models[i], models[j]],
                        design,
                        sigma=float(sigma),
                        eps=eps,
                        n_jobs=max(1, int(n_jobs)),
                    )
                    val = float(lap.get("DJS", float("nan")))
                except Exception:
                    val = float("nan")
                E_all[i, j, d] = val
                E_all[j, i, d] = val
                if pbar is not None:
                    pbar.update(1)
                if not math.isfinite(val):
                    continue
                if val > best_val:
                    best_val = val
                    best_d = d
                    # Met à jour immédiatement B_ij pour le design optimal courant
                    mu_list = lap.get("mu_y", [])
                    Vy_list = lap.get("Vy", [])
                    if mu_list and Vy_list and len(mu_list) == 2 and len(Vy_list) == 2:
                        _, P_pair = _pairwise_chernoff_matrix(mu_list, Vy_list)
                        b_ij = float(P_pair[0][1])
                        best_B_ij = b_ij if math.isfinite(b_ij) and b_ij >= 0.0 else math.nan

            E_max[i, j] = E_max[j, i] = best_val if math.isfinite(best_val) else float("nan")
            best_design_idx[i, j] = best_design_idx[j, i] = best_d
            B_ch[i, j] = B_ch[j, i] = best_B_ij

        E_max[i, i] = 0.0
        best_design_idx[i, i] = -1
        B_ch[i, i] = 0.0

    if pbar is not None:
        pbar.close()

    # Matrice de distance pour linkage
    D_mat = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            if i == j:
                D_mat[i, j] = 0.0
            else:
                b_ij = float(B_ch[i, j])
                if math.isfinite(b_ij) and b_ij >= 0.0:
                    # D = 1 - borne (plus la borne est faible, plus la distance est grande)
                    D_mat[i, j] = 1.0 - max(0.0, min(1.0, b_ij))
                else:
                    # fallback : distance basée sur E_max (clipée)
                    D_mat[i, j] = max(0.0, float(E_max[i, j]))

    # Convertir en vecteur condensé et construire le linkage
    Y = squareform(D_mat, checks=False)
    Z = linkage(Y, method=linkage_method)

    return {
        "E_all": E_all,
        "E_max": E_max,
        "best_design_idx": best_design_idx,
        "D": D_mat,
        "Z": Z,
        "model_labels": labels,
    }


__all__ = ["compute_design_efficiency", "compute_model_separability"]

