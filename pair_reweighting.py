from __future__ import annotations

from typing import List, Sequence, Tuple, Dict, Any

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise RuntimeError("numpy est requis pour pair_reweighting.py") from exc


def iterative_reweight_pairs(
    L_pairs: Sequence[Tuple[int, int]],
    S_init: Sequence[float],
    Corr: Sequence[Sequence[float]],
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """
    Répond à l'algorithme itératif proposé pour combiner séparabilité et corrélation.

    Paramètres
    ----------
    L_pairs : séquence de paires (i, j)
        Liste initiale des paires (i<j), l'ordre initial n'est pas critique.
    S_init : séquence de floats
        Valeurs initiales S[j] associées à chaque paire (par ex. 1 - séparabilité moyenne).
    Corr : matrice (N x N)
        Corrélation entre profils des paires. Corr[a][b] est Corr(a,b).

    Algorithme
    ----------
    On applique pour i = 0..N-1 :
      1. Pour tout j > i :
           S[j] <- S[j] * (1 - max(0, Corr(i,j)))
      2. On réordonne la liste (L_pairs, S, Corr) par S décroissant.

    On retourne S normalisé (somme = 1) et L_pairs dans l'ordre final.
    """
    S = np.asarray(S_init, dtype=float).copy()
    Corr_mat = np.asarray(Corr, dtype=float).copy()
    n = len(S)
    if Corr_mat.shape != (n, n):
        raise ValueError("Corr doit être de taille (N,N) avec N = len(S_init)")

    # Remplace NaN par 0 pour éviter les effets indésirables
    Corr_mat = np.where(np.isfinite(Corr_mat), Corr_mat, 0.0)
    pairs: List[Tuple[int, int]] = list(L_pairs)

    for i in range(n):
        # Mise à jour S[j] pour j > i avec la corrélation actuelle
        for j in range(i + 1, n):
            c = float(Corr_mat[i, j])
            penal = max(0.0, c)
            S[j] *= (1.0 - penal)

        # Réordonne par S décroissant
        order = np.argsort(-S)
        S = S[order]
        Corr_mat = Corr_mat[order][:, order]
        pairs = [pairs[k] for k in order]

    total = float(S.sum())
    if total > 0.0:
        S_norm = (S / total).tolist()
    else:
        # Si tout est nul, on retourne une distribution uniforme
        S_norm = [1.0 / float(n)] * n if n > 0 else []

    return pairs, S_norm


__all__ = ["iterative_reweight_pairs"]
