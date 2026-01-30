from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise RuntimeError("numpy est requis pour plot_model_separability") from exc

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib est requis pour plot_model_separability") from exc

try:
    import seaborn as sns  # type: ignore
    HAVE_SEABORN = True
except Exception:  # pragma: no cover
    HAVE_SEABORN = False

try:
    from scipy.cluster import hierarchy
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scipy.cluster.hierarchy est requis pour plot_model_separability") from exc

def plot_model_separability(
    result: Dict[str, Any],
    *,
    cmap: str = "viridis",
    figsize: Sequence[float] = (12, 5),
    show: bool = True,
) -> Dict[str, Any]:
    """
    Affiche une heatmap de E_max et un dendrogramme des modèles.

    Parameters
    ----------
    result : dict
        Sortie de compute_model_separability.
    cmap : str
        Colormap pour la heatmap.
    figsize : tuple
        Taille de la figure matplotlib.
    show : bool
        Si True, appelle plt.show().
    """
    E_max = result.get("E_max")
    Z = result.get("Z")
    labels = result.get("model_labels", [])
    if E_max is None or Z is None:
        raise ValueError("result doit contenir E_max et Z.")
    E_max = np.asarray(E_max, dtype=float)
    labels = list(labels) if labels else [str(i) for i in range(E_max.shape[0])]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Heatmap E_max
    ax0 = axes[0]
    if HAVE_SEABORN:
        sns.heatmap(E_max, ax=ax0, cmap=cmap, annot=False, xticklabels=labels, yticklabels=labels, cbar=True)
    else:
        im = ax0.imshow(E_max, cmap=cmap)
        ax0.set_xticks(range(len(labels)))
        ax0.set_yticks(range(len(labels)))
        ax0.set_xticklabels(labels, rotation=90)
        ax0.set_yticklabels(labels)
        fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    ax0.set_title("E_max (séparabilité maximale)")

    # Dendrogram
    ax1 = axes[1]
    hierarchy.dendrogram(Z, labels=labels, ax=ax1, orientation="right")
    ax1.set_title("Clustering hiérarchique")

    plt.tight_layout()
    if show:
        plt.show()

    return {"fig": fig, "axes": axes}


__all__ = ["plot_model_separability"]
