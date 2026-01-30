from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence


def plot_online_convergence(
    estimates: Sequence[Dict[str, float]],
    true_params: Optional[Dict[str, float]] = None,
    *,
    param_order: Optional[Sequence[str]] = None,
    title: str = "Online parameter convergence",
    figsize: tuple = (10, 6),
    show: bool = True,
    savepath: Optional[str] = None,
    # New: history of chosen inputs (designs) per step and which variables to plot
    inputs_history: Optional[Sequence[Dict[str, Any]]] = None,
    variables: Optional[Sequence[str]] = None,
):
    """
    Trace l'évolution des paramètres estimés au fil des itérations.

    - estimates: liste des dictionnaires de paramètres par itération
    - true_params: dictionnaire de référence (paramètres vrais) optionnel
    - param_order: ordre des paramètres à tracer; sinon déduit du premier dict
    - figsize, show, savepath: contrôles d'affichage/sauvegarde
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[diag] matplotlib non installé: skip du tracé de convergence.")
        return None, None

    if not estimates:
        raise ValueError("'estimates' ne doit pas être vide")

    keys = list(estimates[0].keys()) if param_order is None else list(param_order)
    steps = list(range(len(estimates)))

    n_var = 0
    if inputs_history:
        if variables is None and inputs_history:
            variables = list(inputs_history[0].keys())
        n_var = len(variables or [])

    # Build figure with 1 (params) + n_var rows, shared x-axis
    if n_var > 0:
        fig, axes = plt.subplots(1 + n_var, 1, figsize=figsize, sharex=True,
                                 gridspec_kw={"height_ratios": [2] + [1]*n_var})
        if not isinstance(axes, (list, tuple)):
            # when matplotlib returns a numpy array
            try:
                axes = axes.ravel()
            except Exception:
                axes = [axes]
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]

    line_colors = {}
    for k in keys:
        ys = [e.get(k, float('nan')) for e in estimates]
        line = ax.plot(steps, ys, marker='o', label=f"{k}")
        # mémoriser la couleur utilisée pour la courbe d'estimation
        if line:
            line_colors[k] = line[0].get_color()

    if true_params is not None:
        for k in keys:
            if k in true_params:
                color = line_colors.get(k, 'k')
                ax.axhline(true_params[k], color=color, linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Parameter value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

    # Plot variables per step, if provided
    if n_var > 0 and inputs_history is not None and variables is not None:
        import numpy as _np  # type: ignore
        from math import isnan as _isnan
        for idx, var in enumerate(variables, start=1):
            axv = axes[idx]
            axv.set_title(f"{var}")
            axv.set_ylabel(var)
            # For each step, scatter the values (scalar or iterable)
            for step, design in enumerate(inputs_history):
                if var not in design:
                    continue
                val = design[var]
                # Normalize to list of numbers
                if isinstance(val, (list, tuple)):
                    ys = list(val)
                else:
                    try:
                        import numpy as np
                        if isinstance(val, np.ndarray):
                            ys = val.flatten().tolist()
                        else:
                            ys = [float(val)]
                    except Exception:
                        ys = [float(val)]
                # Scatter and annotate
                if ys:
                    xs = [step] * len(ys)
                    points = axv.scatter(xs, ys, s=18, color='tab:gray')
                    for (xpt, ypt) in zip(xs, ys):
                        try:
                            txt = f"{ypt:.3g}"
                        except Exception:
                            txt = str(ypt)
                        axv.text(xpt + 0.05, ypt, txt, fontsize=7, alpha=0.8)
            axv.grid(True, linestyle=':', alpha=0.4)
        axes[-1].set_xlabel("Iteration")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    if show:
        plt.show()
    return fig, ax
