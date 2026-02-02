# TO CHECK

from __future__ import annotations

import os
import random
import sys
from typing import List, Tuple

import numpy as np

# Allow running as a script from repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from recode.design.build import build_design_v5, build_fixed_times
from recode.design_optimizer.base import numerical_jacobian
from recode.models.h_actions import build_h_action_function


Event = Tuple[float, float]
ActionTyped = Tuple[int, Event, Event]


def _random_actions(rng: random.Random, t_fixed: List[float], n_types: int) -> List[Tuple[int, float, float, float, float]]:
    actions: List[Tuple[int, float, float, float, float]] = []
    for i in range(len(t_fixed) - 1):
        start = t_fixed[i]
        end = t_fixed[i + 1]
        if end - start < 2.0:
            continue
        t_rew = rng.uniform(start + 1.0, end - 0.5)
        t_eff = rng.uniform(start + 0.1, max(start + 0.2, t_rew - 0.2))
        type_idx = rng.randrange(n_types)
        actions.append((type_idx, start, end, t_eff, t_rew))
    actions.sort(key=lambda x: x[4])
    return actions


def _make_design(rng: random.Random) -> dict:
    t_fixed = build_fixed_times(0.0, 60.0, 10.0)
    action_types = [(0.5, 1.0), (0.25, 0.5), (0.125, 0.25)]
    actions_idx_times = _random_actions(rng, t_fixed, n_types=len(action_types))
    design = build_design_v5(
        action_types=action_types,
        actions_idx_times=actions_idx_times,
        bonuses_t=[],
        rew_bonus_vals=[1.5, 1.0, 2.0],
        t_fixed=t_fixed,
        meas_times=t_fixed,
        meas_sources=None,
    )
    return design


def _plot_heatmap(ax, data: np.ndarray, title: str) -> None:
    im = ax.imshow(data, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("param")
    ax.set_ylabel("t index")
    return im


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for heatmaps") from exc

    rng = random.Random(1234)
    design = _make_design(rng)

    models = []
    for kernel in ["event_weighted", "action_avg"]:
        for update in ["continuous", "event", "action"]:
            f = build_h_action_function(kernel=kernel, update=update, observation="identity")
            models.append(f)

    n_models = len(models)
    n_cols = 3
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.asarray(axes).reshape(-1)

    for idx, f in enumerate(models):
        params = dict(f.parameters)
        J_exact = np.asarray(f.jacobian(design, params), dtype=float)
        J_num = np.asarray(numerical_jacobian(f, design, params, eps=1e-5), dtype=float)
        diff = J_num - J_exact
        ax = axes[idx]
        im = _plot_heatmap(ax, diff, f"{f.name}\n(num - exact)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for k in range(n_models, len(axes)):
        axes[k].axis("off")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
