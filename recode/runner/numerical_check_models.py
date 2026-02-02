# TO CHECK

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np

# Allow running as a script from repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from recode.models.h_actions import build_h_action_function

Event = Tuple[float, float]  # (value, time)
Action = Tuple[int, Event, Event]  # (type_idx, (e_val, t_e), (r_val, t_r))


def _build_forced_actions(
    type_seq_1based: List[int],
    *,
    t_start_rew: float = 20.0,
    dt_rew: float = 20.0,
) -> Tuple[List[Tuple[float, float]], List[Action], List[Event], List[Event]]:
    """
    Build a deterministic A_typed from a forced sequence of action types.
    Types are interpreted as 1-based in the input list.
    """
    # Match the values used in the legacy scripts.
    action_types = [
        (0.5, 1.0),  # type 1 -> idx 0
        (0.25, 0.5),  # type 2 -> idx 1
        (0.125, 0.25),  # type 3 -> idx 2
    ]
    deltas = [12.0, 6.0, 3.0]  # Eff->Rew delays per type

    # Convert 1-based -> 0-based, and clamp.
    type_seq = []
    for k in type_seq_1based:
        kk = int(k) - 1
        if kk < 0:
            kk = 0
        if kk >= len(action_types):
            kk = len(action_types) - 1
        type_seq.append(kk)

    Eff_: List[Event] = []
    Rew_: List[Event] = []
    A_typed: List[Action] = []

    for i, type_idx in enumerate(type_seq):
        t_rew = float(t_start_rew + i * dt_rew)
        delta = deltas[type_idx]
        t_eff = max(0.0, t_rew - delta)
        e_val, r_val = action_types[type_idx]
        eff = (float(e_val), float(t_eff))
        rew = (float(r_val), float(t_rew))
        Eff_.append(eff)
        Rew_.append(rew)
        A_typed.append((type_idx, eff, rew))

    Rew_.sort(key=lambda x: x[1])
    return action_types, A_typed, Eff_, Rew_


def main() -> None:
    # Forced action types: [1,1,1,2,2,2,3,3,3]
    type_seq = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    action_types, A_typed, Eff_, Rew_ = _build_forced_actions(type_seq)

    models = []
    for kernel in ["event_weighted", "action_avg"]:
        for update in ["continuous", "event", "action"]:
            models.append(build_h_action_function(kernel=kernel, update=update, observation="identity"))

    # Evaluation grid
    t_end = max([0.0] + [t for _, t in Rew_]) + 30.0
    t_grid = np.linspace(0.0, t_end, int(max(200, 10 * t_end)) + 1).tolist()

    inputs = {
        "t": t_grid,
        "Eff_": Eff_,
        "Rew_": Rew_,
        "A_typed": A_typed,
        "K_types": len(action_types),
    }

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from exc

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    for f in models:
        y = f.eval(dict(inputs))
        ax.plot(t_grid, y, linewidth=1.2, label=f.name)

    # Visual markers for events
    for v, te in Eff_:
        ax.axvline(te, color="tab:red", alpha=0.15, linewidth=1.0)
    for v, tr in Rew_:
        ax.axvline(tr, color="tab:green", alpha=0.15, linewidth=1.0)

    ax.set_title("h(t) for 6 models on forced action sequence [1,1,1,2,2,2,3,3,3]")
    ax.set_xlabel("t")
    ax.set_ylabel("h(t)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

