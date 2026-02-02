# TO CHECK

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


def build_fixed_times(t_min: float, t_max: float, step: float) -> List[float]:
    """Fixed grid t_min, t_min+step, ..., t_max (inclusive)."""
    if step <= 0:
        return [t_min, t_max]
    vals: List[float] = []
    v = t_min
    while v < t_max:
        vals.append(v)
        v += step
    if not vals or vals[-1] < t_max:
        vals.append(t_max)
    return vals


def build_design_v5(
    action_types: Sequence[Tuple[float, float]],
    actions_idx_times: Sequence[Tuple[int, float, float, float, float]],
    bonuses_t: Sequence[float],
    rew_bonus_vals: Sequence[float],
    t_fixed: Sequence[float],
    meas_times: Sequence[float],
    meas_sources: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    Eff_: List[Tuple[float, float]] = []
    Rew_: List[Tuple[float, float]] = []
    Bonus_: List[Tuple[float, float]] = []
    A: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    A_typed: List[Tuple[int, Tuple[float, float], Tuple[float, float]]] = []

    for type_idx, _start, _end, t_eff, t_rew in actions_idx_times:
        e_val, r_val = action_types[type_idx % len(action_types)]
        Eff_.append((e_val, t_eff))
        Rew_.append((r_val, t_rew))
        A.append(((e_val, t_eff), (r_val, t_rew)))
        A_typed.append((type_idx, (e_val, t_eff), (r_val, t_rew)))

    for idx, tb in enumerate(bonuses_t):
        valb = rew_bonus_vals[idx % len(rew_bonus_vals)]
        Bonus_.append((valb, tb))
        Rew_.append((valb, tb))

    t_meas = sorted(set(t_fixed) | set(meas_times))
    t_all = sorted(set(t_meas) | {t for _, t in Eff_} | {t for _, t in Rew_})
    Rew_sorted = sorted(Rew_, key=lambda p: p[1])

    design: Dict[str, Any] = {
        "t": t_all,
        "t_meas": t_meas,
        "Eff_": Eff_,
        "Rew_": Rew_sorted,
        "Bonus_": Bonus_,
        "A": A,
        "A_typed": A_typed,
        "K_types": max(1, len(action_types)),
    }
    if meas_sources is not None:
        design["meas_times"] = list(meas_times)
        design["meas_sources"] = list(meas_sources)
    return design

