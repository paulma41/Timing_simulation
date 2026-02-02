# TO CHECK

from __future__ import annotations

import random
from typing import List, Sequence, Tuple


ActionTuple = Tuple[int, float, float, float, float]


def sample_measurement_times(
    rng: random.Random,
    actions: Sequence[ActionTuple],
    bonuses_t: Sequence[float],
    n_t: int,
    t_max: float,
    t_min: float = 0.0,
    return_sources: bool = False,
) -> List[float] | Tuple[List[float], List[str]]:
    """
    Samples at most n_t//2 times after Eff and at most n_t//2 after Rew (incl. bonus).
    One action contributes to only one measurement time.
    Remaining times are uniform.
    """
    pairs: List[Tuple[float, str]] = []
    half = max(1, n_t // 2)

    eff_candidates = [
        (idx, t_eff, t_rew)
        for idx, (_, _, _, t_eff, t_rew) in enumerate(actions)
        if (t_rew - t_eff) > 0.5
    ]
    rng.shuffle(eff_candidates)
    used_actions = set()
    for idx, t_eff, t_rew in eff_candidates:
        t_meas = max(t_min, min(t_max, t_eff + 0.5))
        pairs.append((t_meas, "between_eff_rew"))
        used_actions.add(idx)
        if len(pairs) >= half:
            break

    rew_candidates: List[Tuple[int | None, float]] = []
    for idx, (_, _, _, _, t_rew) in enumerate(actions):
        if idx in used_actions:
            continue
        rew_candidates.append((idx, t_rew))
    for tb in bonuses_t:
        rew_candidates.append((None, tb))

    rng.shuffle(rew_candidates)
    for idx_opt, t_rew in rew_candidates:
        t_meas = max(t_min, min(t_max, t_rew + 0.5))
        pairs.append((t_meas, "after_rew"))
        if len(pairs) >= 2 * half:
            break

    while len(pairs) < n_t:
        r = rng.uniform(t_min, t_max)
        pairs.append((r, "uniform"))

    pairs.sort(key=lambda x: x[0])
    times = [p[0] for p in pairs]
    if not return_sources:
        return times
    sources = [p[1] for p in pairs]
    return times, sources


def sample_bonuses(
    rng: random.Random,
    n_bonus: int,
    t_fixed: Sequence[float],
    guard_after_t: float,
    before_margin: float = 8.0,
) -> List[float]:
    bonuses: List[float] = []
    n_intervals = max(0, len(t_fixed) - 1)
    if n_intervals == 0 or n_bonus <= 0:
        return bonuses
    for _ in range(n_bonus):
        m = rng.randrange(0, n_intervals)
        start = t_fixed[m] + guard_after_t
        end = t_fixed[m + 1] - before_margin
        if end <= start:
            continue
        bonuses.append(rng.uniform(start, end))
    bonuses.sort()
    return bonuses


def sample_forced_meas_times(
    rng: random.Random,
    actions_idx_times: Sequence[ActionTuple],
    forced_t_max: float,
    n_t: int,
    type_seq: Sequence[int],
) -> Tuple[List[float], List[str]]:
    if n_t <= 0 or not type_seq:
        return [], []
    seg_len = forced_t_max / float(len(type_seq))
    per_type = n_t // len(type_seq)
    extra = n_t % len(type_seq)
    pairs: List[Tuple[float, str]] = []
    for seg_idx, t_type in enumerate(type_seq):
        n_pick = per_type + (1 if seg_idx < extra else 0)
        if n_pick <= 0:
            continue
        seg_start = seg_idx * seg_len
        seg_end = (seg_idx + 1) * seg_len
        candidates = [
            (t_eff, t_rew)
            for k, _start, _end, t_eff, t_rew in actions_idx_times
            if k == t_type and seg_start <= t_rew <= seg_end
        ]
        rng.shuffle(candidates)
        for j in range(n_pick):
            if j < len(candidates):
                t_eff, t_rew = candidates[j]
                if t_rew - t_eff > 0.5:
                    t_meas = t_eff + 0.5
                else:
                    t_meas = t_eff + 0.5 * max(0.0, t_rew - t_eff)
                if t_meas < seg_start:
                    t_meas = seg_start
                if t_meas > seg_end:
                    t_meas = seg_end
                pairs.append((t_meas, "between_eff_rew"))
            else:
                t_meas = rng.uniform(seg_start, seg_end)
                pairs.append((t_meas, "uniform"))
    pairs.sort(key=lambda x: x[0])
    times = [p[0] for p in pairs]
    sources = [p[1] for p in pairs]
    return times, sources

