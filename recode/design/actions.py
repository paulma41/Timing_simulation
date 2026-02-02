# TO CHECK

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


ActionTuple = Tuple[int, float, float, float, float]


def markov_actions_with_types(
    rng: random.Random,
    t_fixed: List[float],
    guard_after_t: float,
    progress_cb: Optional[Callable[[float], None]] = None,
    forced_type: Optional[int] = None,
) -> Tuple[List[Tuple[float, float]], List[ActionTuple]]:
    """
    Simple Markov chain action generator per interval.
    1/2 no action, 1/6 for each type (0,1,2).
    """
    action_types = [
        (0.5, 1.0),
        (0.25, 0.5),
        (0.125, 0.25),
    ]
    deltas = [12.0, 6.0, 3.0]
    actions: List[ActionTuple] = []
    n_intervals = max(0, len(t_fixed) - 1)
    for m in range(n_intervals):
        start_abs = t_fixed[m]
        end_abs = t_fixed[m + 1]
        local_start = start_abs + 2.0
        local_end = min(start_abs + 29.0, end_abs - guard_after_t)
        if local_end <= local_start:
            continue
        last_rew_time = start_abs + guard_after_t
        t_cur = local_start
        if progress_cb:
            progress_cb(t_cur)
        while t_cur < local_end - 1e-9:
            r = rng.random()
            if r < 0.5:
                t_cur += 1.0
                if progress_cb:
                    progress_cb(t_cur)
                continue
            if forced_type is not None:
                type_idx = int(forced_type)
                if type_idx < 0:
                    type_idx = 0
                if type_idx >= len(deltas):
                    type_idx = len(deltas) - 1
            else:
                r2 = (r - 0.5) / 0.5
                if r2 < 1 / 3:
                    type_idx = 0
                elif r2 < 2 / 3:
                    type_idx = 1
                else:
                    type_idx = 2
            delta_k = deltas[type_idx]
            t_rew = t_cur
            if t_rew > local_end:
                break
            t_eff = max(start_abs + guard_after_t, t_rew - delta_k)
            if t_eff >= t_rew or t_rew > (end_abs - guard_after_t):
                t_cur += 1.0
                if progress_cb:
                    progress_cb(t_cur)
                continue
            if t_eff < last_rew_time:
                t_cur += 1.0
                if progress_cb:
                    progress_cb(t_cur)
                continue
            actions.append((type_idx, start_abs, end_abs, t_eff, t_rew))
            last_rew_time = t_rew
            t_cur += delta_k
            if progress_cb:
                progress_cb(t_cur)
        if progress_cb:
            progress_cb(local_end)
    actions.sort(key=lambda x: x[4])
    return action_types, actions


def magneto_softmax(V: List[float], M: float, a: float, b: float) -> List[float]:
    """Softmax for magneto-like agent; calibrated to p0=1/2 when M=0."""
    logits_base = [a * v for v in V]
    S = sum(math.exp(l) for l in logits_base)
    s0 = math.log(S) if S > 0 else float("-inf")
    logits = [s0] + [lb + b * M for lb in logits_base]
    m = max(logits)
    exp_shift = [math.exp(L - m) for L in logits]
    total = sum(exp_shift)
    if total <= 0:
        return [1.0, 0.0, 0.0, 0.0]
    return [x / total for x in exp_shift]


def event_weighted_h(
    t_eval: float, Eff_hist: List[Tuple[float, float]], Rew_hist: List[Tuple[float, float]], gamma: float
) -> float:
    if gamma < 0:
        gamma = 0.0
    if gamma > 1:
        gamma = 1.0
    ln_g = math.log(gamma if gamma > 0 else 1e-12)
    h_val = 0.0
    for val, te in Eff_hist:
        if te < t_eval:
            dt = t_eval - te
            x = max(min(dt * ln_g, 0.0), -745.0)
            decay = math.exp(x) if gamma > 0 else (1.0 if dt == 0.0 else 0.0)
            h_val += -1.0 * val * decay
    for val, tr in Rew_hist:
        if tr < t_eval:
            dt = t_eval - tr
            x = max(min(dt * ln_g, 0.0), -745.0)
            decay = math.exp(x) if gamma > 0 else (1.0 if dt == 0.0 else 0.0)
            h_val += 1.0 * val * decay
    return h_val


def magneto_like_actions_with_value_learning(
    rng: random.Random,
    t_fixed: List[float],
    guard_after_t: float,
    *,
    gamma_decay: Optional[float] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
    forced_type: Optional[int] = None,
    state: Optional[Dict[str, Any]] = None,
    params_init: Optional[Dict[str, float]] = None,
) -> Tuple[
    List[Tuple[float, float]],
    List[ActionTuple],
    Dict[str, float],
    Dict[str, Any],
]:
    action_types = [
        (0.5, 1.0),
        (0.25, 0.5),
        (0.125, 0.25),
    ]
    deltas = [12.0, 6.0, 3.0]

    if params_init:
        a = float(params_init.get("a", rng.uniform(0.5, 1.0)))
        b = float(params_init.get("b", rng.uniform(0.0, 1.0)))
        if gamma_decay is None:
            gamma_decay = float(params_init.get("gamma", rng.uniform(0.5, 0.99)))
    else:
        a = rng.uniform(0.5, 1.0)
        b = rng.uniform(0.0, 1.0)
        if gamma_decay is None:
            gamma_decay = rng.uniform(0.5, 0.99)
    alpha = 1.0 - gamma_decay

    if state is not None:
        V_est = list(state.get("V_est", [0.0 for _ in action_types]))
        if len(V_est) != len(action_types):
            V_est = [0.0 for _ in action_types]
        Eff_hist = list(state.get("Eff_hist", []))
        Rew_hist = list(state.get("Rew_hist", []))
    else:
        V_est = [0.0 for _ in action_types]
        Eff_hist = []
        Rew_hist = []

    actions: List[ActionTuple] = []

    n_intervals = max(0, len(t_fixed) - 1)
    for m in range(n_intervals):
        start_abs = t_fixed[m]
        end_abs = t_fixed[m + 1]
        local_start = start_abs + 2.0
        local_end = min(start_abs + 29.0, end_abs - guard_after_t)
        if local_end <= local_start:
            continue
        last_rew_time = start_abs + guard_after_t
        t_cur = local_start
        if progress_cb:
            progress_cb(t_cur)
        while t_cur < local_end - 1e-9:
            M = event_weighted_h(t_cur, Eff_hist, Rew_hist, gamma_decay)
            probs = magneto_softmax(V_est, M, a=a, b=b)
            if forced_type is not None:
                forced_idx = int(forced_type)
                if forced_idx < 0:
                    forced_idx = 0
                if forced_idx >= len(action_types):
                    forced_idx = len(action_types) - 1
                keep_idx = forced_idx + 1
                for i in range(1, len(probs)):
                    if i != keep_idx:
                        probs[i] = 0.0
                total = sum(probs)
                if total > 0:
                    probs = [p / total for p in probs]
                else:
                    probs = [1.0] + [0.0 for _ in range(len(action_types))]
            r_draw = rng.random()
            cum = 0.0
            choice = 0
            for idx, p in enumerate(probs):
                cum += p
                if r_draw <= cum:
                    choice = idx
                    break
            if choice == 0:
                t_cur += 1.0
                if progress_cb:
                    progress_cb(t_cur)
                continue
            type_idx = choice - 1
            delta_k = deltas[type_idx]
            t_rew = t_cur
            if t_rew > local_end or t_rew > (end_abs - guard_after_t):
                t_cur += 1.0
                if progress_cb:
                    progress_cb(t_cur)
                continue
            t_eff = max(start_abs + guard_after_t, t_rew - delta_k)
            if t_eff >= t_rew:
                t_cur += 1.0
                if progress_cb:
                    progress_cb(t_cur)
                continue
            if t_eff < last_rew_time:
                t_cur += 1.0
                if progress_cb:
                    progress_cb(t_cur)
                continue
            actions.append((type_idx, start_abs, end_abs, t_eff, t_rew))
            last_rew_time = t_rew

            e_val, r_val = action_types[type_idx]
            Eff_hist.append((e_val, t_eff))
            Rew_hist.append((r_val, t_rew))

            target = r_val - e_val
            V_est[type_idx] += alpha * (target - V_est[type_idx])

            t_cur += delta_k
            if progress_cb:
                progress_cb(t_cur)
        if progress_cb:
            progress_cb(local_end)

    actions.sort(key=lambda x: x[4])
    state_out = {"V_est": V_est[:], "Eff_hist": Eff_hist[:], "Rew_hist": Rew_hist[:]}
    return action_types, actions, {"a": a, "b": b, "gamma": gamma_decay, "V_final": V_est[:], "agent_kind": "magneto"}, state_out


def normalize_type_sequence(type_seq: Optional[Sequence[int]], n_types: int) -> List[int]:
    if not type_seq:
        return list(range(n_types))
    vals = [int(v) for v in type_seq]
    if vals and min(vals) >= 1 and max(vals) <= n_types:
        vals = [v - 1 for v in vals]
    norm = []
    for v in vals:
        if v < 0:
            v = 0
        if v >= n_types:
            v = n_types - 1
        norm.append(v)
    return norm

