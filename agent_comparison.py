from __future__ import annotations

import math
import random
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, Sequence

from design_optimizer.laplace_jsd import laplace_jsd_for_design, _pairwise_chernoff_matrix, _chernoff_gaussian_pair
from test_multi_model import build_six_models, _build_fixed_times, plot_summary
from test_multi_model_bo import make_demo_pools
from test_multi_model_bo_v5 import (
    maximin_chernoff,
    sample_bonuses,
    sample_measurement_times,
    build_design_v5,
)

# Verbose flag for debugging
VERBOSE = False

def markov_actions_with_types(
    rng: random.Random,
    t_fixed: List[float],
    guard_after_t: float,
    progress_cb: Optional[Callable[[float], None]] = None,
    forced_type: Optional[int] = None,
) -> Tuple[List[Tuple[float, float]], List[Tuple[int, float, float, float, float]]]:
    """
    Génère des actions par intervalle via une chaîne de Markov simple :
      - proba 1/2 pour "ne rien faire", 1/6 pour chaque type (0,1,2)
      - si aucune action : t <- t + 1
      - si action de type k : t <- t + delta_k
    Par intervalle, on démarre à t=2 (relatif à l'intervalle) et on s'arrête à t=29 (relatif).
    Types / deltas / valeurs :
      - type 0: effort 0.5, reward 1.0, delta=12
      - type 1: effort 0.25, reward 0.5, delta=6  (scale 6/12)
      - type 2: effort 0.125, reward 0.25, delta=3 (scale 3/12)
    """
    action_types = [
        (0.5, 1.0),            # type 0
        (0.25, 0.5),           # type 1
        (0.125, 0.25),         # type 2
    ]
    deltas = [12.0, 6.0, 3.0]
    actions: List[Tuple[int, float, float, float, float]] = []
    n_intervals = max(0, len(t_fixed) - 1)
    for m in range(n_intervals):
        start_abs = t_fixed[m]
        end_abs = t_fixed[m + 1]
        # Fenêtre locale [2, 29] à l'intérieur de l'intervalle, en respectant le guard
        local_start = start_abs + 2.0
        local_end = min(start_abs + 29.0, end_abs - guard_after_t)
        if local_end <= local_start:
            continue
        last_rew_time = start_abs + guard_after_t
        t_cur = local_start
        # chaîne de Markov simplifiée
        if progress_cb:
            progress_cb(t_cur)
        while t_cur < local_end - 1e-9:
            r = rng.random()
            if r < 0.5:  # ne rien faire
                t_cur += 1.0
                if progress_cb:
                    progress_cb(t_cur)
                continue
            # 0.5 restant réparti uniformément sur 3 types => 1/6 chacun
            if forced_type is not None:
                type_idx = int(forced_type)
                if type_idx < 0:
                    type_idx = 0
                if type_idx >= len(deltas):
                    type_idx = len(deltas) - 1
            else:
                r2 = (r - 0.5) / 0.5  # in [0,1)
                if r2 < 1/3:
                    type_idx = 0
                elif r2 < 2/3:
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
    actions.sort(key=lambda x: x[4])  # tri par t_rew
    return action_types, actions


def _magneto_softmax(V: List[float], M: float, a: float, b: float) -> List[float]:
    """Probabilites comme dans test_MAGNETO, calibrées pour p0=1/2 quand M=0."""
    logits_base = [a * v for v in V]
    S = sum(math.exp(l) for l in logits_base)
    # calibrage : on veut p0 = 1/2 quand M=0 => logit0 = log(S) - log(1)
    s0 = math.log(S) if S > 0 else float("-inf")
    logits = [s0] + [lb + b * M for lb in logits_base]
    m = max(logits)
    exp_shift = [math.exp(L - m) for L in logits]
    total = sum(exp_shift)
    if total <= 0:
        return [1.0, 0.0, 0.0, 0.0]
    return [x / total for x in exp_shift]


def _event_weighted_h(
    t_eval: float, Eff_hist: List[Tuple[float, float]], Rew_hist: List[Tuple[float, float]], gamma: float
) -> float:
    """h(t) global (kernel event_weighted) avec W_effort=-1, W_reward=1, h0=0."""
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
    List[Tuple[int, float, float, float, float]],
    Dict[str, float],
    Dict[str, Any],
]:
    """
    Simule un agent type MAGNETO :
      - valeurs internes V_i mises a jour apres chaque action (delta rule alpha=1-gamma).
      - choix d'action via softmax de test_MAGNETO (a V_i + b M, M=h(t) commun).
    Retourne (action_types, actions_idx_times, params_agent).
    """
    action_types = [
        (0.5, 1.0),            # type 0
        (0.25, 0.5),           # type 1
        (0.125, 0.25),         # type 2
    ]
    deltas = [12.0, 6.0, 3.0]

    # Parametres de choix
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

    # Etat initial
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
    actions: List[Tuple[int, float, float, float, float]] = []

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
            # h(t) global commun a toutes les actions (event_weighted sur historique)
            M = _event_weighted_h(t_cur, Eff_hist, Rew_hist, gamma_decay)
            probs = _magneto_softmax(V_est, M, a=a, b=b)
            if forced_type is not None:
                forced_idx = int(forced_type)
                if forced_idx < 0:
                    forced_idx = 0
                if forced_idx >= len(action_types):
                    forced_idx = len(action_types) - 1
                keep_idx = forced_idx + 1  # 0 = no-action
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

            # Mise a jour V_i <- V_i + alpha * (target - V_i) avec target = r - e
            target = r_val - e_val
            V_est[type_idx] += alpha * (target - V_est[type_idx])

            t_cur += delta_k
            if progress_cb:
                progress_cb(t_cur)
        if progress_cb:
            progress_cb(local_end)

    actions.sort(key=lambda x: x[4])  # tri par t_rew
    state_out = {"V_est": V_est[:], "Eff_hist": Eff_hist[:], "Rew_hist": Rew_hist[:]}
    return action_types, actions, {"a": a, "b": b, "gamma": gamma_decay, "V_final": V_est[:], "agent_kind": "magneto"}, state_out


def _normalize_type_sequence(type_seq: Optional[Sequence[int]], n_types: int) -> List[int]:
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


def _sample_forced_meas_times(
    rng: random.Random,
    actions_idx_times: Sequence[Tuple[int, float, float, float, float]],
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


def run_agent(
    seed: int = 1203,
    n_jobs: int = 4,
    agent_kind: str = "magneto",
    include_models: bool = True,
    progress_cb: Optional[Callable[[float], None]] = None,
    progress_queue: Optional[Any] = None,
    progress_total: Optional[int] = None,
    sigma: float = 0.05,
    light_results: bool = False,
    t_max: float = 7200.0,
    t_step: float = 30.0,
    n_bonus: int = 20,
    n_t: int = 24,
    forced_block: bool = False,
    forced_t_max: float = 900.0,
    forced_t_step: float = 60.0,
    forced_n_t: int = 9,
    forced_type_seq: Optional[Sequence[int]] = None,
    observation: str = "sigmoid",
    obs_temp_mean: float = 1.0,
    obs_temp_std: float = 0.0,
    obs_bias_mean: float = 0.0,
    obs_bias_std: float = 0.0,
) -> Dict[str, Any]:
    if VERBOSE:
        print(f"[dbg] run_agent start seed={seed} agent={agent_kind} n_jobs={n_jobs}", file=sys.stderr, flush=True)
    rng = random.Random(seed)
    models = build_six_models(observation=observation)
    obs_temp = float(obs_temp_mean)
    obs_bias = float(obs_bias_mean)
    if observation == "sigmoid":
        if obs_temp_std > 0.0:
            obs_temp = rng.gauss(float(obs_temp_mean), float(obs_temp_std))
        if obs_bias_std > 0.0:
            obs_bias = rng.gauss(float(obs_bias_mean), float(obs_bias_std))
    for f in models:
        f.parameters["obs_temp"] = obs_temp
        f.parameters["obs_bias"] = obs_bias
        if isinstance(f.sim_priors, dict):
            f.sim_priors["obs_temp"] = {"dist": "Normal", "mu": obs_temp, "sigma": 0.0}
            f.sim_priors["obs_bias"] = {"dist": "Normal", "mu": obs_bias, "sigma": 0.0}
    Eff_pool, Rew_pool = make_demo_pools()  # non utilisé ici mais gardé pour compat

    progress_sent = 0

    def _report_progress(delta: int) -> None:
        nonlocal progress_sent
        if progress_queue is None or delta <= 0:
            return
        progress_sent += delta
        progress_queue.put(delta)

    free_t_max = float(t_max)
    free_t_step = float(t_step)
    free_n_bonus = int(n_bonus)
    free_n_t = int(n_t)

    guard_after_t = 0.5
    forced_offset = float(forced_t_max) if forced_block else 0.0
    free_start = forced_offset
    free_end = free_start + free_t_max

    t_fixed_forced = _build_fixed_times(0.0, forced_t_max, forced_t_step) if forced_block else []
    t_fixed_free = _build_fixed_times(free_start, free_end, free_t_step)
    t_fixed = sorted(set(t_fixed_forced + t_fixed_free))

    actions_forced: List[Tuple[int, float, float, float, float]] = []
    actions_free: List[Tuple[int, float, float, float, float]] = []
    meas_times_forced: List[float] = []
    meas_sources_forced: List[str] = []
    action_types: List[Tuple[float, float]] = []
    agent_params: Dict[str, Any] = {}
    magneto_state: Optional[Dict[str, Any]] = None
    magneto_params: Optional[Dict[str, float]] = None

    type_seq = _normalize_type_sequence(forced_type_seq, 3) if forced_block else []
    if forced_block and type_seq:
        seg_len = forced_t_max / float(len(type_seq))
        for seg_idx, t_type in enumerate(type_seq):
            seg_start = seg_idx * seg_len
            seg_end = (seg_idx + 1) * seg_len
            t_fixed_seg = _build_fixed_times(seg_start, seg_end, forced_t_step)
            if agent_kind == "magneto":
                action_types, actions_seg, agent_params_seg, magneto_state = magneto_like_actions_with_value_learning(
                    rng,
                    t_fixed_seg,
                    guard_after_t,
                    progress_cb=progress_cb,
                    forced_type=t_type,
                    state=magneto_state,
                    params_init=magneto_params,
                )
                magneto_params = {"a": agent_params_seg["a"], "b": agent_params_seg["b"], "gamma": agent_params_seg["gamma"]}
                agent_params = agent_params_seg
            else:
                action_types, actions_seg = markov_actions_with_types(
                    rng, t_fixed_seg, guard_after_t, progress_cb=progress_cb, forced_type=t_type
                )
                agent_params = {"agent_kind": "markov"}
            actions_forced.extend(actions_seg)
        meas_times_forced, meas_sources_forced = _sample_forced_meas_times(
            rng, actions_forced, forced_t_max, forced_n_t, type_seq
        )

    if agent_kind == "magneto":
        action_types_free, actions_free, agent_params_free, magneto_state = magneto_like_actions_with_value_learning(
            rng,
            t_fixed_free,
            guard_after_t,
            progress_cb=progress_cb,
            state=magneto_state,
            params_init=magneto_params,
        )
        if not action_types:
            action_types = action_types_free
        agent_params = agent_params_free
    else:
        action_types_free, actions_free = markov_actions_with_types(
            rng, t_fixed_free, guard_after_t, progress_cb=progress_cb
        )
        if not action_types:
            action_types = action_types_free
        agent_params = {"agent_kind": "markov"}

    bonuses_t = sample_bonuses(
        rng, n_bonus=free_n_bonus, t_fixed=t_fixed_free, guard_after_t=guard_after_t, before_margin=8.0
    )

    if VERBOSE:
        n_actions_total = len(actions_forced) + len(actions_free)
        print(f"[dbg] actions built seed={seed} agent={agent_kind} n_actions={n_actions_total}", file=sys.stderr, flush=True)

    actions_timed = actions_forced + actions_free
    actions_timed.sort(key=lambda x: x[4])

    meas_times_free, meas_sources_free = sample_measurement_times(
        rng,
        actions_free,
        bonuses_t,
        n_t=free_n_t,
        t_max=free_end,
        t_min=free_start,
        return_sources=True,
    )
    meas_times_best = meas_times_forced + meas_times_free
    meas_sources_best = meas_sources_forced + meas_sources_free

    best_design = build_design_v5(
        action_types=action_types,
        actions_idx_times=actions_timed,
        bonuses_t=bonuses_t,
        rew_bonus_vals=Rew_pool,
        t_fixed=t_fixed,
        meas_times=meas_times_best,
        meas_sources=meas_sources_best,
    )

    lap_best = laplace_jsd_for_design(
        models,
        best_design,
        sigma=float(sigma),
        n_jobs=max(1, n_jobs),
        compute_jsd=False,
        progress_cb=_report_progress if progress_queue is not None else None,
        return_lowrank=True,
    )
    if VERBOSE:
        print(f"[dbg] laplace done seed={seed} agent={agent_kind}", file=sys.stderr, flush=True)
    mu_best = lap_best.get("mu_y", [])
    Vy_best = lap_best.get("Vy", [])
    Vy_lowrank = lap_best.get("Vy_lowrank")

    C_best: List[List[float]] = []
    P_best: List[List[float]] = []
    if mu_best and Vy_best:
        if progress_queue is None:
            C_best, P_best = _pairwise_chernoff_matrix(mu_best, Vy_best, cov_lowrank=Vy_lowrank)
        else:
            m = len(mu_best)
            C_best = [[0.0 for _ in range(m)] for _ in range(m)]
            P_best = [[0.0 for _ in range(m)] for _ in range(m)]
            for i in range(m):
                for j in range(i + 1, m):
                    lr_i = Vy_lowrank[i] if Vy_lowrank and i < len(Vy_lowrank) else None
                    lr_j = Vy_lowrank[j] if Vy_lowrank and j < len(Vy_lowrank) else None
                    c_ij = _chernoff_gaussian_pair(
                        mu_best[i],
                        Vy_best[i],
                        mu_best[j],
                        Vy_best[j],
                        n_grid=51,
                        lowrank0=lr_i,
                        lowrank1=lr_j,
                    )
                    C_best[i][j] = c_ij
                    C_best[j][i] = c_ij
                    P_val = 0.5 * math.exp(-c_ij) if math.isfinite(c_ij) else float("nan")
                    P_best[i][j] = P_val
                    P_best[j][i] = P_val
                    _report_progress(1)
    score_best = maximin_chernoff(C_best) if C_best else float("nan")

    result = {
        "score": score_best,
        "C": C_best,
        "P_err_bound": P_best,
        "agent_params": agent_params,
        "agent_kind": agent_kind,
        "observation": observation,
        "obs_temp": obs_temp,
        "obs_bias": obs_bias,
    }
    if not light_results:
        result["design"] = best_design
        result["laplace"] = lap_best
    if include_models:
        result["models"] = models
    if VERBOSE:
        print(f"[dbg] run_agent done seed={seed} agent={agent_kind} score={score_best}", file=sys.stderr, flush=True)
    if progress_queue is not None and progress_total is not None:
        remaining = progress_total - progress_sent
        if remaining > 0:
            progress_queue.put(remaining)
    return result


def _run_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper de parallélisation pour exécuter un agent sans renvoyer les modèles (pickle-friendly)."""
    if VERBOSE:
        print(f"[dbg] worker start seed={task['seed']} agent={task['agent_kind']}", file=sys.stderr, flush=True)
    return {
        "seed": task["seed"],
        **run_agent(
            seed=task["seed"],
            n_jobs=task.get("laplace_jobs", 1),
            agent_kind=task["agent_kind"],
            include_models=False,
            progress_queue=task.get("progress_queue"),
            progress_total=task.get("progress_total"),
            sigma=task.get("sigma", 0.05),
            light_results=task.get("light_results", False),
            t_max=task.get("t_max", 7200.0),
            t_step=task.get("t_step", 30.0),
            n_bonus=task.get("n_bonus", 20),
            n_t=task.get("n_t", 24),
            forced_block=task.get("forced_block", False),
            forced_t_max=task.get("forced_t_max", 900.0),
            forced_t_step=task.get("forced_t_step", 60.0),
            forced_n_t=task.get("forced_n_t", 9),
            forced_type_seq=task.get("forced_type_seq"),
            observation=task.get("observation", "sigmoid"),
            obs_temp_mean=task.get("obs_temp_mean", 1.0),
            obs_temp_std=task.get("obs_temp_std", 0.0),
            obs_bias_mean=task.get("obs_bias_mean", 0.0),
            obs_bias_std=task.get("obs_bias_std", 0.0),
        ),
    }


if __name__ == "__main__":
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        tqdm = None  # type: ignore

    import argparse

    parser = argparse.ArgumentParser(description="Agent comparison (Markov/Magneto)")
    parser.add_argument("--seeds", type=str, default="1203,2203,3203", help="liste de seeds séparées par virgules")
    parser.add_argument("--agents", type=str, default="markov,magneto", help="liste d'agents (markov,magneto)")
    parser.add_argument("--t-max", type=float, default=5400.0, help="t_max du bloc libre")
    parser.add_argument("--t-step", type=float, default=30.0, help="pas des temps fixes du bloc libre")
    parser.add_argument("--n-bonus", type=int, default=18, help="nombre de bonus du bloc libre")
    parser.add_argument("--n-t", type=int, default=18, help="nombre de temps de mesure du bloc libre")
    parser.add_argument("--forced-block", action="store_true", help="active un bloc force avant le bloc libre")
    parser.add_argument("--forced-t-max", type=float, default=900.0, help="t_max du bloc force")
    parser.add_argument("--forced-t-step", type=float, default=60.0, help="pas des temps fixes du bloc force")
    parser.add_argument("--forced-n-t", type=int, default=9, help="nombre de temps de mesure du bloc force")
    parser.add_argument("--forced-type-seq", type=str, default="1,2,3", help="sequence de types (1-based ou 0-based)")
    parser.add_argument("--laplace-jobs", type=int, default=2, help="n_jobs pour laplace_jsd (par tâche)")
    parser.add_argument("--sigma", type=float, default=0.1, help="bruit d'observation (sigma) pour Laplace")
    parser.add_argument("--light-results", action="store_true", help="ne stocke pas laplace/design pour economiser la RAM")
    parser.add_argument(
        "--n-iter",
        type=int,
        default=0,
        help="nombre d'initialisations aleatoires par agent (override --seeds si > 0)",
    )
    parser.add_argument(
        "--observation",
        type=str,
        default="sigmoid",
        choices=["identity", "sigmoid"],
        help="fonction d'observation appliquee a h(t)",
    )
    parser.add_argument("--obs-temp-mean", type=float, default=4.0, help="moyenne de la temperature sigmoide")
    parser.add_argument("--obs-temp-std", type=float, default=0.9, help="ecart-type de la temperature sigmoide")
    parser.add_argument("--obs-bias-mean", type=float, default=-0.0699, help="moyenne du biais sigmoide")
    parser.add_argument("--obs-bias-std", type=float, default=0.46, help="ecart-type du biais sigmoide")
    parser.add_argument("--workers", type=int, default=0, help="max workers pour la parallelisation (0 = auto)")
    args = parser.parse_args()

    if int(args.n_iter) > 0:
        rng_seeds = random.Random()
        seeds = [rng_seeds.randint(0, 2**31 - 1) for _ in range(int(args.n_iter))]
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    agent_kinds = [s.strip() for s in args.agents.split(",") if s.strip()]
    forced_type_seq = [int(s.strip()) for s in args.forced_type_seq.split(",") if s.strip()]

    results: List[Dict[str, Any]] = []
    last_by_kind: Dict[str, Dict[str, Any]] = {}

    # Progression par couple (agent_kind, seed) avec une barre par tâche
    tasks = [
        {
            "seed": sd,
            "agent_kind": ak,
            "t_max": float(args.t_max),
            "t_step": float(args.t_step),
            "n_bonus": int(args.n_bonus),
            "n_t": int(args.n_t),
            "laplace_jobs": int(args.laplace_jobs),
            "sigma": float(args.sigma),
            "light_results": bool(args.light_results),
            "forced_block": bool(args.forced_block),
            "forced_t_max": float(args.forced_t_max),
            "forced_t_step": float(args.forced_t_step),
            "forced_n_t": int(args.forced_n_t),
            "forced_type_seq": forced_type_seq,
            "observation": str(args.observation),
            "obs_temp_mean": float(args.obs_temp_mean),
            "obs_temp_std": float(args.obs_temp_std),
            "obs_bias_mean": float(args.obs_bias_mean),
            "obs_bias_std": float(args.obs_bias_std),
        }
        for ak in agent_kinds
        for sd in seeds
    ]
    # Parallélisation sur les couples (agent_kind, seed)
    if int(args.workers) > 0:
        max_workers = min(len(tasks), int(args.workers))
    else:
        max_workers = min(len(tasks), os.cpu_count() or 1)
    if VERBOSE:
        print(f"[dbg] submitting {len(tasks)} tasks with max_workers={max_workers}", file=sys.stderr, flush=True)

    progress_queue = None
    progress_thread = None
    progress_bar = None
    manager = None
    if tqdm is not None and tasks:
        n_models = len(build_six_models())
        n_pairs = n_models * (n_models - 1) // 2
        per_task_total = n_models + n_pairs
        progress_total = per_task_total * len(tasks)
        if progress_total > 0:
            from multiprocessing import Manager
            import threading

            manager = Manager()
            progress_queue = manager.Queue()
            progress_bar = tqdm(total=progress_total, desc="Global")

            def _progress_worker() -> None:
                while True:
                    delta = progress_queue.get()
                    if delta is None:
                        break
                    progress_bar.update(int(delta))

            progress_thread = threading.Thread(target=_progress_worker, daemon=True)
            progress_thread.start()
            for task in tasks:
                task["progress_queue"] = progress_queue
                task["progress_total"] = per_task_total

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_run_task, task): task for task in tasks}
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                last_by_kind[res.get("agent_kind", "")] = res
    finally:
        if progress_queue is not None:
            progress_queue.put(None)
            if progress_thread is not None:
                progress_thread.join()
            if progress_bar is not None:
                progress_bar.close()
            if manager is not None:
                manager.shutdown()

    # Impression des scores/matrices
    for res in results:
        print(f"[result] Agent={res.get('agent_kind')} | seed={res['seed']} | Chernoff maximin score: {res['score']}")
        if res.get("C"):
            print("C matrix (rounded):")
            for row in res["C"]:
                print([round(x, 3) for x in row])

    # Statistiques Chernoff (moyenne + pire cas) + P(correct|i) par agent
    try:
        import numpy as np  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        np = None  # type: ignore
        plt = None  # type: ignore

    if np is not None and plt is not None:
        by_kind: Dict[str, Dict[str, Any]] = {}
        for res in results:
            kind = res.get("agent_kind", "")
            if not kind:
                continue
            B = res.get("P_err_bound")
            if not B and res.get("C"):
                try:
                    C = np.asarray(res["C"], dtype=float)
                    B = (0.5 * np.exp(-C)).tolist()
                except Exception:
                    B = None
            if not B:
                continue
            entry = by_kind.setdefault(kind, {"B_list": [], "p_correct": []})
            B_arr = np.asarray(B, dtype=float)
            entry["B_list"].append(B_arr)
            m = B_arr.shape[0]
            if not entry["p_correct"]:
                entry["p_correct"] = [[] for _ in range(m)]
            for i in range(m):
                err_i = float(np.nansum([B_arr[i, j] for j in range(m) if j != i]))
                p_corr = max(0.0, 1.0 - err_i)
                entry["p_correct"][i].append(p_corr)

        for kind, entry in by_kind.items():
            B_list = entry.get("B_list", [])
            if not B_list:
                continue
            stack = np.stack(B_list, axis=0)
            mean_B = np.nanmean(stack, axis=0)
            worst_B = np.nanmax(stack, axis=0)
            vmax = float(np.nanmax(worst_B)) if np.isfinite(worst_B).any() else 1.0
            fig_b, axes_b = plt.subplots(1, 2, figsize=(10, 4))
            ax_mean, ax_worst = axes_b
            im0 = ax_mean.imshow(mean_B, cmap="viridis", vmin=0.0, vmax=vmax)
            ax_mean.set_title(f"Chernoff bound mean - {kind}")
            ax_mean.set_xlabel("j (alt model)")
            ax_mean.set_ylabel("i (true model)")
            fig_b.colorbar(im0, ax=ax_mean, fraction=0.046, pad=0.04)
            im1 = ax_worst.imshow(worst_B, cmap="viridis", vmin=0.0, vmax=vmax)
            ax_worst.set_title(f"Chernoff bound worst-case - {kind}")
            ax_worst.set_xlabel("j (alt model)")
            ax_worst.set_ylabel("i (true model)")
            fig_b.colorbar(im1, ax=ax_worst, fraction=0.046, pad=0.04)
            fig_b.tight_layout()
            plt.show()

            p_correct = entry.get("p_correct", [])
            if p_correct:
                m = len(p_correct)
                fig_p, ax_p = plt.subplots(1, 1, figsize=(7, 4))
                data = [np.asarray(vals, dtype=float) for vals in p_correct]
                ax_p.violinplot(data, showmeans=False, showextrema=True)
                means = [float(np.nanmean(v)) if v.size else float("nan") for v in data]
                stds = [float(np.nanstd(v)) if v.size else float("nan") for v in data]
                xs = np.arange(1, m + 1)
                ax_p.errorbar(xs, means, yerr=stds, fmt="o", color="k", capsize=3, label="mean±std")
                for i, v in enumerate(data, start=1):
                    if v.size:
                        jitter = (np.random.rand(v.size) - 0.5) * 0.1
                        ax_p.scatter(np.full_like(v, i, dtype=float) + jitter, v, s=12, alpha=0.4, color="tab:blue")
                ax_p.set_title(f"P(correct|i) distribution - {kind}")
                ax_p.set_xlabel("model i")
                ax_p.set_ylabel("P(correct|i) lower bound")
                ax_p.set_ylim(0.0, 1.0)
                ax_p.legend(loc="best", fontsize=8)
                fig_p.tight_layout()
                plt.show()

    # Plot summary pour le dernier run de chaque agent_kind (reconstruit les models localement)
    for agent_kind, res in last_by_kind.items():
        if res and res.get("C") and res.get("design"):
            try:
                plot_summary(
                    res["C"],
                    res["design"],
                    build_six_models(observation=res.get("observation", "identity")),
                    title=f"Agent={agent_kind} | seed={res.get('seed')}",
                    b_matrix=res.get("P_err_bound"),
                )
            except Exception as exc:
                print(f"[warn] unable to plot summary for {agent_kind}: {exc}")
