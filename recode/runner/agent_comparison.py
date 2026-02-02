# TO CHECK

from __future__ import annotations

import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence

# Allow running as a script from repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from recode.design.actions import (
    markov_actions_with_types,
    magneto_like_actions_with_value_learning,
    normalize_type_sequence,
)
from recode.design.build import build_design_v5, build_fixed_times
from recode.design.measurement import (
    sample_bonuses,
    sample_measurement_times,
    sample_forced_meas_times,
)
from recode.design_optimizer.chernoff import _pairwise_chernoff_matrix, maximin_chernoff
from recode.design_optimizer.laplace import laplace_predictive
from recode.design_optimizer.dsj import jsd_bound_pair
from recode.design_optimizer.laplace_chernoff import laplace_chernoff_risk
from recode.models.h_actions import build_h_action_function
from recode.plotting.summary import plot_summary

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

VERBOSE = False


def build_six_models(observation: str = "identity"):
    models = []
    for kernel in ["event_weighted", "action_avg"]:
        for update in ["continuous", "event", "action"]:
            f = build_h_action_function(kernel=kernel, update=update, observation=observation)
            models.append(f)
    return models


def run_agent(
    seed: int = 1203,
    n_jobs: int = 4,
    agent_kind: str = "magneto",
    include_models: bool = True,
    progress_cb: Optional[Callable[[float], None]] = None,
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
    compare_bounds: bool = False,
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

    guard_after_t = 0.5
    forced_offset = float(forced_t_max) if forced_block else 0.0
    free_start = forced_offset
    free_end = free_start + float(t_max)

    t_fixed_forced = build_fixed_times(0.0, forced_t_max, forced_t_step) if forced_block else []
    t_fixed_free = build_fixed_times(free_start, free_end, t_step)
    t_fixed = sorted(set(t_fixed_forced + t_fixed_free))

    actions_forced: List[Any] = []
    actions_free: List[Any] = []
    meas_times_forced: List[float] = []
    meas_sources_forced: List[str] = []
    action_types: List[Any] = []
    agent_params: Dict[str, Any] = {}
    magneto_state: Optional[Dict[str, Any]] = None
    magneto_params: Optional[Dict[str, float]] = None

    type_seq = normalize_type_sequence(forced_type_seq, 3) if forced_block else []
    if forced_block and type_seq:
        seg_len = forced_t_max / float(len(type_seq))
        for seg_idx, t_type in enumerate(type_seq):
            seg_start = seg_idx * seg_len
            seg_end = (seg_idx + 1) * seg_len
            t_fixed_seg = build_fixed_times(seg_start, seg_end, forced_t_step)
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
        meas_times_forced, meas_sources_forced = sample_forced_meas_times(
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
        rng, n_bonus=int(n_bonus), t_fixed=t_fixed_free, guard_after_t=guard_after_t, before_margin=8.0
    )

    actions_timed = actions_forced + actions_free
    actions_timed.sort(key=lambda x: x[4])

    meas_times_free, meas_sources_free = sample_measurement_times(
        rng,
        actions_free,
        bonuses_t,
        n_t=int(n_t),
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
        rew_bonus_vals=[1.5, 1.0, 2.0],
        t_fixed=t_fixed,
        meas_times=meas_times_best,
        meas_sources=meas_sources_best,
    )

    lap_best = laplace_predictive(
        models,
        best_design,
        sigma=float(sigma),
        n_jobs=max(1, n_jobs),
        return_lowrank=True,
        compute_full=False,
    )
    mu_best = lap_best.get("mu", [])
    Vy_lowrank = lap_best.get("lowrank")

    C_best: List[List[float]] = []
    P_best: List[List[float]] = []
    if mu_best:
        C_best, P_best = _pairwise_chernoff_matrix(mu_best, None, cov_lowrank=Vy_lowrank)
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
    if compare_bounds and mu_best:
        try:
            import numpy as np  # type: ignore
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            np = None  # type: ignore
            plt = None  # type: ignore
        if np is not None and plt is not None:
            # Recompute full covariances once for b_LC (needs full Vy)
            full_pred = laplace_predictive(
                models,
                best_design,
                sigma=float(sigma),
                n_jobs=max(1, n_jobs),
                return_lowrank=False,
                compute_full=True,
            )
            mu_full = [np.asarray(m, dtype=float).reshape(-1) for m in full_pred.get("mu", [])]
            Vy_full = [np.asarray(v, dtype=float) for v in full_pred.get("Vy", [])]
            m = len(models)
            B_lc = np.zeros((m, m), dtype=float)
            B_jsd = np.zeros((m, m), dtype=float)
            B_lc_prob = np.zeros((m, m), dtype=float)
            for i in range(m):
                for j in range(i + 1, m):
                    if not mu_full or not Vy_full:
                        b_ij = float("nan")
                    else:
                        mu_i = mu_full[i]
                        mu_j = mu_full[j]
                        S_i = Vy_full[i]
                        S_j = Vy_full[j]
                        # priors p=[0.5, 0.5]
                        H = -2.0 * (0.5 * math.log(0.5))
                        mu_bar = 0.5 * (mu_i + mu_j)
                        sum_logdet = 0.5 * logdet_psd(S_i.tolist()) + 0.5 * logdet_psd(S_j.tolist())
                        d_i = (mu_i - mu_bar).reshape(-1, 1)
                        d_j = (mu_j - mu_bar).reshape(-1, 1)
                        mix = 0.5 * (d_i @ d_i.T + S_i) + 0.5 * (d_j @ d_j.T + S_j)
                        logdet_mix = logdet_psd(mix.tolist())
                        b_ij = float(H + 0.5 * (sum_logdet - logdet_mix))
                    B_lc[i, j] = b_ij
                    B_lc[j, i] = b_ij
                    p_ij = 0.5 * math.exp(-b_ij) if math.isfinite(b_ij) else float("nan")
                    B_lc_prob[i, j] = p_ij
                    B_lc_prob[j, i] = p_ij
                    # Jensen-Shannon bound b (as in VBA_JensenShannon, base-2)
                    if not mu_full or not Vy_full:
                        b_jsd = float("nan")
                    else:
                        b_jsd = jsd_bound_pair(mu_full[i], Vy_full[i], mu_full[j], Vy_full[j])
                    B_jsd[i, j] = b_jsd
                    B_jsd[j, i] = b_jsd
            B_ch = np.asarray(P_best, dtype=float) if P_best else np.zeros((m, m), dtype=float)
            # Pairwise plot (i > j)
            pairs = [(i, j) for i in range(m) for j in range(i)]
            xs = list(range(len(pairs)))
            cher = [B_ch[i, j] for i, j in pairs]
            lc_half = [0.5 * B_lc[i, j] for i, j in pairs]
            lc_sq = [0.25 * (B_lc[i, j] ** 2) for i, j in pairs]
            b_half = [0.5 * B_jsd[i, j] for i, j in pairs]
            b_sq = [0.25 * (B_jsd[i, j] ** 2) for i, j in pairs]

            fig_line, ax_line = plt.subplots(1, 1, figsize=(10, 4))
            ax_line.scatter(xs, cher, color="tab:blue", label="Chernoff (P_err)", s=18)
            ax_line.scatter(xs, lc_half, color="tab:orange", label="b_LC/2", marker="o", s=18)
            ax_line.scatter(xs, lc_sq, color="tab:orange", label="b_LC^2/4", marker="x", s=24)
            ax_line.scatter(xs, b_half, color="tab:green", label="b/2", marker="o", s=18)
            ax_line.scatter(xs, b_sq, color="tab:green", label="b^2/4", marker="x", s=24)
            ax_line.set_title("Pairwise bounds (i > j)")
            ax_line.set_xlabel("pair index")
            ax_line.set_ylabel("value")
            ax_line.legend(loc="best", fontsize=8, ncol=2)
            ax_line.grid(True, linestyle=":", alpha=0.4)
            fig_line.tight_layout()
            plt.show()

            # Heatmaps
            fig_hm, axes = plt.subplots(2, 3, figsize=(12, 8))
            axes = axes.ravel()
            mats = [
                (B_ch, "Chernoff (P_err)", "viridis"),
                (0.5 * B_lc, "b_LC/2", "viridis"),
                (0.25 * (B_lc ** 2), "b_LC^2/4", "viridis"),
                (0.5 * B_jsd, "b/2", "viridis"),
                (0.25 * (B_jsd ** 2), "b^2/4", "viridis"),
            ]
            for k, (mat, title, cmap) in enumerate(mats):
                im = axes[k].imshow(mat, cmap=cmap)
                axes[k].set_title(title)
                plt.colorbar(im, ax=axes[k], fraction=0.046, pad=0.04)
            axes[-1].axis("off")
            fig_hm.tight_layout()
            plt.show()
    if VERBOSE:
        print(f"[dbg] run_agent done seed={seed} agent={agent_kind} score={score_best}", file=sys.stderr, flush=True)
    return result


def _run_task(task: Dict[str, Any]) -> Dict[str, Any]:
    if VERBOSE:
        print(f"[dbg] worker start seed={task['seed']} agent={task['agent_kind']}", file=sys.stderr, flush=True)
    return {
        "seed": task["seed"],
        **run_agent(
            seed=task["seed"],
            n_jobs=task.get("laplace_jobs", 1),
            agent_kind=task["agent_kind"],
            include_models=False,
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
            compare_bounds=task.get("compare_bounds", False),
        ),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent comparison (Markov/Magneto)")
    parser.add_argument("--seeds", type=str, default="1203,2203,3203", help="comma-separated seeds")
    parser.add_argument("--agents", type=str, default="markov,magneto", help="agents (markov,magneto)")
    parser.add_argument("--t-max", type=float, default=5400.0, help="t_max for free block")
    parser.add_argument("--t-step", type=float, default=30.0, help="step for fixed times")
    parser.add_argument("--n-bonus", type=int, default=0, help="number of bonus in free block")
    parser.add_argument("--n-t", type=int, default=0, help="number of measurement times in free block")
    parser.add_argument("--forced-block", action="store_true", help="enable forced block before free block")
    parser.add_argument("--forced-t-max", type=float, default=900.0, help="t_max for forced block")
    parser.add_argument("--forced-t-step", type=float, default=60.0, help="step for forced block")
    parser.add_argument("--forced-n-t", type=int, default=9, help="number of measurement times in forced block")
    parser.add_argument("--forced-type-seq", type=str, default="1,2,3", help="type sequence (1-based or 0-based)")
    parser.add_argument("--laplace-jobs", type=int, default=2, help="n_jobs for laplace")
    parser.add_argument("--sigma", type=float, default=0.1, help="observation noise sigma")
    parser.add_argument("--light-results", action="store_true", help="skip laplace/design to save RAM")
    parser.add_argument("--n-iter", type=int, default=0, help="random initializations per agent")
    parser.add_argument("--observation", type=str, default="sigmoid", choices=["identity", "sigmoid"])
    parser.add_argument("--obs-temp-mean", type=float, default=4.0)
    parser.add_argument("--obs-temp-std", type=float, default=0.9)
    parser.add_argument("--obs-bias-mean", type=float, default=-0.0699)
    parser.add_argument("--obs-bias-std", type=float, default=0.46)
    parser.add_argument(
        "--compare-bounds",
        action="store_true",
        help="affiche les heatmaps Chernoff vs Laplace-Chernoff (b_LC) et leur difference",
    )
    parser.add_argument("--workers", type=int, default=0, help="max workers for parallelization (0=auto)")
    args = parser.parse_args()

    if int(args.n_iter) > 0:
        rng_seeds = random.Random()
        seeds = [rng_seeds.randint(0, 2**31 - 1) for _ in range(int(args.n_iter))]
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    agent_kinds = [s.strip() for s in args.agents.split(",") if s.strip()]
    forced_type_seq = [int(s.strip()) for s in args.forced_type_seq.split(",") if s.strip()]

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
            "compare_bounds": bool(args.compare_bounds),
        }
        for ak in agent_kinds
        for sd in seeds
    ]

    if int(args.workers) > 0:
        max_workers = min(len(tasks), int(args.workers))
    else:
        max_workers = min(len(tasks), os.cpu_count() or 1)

    results: List[Dict[str, Any]] = []
    last_by_kind: Dict[str, Dict[str, Any]] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_task, task): task for task in tasks}
        iterator = as_completed(futures)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(futures), desc="agent_comparison")
        for fut in iterator:
            res = fut.result()
            results.append(res)
            last_by_kind[res.get("agent_kind", "")] = res

    for res in results:
        print(f"[result] Agent={res.get('agent_kind')} | seed={res['seed']} | Chernoff maximin score: {res['score']}")
        if res.get("C"):
            print("C matrix (rounded):")
            for row in res["C"]:
                print([round(x, 3) for x in row])

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
