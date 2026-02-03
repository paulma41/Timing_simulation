# TO CHECK

from __future__ import annotations

import math
import os
import random
import sys
from datetime import datetime
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


def _parse_magneto_model_idx(agent_kind: str) -> Optional[int]:
    # Accepted forms: "magneto0".."magneto5", "magneto-0", "magneto_model_0"
    s = agent_kind.strip().lower()
    if s == "magneto":
        return None
    for prefix in ("magneto_model_", "magneto-model-", "magneto-", "magneto"):
        if s.startswith(prefix):
            tail = s[len(prefix) :].strip()
            if tail.isdigit():
                return int(tail)
    return None


def _save_figs_if_configured(prefix: str) -> None:
    save_dir = os.environ.get("SAVE_FIGS_DIR", "").strip()
    if not save_dir:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.environ.get("SAVE_FIGS_PREFIX", "").strip() or prefix
    for i, num in enumerate(plt.get_fignums(), start=1):
        fig = plt.figure(num)
        name = f"{base}_{ts}_{i:02d}.png"
        fig.savefig(os.path.join(save_dir, name), dpi=150, bbox_inches="tight")
    plt.close("all")


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
    sample_sim_priors: bool = True,
    disable_h0_wtime: bool = False,
    compare_bounds: bool = False,
) -> Dict[str, Any]:
    if VERBOSE:
        print(f"[dbg] run_agent start seed={seed} agent={agent_kind} n_jobs={n_jobs}", file=sys.stderr, flush=True)
    rng = random.Random(seed)
    all_models = build_six_models(observation=observation)
    magneto_idx = _parse_magneto_model_idx(agent_kind)
    # Always evaluate separability on the full set of 6 models.
    models = all_models
    # But let the magneto agent's internal "mood" be computed from one chosen model.
    mood_model = None
    if magneto_idx is not None:
        if magneto_idx < 0 or magneto_idx >= len(all_models):
            raise ValueError(f"magneto model idx out of range: {magneto_idx}")
        mood_model = all_models[magneto_idx]
    obs_temp = float(obs_temp_mean)
    obs_bias = float(obs_bias_mean)
    if observation == "sigmoid":
        if obs_temp_std > 0.0:
            obs_temp = rng.gauss(float(obs_temp_mean), float(obs_temp_std))
        if obs_bias_std > 0.0:
            obs_bias = rng.gauss(float(obs_bias_mean), float(obs_bias_std))
    sampled_params: Dict[str, float] = {}
    if sample_sim_priors and (not disable_h0_wtime) and all_models and isinstance(all_models[0].sim_priors, dict):
        for pname in ("h0", "W_time"):
            prior = all_models[0].sim_priors.get(pname)
            if isinstance(prior, dict):
                prior_mu = float(prior.get("mu", 0.0))
                prior_sigma = float(prior.get("sigma", 0.0))
                if prior_sigma > 0.0:
                    sampled_params[pname] = rng.gauss(prior_mu, prior_sigma)
                else:
                    sampled_params[pname] = prior_mu

    for f in all_models:
        f.parameters["obs_temp"] = obs_temp
        f.parameters["obs_bias"] = obs_bias
        if isinstance(f.sim_priors, dict):
            f.sim_priors["obs_temp"] = {"dist": "Normal", "mu": obs_temp, "sigma": 0.0}
            f.sim_priors["obs_bias"] = {"dist": "Normal", "mu": obs_bias, "sigma": 0.0}
        if disable_h0_wtime and isinstance(f.sim_priors, dict):
            f.parameters["h0"] = 0.0
            f.parameters["W_time"] = 0.0
            if "h0" in f.sim_priors:
                f.sim_priors["h0"]["mu"] = 0.0
                f.sim_priors["h0"]["sigma"] = 0.0
            if "W_time" in f.sim_priors:
                f.sim_priors["W_time"]["mu"] = 0.0
                f.sim_priors["W_time"]["sigma"] = 0.0
        elif sampled_params:
            for pname, val in sampled_params.items():
                f.parameters[pname] = val

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
            if agent_kind.startswith("magneto"):
                action_types, actions_seg, agent_params_seg, magneto_state = magneto_like_actions_with_value_learning(
                    rng,
                    t_fixed_seg,
                    guard_after_t,
                    progress_cb=progress_cb,
                    forced_type=t_type,
                    state=magneto_state,
                    params_init=magneto_params,
                    mood_model=mood_model,
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

    if agent_kind.startswith("magneto"):
        action_types_free, actions_free, agent_params_free, magneto_state = magneto_like_actions_with_value_learning(
            rng,
            t_fixed_free,
            guard_after_t,
            progress_cb=progress_cb,
            state=magneto_state,
            params_init=magneto_params,
            mood_model=mood_model,
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
        "magneto_model_idx": magneto_idx,
        "observation": observation,
        "obs_temp": obs_temp,
        "obs_bias": obs_bias,
        "n_t": int(n_t),
        "n_bonus": int(n_bonus),
        "sampled_params": sampled_params,
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
            sample_sim_priors=task.get("sample_sim_priors", True),
            disable_h0_wtime=task.get("disable_h0_wtime", False),
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
        "--sample-sim-priors",
        action="store_true",
        default=True,
        help="sample h0 and W_time from their priors for simulation",
    )
    parser.add_argument(
        "--no-sample-sim-priors",
        action="store_false",
        dest="sample_sim_priors",
        help="disable sampling h0 and W_time from priors",
    )
    parser.add_argument(
        "--disable-h0-wtime",
        action="store_true",
        help="force h0 and W_time to 0 and set their prior sigma to 0",
    )
    parser.add_argument(
        "--compare-bounds",
        action="store_true",
        help="affiche les heatmaps Chernoff vs Laplace-Chernoff (b_LC) et leur difference",
    )
    parser.add_argument(
        "--grid-nt-nbonus",
        action="store_true",
        help="sweep n_t and n_bonus on a 0..18 grid and plot mean/worst Chernoff bound per model pair",
    )
    parser.add_argument("--workers", type=int, default=0, help="max workers for parallelization (0=auto)")
    parser.add_argument(
        "--magneto-by-model",
        action="store_true",
        help="expand 'magneto' into 6 agents magneto0..magneto5, each simulated and evaluated on its own model only",
    )
    args = parser.parse_args()

    if bool(args.grid_nt_nbonus):
        # Sweep n_t, n_bonus on a 0..18 grid with 10 agents per type (7 types total).
        seeds = list(range(10))
    elif int(args.n_iter) > 0:
        rng_seeds = random.Random()
        seeds = [rng_seeds.randint(0, 2**31 - 1) for _ in range(int(args.n_iter))]
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if bool(args.grid_nt_nbonus):
        agent_kinds = ["markov"] + [f"magneto{i}" for i in range(6)]
    else:
        agent_kinds = [s.strip() for s in args.agents.split(",") if s.strip()]
    if bool(args.magneto_by_model):
        expanded: List[str] = []
        for ak in agent_kinds:
            if ak.strip().lower() == "magneto":
                expanded.extend([f"magneto{i}" for i in range(6)])
            else:
                expanded.append(ak)
        agent_kinds = expanded
    forced_type_seq = [int(s.strip()) for s in args.forced_type_seq.split(",") if s.strip()]

    if bool(args.grid_nt_nbonus):
        tasks = []
        for n_t in range(19):
            for n_bonus in range(19):
                for ak in agent_kinds:
                    for sd in seeds:
                        tasks.append(
                            {
                                "seed": sd,
                                "agent_kind": ak,
                                "t_max": float(args.t_max),
                                "t_step": float(args.t_step),
                                "n_bonus": int(n_bonus),
                                "n_t": int(n_t),
                                "laplace_jobs": int(args.laplace_jobs),
                                "sigma": float(args.sigma),
                                "light_results": True,
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
                                "sample_sim_priors": bool(args.sample_sim_priors),
                                "disable_h0_wtime": bool(args.disable_h0_wtime),
                                "compare_bounds": False,
                            }
                        )
    else:
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
                "sample_sim_priors": bool(args.sample_sim_priors),
                "disable_h0_wtime": bool(args.disable_h0_wtime),
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

    if not bool(args.grid_nt_nbonus):
        for res in results:
            print(f"[result] Agent={res.get('agent_kind')} | seed={res['seed']} | Chernoff maximin score: {res['score']}")
            if res.get("C"):
                print("C matrix (rounded):")
                for row in res["C"]:
                    print([round(x, 3) for x in row])

    if bool(args.grid_nt_nbonus):
        try:
            import numpy as np  # type: ignore
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            np = None  # type: ignore
            plt = None  # type: ignore

        if np is None or plt is None:
            sys.exit(0)

        m = 6
        mean_grid = np.full((m, m, 19, 19), np.nan, dtype=float)
        worst_grid = np.full((m, m, 19, 19), np.nan, dtype=float)
        bins: Dict[tuple[int, int], List[np.ndarray]] = {}
        for res in results:
            B = res.get("P_err_bound")
            if not B:
                continue
            n_t = int(res.get("n_t", 0))
            n_bonus = int(res.get("n_bonus", 0))
            key = (n_t, n_bonus)
            bins.setdefault(key, []).append(np.asarray(B, dtype=float))

        for (n_t, n_bonus), mats in bins.items():
            stack = np.stack(mats, axis=0)
            mean_B = np.nanmean(stack, axis=0)
            worst_B = np.nanmax(stack, axis=0)
            for i in range(m):
                for j in range(m):
                    mean_grid[i, j, n_t, n_bonus] = mean_B[i, j]
                    worst_grid[i, j, n_t, n_bonus] = worst_B[i, j]

        def _plot_grid(grid: np.ndarray, title: str) -> None:
            fig = plt.figure(figsize=(14, 14))
            gs = fig.add_gridspec(nrows=m, ncols=m, wspace=0.15, hspace=0.2)
            vmin = float(np.nanmin(grid)) if np.isfinite(grid).any() else 0.0
            vmax = float(np.nanmax(grid)) if np.isfinite(grid).any() else 1.0
            last_im = None
            for i in range(m):
                for j in range(m):
                    ax = fig.add_subplot(gs[i, j])
                    if i == j:
                        ax.axis("off")
                        continue
                    mat = grid[i, j]
                    last_im = ax.imshow(
                        mat,
                        origin="lower",
                        vmin=vmin,
                        vmax=vmax,
                        aspect="auto",
                        extent=[0, 18, 0, 18],
                        cmap="viridis",
                    )
                    if i == m - 1:
                        ax.set_xlabel("n_bonus")
                    if j == 0:
                        ax.set_ylabel("n_t")
                    ax.set_title(f"{i}->{j}", fontsize=8)
                    ax.tick_params(labelsize=7)
            if last_im is not None:
                fig.colorbar(last_im, ax=fig.get_axes(), fraction=0.015, pad=0.02)
            fig.suptitle(title, fontsize=12)
            fig.tight_layout()
            plt.show()

        _plot_grid(mean_grid, "Mean Chernoff bound per pair over (n_t, n_bonus)")
        _plot_grid(worst_grid, "Worst-case Chernoff bound per pair over (n_t, n_bonus)")
        if os.environ.get("SAVE_FIGS_DIR"):
            _save_figs_if_configured("grid_nt_nbonus")
        sys.exit(0)

    if int(args.n_iter) > 0:
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
                if kind.lower().startswith("magneto"):
                    kind = "magneto"
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
                entry = by_kind.setdefault(kind, {"B_list": [], "p_correct": [], "p_correct_meta": []})
                B_arr = np.asarray(B, dtype=float)
                entry["B_list"].append(B_arr)
                m = B_arr.shape[0]
                if not entry["p_correct"]:
                    entry["p_correct"] = [[] for _ in range(m)]
                    entry["p_correct_meta"] = [[] for _ in range(m)]
                for i in range(m):
                    err_i = float(np.nansum([B_arr[i, j] for j in range(m) if j != i]))
                    p_corr = max(0.0, 1.0 - err_i)
                    entry["p_correct"][i].append(p_corr)
                    if kind == "magneto":
                        entry["p_correct_meta"][i].append(res.get("magneto_model_idx"))

            for kind, entry in by_kind.items():
                B_list = entry.get("B_list", [])
                if not B_list:
                    continue
                stack = np.stack(B_list, axis=0)
                mean_B = np.nanmean(stack, axis=0)
                worst_B = np.nanmax(stack, axis=0)
                vmax = float(np.nanmax(worst_B)) if np.isfinite(worst_B).any() else 1.0
                p_correct = entry.get("p_correct", [])
                p_correct_meta = entry.get("p_correct_meta", [])
                fig = plt.figure(figsize=(10, 8))
                gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.25)
                ax_mean = fig.add_subplot(gs[0, 0])
                ax_worst = fig.add_subplot(gs[0, 1])
                ax_p = fig.add_subplot(gs[1, :])

                im0 = ax_mean.imshow(mean_B, cmap="viridis", vmin=0.0, vmax=vmax)
                ax_mean.set_title(f"Chernoff bound mean - {kind}")
                ax_mean.set_xlabel("j (alt model)")
                ax_mean.set_ylabel("i (true model)")
                fig.colorbar(im0, ax=ax_mean, fraction=0.046, pad=0.04)
                im1 = ax_worst.imshow(worst_B, cmap="viridis", vmin=0.0, vmax=vmax)
                ax_worst.set_title(f"Chernoff bound worst-case - {kind}")
                ax_worst.set_xlabel("j (alt model)")
                ax_worst.set_ylabel("i (true model)")
                fig.colorbar(im1, ax=ax_worst, fraction=0.046, pad=0.04)

                if p_correct:
                    m = len(p_correct)
                    data = [np.asarray(vals, dtype=float) for vals in p_correct]
                    ax_p.violinplot(data, showmeans=False, showextrema=True)
                    means = [float(np.nanmean(v)) if v.size else float("nan") for v in data]
                    stds = [float(np.nanstd(v)) if v.size else float("nan") for v in data]
                    xs = np.arange(1, m + 1)
                    ax_p.errorbar(xs, means, yerr=stds, fmt="o", color="k", capsize=3, label="mean±std")
                    for i, v in enumerate(data, start=1):
                        if v.size:
                            jitter = (np.random.rand(v.size) - 0.5) * 0.1
                            if kind == "magneto" and p_correct_meta and i - 1 < len(p_correct_meta):
                                meta = p_correct_meta[i - 1]
                                if meta and len(meta) == len(v):
                                    colors = [
                                        "#0072B2",  # blue
                                        "#E69F00",  # orange
                                        "#009E73",  # green
                                        "#D55E00",  # vermillion
                                        "#CC79A7",  # purple
                                        "#56B4E9",  # light blue
                                    ]
                                    for x_val, y_val, k in zip(np.full_like(v, i, dtype=float) + jitter, v, meta):
                                        idx = 0 if k is None else int(k) % len(colors)
                                        ax_p.scatter([x_val], [y_val], s=12, alpha=0.5, color=colors[idx])
                                else:
                                    ax_p.scatter(np.full_like(v, i, dtype=float) + jitter, v, s=12, alpha=0.4, color="tab:blue")
                            else:
                                ax_p.scatter(np.full_like(v, i, dtype=float) + jitter, v, s=12, alpha=0.4, color="tab:blue")
                    ax_p.axhline(1.0 / 6.0, color="tab:gray", linestyle="--", linewidth=1.0, label="chance (1/6)")
                    ax_p.set_title(f"P(correct|i) distribution - {kind}")
                    ax_p.set_xlabel("model i")
                    ax_p.set_ylabel("P(correct|i) lower bound")
                    ax_p.set_ylim(0.0, 1.0)
                    if kind == "magneto":
                        colors = [
                            "#0072B2",
                            "#E69F00",
                            "#009E73",
                            "#D55E00",
                            "#CC79A7",
                            "#56B4E9",
                        ]
                        handles = [
                            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markersize=6)
                            for c in colors
                        ]
                        labels = [f"magneto{i}" for i in range(len(colors))]
                        ax_p.legend(handles, labels, loc="best", fontsize=8)
                    else:
                        ax_p.legend(loc="best", fontsize=8)
                else:
                    ax_p.text(0.5, 0.5, "No P(correct|i) data", ha="center", va="center")

                fig.tight_layout()
                if os.environ.get("SAVE_FIGS_DIR"):
                    _save_figs_if_configured(f"niter_summary_{kind}")
                else:
                    plt.show()

        # For n_iter runs, only show the aggregated summary plots (no per-run summary).

    if int(args.n_iter) <= 0:
        for agent_kind, res in last_by_kind.items():
            if res and res.get("C") and res.get("design"):
                try:
                    models_plot = build_six_models(observation="identity")
                    sampled = res.get("sampled_params") or {}
                    if sampled:
                        for f in models_plot:
                            for pname, val in sampled.items():
                                f.parameters[pname] = val
                    plot_summary(
                        res["C"],
                        res["design"],
                        models_plot,
                        title=f"Agent={agent_kind} | seed={res.get('seed')}",
                        b_matrix=res.get("P_err_bound"),
                    )
                except Exception as exc:
                    print(f"[warn] unable to plot summary for {agent_kind}: {exc}")
        if os.environ.get("SAVE_FIGS_DIR"):
            _save_figs_if_configured("summary")
