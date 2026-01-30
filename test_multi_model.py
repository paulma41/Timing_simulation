import argparse
import os
import random
import pickle
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from functions import build_h_action_function
from design_optimizer import expected_log_bayes_factor_matrix_for_design


def make_demo_pools() -> Tuple[List[float], List[float]]:
    """Pools de valeurs possibles pour les efforts et rewards."""
    Eff_pool = [-1.0, -0.5, -0.8]
    Rew_pool = [1.5, 1.0, 2.0]
    return Eff_pool, Rew_pool


def build_six_models(observation: str = "identity"):
    models = []
    for kernel in ["event_weighted", "action_avg"]:
        for update in ["continuous", "event", "action"]:
            f = build_h_action_function(kernel=kernel, update=update, observation=observation)
            models.append(f)
    return models


def sample_times_with_gap(
    rng: random.Random,
    n: int,
    t_min: float,
    t_max: float,
    min_gap: float,
) -> List[float]:
    """
    Echantillonne n temps dans [t_min, t_max] avec un espacement minimal min_gap.
    Si le filtrage reduit trop le nombre de points, on repasse a un espacement uniforme.
    """
    if n <= 0:
        return []
    vals = sorted(rng.uniform(t_min, t_max) for _ in range(n))
    spaced: List[float] = []
    last = None
    for v in vals:
        if last is None or v - last >= min_gap:
            spaced.append(v)
            last = v
    if len(spaced) >= max(1, n // 2):
        return spaced
    step = max(min_gap, (t_max - t_min) / max(1, n))
    return [t_min + i * step for i in range(n) if t_min + i * step <= t_max]


def sample_latent_actions(
    rng: random.Random,
    Eff_pool: Sequence[float],
    Rew_pool: Sequence[float],
    N_eff: int,
    t_min: float,
    t_max: float,
    delta_er: float,
) -> Dict[str, List[float]]:
    """
    Echantillonne des variables latentes pour les actions :
    - Eff_idx, Eff_t
    - Rew_idx, Rew_t
    """
    n_eff = min(N_eff, len(Eff_pool), len(Rew_pool))
    eff_indices = rng.sample(range(len(Eff_pool)), n_eff)
    rew_indices = rng.sample(range(len(Rew_pool)), n_eff)

    Eff_t = sorted(rng.uniform(t_min, t_max - delta_er) for _ in range(n_eff))
    Rew_t: List[float] = []
    for te in Eff_t:
        tr = rng.uniform(te + delta_er, t_max)
        Rew_t.append(tr)

    return {
        "Eff_idx": eff_indices,
        "Eff_t": Eff_t,
        "Rew_idx": rew_indices,
        "Rew_t": Rew_t,
    }


def decode_design(
    t: List[float],
    latent_actions: Dict[str, List[float]],
    Eff_pool: Sequence[float],
    Rew_pool: Sequence[float],
    t_min: float,
    t_max: float,
    min_gap: float,
    delta_er: float,
) -> Dict[str, Any]:
    """
    Projette les variables latentes (t, Eff_idx/Eff_t, Rew_idx/Rew_t)
    en un design complet (t, Eff_, Rew_, A).
    """
    rng = random.Random()
    t_proj = sample_times_with_gap(rng, len(t), t_min, t_max, min_gap)

    Eff_idx = latent_actions["Eff_idx"]
    Eff_t_raw = latent_actions["Eff_t"]
    Rew_idx = latent_actions["Rew_idx"]
    Rew_t_raw = latent_actions["Rew_t"]

    n_eff = min(len(Eff_idx), len(Eff_t_raw), len(Rew_idx), len(Rew_t_raw))
    Eff_t: List[float] = []
    Rew_t: List[float] = []
    for k in range(n_eff):
        te = max(t_min, min(t_max - delta_er, Eff_t_raw[k]))
        tr = max(te + delta_er, min(t_max, Rew_t_raw[k]))
        Eff_t.append(te)
        Rew_t.append(tr)

    Eff_val = [Eff_pool[i] for i in Eff_idx[:n_eff]]
    Rew_val = [Rew_pool[j] for j in Rew_idx[:n_eff]]

    Eff_ = list(zip(Eff_val, Eff_t))
    Rew_ = list(zip(Rew_val, Rew_t))
    A = list(zip(Eff_, Rew_))

    return {
        "t": t_proj,
        "Eff_": Eff_,
        "Rew_": Rew_,
        "A": A,
    }


def _build_fixed_times(t_min: float, t_max: float, step: float) -> List[float]:
    """Construit les temps fixes t_min, t_min+step, ..., t_max (inclus)."""
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


def decode_design_structured(
    x: List[float],
    *,
    t_min: float,
    t_max: float,
    min_gap_t: float,
    delta_er: float,
    Eff_pool: Sequence[float],
    Rew_pool: Sequence[float],
    t_step: float,
    rng: random.Random,
) -> Dict[str, Any]:
    """
    Structure de design : t fixes tous les t_step + un t optimise par intervalle,
    un seul couple Eff/Rew par intervalle (toutes les combinaisons sont couvertes).
    """
    t_fixed = _build_fixed_times(t_min, t_max, t_step)
    n_intervals = max(0, len(t_fixed) - 1)
    if len(x) != n_intervals:
        raise ValueError(f"Expected {n_intervals} t values, got {len(x)}")

    t_opt: List[float] = []
    for i in range(n_intervals):
        low = t_fixed[i] + min_gap_t
        high = t_fixed[i + 1] - min_gap_t
        if high <= low:
            t_val = (t_fixed[i] + t_fixed[i + 1]) / 2.0
        else:
            t_val = max(low, min(high, x[i]))
        t_opt.append(t_val)

    t_all = sorted(set(t_fixed + t_opt))

    combos = [(e, r) for e in Eff_pool for r in Rew_pool]
    rng.shuffle(combos)

    Eff_: List[Tuple[float, float]] = []
    Rew_: List[Tuple[float, float]] = []
    A: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    for i in range(n_intervals):
        start = t_fixed[i]
        end = t_fixed[i + 1]
        if end - start <= delta_er:
            continue
        eff_val, rew_val = combos[i % len(combos)]
        eff_t = rng.uniform(start, end - delta_er)
        rew_t = rng.uniform(eff_t + delta_er, end)
        Eff_.append((eff_val, eff_t))
        Rew_.append((rew_val, rew_t))
        A.append(((eff_val, eff_t), (rew_val, rew_t)))

    return {"t": t_all, "Eff_": Eff_, "Rew_": Rew_, "A": A}


def plot_summary(U, best_inputs, models, title="Jeffreys multi-model summary", show_indices=None, b_matrix=None):
    try:
        import matplotlib.pyplot as plt

        import numpy as np
        from matplotlib.widgets import CheckButtons
    except Exception:
        print("[diag] matplotlib non installe : skip du plot resume.")
        return
    U = U or []
    m = len(models)

    if b_matrix is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False)
        flat_axes = axes.ravel()
        ax, ax_b, ax2, ax3 = flat_axes[:4]
    else:
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)
        ax, ax2, ax3 = axes

    # Heatmap of U
    if U and len(U) == m:
        M = np.array(U, dtype=float)
        im = ax.imshow(M, cmap='viridis')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"U matrix (log-BF expected) - {title}")
        ax.set_xlabel("j (alt model)")
        ax.set_ylabel("i (true model)")
    else:
        ax.text(0.5, 0.5, "No U matrix", ha='center')

    # Optional b_matrix heatmap (here used for Chernoff error bounds)
    if b_matrix is not None:
        try:
            B = np.asarray(b_matrix, dtype=float)
        except Exception:
            B = None
        if B is not None and B.shape[0] == m and B.shape[1] == m:
            im_b = ax_b.imshow(B, cmap='viridis', vmin = 0.0, vmax = 1/m)
            fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
            ax_b.set_title(f"Chernoff error bound matrix - {title}")
            ax_b.set_xlabel("j (alt model)")
            ax_b.set_ylabel("i (true model)")
        else:
            ax_b.text(0.5, 0.5, "No b matrix", ha='center')
    # Plot h(t) for each model with current params on best t
    t = best_inputs.get('t', []) if isinstance(best_inputs, dict) else []
    if not t:
        ax2.text(0.5, 0.5, "No t to plot", ha='center')
    else:
        Eff_ = best_inputs.get('Eff_', [])
        Rew_ = best_inputs.get('Rew_', [])
        A = best_inputs.get('A', [])

        import numpy as np
        t_min_plot, t_max_plot = 0.0, max(t)
        t_measured = best_inputs.get('t_meas', [])
        # grille dense incluant explicitement les temps de mesure
        base_grid = np.linspace(t_min_plot, t_max_plot, max(1, int(2 * t_max_plot) + 1)).tolist()
        t_grid = sorted(set(base_grid) | set(t_measured))

        indices = list(range(len(models))) if show_indices is None else [i for i in show_indices if 0 <= i < len(models)]
        model_lines = []
        model_labels = []
        for idx in indices:
            f = models[idx]
            y_full = f.eval({'t': t_grid, 'Eff_': Eff_, 'Rew_': Rew_, 'A': A, 'A_typed': best_inputs.get('A_typed', []), 'S1': Eff_, 'S2': Rew_})
            if not isinstance(y_full, list):
                y_full = [y_full]
            line = ax2.plot(t_grid, y_full, linestyle='--', linewidth=1.2, label=f.name)
            color = line[0].get_color()
            model_lines.append(line[0])
            model_labels.append(f.name)
            if t_measured:
                y_meas = f.eval({'t': t_measured, 'Eff_': Eff_, 'Rew_': Rew_, 'A': A, 'A_typed': best_inputs.get('A_typed', []), 'S1': Eff_, 'S2': Rew_})
                ax2.plot(t_measured, y_meas, marker='x', linestyle='none', color=color)


        ax2.set_title("h(t) on best design (dashed=continuous, x=measured)")
        ax2.set_xlabel("t")
        ax2.set_ylabel("h(t)")
        ax2.grid(True, linestyle=':', alpha=0.4)

        if model_lines:
            ax2.legend(model_lines, model_labels, fontsize=8, ncol=2)
    Eff_ = best_inputs.get('Eff_', []) if isinstance(best_inputs, dict) else []
    Rew_ = best_inputs.get('Rew_', []) if isinstance(best_inputs, dict) else []
    if Eff_:
        ex = [et for _, et in Eff_]
        ey = [ev for ev, _ in Eff_]
        ax3.scatter(ex, ey, color='tab:red', label='Eff_', marker='x')
        for x, y in zip(ex, ey):
            ax3.text(x, y, f"{y:.3g}", fontsize=7, color='tab:red', va='bottom', ha='left')
    if Rew_:
        rx = [rt for _, rt in Rew_]
        ry = [rv for rv, _ in Rew_]
        ax3.scatter(rx, ry, color='tab:green', label='Rew_', marker='o')
        for x, y in zip(rx, ry):
            ax3.text(x, y, f"{y:.3g}", fontsize=7, color='tab:green', va='bottom', ha='left')
    if isinstance(best_inputs, dict):
        tvals = best_inputs.get('t_meas', None)
        if tvals:
            ax3.scatter(tvals, [0.0]*len(tvals), color='k', marker='|', s=40, label='t')
            for x in tvals:
                ax3.text(x, 0.0, f"{x:.3g}", fontsize=7, color='k', va='bottom', ha='left')
    ax3.axhline(0.0, color='k', linewidth=0.5, alpha=0.5)
    ax3.set_title("Design variables (Eff_, Rew_, t)")
    ax3.set_xlabel("time")
    ax3.set_ylabel("value")
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, linestyle=':', alpha=0.4)

    fig.tight_layout()
    plt.show()

    if isinstance(best_inputs, dict):
        sources = list(best_inputs.get("meas_sources", []) or [])
        if sources:
            labels = ["between_eff_rew", "after_rew", "uniform"]
            counts = [sum(1 for s in sources if s == lbl) for lbl in labels]
            other = len(sources) - sum(counts)
            if other > 0:
                labels.append("other")
                counts.append(other)
            total = max(1, len(sources))
            proportions = [c / total for c in counts]
            fig_src, ax_src = plt.subplots(1, 1, figsize=(6, 3.5))
            ax_src.bar(labels, proportions, color="tab:blue", edgecolor="k", alpha=0.8)
            ax_src.set_ylim(0.0, 1.0)
            ax_src.set_ylabel("proportion")
            ax_src.set_title("Measurement time sources (n_t)")
            for i, (p, c) in enumerate(zip(proportions, counts)):
                ax_src.text(i, p + 0.02, f"{p:.2f} ({c})", ha="center", va="bottom", fontsize=8)
            fig_src.tight_layout()
            plt.show()

    # --- Extra summary figure: histograms (t gaps, Eff values, Rew values) + scatter/violin for Δt ---
    try:
        # Temps de mesure : si t_meas est fourni, on ne plot que ceux-là ; sinon pas de croix.
        t_measured = best_inputs.get('t_meas', None) if isinstance(best_inputs, dict) else None
        t_all      = best_inputs.get('t', None)
        t_all      = sorted(t_all)
        if isinstance(best_inputs, dict) and t_measured is not None:
            t_list = sorted(t_measured)
        else:
            t_list = []
        # Mesures explicites si fournies, sinon on retombe sur t_list
        t_measured = sorted(best_inputs.get('t_meas', t_list)) if isinstance(best_inputs, dict) else []
        gaps = [t_list[i + 1] - t_list[i] for i in range(len(t_list) - 1)] if len(t_list) > 1 else []
        eff_vals = [ev for ev, _ in best_inputs.get('Eff_', [])] if isinstance(best_inputs, dict) else []
        rew_vals = [rv for rv, _ in best_inputs.get('Rew_', [])] if isinstance(best_inputs, dict) else []
        gaps_all = [t_all[i + 1] - t_all[i] for i in range(len(t_list) - 1)] if len(t_list) > 1 else []
        gaps_meas = list(np.diff(t_measured)) if len(t_measured) > 1 else []

        if t_list or eff_vals or rew_vals:
            fig_hist, (ax_gap, ax_val, ax_violin) = plt.subplots(1, 3, figsize=(14, 4))
            # gaps histogram + scatter
            if gaps_all:
                try:
                    bins_gap = np.arange(-0.25, max(gaps_all) + 0.75, 0.5)
                    if len(bins_gap) < 2:
                        bins_gap = 'auto'
                except Exception:
                    bins_gap = 'auto'
                    #ax.set_xticks(np.arange(0.0, max(gaps_all) + 1.))
                counts, bins_used, _ = ax_gap.hist(gaps_all, bins=bins_gap, color='steelblue', alpha=0.7, edgecolor='k')
                max_count = float(max(counts)) if len(counts) > 0 else 1.0

                # Scatter Δt (tous événements)
                jitter_x = np.array(gaps_all)
                jitter_y = np.random.uniform(0.05 * max_count, 0.2 * max_count, size=len(gaps_all))
                ax_gap.scatter(jitter_x, jitter_y, color='black', alpha=0.6, s=8, label='Δt (all t)')

                # Δt (mesures seulement)
                if gaps_meas:
                    jitter_xm = np.array(gaps_meas)
                    jitter_ym = np.random.uniform(0.25 * max_count, 0.45 * max_count, size=len(gaps_meas))
                    ax_gap.scatter(jitter_xm, jitter_ym, color='orange', alpha=0.7, s=10, label='Δt (measures only)')
                    ax_gap.legend(fontsize=8)

                ax_gap.set_title("Δt (successive t)")
                ax_gap.set_xlabel("Δt")
                ax_gap.set_ylabel("count / points")
                ax_gap.set_ylim(bottom=0.0)
            else:
                ax_gap.text(0.5, 0.5, "No successive t", ha='center')

            # values histogram
            if eff_vals or rew_vals:
                if eff_vals:
                    ax_val.hist(eff_vals, bins='auto', color='tab:red', alpha=0.6, edgecolor='k', label='Eff')
                if rew_vals:
                    ax_val.hist(rew_vals, bins='auto', color='tab:green', alpha=0.6, edgecolor='k', label='Rew')
                ax_val.set_title("Histogram of Eff/Rew values")
                ax_val.set_xlabel("value")
                ax_val.set_ylabel("count")
                ax_val.legend()
            else:
                ax_val.text(0.5, 0.5, "No Eff/Rew values", ha='center')
            # violin + scatter pour Δt (mesures uniquement si disponibles)
            violin_data = gaps_meas if gaps_meas else gaps
            if violin_data:
                ax_violin.violinplot(violin_data, showmeans=True, widths=0.7)
                jitter_x = np.random.normal(loc=1.0, scale=0.03, size=len(violin_data))
                ax_violin.scatter(jitter_x, violin_data, color='black', alpha=0.5, s=8)
                ax_violin.set_xticks([1])
                ax_violin.set_xticklabels(["Δt meas" if gaps_meas else "Δt all"])
                ax_violin.set_title("Δt distribution (measures)" if gaps_meas else "Δt distribution (all t)")
                ax_violin.set_ylabel("Δt")
            else:
                ax_violin.text(0.5, 0.5, "No Δt", ha='center')
            fig_hist.tight_layout()
            plt.show()
    except Exception as exc:
        print(f"[diag] unable to plot extra histograms: {exc}")

    # Action choice distribution (type counts + timeline)
    try:
        A_typed = best_inputs.get('A_typed', []) if isinstance(best_inputs, dict) else []
        type_idx: List[int] = []
        t_rew_list: List[float] = []
        for entry in A_typed:
            try:
                k, _eff, rew = entry
                type_idx.append(int(k))
                t_rew_list.append(float(rew[1]))
            except Exception:
                continue
        if type_idx:
            max_type = max(type_idx)
            counts = [0 for _ in range(max_type + 1)]
            for k in type_idx:
                if k >= 0:
                    counts[k] += 1
            xs = list(range(len(counts)))
            fig_act, (ax_count, ax_time) = plt.subplots(1, 2, figsize=(10, 4))
            ax_count.bar(xs, counts, color='tab:blue', alpha=0.7, edgecolor='k')
            ax_count.set_title("Action type counts")
            ax_count.set_xlabel("action type")
            ax_count.set_ylabel("count")
            ax_count.set_xticks(xs)

            ax_time.scatter(t_rew_list, type_idx, s=10, alpha=0.7, color='tab:orange')
            ax_time.set_title("Action types over time (reward time)")
            ax_time.set_xlabel("time (t_rew)")
            ax_time.set_ylabel("action type")
            ax_time.set_yticks(xs)
            ax_time.grid(True, linestyle=':', alpha=0.4)
            fig_act.tight_layout()
            plt.show()
    except Exception as exc:
        print(f"[diag] unable to plot action distribution: {exc}")

    # Additional histograms: distances between t and Eff/Rew times
    try:
        if isinstance(best_inputs, dict):
            t_list = sorted(best_inputs.get('t', []))
            eff_times = [t for _, t in best_inputs.get('Eff_', [])]
            rew_times = [t for _, t in best_inputs.get('Rew_', [])]
        else:
            t_list, eff_times, rew_times = [], [], []

        dist_eff: List[float] = []
        dist_rew: List[float] = []
        if t_list:
            for te in eff_times:
                dist_eff.append(min(abs(te - tv) for tv in t_list))
            for tr in rew_times:
                dist_rew.append(min(abs(tr - tv) for tv in t_list))

        if dist_eff or dist_rew:
            fig_d, (ax_de, ax_dr) = plt.subplots(1, 2, figsize=(10, 4))
            if dist_eff:
                try:
                    bins_de = np.linspace(0.0, max(dist_eff) + 1e-9, num=max(5, int(max(dist_eff)) + 2))
                    if len(bins_de) < 2:
                        bins_de = 'auto'
                except Exception:
                    bins_de = 'auto'
                ax_de.hist(dist_eff, bins=bins_de, color='tab:red', alpha=0.7, edgecolor='k')
                ax_de.set_title("Distance |t_eff - t| (nearest t)")
                ax_de.set_xlabel("distance")
                ax_de.set_ylabel("count")
            else:
                ax_de.text(0.5, 0.5, "No Eff times", ha='center')
            if dist_rew:
                try:
                    bins_dr = np.linspace(0.0, max(dist_rew) + 1e-9, num=max(5, int(max(dist_rew)) + 2))
                    if len(bins_dr) < 2:
                        bins_dr = 'auto'
                except Exception:
                    bins_dr = 'auto'
                ax_dr.hist(dist_rew, bins=bins_dr, color='tab:green', alpha=0.7, edgecolor='k')
                ax_dr.set_title("Distance |t_rew - t| (nearest t)")
                ax_dr.set_xlabel("distance")
                ax_dr.set_ylabel("count")
            else:
                ax_dr.text(0.5, 0.5, "No Rew times", ha='center')
            fig_d.tight_layout()
            plt.show()
    except Exception as exc:
        print(f"[diag] unable to plot distance histograms: {exc}")

    # Print summary from Chernoff matrix if provided (approx. error/prob. correct)
    try:
        if b_matrix is not None:
            import numpy as np  # type: ignore
            B = np.asarray(b_matrix, dtype=float)
            if B.shape[0] == m and B.shape[1] == m:
                # Per-model upper bound on misclassification (union bound)
                per_model_err = []
                for i in range(m):
                    err_i = float(np.nansum([B[i, j] for j in range(m) if j != i]))
                    per_model_err.append(err_i)
                avg_err = float(np.nanmean(per_model_err)) if per_model_err else float("nan")
                max_err = float(np.nanmax(per_model_err)) if per_model_err else float("nan")
                approx_p_correct_avg = max(0.0, 1.0 - avg_err) if math.isfinite(avg_err) else float("nan")
                approx_p_correct_worst = max(0.0, 1.0 - max_err) if math.isfinite(max_err) else float("nan")
                print("[cher] Per-model union-bound error (row sums of B):")
                for i, err_i in enumerate(per_model_err):
                    print(f"  model {i}: err <= {err_i:.3g}, P(correct|i) >= {max(0.0, 1.0-err_i):.3g}")
                print(f"[cher] Avg err <= {avg_err:.3g}  =>  P(correct) >= {approx_p_correct_avg:.3g}")
                print(f"[cher] Worst-case err <= {max_err:.3g}  =>  P(correct) >= {approx_p_correct_worst:.3g}")
    except Exception as exc:
        print(f"[diag] unable to print Chernoff summary: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Multi-model design optimization (Jeffreys)")
    parser.add_argument("--verbose", type=int, default=1, help="verbosity level (0,1,2)")
    parser.add_argument("--budget", type=int, default=20, help="number of candidate designs")
    parser.add_argument("--outer", type=int, default=100, help="K_outer Monte Carlo samples")
    parser.add_argument("--inner", type=int, default=100, help="K_inner Monte Carlo samples per model")
    parser.add_argument("--sigma", type=float, default=0.1, help="obs noise stddev")
    parser.add_argument("--no-progress", action='store_true', help="disable progress bars")
    parser.add_argument("--show-models", type=str, default=None,
                        help="comma-separated indices of models to plot (0-based); default shows all")
    parser.add_argument("--bo", action='store_true', default=True, help="use Bayesian Optimization (skopt) instead of random search (default: on)")
    parser.add_argument("--no-bo", action='store_false', dest='bo', help="disable BO and use random search")
    parser.add_argument("--t-step", type=float, default=30.0, help="fixed step between measurement times")
    parser.add_argument("--min-gap", type=float, default=5.0, help="minimum gap between times")
    parser.add_argument("--delta-er", type=float, default=5.0, help="minimum Eff/Rew separation")
    parser.add_argument("--cache-file", type=str, default=".test_multi_model_cache.pkl", help="pickle file to persist objective evaluations across runs")
    parser.add_argument("--no-convergence-plot", action="store_true", help="disable BO convergence plot")
    parser.add_argument("--pairs", type=str, default=None, help="list of i-j pairs to optimize (comma-separated, e.g. '0-1,2-3'); empty=all pairs")
    parser.add_argument("--cov-plot", action="store_true", help="show covariance heatmap of U[i,j] over random search evaluations")
    parser.add_argument("--mc-convergence", action="store_true", help="run a 5-design Monte Carlo convergence check (vary K_outer checkpoints) and plot scores")
    parser.add_argument("--adaptive-utility", action="store_true", help="use adaptive per-row MC sampling (stops when SE below tol)")
    parser.add_argument("--outer-min", type=int, default=None, help="minimum K_outer for adaptive utility (default: auto)")
    parser.add_argument("--inner-min", type=int, default=None, help="minimum K_inner for adaptive utility (default: auto)")
    parser.add_argument("--tol-rel", type=float, default=0.05, help="relative tolerance for adaptive utility stop")
    parser.add_argument("--tol-abs", type=float, default=0.05, help="absolute tolerance for adaptive utility stop")
    parser.add_argument("--check-covariance", action="store_true", help="force random evaluations and plot convergence + covariance heatmap of U[i,j]")
    args = parser.parse_args()

    if args.verbose >= 1:
        print("[info] Building models.")
    models = build_six_models()
    Eff_pool, Rew_pool = make_demo_pools()

    t_min, t_max = 0.0, 600.0
    t_step = float(args.t_step)
    min_gap_t = float(args.min_gap)
    delta_er = float(args.delta_er)
    adaptive_outer_min = args.outer_min if args.outer_min is not None else max(5, args.outer // 4)
    adaptive_outer_max = max(args.outer, adaptive_outer_min)
    adaptive_inner_min = args.inner_min if args.inner_min is not None else max(5, args.inner // 2)
    adaptive_inner_max = max(args.inner, adaptive_inner_min)

    t_fixed = _build_fixed_times(t_min, t_max, t_step)
    bounds: List[Tuple[float, float]] = []
    for i in range(len(t_fixed) - 1):
        low = t_fixed[i] + min_gap_t
        high = t_fixed[i + 1] - min_gap_t
        bounds.append((low, high if high > low else low))
    n_intervals = len(bounds)

    if args.verbose >= 1:
        print(f"[info] Design structure: {len(t_fixed)} fixed t (step={t_step}), {n_intervals} optimised t.")

    rng_main = random.Random(123)
    cache_path = Path(args.cache_file)
    objective_cache: Dict[Tuple[float, ...], float] = {}
    eval_history: List[Dict[Tuple[int, int], float]] = []  # valeurs U par eval
    obj_history: List[float] = []  # valeur objective (-score) par eval
    if args.check_covariance:
        args.bo = False  # force random evaluations to study covariance
    if cache_path.exists():
        try:
            with cache_path.open("rb") as fh:
                objective_cache = pickle.load(fh)
            if args.verbose >= 1:
                print(f"[info] Loaded cache with {len(objective_cache)} entries from {cache_path}")
        except Exception as exc:
            if args.verbose >= 1:
                print(f"[warn] Unable to load cache {cache_path}: {exc}")
            objective_cache = {}

    def _parse_pairs(m: int) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        if args.pairs:
            for tok in args.pairs.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                if "-" in tok:
                    a, b = tok.split("-", 1)
                elif ":" in tok:
                    a, b = tok.split(":", 1)
                else:
                    continue
                try:
                    i = int(a)
                    j = int(b)
                except ValueError:
                    continue
                if 0 <= i < m and 0 <= j < m and i != j:
                    pairs.append((i, j))
        return pairs

    def objective(x: List[float]) -> float:
        key = tuple(round(float(v), 6) for v in x)
        if key in objective_cache:
            return objective_cache[key]
        rng_design = random.Random(hash(key) & 0x7FFFFFFF)
        design_inputs = decode_design_structured(
            list(x),
            t_min=t_min,
            t_max=t_max,
            min_gap_t=min_gap_t,
            delta_er=delta_er,
            Eff_pool=Eff_pool,
            Rew_pool=Rew_pool,
            t_step=t_step,
            rng=rng_design,
        )
        try:
            U = expected_log_bayes_factor_matrix_for_design(
                models,
                design_inputs,
                sigma=args.sigma,
                K_outer=args.outer,
                K_inner=args.inner,
                rng=rng_main,
                n_jobs=min(len(models), os.cpu_count() or 1),
                progress=not args.no_progress and args.verbose >= 1,
                adaptive=args.adaptive_utility,
                K_outer_min=adaptive_outer_min,
                K_outer_max=adaptive_outer_max,
                K_inner_min=adaptive_inner_min,
                K_inner_max=adaptive_inner_max,
                tol_rel=args.tol_rel,
                tol_abs=args.tol_abs,
            )
        except Exception:
            return float("inf")

        m = len(models)
        pair_vals: Dict[Tuple[int, int], float] = {}
        pairs = _parse_pairs(m)
        if pairs:
            vals = []
            for (i, j) in pairs:
                pair_vals[(i, j)] = U[i][j]
                vals.append(U[i][j])
            score = min(vals) if vals else float("-inf")
        else:
            row_min: List[float] = []
            for i in range(m):
                vals = [U[i][j] for j in range(m) if j != i]
                row_min.append(min(vals) if vals else float("-inf"))
            score = min(row_min)
        # Toujours stocker toutes les paires pour la convergence
        for i in range(m):
            for j in range(m):
                if i != j:
                    pair_vals[(i, j)] = U[i][j]
        val = -score if score == score else float("inf")
        objective_cache[key] = val
        eval_history.append(dict(pair_vals))
        obj_history.append(val)
        return val

    best_x: List[float] = []
    best_val = float("inf")

    # Mode de convergence MC: 5 designs al�atoires, pas d'optimisation
    if args.mc_convergence:
        rng_mc = random.Random(999)
        checkpoints: List[int] = []
        k = 1
        while k < args.outer:
            checkpoints.append(k)
            k *= 2
        if checkpoints and checkpoints[-1] != args.outer:
            checkpoints.append(args.outer)
        if not checkpoints:
            checkpoints = [args.outer]
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as exc:
            print(f"[warn] Unable to import matplotlib for mc-convergence: {exc}")
            return
        rows = 5
        fig, axes = plt.subplots(rows, 1, figsize=(8, 3 * rows), squeeze=False)
        m = len(models)
        for idx in range(rows):
            x = [rng_mc.uniform(lo, hi) for lo, hi in bounds]
            design_inputs = decode_design_structured(
                list(x),
                t_min=t_min,
                t_max=t_max,
                min_gap_t=min_gap_t,
                delta_er=delta_er,
                Eff_pool=Eff_pool,
                Rew_pool=Rew_pool,
                t_step=t_step,
                rng=random.Random(rng_mc.randint(0, 2**31 - 1)),
            )
            ys: List[float] = []
            for k_outer in checkpoints:
                U = expected_log_bayes_factor_matrix_for_design(
                    models,
                    design_inputs,
                    sigma=args.sigma,
                    K_outer=k_outer,
                    K_inner=args.inner,
                    rng=rng_main,
                    n_jobs=min(len(models), os.cpu_count() or 1),
                    progress=False,
                    adaptive=args.adaptive_utility,
                    K_outer_min=adaptive_outer_min,
                    K_outer_max=adaptive_outer_max,
                    K_inner_min=adaptive_inner_min,
                    K_inner_max=adaptive_inner_max,
                    tol_rel=args.tol_rel,
                    tol_abs=args.tol_abs,
                )
                pairs = _parse_pairs(m)
                if pairs:
                    vals = [U[i][j] for (i, j) in pairs]
                    score = min(vals) if vals else float("-inf")
                else:
                    row_min = []
                    for i in range(m):
                        vals = [U[i][j] for j in range(m) if j != i]
                        row_min.append(min(vals) if vals else float("-inf"))
                    score = min(row_min)
                ys.append(score)
            ax = axes[idx][0]
            ax.plot(checkpoints, ys, marker="o", linestyle="-")
            ax.set_title(f"MC convergence design {idx+1}")
            ax.set_xlabel("K_outer")
            ax.set_ylabel("score")
            ax.grid(True, linestyle=":", alpha=0.4)
        plt.tight_layout()
        plt.show()
        return

    if args.bo:
        try:
            from skopt import gp_minimize
            from skopt.space import Real
            from skopt.sampler import Sobol  # type: ignore
        except Exception as exc:
            raise RuntimeError("scikit-optimize (skopt) est requis pour --bo") from exc

        n_calls = max(1, args.budget)
        n_init = max(1, min(n_calls // 2, n_calls))
        space = [Real(lo, hi, name=f"t_opt_{i}") for i, (lo, hi) in enumerate(bounds)]
        initial_point_generator = "lhs"
        try:
            # Sobol requiert n_init puissance de 2 pour ne pas spammer de warnings
            if n_init & (n_init - 1) == 0:
                initial_point_generator = Sobol()
        except Exception:
            initial_point_generator = "random"
        if args.verbose >= 1:
            print(f"[info] Optimizing with BO: n_calls={n_calls}, n_init={n_init}")
        pbar_bo = None
        try:
            if not args.no_progress:
                from tqdm import tqdm  # type: ignore
                pbar_bo = tqdm(total=n_calls, desc="BO n_calls", leave=True)
        except Exception:
            pbar_bo = None

        def _cb(_res=None) -> None:
            if pbar_bo is not None:
                pbar_bo.update(1)

        x0 = [list(k) for k in objective_cache.keys() if len(k) == len(space)]
        y0 = [objective_cache[tuple(k)] for k in x0]
        res = gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            n_initial_points=n_init,
            initial_point_generator=initial_point_generator,
            random_state=rng_main.randint(0, 2**31 - 1),
            x0=x0 if x0 else None,
            y0=y0 if y0 else None,
            callback=_cb if pbar_bo is not None else None,
        )
        if pbar_bo is not None:
            pbar_bo.close()
        best_x = res.x
        best_val = res.fun
    else:
        if args.verbose >= 1:
            print(f"[info] Optimizing with random search: budget={args.budget}")
        try:
            from tqdm import tqdm  # type: ignore
            iterator = tqdm(range(max(1, args.budget)), desc="Random search", leave=True)
        except Exception:
            iterator = range(max(1, args.budget))
        for _ in iterator:
            x = [rng_main.uniform(lo, hi) for lo, hi in bounds]
            val = objective(x)
            if val < best_val:
                best_val = val
                best_x = x
        # Analyse de covariation des U[i,j] si assez d'�valuations
        if len(eval_history) > 1:
            try:
                import numpy as np  # type: ignore

                pair_keys = sorted(set().union(*[set(h.keys()) for h in eval_history]))
                data = np.array([[h.get(k, np.nan) for k in pair_keys] for h in eval_history], dtype=float)
                # Retire les colonnes toutes NaN
                mask_valid = ~np.all(np.isnan(data), axis=0)
                data = data[:, mask_valid]
                pair_keys = [k for k, m in zip(pair_keys, mask_valid) if m]
                if data.shape[1] > 1:
                    # Remplace les NaN par la moyenne de colonne pour calculer corr
                    col_means = np.nanmean(data, axis=0)
                    inds = np.where(np.isnan(data))
                    data[inds] = np.take(col_means, inds[1])
                    corr = np.corrcoef(data, rowvar=False)
                    # Top corr�lations en valeur absolue
                    top = []
                    for i in range(len(pair_keys)):
                        for j in range(i + 1, len(pair_keys)):
                            top.append((abs(corr[i, j]), corr[i, j], pair_keys[i], pair_keys[j]))
                    top.sort(reverse=True, key=lambda t: t[0])
                    if args.verbose >= 1 and top:
                        print("[diag] Top pairwise corr(|rho|):")
                        for rho_abs, rho, k1, k2 in top[:5]:
                            print(f"  U{k1} vs U{k2}: rho={rho:.3f}")
                    if args.cov_plot:
                        import matplotlib.pyplot as plt  # type: ignore

                        fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(pair_keys)), max(4, 0.4 * len(pair_keys))))
                        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
                        ax.set_xticks(range(len(pair_keys)))
                        ax.set_yticks(range(len(pair_keys)))
                        ax.set_xticklabels([f"{i},{j}" for i, j in pair_keys], rotation=90, fontsize=6)
                        ax.set_yticklabels([f"{i},{j}" for i, j in pair_keys], fontsize=6)
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="corr(U[i,j])")
                        ax.set_title("Correlation of U[i,j] over random search evaluations")
                        plt.tight_layout()
                        plt.show()
            except Exception as exc:
                if args.verbose >= 1:
                    print(f"[warn] Unable to compute/plot covariance: {exc}")

    best_design = decode_design_structured(
        list(best_x),
        t_min=t_min,
        t_max=t_max,
        min_gap_t=min_gap_t,
        delta_er=delta_er,
        Eff_pool=Eff_pool,
        Rew_pool=Rew_pool,
        t_step=t_step,
        rng=random.Random(0),
    )
    best_U = expected_log_bayes_factor_matrix_for_design(
        models,
        best_design,
        sigma=args.sigma,
        K_outer=args.outer,
        K_inner=args.inner,
        rng=rng_main,
        n_jobs=min(len(models), os.cpu_count() or 1),
        progress=False,
        adaptive=args.adaptive_utility,
        K_outer_min=adaptive_outer_min,
        K_outer_max=adaptive_outer_max,
        K_inner_min=adaptive_inner_min,
        K_inner_max=adaptive_inner_max,
        tol_rel=args.tol_rel,
        tol_abs=args.tol_abs,
    )
    m = len(models)
    row_min = []
    for i in range(m):
        vals = [best_U[i][j] for j in range(m) if j != i]
        row_min.append(min(vals) if vals else float("-inf"))
    best_score = min(row_min)

    if args.verbose >= 1:
        print("[result] Best utility (maximin):", best_score)
        print("[result] Best t:", best_design.get("t"))
        print("[result] U matrix:")
        for row in best_U:
            print([round(x, 3) for x in row])
        try:
            import matplotlib.pyplot as plt  # type: ignore
            import numpy as np  # type: ignore
        except Exception as exc:
            plt = None  # type: ignore
            if args.bo and not args.no_convergence_plot:
                print(f"[warn] Unable to show convergence plot: {exc}")
        # BO convergence
        if args.bo and not args.no_convergence_plot and plt is not None:
            try:
                from skopt.plots import plot_convergence  # type: ignore
                plot_convergence(res)
                plt.show()
            except Exception as exc:
                print(f"[warn] Unable to show convergence plot: {exc}")
        # Convergence par paires (best-so-far) et covariance si demandé
        if ((args.bo and not args.no_convergence_plot) or args.check_covariance) and plt is not None:
            xs = list(range(1, len(eval_history) + 1))
            best_pairs_traj: List[Dict[Tuple[int, int], float]] = []
            best_val_so_far = float("inf")
            best_pairs: Dict[Tuple[int, int], float] = {}
            for val, pv in zip(obj_history, eval_history):
                if val < best_val_so_far:
                    best_val_so_far = val
                    best_pairs = pv
                best_pairs_traj.append(best_pairs)

            pairs_specific: List[Tuple[int, int]] = []
            if args.pairs:
                for tok in args.pairs.split(","):
                    tok = tok.strip()
                    if not tok:
                        continue
                    if "-" in tok:
                        a, b = tok.split("-", 1)
                    elif ":" in tok:
                        a, b = tok.split(":", 1)
                    else:
                        continue
                    try:
                        i = int(a)
                        j = int(b)
                    except ValueError:
                        continue
                    if 0 <= i < len(models) and 0 <= j < len(models) and i != j:
                        pairs_specific.append((i, j))
            if best_pairs_traj and pairs_specific:
                cols = 2
                rows = math.ceil(len(pairs_specific) / cols)
                fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
                for idx, (i, j) in enumerate(pairs_specific):
                    r = idx // cols
                    c = idx % cols
                    ax = axes[r][c]
                    ys = [h.get((i, j), float("nan")) for h in best_pairs_traj]
                    ax.plot(xs, ys, marker="o", linestyle="-")
                    ax.set_title(f"Convergence U[{i},{j}]")
                    ax.set_xlabel("evaluation")
                    ax.set_ylabel("U value")
                    ax.grid(True, linestyle=":", alpha=0.4)
                plt.tight_layout()
                plt.show()
            if best_pairs_traj and plt is not None:
                pairs_all = [(i, j) for i in range(len(models)) for j in range(len(models)) if i != j]
                cols_all = 3
                rows_all = math.ceil(len(pairs_all) / cols_all)
                fig_all, axes_all = plt.subplots(rows_all, cols_all, figsize=(5 * cols_all, 3.5 * rows_all), squeeze=False)
                for idx, (i, j) in enumerate(pairs_all):
                    r = idx // cols_all
                    c = idx % cols_all
                    ax = axes_all[r][c]
                    ys = [h.get((i, j), float("nan")) for h in best_pairs_traj]
                    ax.plot(xs, ys, marker=".", linestyle="-", linewidth=1.0)
                    ax.set_title(f"U[{i},{j}]")
                    ax.grid(True, linestyle=":", alpha=0.3)
                for k in range(len(pairs_all), rows_all * cols_all):
                    r = k // cols_all
                    c = k % cols_all
                    axes_all[r][c].axis("off")
                plt.tight_layout()
                plt.show()
            # Heatmap de corrélation sur les U bruts si check_covariance
            if args.check_covariance and plt is not None and len(eval_history) > 1:
                pair_keys = sorted(set().union(*[set(h.keys()) for h in eval_history]))
                data = np.array([[h.get(k, np.nan) for k in pair_keys] for h in eval_history], dtype=float)
                mask_valid = ~np.all(np.isnan(data), axis=0)
                data = data[:, mask_valid]
                pair_keys = [k for k, m in zip(pair_keys, mask_valid) if m]
                if data.shape[1] > 1:
                    col_means = np.nanmean(data, axis=0)
                    inds = np.where(np.isnan(data))
                    data[inds] = np.take(col_means, inds[1])
                    corr = np.corrcoef(data, rowvar=False)
                    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(pair_keys)), max(4, 0.4 * len(pair_keys))))
                    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
                    ax.set_xticks(range(len(pair_keys)))
                    ax.set_yticks(range(len(pair_keys)))
                    ax.set_xticklabels([f"{i},{j}" for i, j in pair_keys], rotation=90, fontsize=6)
                    ax.set_yticklabels([f"{i},{j}" for i, j in pair_keys], fontsize=6)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="corr(U[i,j])")
                    ax.set_title("Correlation of U[i,j] over evaluations")
                    plt.tight_layout()
                    plt.show()

    try:
        with cache_path.open("wb") as fh:
            pickle.dump(objective_cache, fh)
        if args.verbose >= 1:
            print(f"[info] Saved cache with {len(objective_cache)} entries to {cache_path}")
    except Exception as exc:
        if args.verbose >= 1:
            print(f"[warn] Unable to save cache to {cache_path}: {exc}")

    show_indices = None
    if args.show_models:
        try:
            show_indices = [int(x) for x in args.show_models.split(',') if x.strip() != ""]
        except ValueError:
            print("[warn] --show-models doit être une liste d'indices entiers séparés par des virgules.")
            show_indices = None

    plot_summary(best_U, best_design, models, title="maximin", show_indices=show_indices)


if __name__ == "__main__":
    main()

