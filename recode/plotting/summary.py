# TO CHECK

from __future__ import annotations

from typing import Any, List, Sequence


def plot_summary(U, best_inputs, models, title="Jeffreys multi-model summary", show_indices=None, b_matrix=None):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        print("[diag] matplotlib not installed: skip plot summary.")
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

    if U and len(U) == m:
        M = np.array(U, dtype=float)
        im = ax.imshow(M, cmap="viridis")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"U matrix (log-BF expected) - {title}")
        ax.set_xlabel("j (alt model)")
        ax.set_ylabel("i (true model)")
    else:
        ax.text(0.5, 0.5, "No U matrix", ha="center")

    if b_matrix is not None:
        try:
            B = np.asarray(b_matrix, dtype=float)
        except Exception:
            B = None
        if B is not None and B.shape[0] == m and B.shape[1] == m:
            im_b = ax_b.imshow(B, cmap="viridis", vmin=0.0, vmax=1 / max(1, m))
            fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
            ax_b.set_title(f"Chernoff error bound matrix - {title}")
            ax_b.set_xlabel("j (alt model)")
            ax_b.set_ylabel("i (true model)")
        else:
            ax_b.text(0.5, 0.5, "No b matrix", ha="center")

    t = best_inputs.get("t", []) if isinstance(best_inputs, dict) else []
    if not t:
        ax2.text(0.5, 0.5, "No t to plot", ha="center")
    else:
        Eff_ = best_inputs.get("Eff_", [])
        Rew_ = best_inputs.get("Rew_", [])
        A = best_inputs.get("A", [])

        t_min_plot, t_max_plot = 0.0, max(t)
        t_measured = best_inputs.get("t_meas", [])
        base_grid = np.linspace(t_min_plot, t_max_plot, max(1, int(2 * t_max_plot) + 1)).tolist()
        t_grid = sorted(set(base_grid) | set(t_measured))

        indices = list(range(len(models))) if show_indices is None else [i for i in show_indices if 0 <= i < len(models)]
        model_lines = []
        model_labels = []
        for idx in indices:
            f = models[idx]
            y_full = f.eval({
                "t": t_grid,
                "Eff_": Eff_,
                "Rew_": Rew_,
                "A": A,
                "A_typed": best_inputs.get("A_typed", []),
                "K_types": best_inputs.get("K_types"),
            })
            if not isinstance(y_full, list):
                y_full = [y_full]
            line = ax2.plot(t_grid, y_full, linestyle="--", linewidth=1.2, label=f.name)
            color = line[0].get_color()
            model_lines.append(line[0])
            model_labels.append(f.name)
            if t_measured:
                y_meas = f.eval({
                    "t": t_measured,
                    "Eff_": Eff_,
                    "Rew_": Rew_,
                    "A": A,
                    "A_typed": best_inputs.get("A_typed", []),
                    "K_types": best_inputs.get("K_types"),
                })
                ax2.plot(t_measured, y_meas, marker="x", linestyle="none", color=color)

        ax2.set_title("h(t) on best design (dashed=continuous, x=measured)")
        ax2.set_xlabel("t")
        ax2.set_ylabel("h(t)")
        ax2.grid(True, linestyle=":", alpha=0.4)

        if model_lines:
            ax2.legend(model_lines, model_labels, fontsize=8, ncol=2)

    Eff_ = best_inputs.get("Eff_", []) if isinstance(best_inputs, dict) else []
    Rew_ = best_inputs.get("Rew_", []) if isinstance(best_inputs, dict) else []
    if Eff_:
        ex = [et for _, et in Eff_]
        ey = [ev for ev, _ in Eff_]
        ax3.scatter(ex, ey, color="tab:red", label="Eff_", marker="x")
        for x, y in zip(ex, ey):
            ax3.text(x, y, f"{y:.3g}", fontsize=7, color="tab:red", va="bottom", ha="left")
    if Rew_:
        rx = [rt for _, rt in Rew_]
        ry = [rv for rv, _ in Rew_]
        ax3.scatter(rx, ry, color="tab:green", label="Rew_", marker="o")
        for x, y in zip(rx, ry):
            ax3.text(x, y, f"{y:.3g}", fontsize=7, color="tab:green", va="bottom", ha="left")
    if isinstance(best_inputs, dict):
        tvals = best_inputs.get("t_meas", None)
        if tvals:
            ax3.scatter(tvals, [0.0] * len(tvals), color="k", marker="|", s=40, label="t")
            for x in tvals:
                ax3.text(x, 0.0, f"{x:.3g}", fontsize=7, color="k", va="bottom", ha="left")
    ax3.axhline(0.0, color="k", linewidth=0.5, alpha=0.5)
    ax3.set_title("Design variables (Eff_, Rew_, t)")
    ax3.set_xlabel("time")
    ax3.set_ylabel("value")
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(True, linestyle=":", alpha=0.4)

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

    try:
        t_list = sorted(best_inputs.get("t_meas", [])) if isinstance(best_inputs, dict) else []
        t_all = sorted(best_inputs.get("t", [])) if isinstance(best_inputs, dict) else []
        gaps_all = [t_all[i + 1] - t_all[i] for i in range(len(t_all) - 1)] if len(t_all) > 1 else []
        gaps_meas = list(np.diff(t_list)) if len(t_list) > 1 else []
        eff_vals = [ev for ev, _ in best_inputs.get("Eff_", [])] if isinstance(best_inputs, dict) else []
        rew_vals = [rv for rv, _ in best_inputs.get("Rew_", [])] if isinstance(best_inputs, dict) else []

        if t_list or eff_vals or rew_vals:
            fig_hist, (ax_gap, ax_val, ax_violin) = plt.subplots(1, 3, figsize=(14, 4))
            if gaps_all:
                bins_gap = "auto"
                counts, bins_used, _ = ax_gap.hist(gaps_all, bins=bins_gap, color="steelblue", alpha=0.7, edgecolor="k")
                max_count = float(max(counts)) if len(counts) > 0 else 1.0

                jitter_x = np.array(gaps_all)
                jitter_y = np.random.uniform(0.05 * max_count, 0.2 * max_count, size=len(gaps_all))
                ax_gap.scatter(jitter_x, jitter_y, color="black", alpha=0.6, s=8, label="dt (all t)")

                if gaps_meas:
                    jitter_xm = np.array(gaps_meas)
                    jitter_ym = np.random.uniform(0.25 * max_count, 0.45 * max_count, size=len(gaps_meas))
                    ax_gap.scatter(jitter_xm, jitter_ym, color="orange", alpha=0.7, s=10, label="dt (measures only)")
                    ax_gap.legend(fontsize=8)

                ax_gap.set_title("dt (successive t)")
                ax_gap.set_xlabel("dt")
                ax_gap.set_ylabel("count / points")
                ax_gap.set_ylim(bottom=0.0)
            else:
                ax_gap.text(0.5, 0.5, "No successive t", ha="center")

            if eff_vals or rew_vals:
                if eff_vals:
                    ax_val.hist(eff_vals, bins="auto", color="tab:red", alpha=0.6, edgecolor="k", label="Eff")
                if rew_vals:
                    ax_val.hist(rew_vals, bins="auto", color="tab:green", alpha=0.6, edgecolor="k", label="Rew")
                ax_val.set_title("Histogram of Eff/Rew values")
                ax_val.set_xlabel("value")
                ax_val.set_ylabel("count")
                ax_val.legend()
            else:
                ax_val.text(0.5, 0.5, "No Eff/Rew values", ha="center")

            violin_data = gaps_meas if gaps_meas else gaps_all
            if violin_data:
                ax_violin.violinplot(violin_data, showmeans=True, widths=0.7)
                jitter_x = np.random.normal(loc=1.0, scale=0.03, size=len(violin_data))
                ax_violin.scatter(jitter_x, violin_data, color="black", alpha=0.5, s=8)
                ax_violin.set_xticks([1])
                ax_violin.set_xticklabels(["dt meas" if gaps_meas else "dt all"])
                ax_violin.set_title("dt distribution (measures)" if gaps_meas else "dt distribution (all t)")
                ax_violin.set_ylabel("dt")
            else:
                ax_violin.text(0.5, 0.5, "No dt", ha="center")
            fig_hist.tight_layout()
            plt.show()
    except Exception as exc:
        print(f"[diag] unable to plot extra histograms: {exc}")

    try:
        A_typed = best_inputs.get("A_typed", []) if isinstance(best_inputs, dict) else []
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
            ax_count.bar(xs, counts, color="tab:blue", alpha=0.7, edgecolor="k")
            ax_count.set_title("Action type counts")
            ax_count.set_xlabel("action type")
            ax_count.set_ylabel("count")
            ax_count.set_xticks(xs)

            ax_time.scatter(t_rew_list, type_idx, s=10, alpha=0.7, color="tab:orange")
            ax_time.set_title("Action types over time (reward time)")
            ax_time.set_xlabel("time (t_rew)")
            ax_time.set_ylabel("action type")
            ax_time.set_yticks(xs)
            ax_time.grid(True, linestyle=":", alpha=0.4)
            fig_act.tight_layout()
            plt.show()
    except Exception as exc:
        print(f"[diag] unable to plot action distribution: {exc}")

    try:
        if isinstance(best_inputs, dict):
            t_list = sorted(best_inputs.get("t", []))
            eff_times = [t for _, t in best_inputs.get("Eff_", [])]
            rew_times = [t for _, t in best_inputs.get("Rew_", [])]
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
                ax_de.hist(dist_eff, bins="auto", color="tab:red", alpha=0.7, edgecolor="k")
                ax_de.set_title("Distance |t_eff - t| (nearest t)")
                ax_de.set_xlabel("distance")
                ax_de.set_ylabel("count")
            else:
                ax_de.text(0.5, 0.5, "No Eff times", ha="center")
            if dist_rew:
                ax_dr.hist(dist_rew, bins="auto", color="tab:green", alpha=0.7, edgecolor="k")
                ax_dr.set_title("Distance |t_rew - t| (nearest t)")
                ax_dr.set_xlabel("distance")
                ax_dr.set_ylabel("count")
            else:
                ax_dr.text(0.5, 0.5, "No Rew times", ha="center")
            fig_d.tight_layout()
            plt.show()
    except Exception as exc:
        print(f"[diag] unable to plot distance histograms: {exc}")

