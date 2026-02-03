# TO CHECK

from __future__ import annotations

from typing import Any, Sequence


def plot_summary(
    U: Any,
    best_inputs: Any,
    models: Sequence[Any],
    title: str = "Jeffreys multi-model summary",
    show_indices: Any = None,
    b_matrix: Any = None,
) -> None:
    """
    Plot a compact summary of a design:
    - Chernoff error bound matrix (left)
    - h(t) curves (top-right)
    - Design variables (bottom-right), sharing the x-axis with h(t)

    Notes:
    - `U` is accepted for API compatibility but intentionally not plotted.
    - Only the design-variable axis gets action-window shading/markers.
    """

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        print("[diag] matplotlib not installed: skip plot summary.")
        return

    m = len(models)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[1.0, 1.8],
        height_ratios=[1.0, 1.0, 0.65],
        wspace=0.25,
        hspace=0.12,
    )
    ax_b = fig.add_subplot(gs[0:2, 0])
    ax_count = fig.add_subplot(gs[2, 0])
    ax_h = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1], sharex=ax_h)

    # --- Left: Chernoff bound matrix heatmap ---
    B = None
    if b_matrix is not None:
        try:
            B = np.asarray(b_matrix, dtype=float)
        except Exception:
            B = None

    if B is not None and B.shape == (m, m):
        # Typical multi-class scale is <= 1/m (uniform prior), but keep vmax robust.
        vmax = float(np.nanmax(B)) if np.isfinite(B).any() else 1.0 / max(1, m)
        vmax = max(vmax, 1.0 / max(1, m))
        im = ax_b.imshow(B, cmap="viridis", vmin=0.0, vmax=vmax)
        fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
        ax_b.set_title(f"Chernoff error bound matrix - {title}")
        ax_b.set_xlabel("j (alt model)")
        ax_b.set_ylabel("i (true model)")
    else:
        ax_b.text(0.5, 0.5, "No Chernoff bound matrix", ha="center", va="center")
        ax_b.set_axis_off()

    # --- Right top: h(t) ---
    t = best_inputs.get("t", []) if isinstance(best_inputs, dict) else []
    if not t:
        ax_h.text(0.5, 0.5, "No t to plot", ha="center", va="center")
    else:
        Eff_ = best_inputs.get("Eff_", [])
        Rew_ = best_inputs.get("Rew_", [])
        A = best_inputs.get("A", [])

        t_min_plot, t_max_plot = 0.0, max(t)
        t_measured = best_inputs.get("t_meas", [])
        base_grid = np.linspace(t_min_plot, t_max_plot, max(1, int(2 * t_max_plot) + 1)).tolist()
        t_grid = sorted(set(base_grid) | set(t_measured))

        indices = list(range(len(models))) if show_indices is None else [i for i in show_indices if 0 <= i < len(models)]
        for idx in indices:
            f = models[idx]
            y_full = f.eval(
                {
                    "t": t_grid,
                    "Eff_": Eff_,
                    "Rew_": Rew_,
                    "A": A,
                    "A_typed": best_inputs.get("A_typed", []),
                    "K_types": best_inputs.get("K_types"),
                }
            )
            if not isinstance(y_full, list):
                y_full = [y_full]
            line = ax_h.plot(t_grid, y_full, linestyle="--", linewidth=1.2, label=getattr(f, "name", f"model{idx}"))
            color = line[0].get_color()
            if t_measured:
                y_meas = f.eval(
                    {
                        "t": t_measured,
                        "Eff_": Eff_,
                        "Rew_": Rew_,
                        "A": A,
                        "A_typed": best_inputs.get("A_typed", []),
                        "K_types": best_inputs.get("K_types"),
                    }
                )
                ax_h.plot(t_measured, y_meas, marker="x", linestyle="none", color=color)

        ax_h.set_title("h(t) on best design (dashed=continuous, x=measured)")
        ax_h.set_ylabel("h(t)")
        ax_h.grid(True, linestyle=":", alpha=0.4)
        if indices:
            ax_h.legend(fontsize=8, ncol=2)
        ax_h.tick_params(labelbottom=False)

    # --- Right bottom: design variables ---
    Eff_ = best_inputs.get("Eff_", []) if isinstance(best_inputs, dict) else []
    Rew_ = best_inputs.get("Rew_", []) if isinstance(best_inputs, dict) else []
    A_typed = best_inputs.get("A_typed", []) if isinstance(best_inputs, dict) else []

    if A_typed:
        # Shade each action window [t_eff, t_rew] with one color per action type and
        # use a different marker per action type.
        marker_by_type = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
        # High-contrast, colorblind-friendly palette (repeat if needed).
        type_colors = [
            "#0072B2",  # blue
            "#E69F00",  # orange
            "#009E73",  # green
            "#D55E00",  # vermillion
            "#CC79A7",  # purple
            "#56B4E9",  # light blue
            "#F0E442",  # yellow
            "#000000",  # black
            "#8B0000",  # dark red
            "#228B22",  # forest green
        ]

        actions = []
        for entry in A_typed:
            try:
                k, eff, rew = entry
                e_val, t_eff = eff
                r_val, t_rew = rew
                actions.append((int(k), float(e_val), float(t_eff), float(r_val), float(t_rew)))
            except Exception:
                continue
        actions.sort(key=lambda x: x[4])  # sort by t_rew

        # Background shading: actions only (no pause shading), colored by type.
        for k, e_val, t_eff, r_val, t_rew in actions:
            c = type_colors[int(k) % len(type_colors)]
            c_rgba = plt.matplotlib.colors.to_rgba(c, alpha=0.20)
            ax_d.axvspan(t_eff, t_rew, color=c_rgba, lw=0)

        # Markers: effort (open) and reward (filled), marker shape + color encode type.
        for k, e_val, t_eff, r_val, t_rew in actions:
            c = type_colors[int(k) % len(type_colors)]
            mk = marker_by_type[int(k) % len(marker_by_type)]
            ax_d.scatter([t_eff], [e_val], marker=mk, facecolors="none", edgecolors=[c], linewidths=1.2, s=45)
            ax_d.scatter([t_rew], [r_val], marker=mk, facecolors=[c], edgecolors="k", linewidths=0.4, s=55)

        # Bonus rewards: Rew_ entries not matching typed action reward times (if any).
        rew_action_times = {t_rew for _k, _e, _te, _r, t_rew in actions}
        bonus = [(v, t_) for (v, t_) in Rew_ if float(t_) not in rew_action_times]
        if bonus:
            bx = [t_ for _v, t_ in bonus]
            by = [v for v, _t in bonus]
            ax_d.scatter(bx, by, color="tab:green", label="Bonus", marker="+", s=60, linewidths=1.2)

        # Legend: marker + color per type.
        try:
            K_types = int(best_inputs.get("K_types", 0)) if isinstance(best_inputs, dict) else 0
        except Exception:
            K_types = 0
        if K_types <= 0:
            K_types = 1 + max(k for k, *_rest in actions) if actions else 1

        handles = []
        labels = []
        for k in range(max(1, K_types)):
            mk = marker_by_type[k % len(marker_by_type)]
            c = type_colors[int(k) % len(type_colors)]
            h = ax_d.scatter([], [], marker=mk, facecolors="none", edgecolors=[c], s=45, linewidths=1.2)
            handles.append(h)
            labels.append(f"type {k}")
        if bonus:
            hb = ax_d.scatter([], [], marker="+", color="tab:green", s=60, linewidths=1.2)
            handles.append(hb)
            labels.append("Bonus")
        ax_d.legend(handles, labels, loc="best", fontsize=8, ncol=2)
    else:
        if Eff_:
            ex = [et for _, et in Eff_]
            ey = [ev for ev, _ in Eff_]
            ax_d.scatter(ex, ey, color="tab:red", label="Eff_", marker="x")
        if Rew_:
            rx = [rt for _, rt in Rew_]
            ry = [rv for rv, _ in Rew_]
            ax_d.scatter(rx, ry, color="tab:green", label="Rew_", marker="o")

    if isinstance(best_inputs, dict):
        tvals = best_inputs.get("t_meas", None)
        if tvals:
            ax_d.scatter(tvals, [0.0] * len(tvals), color="k", marker="|", s=40, label="t")

    ax_d.axhline(0.0, color="k", linewidth=0.5, alpha=0.5)
    ax_d.set_title("Design variables (Eff_, Rew_, t)")
    ax_d.set_xlabel("time")
    ax_d.set_ylabel("value")
    if not A_typed:
        ax_d.legend(loc="best", fontsize=8)
    ax_d.grid(True, linestyle=":", alpha=0.4)

    # --- Left bottom: count per action type + % rest ---
    rest_pct = None
    if isinstance(best_inputs, dict):
        t_all = list(best_inputs.get("t", []) or [])
    else:
        t_all = []

    if A_typed:
        type_idx = []
        windows = []
        for entry in A_typed:
            try:
                k, eff, rew = entry
                _e_val, t_eff = eff
                _r_val, t_rew = rew
                type_idx.append(int(k))
                windows.append((float(t_eff), float(t_rew)))
            except Exception:
                continue

        if t_all and windows:
            n_rest = 0
            for tt in t_all:
                inside = any((t0 <= tt <= t1) for t0, t1 in windows)
                if not inside:
                    n_rest += 1
            rest_pct = 100.0 * n_rest / max(1, len(t_all))

        if type_idx:
            max_type = max(type_idx)
            counts = [0 for _ in range(max_type + 1)]
            for k in type_idx:
                if k >= 0:
                    counts[k] += 1
            xs = list(range(len(counts)))
            bar_colors = [type_colors[k % len(type_colors)] for k in xs]
            ax_count.bar(xs, counts, color=bar_colors, alpha=0.85, edgecolor="k")
            ax_count.set_xlabel("action type")
            ax_count.set_ylabel("count")
            ax_count.set_xticks(xs)
        else:
            ax_count.text(0.5, 0.5, "No action types", ha="center", va="center")
    else:
        ax_count.text(0.5, 0.5, "No action types", ha="center", va="center")

    if rest_pct is None:
        ax_count.set_title("Action counts by type (rest: n/a)")
    else:
        ax_count.set_title(f"Action counts by type (rest: {rest_pct:.1f}%)")

    fig.tight_layout()
    plt.show()

    # --- Additional diagnostics in a separate figure (legacy plots + dt_event-measure) ---
    if not isinstance(best_inputs, dict):
        return

    try:
        import numpy as np
    except Exception:
        return

    t_list = sorted(best_inputs.get("t_meas", []) or [])
    t_all = sorted(best_inputs.get("t", []) or [])
    eff_times = [t for _, t in best_inputs.get("Eff_", [])]
    rew_times = [t for _, t in best_inputs.get("Rew_", [])]
    eff_vals = [ev for ev, _ in best_inputs.get("Eff_", [])]
    rew_vals = [rv for rv, _ in best_inputs.get("Rew_", [])]

    bonus_times = [t for _, t in best_inputs.get("Bonus_", [])] if isinstance(best_inputs, dict) else []
    t_event = sorted(set(eff_times + rew_times + bonus_times + t_list))
    gaps_all = [t_event[i + 1] - t_event[i] for i in range(len(t_event) - 1)] if len(t_event) > 1 else []
    gaps_meas = list(np.diff(t_list)) if len(t_list) > 1 else []

    dist_eff = [min(abs(te - tv) for tv in t_list) for te in eff_times] if t_list and eff_times else []
    dist_rew = [min(abs(tr - tv) for tv in t_list) for tr in rew_times] if t_list and rew_times else []

    # New: distance from each event (eff/rew) to nearest measurement time.
    event_times = eff_times + rew_times
    dt_event_meas = [min(abs(te - tm) for tm in t_list) for te in event_times] if t_list and event_times else []

    fig2 = plt.figure(figsize=(14, 10))
    gs2 = fig2.add_gridspec(nrows=3, ncols=2, height_ratios=[1.0, 1.0, 1.0], wspace=0.25, hspace=0.35)

    ax_src = fig2.add_subplot(gs2[0, 0])
    ax_gap = fig2.add_subplot(gs2[0, 1])
    ax_val = fig2.add_subplot(gs2[1, 0])
    ax_violin = fig2.add_subplot(gs2[1, 1])
    ax_act = fig2.add_subplot(gs2[2, 0])
    ax_dist = fig2.add_subplot(gs2[2, 1])

    # Measurement time sources
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
        ax_src.bar(labels, proportions, color="tab:blue", edgecolor="k", alpha=0.8)
        ax_src.set_ylim(0.0, 1.0)
        ax_src.set_ylabel("proportion")
        ax_src.set_title("Measurement time sources (n_t)")
        for i, (p, c) in enumerate(zip(proportions, counts)):
            ax_src.text(i, p + 0.02, f"{p:.2f} ({c})", ha="center", va="bottom", fontsize=8)
    else:
        ax_src.text(0.5, 0.5, "No meas_sources", ha="center", va="center")

    # dt histograms (gaps between event/measure times)
    if gaps_all:
        counts, _, _ = ax_gap.hist(gaps_all, bins="auto", color="steelblue", alpha=0.7, edgecolor="k")
        max_count = float(max(counts)) if len(counts) > 0 else 1.0
        jitter_x = np.array(gaps_all)
        jitter_y = np.random.uniform(0.05 * max_count, 0.2 * max_count, size=len(gaps_all))
        ax_gap.scatter(jitter_x, jitter_y, color="black", alpha=0.6, s=8, label="dt (events + measures)")
        if gaps_meas:
            jitter_xm = np.array(gaps_meas)
            jitter_ym = np.random.uniform(0.25 * max_count, 0.45 * max_count, size=len(gaps_meas))
            ax_gap.scatter(jitter_xm, jitter_ym, color="orange", alpha=0.7, s=10, label="dt (measures only)")
            ax_gap.legend(fontsize=8)
        ax_gap.set_title("dt between successive event/measure times")
        ax_gap.set_xlabel("dt")
        ax_gap.set_ylabel("count / points")
        ax_gap.set_ylim(bottom=0.0)
    else:
        ax_gap.text(0.5, 0.5, "No successive t", ha="center", va="center")

    # Eff/Rew value hist
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
        ax_val.text(0.5, 0.5, "No Eff/Rew values", ha="center", va="center")

    # Violin of dt
    violin_data = gaps_meas if gaps_meas else gaps_all
    if violin_data:
        ax_violin.violinplot(violin_data, showmeans=True, widths=0.7)
        jitter_x = np.random.normal(loc=1.0, scale=0.03, size=len(violin_data))
        ax_violin.scatter(jitter_x, violin_data, s=10, alpha=0.5, color="black")
        ax_violin.set_xticks([1])
        ax_violin.set_xticklabels(["dt meas" if gaps_meas else "dt all"])
        ax_violin.set_title("dt distribution")
        ax_violin.set_ylabel("dt")
    else:
        ax_violin.text(0.5, 0.5, "No dt", ha="center", va="center")

    # Action distribution: counts per type
    type_idx = []
    A_typed = best_inputs.get("A_typed", []) if isinstance(best_inputs, dict) else []
    for entry in A_typed:
        try:
            k, _eff, _rew = entry
            type_idx.append(int(k))
        except Exception:
            continue
    if type_idx:
        max_type = max(type_idx)
        counts = [0 for _ in range(max_type + 1)]
        for k in type_idx:
            if k >= 0:
                counts[k] += 1
        xs = list(range(len(counts)))
        ax_act.bar(xs, counts, color="tab:blue", alpha=0.7, edgecolor="k")
        ax_act.set_title("Action type counts")
        ax_act.set_xlabel("action type")
        ax_act.set_ylabel("count")
        ax_act.set_xticks(xs)
    else:
        ax_act.text(0.5, 0.5, "No action types", ha="center", va="center")

    # Distances to nearest t + new dt_event-measure
    if dist_eff or dist_rew or dt_event_meas:
        if dist_eff:
            ax_dist.hist(dist_eff, bins="auto", color="tab:red", alpha=0.6, edgecolor="k", label="|t_eff - t_meas|")
        if dist_rew:
            ax_dist.hist(dist_rew, bins="auto", color="tab:green", alpha=0.6, edgecolor="k", label="|t_rew - t_meas|")
        if dt_event_meas:
            ax_dist.hist(dt_event_meas, bins="auto", color="tab:purple", alpha=0.5, edgecolor="k", label="|t_event - t_meas|")
        ax_dist.set_title("Distances to nearest measurement time")
        ax_dist.set_xlabel("distance")
        ax_dist.set_ylabel("count")
        ax_dist.legend(fontsize=8)
    else:
        ax_dist.text(0.5, 0.5, "No distance data", ha="center", va="center")

    fig2.tight_layout()
    plt.show()
