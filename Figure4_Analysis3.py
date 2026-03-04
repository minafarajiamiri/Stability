import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch



FILE_SCORES_RADIO = "robustness_scores_Radiorag_dataset.csv"
FILE_SCORES_TUM = "robustness_scores_internal_TUM_dataset.csv"
FILE_TRANS_RADIO = "robustness_transitions_Radiorag_dataset.csv"
FILE_TRANS_TUM = "robustness_transitions_internal_TUM_dataset.csv"
FILE_SUMMARY_RADIO = "table_3_summary_Radiorag_dataset.csv"
FILE_SUMMARY_TUM = "table_3_summary_internal_TUM_dataset.csv"

OUTPUT_FIGURE = "Figure4_Analysis3.png"

RNG_SEED = 42


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.frameon'] = False
plt.rcParams["axes.grid"] = False

PALETTE_DATASETS = {"Benchmark-RadQA": "#1f77b4", "Board-RadQA": "#2ca02c"}
PALETTE_METHODS = {"zero-shot": "tab:blue", "agentic": "tab:green"}

PALETTE_BINS = {
    "Low": "#d62728",
    "Medium": "#4c72b0",  # Figure 6 "Moderate"
    "High": "#2ca02c", 
    }

PALETTE_CATS = {"Improved": "#2ca02c", "Worsened": "#d62728"}
PALETTE_TRANS = {"Improved": "#2ca02c", "No Change": "#bbbbbb", "Decreased": "#d62728"}

def standardize_dataset_names(df):
    if df is None or df.empty:
        return df
    return df.replace(
        {
            "Radiorag dataset": "Benchmark-RadQA",
            "internal TUM dataset": "Board-RadQA",
        }
    )


def safe_float_from_summary(summary_df, metric_name, default=np.nan):
    if summary_df is None or summary_df.empty:
        return float(default)
    try:
        val = summary_df.loc[summary_df["Metric"] == metric_name, "Value"].values[0]
        return float(val)
    except Exception:
        return float(default)


def p_to_str(p):
    if p is None or np.isnan(p):
        return "P=?"
    if p < 0.001:
        return "P<0.001"
    return f"P={p:.3f}"


def tukey_bounds(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size < 4:
        return None, None
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return lo, hi


def add_jitter_points(ax, x_pos, y_vals, rng, jitter=0.06, size=7, alpha=0.35,
                      c=None, ec=None, lw=0.6, zorder=2):
    y = np.asarray(y_vals, dtype=float)
    y = y[~np.isnan(y)]
    x = x_pos + rng.normal(0, jitter, size=y.size)
    ax.scatter(x, y, s=size, alpha=alpha, c=c, edgecolors=ec, linewidths=lw, zorder=zorder)
    return x, y


def load_csv_checked(filepath):
    if not os.path.exists(filepath):
        sys.exit(f"Error: File not found at {filepath}")
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        sys.exit(f"Error reading CSV '{filepath}': {e}")

def pad_axes_in_container(ax, pad_left=0.06, pad_right=0.06, pad_bottom=0.10, pad_top=0.08):
    pos = ax.get_position()
    x0, y0, w, h = pos.x0, pos.y0, pos.width, pos.height
    new_x0 = x0 + pad_left * w
    new_y0 = y0 + pad_bottom * h
    new_w  = w * (1 - pad_left - pad_right)
    new_h  = h * (1 - pad_bottom - pad_top)
    ax.set_position([new_x0, new_y0, new_w, new_h])

def plot_panel_a_scatter(ax, scores_df):
    df = standardize_dataset_names(scores_df.copy())

    zs = df[df["method"] == "zero-shot"][["dataset", "question_id", "robustness_score"]].rename(
        columns={"robustness_score": "R_zeroshot"}
    )
    ag = df[df["method"] == "agentic"][["dataset", "question_id", "robustness_score"]].rename(
        columns={"robustness_score": "R_agentic"}
    )
    merged = pd.merge(zs, ag, on=["dataset", "question_id"], how="inner")

    max_val = max(merged["R_zeroshot"].max(), merged["R_agentic"].max()) * 1.1
    if not np.isfinite(max_val) or max_val <= 0:
        max_val = 1.0

    x_fill = np.linspace(0, max_val, 100)

    # Background shading
    ax.fill_between(x_fill, 0, x_fill, color=PALETTE_CATS["Worsened"],
                    alpha=0.10, zorder=0, edgecolor="none")
    ax.fill_between(x_fill, x_fill, max_val, color=PALETTE_CATS["Improved"],
                    alpha=0.10, zorder=0, edgecolor="none")

    ax.plot([0, max_val], [0, max_val], ls="-", c="black", lw=1.0, zorder=1)

    order = ["Board-RadQA", "Benchmark-RadQA"]
    merged["dataset"] = pd.Categorical(merged["dataset"], categories=order, ordered=True)

    sns.scatterplot(
        data=merged,
        x="R_zeroshot",
        y="R_agentic",
        hue="dataset",
        hue_order=order,
        style="dataset",
        style_order=order,
        palette={"Board-RadQA": "#2ca02c", "Benchmark-RadQA": "#1f77b4"},
        markers={"Board-RadQA": "o", "Benchmark-RadQA": "^"},
        alpha=0.70,
        s=50,
        ax=ax,
        zorder=2,
        #edgecolor="black",
        linewidth=0.4,
    )

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    ax.set_aspect("auto")

    ax.set_xlabel("Robustness score (zero-shot)")
    ax.set_ylabel("Robustness score (agentic)")

    ax.text(max_val * 0.70, max_val * 0.30, "Worsened\nrobustness",
            ha="center", va="center", color=PALETTE_CATS["Worsened"],
            fontweight="bold", zorder=3, fontsize=11)
    ax.text(max_val * 0.30, max_val * 0.60, "Improved\nrobustness",
            ha="center", va="center", color=PALETTE_CATS["Improved"],
            fontweight="bold", zorder=3, fontsize=11)

    ax.legend(title="", loc="upper left")
    ax.set_title(r'$\mathbf{a}$   Paired robustness score (zero-shot vs agentic)',
                 loc='left', fontsize=14)
    ax.set_axisbelow(True)



def plot_panel_b_distributions(ax_container, scores_df, summary_radio, summary_tum):
    ax_container.set_title(r"$\mathbf{b}$  Robustness score distributions",
                           loc="left", fontsize=14, y=1.08)

    gs_inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=ax_container, wspace=0.3)
    axes = [plt.Subplot(ax_container.figure, gs_inner[i]) for i in range(3)]
    for ax in axes:
        ax_container.figure.add_subplot(ax)

    df = standardize_dataset_names(scores_df.copy())
    rng = np.random.default_rng(RNG_SEED)

    groups = [
        ("Pooled", df, None),
        ("Benchmark-RadQA", df[df["dataset"] == "Benchmark-RadQA"], summary_radio),
        ("Board-RadQA", df[df["dataset"] == "Board-RadQA"], summary_tum),
    ]

    for ax, (group, subdf, summ) in zip(axes, groups):
        zs = subdf[subdf["method"] == "zero-shot"]["robustness_score"].to_numpy()
        ag = subdf[subdf["method"] == "agentic"]["robustness_score"].to_numpy()

        data = [zs, ag]
        positions = [1, 2]

        x_zs, y_zs = add_jitter_points(ax, 1, zs, rng, c="tab:orange", size=7, alpha=0.35, jitter=0.06, zorder=2)
        x_ag, y_ag = add_jitter_points(ax, 2, ag, rng, c="tab:red",    size=7, alpha=0.35, jitter=0.06, zorder=2)

        vp = ax.violinplot(
            data,
            positions=positions,
            widths=0.75,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, col in zip(vp["bodies"], [PALETTE_METHODS["zero-shot"], PALETTE_METHODS["agentic"]]):
            body.set_facecolor(col)
            body.set_alpha(0.22)
            body.set_edgecolor("none")
            body.set_zorder(1)

        ax.boxplot(
            data,
            positions=positions,
            widths=0.25,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor="none", linewidth=1.0),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            medianprops=dict(color="black", linewidth=1.3),
        )

        lo_zs, hi_zs = tukey_bounds(zs)
        lo_ag, hi_ag = tukey_bounds(ag)

        if lo_zs is not None:
            mask_zs = (y_zs < lo_zs) | (y_zs > hi_zs)
            if mask_zs.any():
                ax.scatter(
                    x_zs[mask_zs], y_zs[mask_zs],
                    s=26, marker="o",
                    facecolors="none",
                    edgecolors="black",  # or PALETTE_METHODS["zero-shot"]
                    linewidths=0.9,
                    zorder=6
                )

        if lo_ag is not None:
            mask_ag = (y_ag < lo_ag) | (y_ag > hi_ag)
            if mask_ag.any():
                ax.scatter(
                    x_ag[mask_ag], y_ag[mask_ag],
                    s=26, marker="o",
                    facecolors="none",
                    edgecolors="black",  # or PALETTE_METHODS["agentic"]
                    linewidths=0.9,
                    zorder=6
                )

        # facet label
        ax.text(0.5, 1.02, group, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=12)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Zero-shot", "Agentic"])
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Robustness score", fontsize=12)
        ax.set_axisbelow(True)

        if summ is not None:
            mean_delta = safe_float_from_summary(summ, "Mean Δrobustness")
            p_val = safe_float_from_summary(summ, "Wilcoxon p-value")
            eff_r = safe_float_from_summary(summ, "Effect size (r)")
            _ = f"ΔR̄={mean_delta:.3f}\n{p_to_str(p_val)}\nr={eff_r:.2f}"

    ax_container.axis("off")


def plot_panel_c_bins(ax, scores_df):
    df = standardize_dataset_names(scores_df.copy())

    datasets = ["Benchmark-RadQA", "Board-RadQA"]
    methods = ["zero-shot", "agentic"]
    bins_order = ["Low", "Medium", "High"]

    perc = {}
    for ds in datasets:
        for m in methods:
            sub = df[(df["dataset"] == ds) & (df["method"] == m)]
            total = len(sub)
            counts = sub["robustness_bin"].value_counts()
            perc[(ds, m)] = {b: (counts.get(b, 0) / total * 100 if total else 0.0) for b in bins_order}

    centers = np.arange(len(datasets))
    width = 0.35
    gap = 0.06                 
    
    x_zs = centers - (width/2 + gap/2)
    x_ag = centers + (width/2 + gap/2)

    def stacked_one(x, ds, m):
        low = perc[(ds, m)]["Low"]
        med = perc[(ds, m)]["Medium"]
        high = perc[(ds, m)]["High"]

        ax.bar(x, low, width, color=PALETTE_BINS["Low"], edgecolor="white", linewidth=1.0)
        ax.bar(x, med, width, bottom=low, color=PALETTE_BINS["Medium"], edgecolor="white", linewidth=1.0)
        ax.bar(x, high, width, bottom=low + med, color=PALETTE_BINS["High"], edgecolor="white", linewidth=1.0)

        for y0, val in [(0, low), (low, med), (low + med, high)]:
            if val >= 6:
                ax.text(x, y0 + val / 2, f"{val:.0f}",
                        ha="center", va="center", fontsize=12, fontweight="bold", color="white")

    for i, ds in enumerate(datasets):
        stacked_one(x_zs[i], ds, "zero-shot")
        stacked_one(x_ag[i], ds, "agentic")

    ax.set_ylabel("Proportion (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(centers)
    ax.set_xticklabels(datasets)

    bar_positions = np.ravel(np.column_stack([x_zs, x_ag]))
    ax.set_xticks(bar_positions, minor=True)
    ax.set_xticklabels(["Zero-shot", "Agentic"] * len(datasets), minor=True, fontsize=11)

    ax.tick_params(axis="x", which="major", pad=14, length=0)
    ax.tick_params(axis="x", which="minor", pad=2, length=0)

    handles = [
        mpatches.Patch(color=PALETTE_BINS["Low"], label="Low"),
        mpatches.Patch(color=PALETTE_BINS["Medium"], label="Medium"),
        mpatches.Patch(color=PALETTE_BINS["High"], label="High"),
    ]
    ax.legend(
        handles=handles,
        loc="center left",          # legend’s left edge is placed at bbox anchor
        bbox_to_anchor=(1.02, 0.5), # push it to the right of the axes
        ncol=1,
        fontsize=12,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    ax.set_axisbelow(True)
    ax.set_title(r'$\mathbf{c}$  Robustness bin proportions', loc='left', fontsize=14)#, y=1.08)


def draw_alluvial_one(ax, trans_df, title, show_panel_header=False):
  
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    bins_visual = ["High", "Medium", "Low"]
    y = {"High": 0.78, "Medium": 0.50, "Low": 0.22}
    h = 0.16
    rank = {"Low": 0, "Medium": 1, "High": 2}

    left_x, right_x = 0.10, 0.72
    box_w = 0.16

    left_counts = {b: int((trans_df["zero_shot_bin"] == b).sum()) for b in bins_visual}
    right_counts = {b: int((trans_df["agentic_bin"] == b).sum()) for b in bins_visual}

    for b in bins_visual:
        ax.add_patch(
            FancyBboxPatch(
                (left_x, y[b] - h / 2), box_w, h,
                boxstyle="round,pad=0.01",
                facecolor=PALETTE_BINS[b],
                edgecolor="black", lw=0.8, alpha=0.75
            )
        )
        ax.text(left_x + box_w / 2, y[b], f"{left_counts[b]}",
                ha="center", va="center", fontsize=11, fontweight="bold", color="white")

        ax.add_patch(
            FancyBboxPatch(
                (right_x, y[b] - h / 2), box_w, h,
                boxstyle="round,pad=0.01",
                facecolor=PALETTE_BINS[b],
                edgecolor="black", lw=0.8, alpha=0.75
            )
        )
        ax.text(right_x + box_w / 2, y[b], f"{right_counts[b]}",
                ha="center", va="center", fontsize=11, fontweight="bold", color="white")

    flow_counts = []
    for fb in ["Low", "Medium", "High"]:
        for tb in ["Low", "Medium", "High"]:
            c = int(((trans_df["zero_shot_bin"] == fb) & (trans_df["agentic_bin"] == tb)).sum())
            if c > 0:
                flow_counts.append(c)
    max_c = max(flow_counts) if flow_counts else 1

    for fb in ["Low", "Medium", "High"]:
        for tb in ["Low", "Medium", "High"]:
            c = int(((trans_df["zero_shot_bin"] == fb) & (trans_df["agentic_bin"] == tb)).sum())
            if c == 0:
                continue

            if fb == tb:
                col, a = PALETTE_TRANS["No Change"], 0.22
            elif rank[tb] > rank[fb]:
                col, a = PALETTE_TRANS["Improved"], 0.45
            else:
                col, a = PALETTE_TRANS["Decreased"], 0.60

            lw = 1.0 + 6.0 * (c / max_c)

            ax.add_patch(
                FancyArrowPatch(
                    (left_x + box_w, y[fb]),
                    (right_x, y[tb]),
                    connectionstyle="arc3,rad=0.25",
                    arrowstyle="-",
                    lw=lw, color=col, alpha=a, zorder=0
                )
            )

    ax.text(left_x + box_w / 2, 0.95, "Zero-shot", ha="center", va="top", fontsize=10)
    ax.text(right_x + box_w / 2, 0.95, "Agentic", ha="center", va="top", fontsize=10)
    ax.text(0.5, 0.02, title, ha="center", va="bottom", fontsize=11, fontweight="normal")



def plot_panel_d_transitions(ax_container, trans_radio, trans_tum):
    needed = {"zero_shot_bin", "agentic_bin"}
    if not needed.issubset(set(trans_radio.columns)) or not needed.issubset(set(trans_tum.columns)):
        raise ValueError("Transition CSVs must include columns: zero_shot_bin, agentic_bin")

    ax_container.set_title(r"$\mathbf{d}$  Robustness bin transitions",
                           loc="left", fontsize=14)#, y=1.08)

    gs_inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=ax_container, wspace=0.2)
    
    
    axes = [plt.Subplot(ax_container.figure, gs_inner[i]) for i in range(2)]
    for ax in axes:
        ax_container.figure.add_subplot(ax)

    ax_left, ax_right = axes
    pad_axes_in_container(ax_left,  pad_left=0.06, pad_right=0.0, pad_bottom=0.12, pad_top=0.10)
    pad_axes_in_container(ax_right, pad_left=0.03, pad_right=0.06, pad_bottom=0.12, pad_top=0.10)
    
    draw_alluvial_one(ax_left, trans_radio, "Benchmark-RadQA", show_panel_header=False)
    draw_alluvial_one(ax_right, trans_tum, "Board-RadQA", show_panel_header=False)

    draw_alluvial_one(ax_left, trans_radio, "Benchmark-RadQA", show_panel_header=False)
    draw_alluvial_one(ax_right, trans_tum, "Board-RadQA", show_panel_header=False)

    for ax in (ax_left, ax_right):
        for s in ax.spines.values():
            s.set_visible(False)

    # BOX around the whole panel d (container spines on)
    ax_container.set_xticks([])
    ax_container.set_yticks([])
    ax_container.set_xlabel("")
    ax_container.set_ylabel("")
    ax_container.set_facecolor("none")

    for spine in ax_container.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    ax_container.set_zorder(10)
    ax_container.patch.set_alpha(0.0)


def create_figure_4(script_dir):
    print("Loading data...")

    scores_radio = load_csv_checked(os.path.join(script_dir, FILE_SCORES_RADIO))
    scores_tum = load_csv_checked(os.path.join(script_dir, FILE_SCORES_TUM))
    trans_radio = load_csv_checked(os.path.join(script_dir, FILE_TRANS_RADIO))
    trans_tum = load_csv_checked(os.path.join(script_dir, FILE_TRANS_TUM))
    summary_radio = load_csv_checked(os.path.join(script_dir, FILE_SUMMARY_RADIO))
    summary_tum = load_csv_checked(os.path.join(script_dir, FILE_SUMMARY_TUM))

    scores_df = pd.concat([scores_radio, scores_tum], ignore_index=True)
    scores_df = standardize_dataset_names(scores_df)

    fig = plt.figure(figsize=(10, 12))

    outer = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[1.1, 1.1, 1],
        width_ratios=[1, 1],
        wspace=0.32,
        hspace=0.4,
    )

    # Panel a spans full first row
    ax_a = fig.add_subplot(outer[0, :])

    # Panel b container spans full second row
    ax_b = fig.add_subplot(outer[1, :])

    # Bottom row: panel c (left), panel d container (right)
    ax_c = fig.add_subplot(outer[2, 0])
    ax_d = fig.add_subplot(outer[2, 1])

    # Draw panels
    plot_panel_a_scatter(ax_a, scores_df)
    plot_panel_b_distributions(ax_b, scores_df, summary_radio, summary_tum)
    plot_panel_c_bins(ax_c, scores_df)
    plot_panel_d_transitions(ax_d, trans_radio, trans_tum)

    out_path = os.path.join(script_dir, OUTPUT_FIGURE)
    print(f"Saving: {out_path}")
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")

    plt.show()
    plt.close(fig)
    print("Done.")
    return out_path


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Running from: {script_dir}")

    create_figure_4(script_dir)
