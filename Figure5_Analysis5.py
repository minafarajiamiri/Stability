# =============================================================================
# Figure 5: Consensus–Robustness Coupling Analysis
# =============================================================================
# Requirements: pandas, numpy, matplotlib, seaborn, openpyxl
#
# Put all input files in the SAME folder as this script (recommended).
# Output will be saved to the same folder unless OUTPUT_DIR is changed.
# =============================================================================

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # kept for consistency across your figure scripts
from matplotlib.patches import Rectangle


# =============================================================================
# --- USER CONFIGURATION ---
# =============================================================================
FILE_CORR = "step2_correlations.csv"
FILE_CONSENSUS = "step3_consensus_correctness.csv"
FILE_ANOMALOUS = "step4_anomalous_cases_dataset.csv"

FILE_EXCEL = "analyse2_final_v2.xlsx"
SHEET_MAJORITY = "majority"

FILE_ROBUST_RADIO = "robustness_scores_Radiorag_dataset.csv"
FILE_ROBUST_TUM = "robustness_scores_internal_TUM_dataset.csv"

OUTPUT_NAME = "Figure5_Analysis5.png"
OUTPUT_DIR = None  # None -> script folder

RNG_SEED = 42

THRESH_M_HIGH = 0.80
THRESH_R_LOW = 0.40

COLOR_INCORRECT_REGION = "#d62728"


# =============================================================================
# --- STYLE SETUP ---
# =============================================================================
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["legend.frameon"] = False
plt.rcParams["axes.grid"] = False


# =============================================================================
# --- IO HELPERS ---
# =============================================================================
def load_csv_checked(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def load_excel_checked(path, sheet_name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")


def standardize_dataset_names(df):
    """
    Normalize dataset labels across inputs (defensive).
    Adjust mapping here if your upstream files change naming.
    """
    if df is None or df.empty:
        return df

    mapping = {
        "Radiorag dataset": "Benchmark-RadQA",
        "RadioRAG": "Benchmark-RadQA",
        "radiorag": "Benchmark-RadQA",
        "internal TUM dataset": "Board-RadQA",
        "Internal_TUM": "Board-RadQA",
        "internal_tum": "Board-RadQA",
        "Internal TUM": "Board-RadQA",
    }
    return df.replace(mapping)


# =============================================================================
# --- PLOT HELPERS ---
# =============================================================================
def loess_smooth_interp(x, y, n=120):
    """
    Lightweight smoothing (sorted linear interpolation; stable, no extra deps).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 5:
        return x, y

    sidx = np.argsort(x)
    xs = x[sidx]
    ys = y[sidx]
    xg = np.linspace(xs.min(), xs.max(), n)
    yg = np.interp(xg, xs, ys)
    return xg, yg


def add_jitter_points(ax, x_pos, y_vals, rng, jitter=0.06, size=12, alpha=0.55,
                      c=None, ec="white", lw=0.4, zorder=3):
    y = np.asarray(y_vals, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return
    x = x_pos + rng.normal(0, jitter, size=y.size)
    ax.scatter(x, y, s=size, alpha=alpha, c=c, edgecolors=ec, linewidths=lw, zorder=zorder)


# =============================================================================
# --- LOAD & MERGE DATA ---
# =============================================================================
def load_and_merge_data(data_dir):
    print(f"Loading data from: {os.path.abspath(data_dir)}")

    corr_df = load_csv_checked(os.path.join(data_dir, FILE_CORR))
    consensus_df = load_csv_checked(os.path.join(data_dir, FILE_CONSENSUS))
    anomalous_df = load_csv_checked(os.path.join(data_dir, FILE_ANOMALOUS))
    majority_df = load_excel_checked(os.path.join(data_dir, FILE_EXCEL), SHEET_MAJORITY)

    robust_radio = load_csv_checked(os.path.join(data_dir, FILE_ROBUST_RADIO))
    robust_tum = load_csv_checked(os.path.join(data_dir, FILE_ROBUST_TUM))

    robust_radio["dataset"] = "Benchmark-RadQA"
    robust_tum["dataset"] = "Board-RadQA"
    robust_df = pd.concat([robust_radio, robust_tum], ignore_index=True)

    # Standardize names (defensive)
    majority_df = standardize_dataset_names(majority_df)
    robust_df = standardize_dataset_names(robust_df)

    needed_major = {"dataset", "question_id", "method"}
    needed_rob = {"dataset", "question_id", "method", "robustness_score"}

    missing_major = sorted(list(needed_major - set(majority_df.columns)))
    missing_rob = sorted(list(needed_rob - set(robust_df.columns)))

    if missing_major:
        raise ValueError(f"Majority sheet missing columns: {missing_major}")
    if missing_rob:
        raise ValueError(f"Robustness CSVs missing columns: {missing_rob}")

    majority_df["match_key"] = (
        majority_df["dataset"].astype(str).str.strip()
        + "|"
        + majority_df["question_id"].astype(str)
        + "|"
        + majority_df["method"].astype(str).str.strip()
    )
    robust_df["match_key"] = (
        robust_df["dataset"].astype(str).str.strip()
        + "|"
        + robust_df["question_id"].astype(str)
        + "|"
        + robust_df["method"].astype(str).str.strip()
    )

    merged_df = majority_df.merge(
        robust_df[["match_key", "robustness_score"]],
        on="match_key",
        how="left",
    )

    # Numeric coercion
    if "majority_fraction" in merged_df.columns:
        merged_df["majority_fraction"] = pd.to_numeric(merged_df["majority_fraction"], errors="coerce")
    merged_df["robustness_score"] = pd.to_numeric(merged_df["robustness_score"], errors="coerce")

    print("✓ Data loaded successfully\n")
    return corr_df, consensus_df, anomalous_df, merged_df


# =============================================================================
# --- PANEL a (container + diagram in one function) ---
# =============================================================================
def plot_panel_a(fig, gs_cell):
    """
    Panel a: Metric schematic
    Container axis holds the title; two inner axes show example bar charts.
    """
    ax_panel = fig.add_subplot(gs_cell)
    ax_panel.axis("off")
    # Put panel label on the container axis so it doesn't mess with subplot titles
    # ax_panel.text(
    #     0.0, 1.2, r"$\mathbf{a}$  Metric schematic",
    #     transform=ax_panel.transAxes,  ha="left", va="bottom", fontsize=14
    # )
    
    sub_gs = gs_cell.subgridspec(1, 2, wspace=0.35)
    ax_left = fig.add_subplot(sub_gs[0, 0])
    ax_right = fig.add_subplot(sub_gs[0, 1])

    ax_left.set_title(r"$\mathbf{a}$  Metric schematic", loc="left", pad=4, fontsize=14, y=1.1)
    # Example 1: High M, High R
    answers_1 = ["C"] * 1 + ["A"] * 2 + ["B"] * 21 + ["D"] * 1
    counts_1 = pd.Series(answers_1).value_counts().reindex(["A", "B", "C", "D"], fill_value=0)
    colors_1 = ["#2ca02c" if a == "B" else "#cccccc" for a in counts_1.index]

    ax_left.bar(range(4), counts_1.values, color=colors_1, edgecolor="black", linewidth=1.0)
    ax_left.set_xticks(range(4))
    ax_left.set_xticklabels(["A", "B", "C", "D"])
    ax_left.set_ylabel("Number of models")
    ax_left.set_ylim(0, 25)
    ax_left.set_title("Example 1: High M, High R", fontsize=11, pad=8)
    ax_left.text(
        0.98, 0.88, "M = 0.87", transform=ax_left.transAxes, ha="right", va="center",
        fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5, edgecolor="none")
    )
    ax_left.text(
        0.98, 0.74, "R = 0.84", transform=ax_left.transAxes, ha="right", va="center",
        fontsize=10, bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5, edgecolor="none")
    )
    ax_left.set_xlabel("Answer option", fontsize=11)

    # Example 2: High M, Low R
    answers_2 = ["A"] * 20 + ["C"] * 1 + ["D"] * 4
    counts_2 = pd.Series(answers_2).value_counts().reindex(["A", "B", "C", "D"], fill_value=0)
    colors_2 = ["#2ca02c" if a == "C" else "#cccccc" for a in counts_2.index]

    ax_right.bar(range(4), counts_2.values, color=colors_2, edgecolor="black", linewidth=1.0)
    ax_right.set_xticks(range(4))
    ax_right.set_xticklabels(["A", "B", "C", "D"])
    #ax_right.set_ylabel("Number of models")
    ax_right.set_ylim(0, 25)
    ax_right.set_title("Example 2: High M, Low R", fontsize=11, pad=8)
    ax_right.text(
        0.98, 0.88, "M = 0.83", transform=ax_right.transAxes, ha="right", va="center",
        fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5, edgecolor="none")
    )
    ax_right.text(
        0.98, 0.74, "R = 0.04", transform=ax_right.transAxes, ha="right", va="center",
        fontsize=10, bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5, edgecolor="none")
    )
    ax_right.set_xlabel("Answer option", fontsize=11)
    #ax_left.set_title(r"$\mathbf{a}$  Metric schematic", loc="left", pad=12, fontsize=14, y=1.1)
    return ax_panel


# =============================================================================
# --- PANEL b (container + 2 scatters in one function) ---
# =============================================================================
# def plot_panel_b(fig, gs_cell, merged_df):
#     """
#     Panel b: Consensus–robustness coupling
#     Title is attached using ax.set_title()
#     """

#     # --- Container axis ---
#     ax_panel = fig.add_subplot(gs_cell)
#     ax_panel.set_xticks([])
#     ax_panel.set_yticks([])
#     ax_panel.set_frame_on(False)

#     ax_panel.set_title(r"$\mathbf{b}$  Consensus–robustness coupling", loc="left", pad=12, fontsize=14, y=1.12)

#     # --- Inner layout ---
#     sub_gs = gs_cell.subgridspec(1, 2, wspace=0.45)
#     ax_b1 = fig.add_subplot(sub_gs[0, 0])
#     ax_b2 = fig.add_subplot(sub_gs[0, 1])

#     def loess_smooth_interp(x, y, n=120):
#         x = np.asarray(x, dtype=float)
#         y = np.asarray(y, dtype=float)
#         m = np.isfinite(x) & np.isfinite(y)
#         x, y = x[m], y[m]
#         if x.size < 5:
#             return x, y
#         sidx = np.argsort(x)
#         xs, ys = x[sidx], y[sidx]
#         xg = np.linspace(xs.min(), xs.max(), n)
#         yg = np.interp(xg, xs, ys)
#         return xg, yg

#     def scatter_facet(ax, method, dataset, facet_title):
#         sub = merged_df[(merged_df["method"] == method) &
#                         (merged_df["dataset"] == dataset)].copy()

#         ax.scatter(
#             sub["majority_fraction"],
#             sub["robustness_score"],
#             s=45,
#             alpha=0.6,
#             color="#555555",
#             edgecolors="none",
#         )

#         valid = sub.dropna(subset=["majority_fraction", "robustness_score"])
#         if len(valid) >= 5:
#             xs, ys = loess_smooth_interp(
#                 valid["majority_fraction"].values,
#                 valid["robustness_score"].values )
#             ax.plot(xs, ys, color="#1f77b4", lw=2)

#         ax.set_xlim(0, 1.05)
#         ax.set_ylim(0, 1.05)
#         ax.set_xlabel("Majority fraction (M)",fontsize=11)
#         ax.set_ylabel("Robustness (R)",fontsize=11)
#         ax.set_title(facet_title, fontsize=11)#, style="italic")

#     scatter_facet(ax_b1, "zero-shot", "Benchmark-RadQA", "Benchmark-RadQA | Zero-shot")
#     scatter_facet(ax_b2, "zero-shot", "Board-RadQA", "Board-RadQA | Zero-shot")

#     return ax_panel
def plot_panel_b(fig, gs_cell, merged_df):
    """
    Panel b: Consensus–robustness coupling
    2x2 grid: top row = zero-shot, bottom row = agentic
    """

    # --- Container axis ---
    ax_panel = fig.add_subplot(gs_cell)
    ax_panel.set_xticks([])
    ax_panel.set_yticks([])
    ax_panel.set_frame_on(False)

    ax_panel.set_title(r"$\mathbf{b}$  Consensus–robustness coupling", loc="left", pad=4, fontsize=14, y=1.12)

    # --- Inner layout (2 rows x 2 cols) ---
    sub_gs = gs_cell.subgridspec(2, 2, wspace=0.45, hspace=0.68)
    ax_b1 = fig.add_subplot(sub_gs[0, 0])
    ax_b2 = fig.add_subplot(sub_gs[0, 1])
    ax_b3 = fig.add_subplot(sub_gs[1, 0])
    ax_b4 = fig.add_subplot(sub_gs[1, 1])

    def loess_smooth_interp(x, y, n=120):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size < 5:
            return x, y
        sidx = np.argsort(x)
        xs, ys = x[sidx], y[sidx]
        xg = np.linspace(xs.min(), xs.max(), n)
        yg = np.interp(xg, xs, ys)
        return xg, yg

    def scatter_facet(ax, method, dataset, facet_title):
        sub = merged_df[(merged_df["method"] == method) &
                        (merged_df["dataset"] == dataset)].copy()

        ax.scatter(
            sub["majority_fraction"],
            sub["robustness_score"],
            s=45,
            alpha=0.6,
            color="#555555",
            edgecolors="none",
        )

        valid = sub.dropna(subset=["majority_fraction", "robustness_score"])
        if len(valid) >= 5:
            xs, ys = loess_smooth_interp(
                valid["majority_fraction"].values,
                valid["robustness_score"].values )
            ax.plot(xs, ys, color="#1f77b4", lw=2)

        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.5, 1.0])
        ax.set_xlabel("Majority fraction (M)", fontsize=11)
        ax.set_ylabel("Robustness (R)", fontsize=11)
        ax.set_title(facet_title, fontsize=11)

    scatter_facet(ax_b1, "zero-shot", "Benchmark-RadQA", "Benchmark-RadQA | Zero-shot")
    scatter_facet(ax_b2, "zero-shot", "Board-RadQA", "Board-RadQA | Zero-shot")
    scatter_facet(ax_b3, "agentic", "Benchmark-RadQA", "Benchmark-RadQA | Agentic")
    scatter_facet(ax_b4, "agentic", "Board-RadQA", "Board-RadQA | Agentic")

    return ax_panel

# =============================================================================
# --- PANEL c (container + 4 violins in one function) ---
# =============================================================================
def plot_panel_c(fig, gs_cell, merged_df, rng_seed=42):
    """
    Panel c: Consensus by majority correctness
    One function creates:
      - 1 container axis (title via ax.set_title)
      - 4 inner violin facets (1 x 4)
    """

    # --- Container axis ---
    ax_panel = fig.add_subplot(gs_cell)
    ax_panel.set_xticks([])
    ax_panel.set_yticks([])
    ax_panel.set_frame_on(False)

    ax_panel.set_title(r"$\mathbf{c}$  Consensus by majority correctness", loc="left", pad=12, fontsize=14, y=1.1)

    # --- Inner layout (1 row x 4 cols) ---
    sub_gs = gs_cell.subgridspec(1, 4, wspace=0.45)
    ax_c1 = fig.add_subplot(sub_gs[0, 0])
    ax_c2 = fig.add_subplot(sub_gs[0, 1])
    ax_c3 = fig.add_subplot(sub_gs[0, 2])
    ax_c4 = fig.add_subplot(sub_gs[0, 3])

    rng = np.random.default_rng(rng_seed)

    def add_jitter_points(ax, x_pos, y_vals, jitter=0.06, size=12, alpha=0.55,
                          c=None, ec="white", lw=0.4, zorder=3):
        y = np.asarray(y_vals, dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            return
        x = x_pos + rng.normal(0, jitter, size=y.size)
        ax.scatter(x, y, s=size, alpha=alpha, c=c, edgecolors=ec, linewidths=lw, zorder=zorder)

    def violin_facet(ax, method, dataset, facet_title):
        sub = merged_df[(merged_df["method"] == method) &
                        (merged_df["dataset"] == dataset)].copy()

        if sub.empty or ("majority_correct" not in sub.columns) or ("majority_fraction" not in sub.columns):
            ax.text(0.5, 0.5, "No / incomplete data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return

        correct = sub.loc[sub["majority_correct"] == 1, "majority_fraction"].astype(float).dropna().to_numpy()
        incorrect = sub.loc[sub["majority_correct"] == 0, "majority_fraction"].astype(float).dropna().to_numpy()

        # Jittered points
        add_jitter_points(ax, 1, correct, c="tab:orange")
        add_jitter_points(ax, 2, incorrect, c="tab:red")

        data = [correct, incorrect]

        # Violin plot
        vp = ax.violinplot(
            data, positions=[1, 2], widths=0.75,
            showmeans=False, showmedians=True, showextrema=True
        )

        # Violin bodies
        for body, col in zip(vp["bodies"], ["tab:blue", "tab:green"]):
            body.set_facecolor(col)
            body.set_edgecolor("none")
            body.set_alpha(0.22)
            body.set_zorder(1)

        # Violin lines
        for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
            vp[part].set_color("grey")
            vp[part].set_linewidth(1.0)
            vp[part].set_zorder(4)

        # Box overlay
        ax.boxplot(
            data, positions=[1, 2], widths=0.25, showfliers=False, patch_artist=True,
            boxprops=dict(facecolor="none", linewidth=1.0),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            medianprops=dict(color="black", linewidth=1.3),
        )

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Correct", "Incorrect"])
        ax.set_ylabel("Majority fraction")
        ax.set_ylim(0, 1.05)
        ax.set_title(facet_title, fontsize=11)#, style="italic")

    # --- 4 facets ---
    violin_facet(ax_c1, "zero-shot", "Benchmark-RadQA", "Benchmark-RadQA | Zero-shot")
    violin_facet(ax_c2, "zero-shot", "Board-RadQA", "Board-RadQA | Zero-shot")
    violin_facet(ax_c3, "agentic", "Benchmark-RadQA", "Benchmark-RadQA | Agentic")
    violin_facet(ax_c4, "agentic", "Board-RadQA", "Board-RadQA | Agentic")

    return ax_panel


# =============================================================================
# --- PANEL d (full-width anomalous plot) ---
# =============================================================================
def plot_panel_d_anomalous(ax, merged_df, anomalous_df):
    df = merged_df.copy()
    df = standardize_dataset_names(df)

    if ("majority_fraction" not in df.columns) or ("robustness_score" not in df.columns):
        ax.text(0.5, 0.5, "Missing columns", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    high = df[df["majority_fraction"] >= THRESH_M_HIGH].copy()

    if "majority_correct" in high.columns:
        correct = high[high["majority_correct"] == 1]
        incorrect = high[high["majority_correct"] == 0]
    else:
        correct = high
        incorrect = high.iloc[0:0].copy()

    ax.scatter(
        correct["majority_fraction"], correct["robustness_score"],
        s=55, alpha=0.25, color="#999999", edgecolors="none",
        label="High M, Correct", zorder=1
    )
    ax.scatter(
        incorrect["majority_fraction"], incorrect["robustness_score"],
        s=95, alpha=0.85, facecolors="tab:green",
        edgecolors="darkred", linewidths=1.5,
        label="High M, Incorrect", zorder=3
    )

    ax.axvline(THRESH_M_HIGH, color="gray", linestyle="--", lw=1.0, alpha=0.7)
    ax.axhline(THRESH_R_LOW, color="gray", linestyle="--", lw=1.0, alpha=0.7)

    ax.add_patch(
        Rectangle(
            (THRESH_M_HIGH, 0.0),
            1.05 - THRESH_M_HIGH,
            THRESH_R_LOW,
            facecolor=COLOR_INCORRECT_REGION,
            alpha=0.08,
            edgecolor="none",
            zorder=0,
        )
    )

    ax.text(
        0.92, 0.20,
        f"Anomalous region\n(M≥{THRESH_M_HIGH:.2f}, R<{THRESH_R_LOW:.2f})",
        ha="center", va="center", fontsize=11, style="italic",
        color=COLOR_INCORRECT_REGION, fontweight="bold",
        transform=ax.transAxes,
    )

    zero = 0
    agen = 0
    if anomalous_df is not None and not anomalous_df.empty and ("Method" in anomalous_df.columns):
        counts = anomalous_df.groupby("Method").size()
        zero = int(counts.get("zero-shot", 0))
        agen = int(counts.get("agentic", 0))

    ax.text(
        0.02, 0.98,
        f"Anomalous cases:\nZero-shot: {zero}\nAgentic: {agen}",
        transform=ax.transAxes, ha="left", va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.90, edgecolor="gray", linewidth=1.0),
    )

    ax.set_xlim(0.75, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Majority fraction (M)", fontsize=11)
    ax.set_ylabel("Robustness (R)")
    ax.legend(loc="lower left", fontsize=11, frameon=True, edgecolor="gray")
    ax.set_title(r"$\mathbf{d}$  Coordinated but incorrect convergence", loc="left", fontsize=14, pad=12)


# =============================================================================
# --- MAIN FIGURE BUILDER ---
# =============================================================================
def create_figure_5(data_dir, output_dir):
    corr_df, consensus_df, anomalous_df, merged_df = load_and_merge_data(data_dir)

    # fig = plt.figure(figsize=(12, 10))

    # # Outer layout: (a | b) on top row, c full row, d full row
    # outer = fig.add_gridspec(
    #     nrows=5, ncols=2,
    #     height_ratios=[1.0, 0.25, 1.3, 0.0, 1.15],
    #     width_ratios=[1.0, 1.0],
    #     wspace=0.35, hspace=0.4 )

    # Change figure height from 10 to ~13
    fig = plt.figure(figsize=(12, 13))

    # Increase first height ratio to give panels a & b more room
    outer = fig.add_gridspec(
        nrows=5, ncols=2,
        height_ratios=[1.8, 0.25, 1.3, 0.0, 1.15],
        width_ratios=[1.0, 1.0],
        wspace=0.35, hspace=0.3
    )
    # Panel a and b
    plot_panel_a(fig, outer[0, 0])
    plot_panel_b(fig, outer[0, 1], merged_df)

    # Panel c (full width)
    plot_panel_c(fig, outer[2, :], merged_df)

    # Panel d (full width)
    ax_d = fig.add_subplot(outer[4, :])
    plot_panel_d_anomalous(ax_d, merged_df, anomalous_df)
    

    # Save
    out_path = os.path.join(output_dir, OUTPUT_NAME)
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    print(f"✓ Saved: {os.path.abspath(out_path)}")

    return fig, out_path

# =============================================================================
# --- RUN ---
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Figure 5: Consensus–Robustness Coupling")
    print("=" * 60 + "\n")

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = script_dir
        output_dir = OUTPUT_DIR if OUTPUT_DIR is not None else script_dir

        fig, out_path = create_figure_5(data_dir=data_dir, output_dir=output_dir)
        plt.show()
        plt.close(fig)

        print("✓ Complete!\n")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
