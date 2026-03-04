# -*- coding: utf-8 -*-
"""
Figure 6: Comprehensive clinical severity assessment of incorrect model decisions
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


# ==========================================
# --- USER CONFIGURATION ---
# ==========================================
YOUR_EXCEL_FILE = "analyse6_final_v3.xlsx"
YOUR_SHEET_NAME = "Combine"

BAR_W = 0.25
BAR_SPACING = 0.2

# --- Style Setup ---
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.frameon"] = False

# --- Color Palettes ---
PALETTE_SEVERITY = {"Low": "#2ca02c", "Moderate": "#4c72b0", "High": "#d62728"}
PALETTE_AGREEMENT = {
    "Unanimous (3/3)": "#08519c",
    "Majority (2/3)": "#6baed6",
    "No consensus": "#bdbdbd",
}
COLOR_OBSERVED = "#1f77b4"
COLOR_EXPECTED = "#aec7e8"
# ==============================================================================
# --- HELPER FUNCTIONS (plots) ---
# ==============================================================================
def tukey_outliers(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 4:
        return np.array([], dtype=float)
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    return v[(v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)]


def add_jitter_points(ax, x_pos, y_vals, rng, jitter=0.06, size=14, alpha=0.6, c=None, ec=None, lw=0.6):
    y = np.asarray(y_vals, dtype=float)
    y = y[~np.isnan(y)]
    if y.size == 0:
        return
    x = x_pos + rng.normal(0, jitter, size=y.size)
    ax.scatter(x, y, s=size, alpha=alpha, c=c, edgecolors=ec, linewidths=lw, zorder=2)


def draw_styled_violin_single(ax, data, pt_color, vn_color):
    rng = np.random.default_rng(42)
    data = pd.to_numeric(pd.Series(data), errors="coerce").dropna().values
    if data.size == 0:
        ax.set_xticks([])
        return

    add_jitter_points(ax, 1, data, rng, size=15, alpha=0.5, c=pt_color, ec="none")

    vp = ax.violinplot([data], positions=[1], showmeans=False, showmedians=True, showextrema=True, widths=0.8)
    for body in vp["bodies"]:
        body.set_facecolor(vn_color)
        body.set_edgecolor("none")
        body.set_alpha(0.25)
        body.set_zorder(1)
    for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
        vp[part].set_color("grey")
        vp[part].set_linewidth(1)
        vp[part].set_zorder(3)

    ax.boxplot(
        [data],
        positions=[1],
        widths=0.2,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "none", "linewidth": 1.2, "zorder": 4},
        whiskerprops={"linewidth": 1.2, "zorder": 4},
        capprops={"linewidth": 1.2, "zorder": 4},
        medianprops={"color": "black", "linewidth": 1.5, "zorder": 5},
    )

    outs = tukey_outliers(data)
    if outs.size:
        jitter = rng.normal(0, 0.03, size=outs.size)
        ax.scatter(
            np.full(outs.size, 1) + jitter,
            outs,
            s=45,
            facecolors=pt_color,
            edgecolors="black",
            linewidths=1,
            alpha=0.9,
            zorder=6,
        )

    ax.set_xticks([])
# ==============================================================================
# --- Fleiss' kappa + CI + sensitivity ---
# ==============================================================================
def calculate_fleiss_kappa_from_codes(df_codes, code_cols, k):
    
    if df_codes.empty or not all(c in df_codes.columns for c in code_cols):
        return np.nan, np.nan, np.nan, np.array([])

    data = df_codes[code_cols].to_numpy()
    N, n = data.shape
    if N == 0 or n < 2:
        return np.nan, np.nan, np.nan, np.array([])

    n_ij = np.zeros((N, k), dtype=float)
    for j in range(k):
        n_ij[:, j] = np.sum(data == j, axis=1)

    P_i = (np.sum(n_ij ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = np.mean(P_i)

    p_j = np.sum(n_ij, axis=0) / (N * n)
    P_e = np.sum(p_j ** 2)

    denom = 1 - P_e
    if np.isclose(denom, 0):
        kappa = 1.0 if np.isclose(P_bar, 1.0) else np.nan
    else:
        kappa = (P_bar - P_e) / denom

    return P_bar, P_e, kappa, P_i


def bootstrap_fleiss_kappa_ci(df_codes, code_cols, k, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)

    _, _, kappa_hat, _ = calculate_fleiss_kappa_from_codes(df_codes, code_cols, k)
    if df_codes.empty or not np.isfinite(kappa_hat):
        return kappa_hat, np.nan, np.nan, np.array([])

    N = len(df_codes)
    idx = np.arange(N)
    boot = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        sample_idx = rng.choice(idx, size=N, replace=True)
        sample = df_codes.iloc[sample_idx]
        _, _, kb, _ = calculate_fleiss_kappa_from_codes(sample, code_cols, k)
        boot[b] = kb

    boot = boot[np.isfinite(boot)]
    if boot.size < max(30, 0.1 * n_boot):
        return kappa_hat, np.nan, np.nan, boot

    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    return kappa_hat, float(ci_low), float(ci_high), boot


def build_kappa_df_three_class(df, rater_cols=("Sev_LA", "Sev_TTN", "Sev_FBO")):
    valid_map = {"Low": 0, "Moderate": 1, "High": 2}
    df2 = df.copy()
    for c in rater_cols:
        df2[c] = df2[c].astype(str).str.strip()
    mask = df2[list(rater_cols)].isin(list(valid_map.keys())).all(axis=1)
    out = df2.loc[mask, list(rater_cols) + (["question_id"] if "question_id" in df2.columns else [])].copy()
    for c in rater_cols:
        out[c + "_code"] = out[c].map(valid_map)
    return out


def build_kappa_df_binary(df, rater_cols=("Sev_LA", "Sev_TTN", "Sev_FBO")):
    valid_map = {"Low": 0, "Moderate": 1, "High": 1}
    df2 = df.copy()
    for c in rater_cols:
        df2[c] = df2[c].astype(str).str.strip()
    mask = df2[list(rater_cols)].isin(list(valid_map.keys())).all(axis=1)
    out = df2.loc[mask, list(rater_cols) + (["question_id"] if "question_id" in df2.columns else [])].copy()
    for c in rater_cols:
        out[c + "_code"] = out[c].map(valid_map)
    return out


def kappa_with_ci_and_sensitivity(df_full, n_boot=2000, seed=42):
    rater_cols = ("Sev_LA", "Sev_TTN", "Sev_FBO")

    df_k3 = build_kappa_df_three_class(df_full, rater_cols=rater_cols)
    code_cols_3 = [c + "_code" for c in rater_cols]
    Pbar3, Pe3, kappa3, P_i3 = calculate_fleiss_kappa_from_codes(df_k3, code_cols_3, k=3)
    kappa3_hat, lo3, hi3, _ = bootstrap_fleiss_kappa_ci(df_k3, code_cols_3, k=3, n_boot=n_boot, seed=seed)

    df_k2 = build_kappa_df_binary(df_full, rater_cols=rater_cols)
    code_cols_2 = [c + "_code" for c in rater_cols]
    Pbar2, Pe2, kappa2, _ = calculate_fleiss_kappa_from_codes(df_k2, code_cols_2, k=2)
    kappa2_hat, lo2, hi2, _ = bootstrap_fleiss_kappa_ci(df_k2, code_cols_2, k=2, n_boot=n_boot, seed=seed)

    return {
        "three_class": {
            "df_kappa": df_k3,
            "Pbar": Pbar3,
            "Pe": Pe3,
            "kappa": kappa3_hat,
            "ci_low": lo3,
            "ci_high": hi3,
            "P_i": P_i3,
        },
        "binary": {
            "df_kappa": df_k2,
            "Pbar": Pbar2,
            "Pe": Pe2,
            "kappa": kappa2_hat,
            "ci_low": lo2,
            "ci_high": hi2,
        },
    }


# ==============================================================================
# --- DATA LOADING & PREPARATION ---
# ==============================================================================
def _derive_sev_final_from_counts(df):
    needed = ["n_low", "n_moderate", "n_high", "max_count"]
    if not all(c in df.columns for c in needed):
        return df

    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def _row_sev_final(row):
        if pd.isna(row["max_count"]) or row["max_count"] < 2:
            return np.nan
        counts = {"Low": row["n_low"], "Moderate": row["n_moderate"], "High": row["n_high"]}
        if all(pd.isna(v) for v in counts.values()):
            return np.nan
        return max(counts, key=lambda k: (-np.inf if pd.isna(counts[k]) else counts[k]))

    df["Sev_final"] = df.apply(_row_sev_final, axis=1)
    return df


def load_data(excel_path, sheet_name):
    if not os.path.exists(excel_path):
        sys.exit(f"Error: File not found at {excel_path}")

    print(f"Reading: {os.path.basename(excel_path)} | Sheet: {sheet_name}")
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    except Exception as e:
        sys.exit(f"Error reading Excel file: {e}")

    if "is_correct" in df.columns:
        df = df[df["is_correct"] == 0].copy()

    if "max_count" in df.columns:
        df["max_count"] = pd.to_numeric(df["max_count"], errors="coerce")
        df["agreement_label"] = np.select(
            [df["max_count"] == 3, df["max_count"] == 2],
            ["Unanimous (3/3)", "Majority (2/3)"],
            default="No consensus",
        )
    else:
        df["agreement_label"] = "No consensus"
        print("Warning: 'max_count' missing -> agreement_label set to 'No consensus' for all rows.")

    valid_cats = ["Low", "Moderate", "High"]
    if "Sev_final" in df.columns:
        df["Sev_final"] = df["Sev_final"].astype(str).str.strip()
        df.loc[~df["Sev_final"].isin(valid_cats), "Sev_final"] = np.nan
    else:
        df = _derive_sev_final_from_counts(df)

    return df


def calculate_entropy_and_agreement(df_stats):
    if df_stats.empty or "question_id" not in df_stats.columns:
        return pd.DataFrame({"severity_entropy": [], "mean_agreement": []})

    def get_entropy(series):
        s = pd.Series(series).dropna()
        if s.empty:
            return np.nan
        counts = s.value_counts(normalize=True)
        return float(-(counts * np.log2(counts)).sum())

    grouped = df_stats.groupby("question_id", observed=False)
    entropy_df = grouped["Sev_final"].apply(get_entropy).reset_index(name="severity_entropy")
    agreement_df = grouped["P_i_calc"].mean().reset_index(name="mean_agreement")
    return entropy_df.merge(agreement_df, on="question_id", how="left")


# ==============================================================================
# --- PLOTTING ---
# ==============================================================================
def plot_combined_analysis6(df_full, kappa_results):
    three = kappa_results["three_class"]
    binary = kappa_results["binary"]

    P_bar = three["Pbar"]
    P_e = three["Pe"]
    kappa = three["kappa"]
    ci_low = three["ci_low"]
    ci_high = three["ci_high"]

    df_kappa_3 = three["df_kappa"].copy()
    if not df_kappa_3.empty and "Sev_final" in df_full.columns:
        df_kappa_3["Sev_final"] = df_full.loc[df_kappa_3.index, "Sev_final"] if len(df_full.index.intersection(df_kappa_3.index)) else np.nan
    else:
        df_kappa_3["Sev_final"] = np.nan

    if len(three["P_i"]) == len(df_kappa_3):
        df_kappa_3["P_i_calc"] = three["P_i"]
    else:
        df_kappa_3["P_i_calc"] = np.nan

    question_stats_df = calculate_entropy_and_agreement(df_kappa_3)

    fig = plt.figure(figsize=(11, 12))
    gs = gridspec.GridSpec(3, 6, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.85)

    # ================= Panel a: agreement levels =================
    ax_a = fig.add_subplot(gs[0, 0:2])
    if "agreement_label" in df_full.columns and not df_full.empty:
        counts_a = df_full["agreement_label"].value_counts(normalize=True)
        order_a = ["Unanimous (3/3)", "Majority (2/3)", "No consensus"]
        counts_a = counts_a.reindex(order_a, fill_value=0) * 100

        BAR_SPACING_A = 0.5
        x_pos = np.arange(len(order_a)) * BAR_SPACING_A

        colors = [PALETTE_AGREEMENT[label] for label in order_a]
        values = [counts_a[label] for label in order_a]

        bars = ax_a.bar(x_pos, values, color=colors, edgecolor="white", width=BAR_W)
        for bar in bars:
            h = bar.get_height()
            ax_a.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.0f}%", ha="center", va="bottom", fontsize=12)

        ax_a.set_xticks(x_pos)
        ax_a.set_xticklabels([l.replace(" ", "\n") for l in order_a])

        # Optional: quick sanity print (won't affect plot)
        # print("Panel A %:", counts_a.to_dict())

    ax_a.set_ylim(0, 80)
    ax_a.set_ylabel("Proportion (%)")
    ax_a.set_title(r"$\mathbf{a}$   Inter-rater agreement levels", loc="left", fontsize=14)
    # ================= Panel b: observed vs expected + kappa CI + sensitivity =================
    ax_b = fig.add_subplot(gs[0, 2:4])

    bars_labels = [r"Observed ($\bar{P}$)", r"Expected ($\bar{P}_e$)"]
    values = [P_bar, P_e]

    BAR_SPACING_B = 0.4
    x = np.array([-0.5, 0.5]) * BAR_SPACING_B

    ax_b.bar(x, values, color=[COLOR_OBSERVED, COLOR_EXPECTED], width=BAR_W * 0.7)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(bars_labels)

    vmax = np.nanmax(values) if np.isfinite(np.nanmax(values)) else 0
    ax_b.set_ylim(0, (vmax * 1.35) if vmax > 0 else 1.0)
    ax_b.set_ylabel("Agreement proportion")
    ax_b.set_xlim(-BAR_SPACING_B * 1.2, BAR_SPACING_B * 1.2)

    for i, v in enumerate(values):
        ax_b.text(x[i], (v + 0.01) if np.isfinite(v) else 0.02, f"{v:.2f}" if np.isfinite(v) else "NA",
                  ha="center", fontsize=12)

    # Kappa + CI text
    kappa_txt = f"{kappa:.2f}" if np.isfinite(kappa) else "NA"
    ci_txt = f"[{ci_low:.2f}, {ci_high:.2f}]" if np.isfinite(ci_low) and np.isfinite(ci_high) else "[NA, NA]"
    ax_b.text(0.5, 0.93, f"Fleiss' $\\kappa$ = {kappa_txt} (95% CI {ci_txt})",
              transform=ax_b.transAxes, ha="center", fontsize=11)

    # Sensitivity (binary) text
    k2 = binary["kappa"]
    lo2 = binary["ci_low"]
    hi2 = binary["ci_high"]
    k2_txt = f"{k2:.3f}" if np.isfinite(k2) else "NA"
    ci2_txt = f"[{lo2:.3f}, {hi2:.3f}]" if np.isfinite(lo2) and np.isfinite(hi2) else "[NA, NA]"

    ax_b.set_title(r"$\mathbf{b}$   Observed vs. expected agreement", loc="left", fontsize=14)
    # ================= Panel c: overall severity composition =================
    ax_c = fig.add_subplot(gs[0, 4:6])
    valid_cats = ["Low", "Moderate", "High"]

    if "Sev_final" in df_full.columns and not df_full.empty:
        sev = df_full["Sev_final"]
        counts_c = sev.value_counts(dropna=True).reindex(valid_cats, fill_value=0)
        total_c = counts_c.sum()
        props_c = counts_c / total_c * 100 if total_c > 0 else counts_c * 0

        x_pos = np.arange(len(valid_cats)) * BAR_SPACING
        colors = [PALETTE_SEVERITY[cat] for cat in valid_cats]
        vals_w = [props_c[cat] for cat in valid_cats]
        vals_n = [counts_c[cat] for cat in valid_cats]

        bars_c = ax_c.bar(x_pos, vals_w, color=colors, edgecolor="white", width=0.10)
        for i, bar in enumerate(bars_c):
            h = bar.get_height()
            ax_c.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.0f}%\n(N={vals_n[i]})",
                      ha="center", va="bottom", fontsize=12)

        ax_c.set_xticks(x_pos)
        ax_c.set_xticklabels(valid_cats)

    ax_c.set_ylim(0, 70)
    ax_c.set_ylabel("Proportion of incorrect options (%)")
    ax_c.set_title(r"$\mathbf{c}$   Overall severity composition", loc="left", fontsize=14)
    # ================= Panel d: severity profile by dataset =================
    ax_d = fig.add_subplot(gs[1, :])
    if "dataset" in df_full.columns and "Sev_final" in df_full.columns and not df_full.empty:
        df_plot = df_full.copy()

        df_pooled = df_plot.copy()
        df_pooled["dataset_clean"] = "Pooled"

        df_split = df_plot.copy()
        df_split["dataset_clean"] = df_split["dataset"].replace(
            {
                "Internal_TUM": "Board-RadQA",
                "Board-RadQA": "Board-RadQA",
                "Benchmark-RadQA": "Benchmark-RadQA",
                "RadioRAG": "Benchmark-RadQA",
            }
        )

        df_combined = pd.concat([df_pooled, df_split], ignore_index=True)
        df_combined = df_combined[df_combined["Sev_final"].isin(valid_cats)].copy()

        group_order = ["Pooled", "Benchmark-RadQA", "Board-RadQA"]
        existing_groups = [g for g in group_order if g in df_combined["dataset_clean"].unique()]

        if existing_groups and not df_combined.empty:
            props = (
                df_combined.groupby(["dataset_clean", "Sev_final"], observed=False)
                .size()
                .reset_index(name="count")
            )
            totals = props.groupby("dataset_clean")["count"].transform("sum")
            props["percent"] = (props["count"] / totals) * 100

            sns.barplot(
                data=props,
                x="dataset_clean",
                y="percent",
                hue="Sev_final",
                palette=PALETTE_SEVERITY,
                hue_order=valid_cats,
                order=existing_groups,
                ax=ax_d,
                saturation=0.9,
                width=0.5,
            )

            GROUP_COMPRESS = 0.9
            HUE_GAP = 0.7

            for p in ax_d.patches:
                x_center = p.get_x() + p.get_width() / 2
                new_center = x_center * GROUP_COMPRESS
                p.set_x(new_center - p.get_width() / 2)

            ticks = np.arange(len(existing_groups)) * GROUP_COMPRESS
            ax_d.set_xticks(ticks)
            ax_d.set_xticklabels(existing_groups)

            for p in ax_d.patches:
                w = p.get_width()
                new_w = w * HUE_GAP
                p.set_x(p.get_x() + (w - new_w) / 2)
                p.set_width(new_w)

            for c in ax_d.containers:
                ax_d.bar_label(c, fmt="%.0f%%", padding=1, fontsize=12)

            ax_d.legend(title="", loc="upper right", frameon=True, fontsize=12, bbox_to_anchor=(1, 1))

    ax_d.set_ylabel("Proportion (%)")
    ax_d.set_xlabel("")
    ax_d.set_ylim(0, 70)
    ax_d.set_title(r"$\mathbf{d}$   Severity profile by dataset", loc="left", fontsize=14)
    # ================= Panel e: per-question agreement distribution =================
    ax_e = fig.add_subplot(gs[2, 0:2])
    data_e = question_stats_df["mean_agreement"] if "mean_agreement" in question_stats_df.columns else pd.Series(dtype=float)

    if not data_e.empty:
        draw_styled_violin_single(ax_e, data_e, pt_color="tab:orange", vn_color="tab:blue")
        if np.isfinite(P_e):
            ax_e.axhline(P_e, color="black", linestyle="--", label=r"Exp ($P_e$)")

    ax_e.set_ylabel("Mean agreement per question")
    ax_e.set_ylim(0, 1.1)
    if np.isfinite(P_e):
        ax_e.legend(fontsize=12)
    ax_e.set_title(r"$\mathbf{e}$   Per-question agreement distribution", loc="left", fontsize=14)
    # ================= Panel f: per-question severity entropy =================
    ax_f = fig.add_subplot(gs[2, 2:6])
    if not question_stats_df.empty and "severity_entropy" in question_stats_df.columns:
        sns.histplot(
            question_stats_df["severity_entropy"].dropna(),
            ax=ax_f,
            element="step",
            fill=True,
            color="#1f77b4",
            alpha=0.6,
            bins=15,
        )

    ax_f.set_xlabel("Severity entropy")
    ax_f.set_ylabel("Number of questions")
    ax_f.set_title(r"$\mathbf{f}$   Per-question severity entropy", loc="left", fontsize=14)

    plt.subplots_adjust(top=0.93, bottom=0.05, left=0.04, right=0.98)

    output_filename = "Figure6_Analysis6_V3.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {output_filename}")
    plt.show()
# ==============================================================================
# --- MAIN ---
# ==============================================================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(script_dir, YOUR_EXCEL_FILE)

    print(f"Loading data from: {excel_path}")
    df_full = load_data(excel_path, YOUR_SHEET_NAME)
    print(f"Loaded df_full rows (incorrect options only): {len(df_full)}")

    kappa_results = kappa_with_ci_and_sensitivity(df_full, n_boot=2000, seed=42)

    three = kappa_results["three_class"]
    binary = kappa_results["binary"]
    print("\n=== Fleiss' kappa (3-class: Low/Moderate/High) ===")
    print(f"N used: {len(three['df_kappa'])}")
    print(f"Observed P̄: {three['Pbar']:.3f} | Expected P̄e: {three['Pe']:.3f}")
    print(f"κ: {three['kappa']:.3f} | 95% CI: [{three['ci_low']:.3f}, {three['ci_high']:.3f}]")

    print("\n=== Sensitivity (binary: Low vs Moderate/High) ===")
    print(f"N used: {len(binary['df_kappa'])}")
    print(f"Observed P̄: {binary['Pbar']:.3f} | Expected P̄e: {binary['Pe']:.3f}")
    print(f"κ: {binary['kappa']:.3f} | 95% CI: [{binary['ci_low']:.3f}, {binary['ci_high']:.3f}]")

    plot_combined_analysis6(df_full, kappa_results)
