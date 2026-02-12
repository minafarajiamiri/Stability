# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 12:14:25 2026

@author: amiri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sys
import os
 
# ==========================================
# --- USER CONFIGURATION ---
# ==========================================
YOUR_EXCEL_FILE = 'analyse6_final_v2.xlsx'
YOUR_SHEET_NAME = 'Combine'

BAR_W = 0.25  # same width for ALL bar charts 
BAR_SPACING = 0.2   # <1 = bars closer, >1 = bars farther apart


# --- Style Setup ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.frameon'] = False

# --- Color Palettes ---
PALETTE_SEVERITY = {'Low': '#2ca02c', 'Moderate': '#4c72b0', 'High': '#d62728'} #'Moderate': '#ff7f0e',
PALETTE_AGREEMENT = {'Unanimous (3/3)': '#08519c', 'Majority (2/3)': '#6baed6', 'No consensus': '#bdbdbd'}
COLOR_OBSERVED = '#1f77b4'
COLOR_EXPECTED = '#aec7e8'

# ==============================================================================
# --- HELPER FUNCTIONS ---
# ==============================================================================
def tukey_outliers(values):
    """Return outlier values using Tukey rule (1.5*IQR)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 4:
        return np.array([], dtype=float)
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    return v[(v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)]

def add_jitter_points(ax, x_pos, y_vals, rng, jitter=0.06, size=14, alpha=0.6, c=None, ec=None, lw=0.6):
    """Adds jittered points to a plot."""
    y = np.asarray(y_vals, dtype=float)
    y = y[~np.isnan(y)]
    x = x_pos + rng.normal(0, jitter, size=y.size)
    ax.scatter(x, y, s=size, alpha=alpha, c=c, edgecolors=ec, linewidths=lw, zorder=2)

def draw_styled_violin_single(ax, data, pt_color, vn_color):
    """
    Helper function to draw a single styled violin plot.
    Matches the reference style: Jittered points -> Violin -> Box -> Outliers.
    """
    rng = np.random.default_rng(42)

    # 1. Jittered Points
    add_jitter_points(ax, 1, data, rng, size=15, alpha=0.5, c=pt_color, ec="none")

    # 2. Violin
    vp = ax.violinplot([data], positions=[1], showmeans=False, showmedians=True, showextrema=True, widths=0.8)
    for body in vp["bodies"]:
        body.set_facecolor(vn_color)
        body.set_edgecolor('none')
        body.set_alpha(0.25)
        body.set_zorder(1)
    for part in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
        vp[part].set_color('grey')
        vp[part].set_linewidth(1)
        vp[part].set_zorder(3)

    # 3. Box plot
    ax.boxplot([data], positions=[1], widths=0.2, showfliers=False, patch_artist=True,
               boxprops={'facecolor': 'none', 'linewidth': 1.2, 'zorder': 4},
               whiskerprops={'linewidth': 1.2, 'zorder': 4},
               capprops={'linewidth': 1.2, 'zorder': 4},
               medianprops={'color': 'black', 'linewidth': 1.5, 'zorder': 5})

    # 4. Outliers
    outs = tukey_outliers(data)
    if outs.size:
        jitter = rng.normal(0, 0.03, size=outs.size)
        ax.scatter(np.full(outs.size, 1) + jitter, outs, s=45,
                   facecolors=pt_color, edgecolors="black", linewidths=1, alpha=0.9, zorder=6)

    ax.set_xticks([])

# ==============================================================================
# --- DATA LOADING & PREPARATION ---
# ==============================================================================
def load_data(excel_path, sheet_name):
    if not os.path.exists(excel_path):
        sys.exit(f"Error: File not found at {excel_path}")

    print(f"Attempting to read Excel file: '{os.path.basename(excel_path)}', Sheet: '{sheet_name}'...")
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, engine='openpyxl')
    except Exception as e:
        sys.exit(f"Error reading Excel file: {e}")

    # 1. Filter for incorrect options
    if 'is_correct' in df.columns:
        df = df[df['is_correct'] == 0].copy()

    # 2. Agreement Label
    if 'max_count' in df.columns:
        df['max_count'] = pd.to_numeric(df['max_count'], errors='coerce')
        df['agreement_label'] = np.select(
            [df['max_count'] == 3, df['max_count'] == 2],
            ['Unanimous (3/3)', 'Majority (2/3)'],
            default='No Consensus'
        )
    else:
        print("Warning: 'max_count' column missing.")

    # 3. Kappa Data
    raters = ['Sev_LA', 'Sev_TTN', 'Sev_FBO']
    valid_cats = ['Low', 'Moderate', 'High']

    for r in raters:
        if r in df.columns:
            df[r] = df[r].astype(str).str.strip()

    if 'Sev_final' in df.columns:
        df['Sev_final'] = df['Sev_final'].astype(str).str.strip()
        df = df[df['Sev_final'].isin(valid_cats)].copy()
    else:
        print("Warning: 'Sev_final' missing.")

    if all(r in df.columns for r in raters):
        mask_kappa = df[raters].isin(valid_cats).all(axis=1)
        df_kappa = df[mask_kappa].copy()
        cat_map = {'Low': 0, 'Moderate': 1, 'High': 2}
        for r in raters:
            df_kappa[f'{r}_code'] = df_kappa[r].map(cat_map)
    else:
        df_kappa = pd.DataFrame()

    return df, df_kappa

def calculate_fleiss_kappa(df):
    if df.empty:
        return 0, 0, 0, []
    raters = ['Sev_LA_code', 'Sev_TTN_code', 'Sev_FBO_code']
    data = df[raters].values
    N, n = data.shape
    k = 3
    n_ij = np.zeros((N, k))
    for j in range(k):
        n_ij[:, j] = np.sum(data == j, axis=1)
    if n < 2:
        P_i = np.zeros(N)
    else:
        P_i = (np.sum(n_ij**2, axis=1) - n) / (n * (n - 1))
    P_bar = np.mean(P_i)
    p_j = np.sum(n_ij, axis=0) / (N * n)
    P_e = np.sum(p_j**2)
    if P_e == 1:
        kappa = 1 if P_bar == 1 else 0
    else:
        kappa = (P_bar - P_e) / (1 - P_e)
    return P_bar, P_e, kappa, P_i

def calculate_entropy_and_agreement(df, P_i_series):
    if df.empty:
        return pd.DataFrame({'severity_entropy': [], 'mean_agreement': []})

    def get_entropy(series):
        counts = series.value_counts(normalize=True)
        return -np.sum(counts * np.log2(counts + 1e-9))

    df = df.copy()
    df['P_i_calc'] = P_i_series[df.index]
    grouped = df.groupby('question_id')
    entropy_df = grouped['Sev_final'].apply(get_entropy).reset_index(name='severity_entropy')
    agreement_df = grouped['P_i_calc'].mean().reset_index(name='mean_agreement')
    return entropy_df.merge(agreement_df, on='question_id', how='left')

# ==============================================================================
# --- PLOTTING FUNCTION ---
# ==============================================================================
def plot_combined_analysis6(df_full, df_kappa):
    P_bar, P_e, kappa, P_i = calculate_fleiss_kappa(df_kappa)
    if not df_kappa.empty:
        P_i_series = pd.Series(P_i, index=df_kappa.index)
        question_stats_df = calculate_entropy_and_agreement(df_kappa, P_i_series)
    else:
        question_stats_df = pd.DataFrame({'severity_entropy': [], 'mean_agreement': []})
        P_bar, P_e, kappa = 0, 0, 0

    fig = plt.figure(figsize=(11, 12))

    gs = gridspec.GridSpec(
        3, 6,
        height_ratios=[1, 1, 1],
        hspace=0.3,  # tighter vertical spacing
        wspace=0.85   # tighter horizontal spacing
    )

    # ================= Panel a: Inter-rater agreement levels =================
    ax_a = fig.add_subplot(gs[0, 0:2])
    if 'agreement_label' in df_full.columns:
        counts_a = df_full['agreement_label'].value_counts(normalize=True)
        order_a = ['Unanimous (3/3)', 'Majority (2/3)', 'No consensus']
        counts_a = counts_a.reindex(order_a, fill_value=0) * 100

        BAR_SPACING = 0.5   
        x_pos = np.arange(len(order_a)) * BAR_SPACING
        colors = [PALETTE_AGREEMENT[label] for label in order_a]
        values = [counts_a[label] for label in order_a]

        bars = ax_a.bar(x_pos, values, color=colors, edgecolor='white', width=BAR_W)
                

        for bar in bars:
            h = bar.get_height()
            ax_a.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.0f}%",
                      ha='center', va='bottom', color='black', fontsize=12) #, fontweight='bold'

        ax_a.set_xticks(x_pos)
        wrapped_labels = [l.replace(' ', '\n') for l in order_a]
        ax_a.set_xticklabels(wrapped_labels)

    ax_a.set_ylim(0, 80)
    ax_a.set_ylabel('Proportion (%)')
    ax_a.set_title(r'$\mathbf{a}$   Inter-rater agreement levels', loc='left', fontsize=14)
    
    # ================= Panel b: Observed vs expected agreement =================
    ax_b = fig.add_subplot(gs[0, 2:4])
    
    bars_labels = [r'Observed ($\bar{P}$)', r'Expected ($\bar{P}_e$)']
    values = [P_bar, P_e]
    
    # --- Control spacing between the two bars (smaller = closer, larger = farther) ---
    BAR_SPACING_B = 0.4  
    
    x = np.array([-0.5, 0.5]) * BAR_SPACING_B   

    
    ax_b.bar(x, values, color=[COLOR_OBSERVED, COLOR_EXPECTED], width=BAR_W*0.7)
    
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(bars_labels)
    
    ymax = max(values) * 1.3 if max(values) > 0 else 1.0
    ax_b.set_ylim(0, ymax)
    ax_b.set_ylabel('Agreement proportion')
    
    pad = BAR_SPACING_B * 1.2
    ax_b.set_xlim(-pad, pad)
    
    for i, v in enumerate(values):
        ax_b.text(x[i], v + 0.01, f"{v:.2f}", ha='center', fontsize=12) #, fontweight='bold'
    
    ax_b.text(0.5, 0.92, f"Fleiss' $\kappa$ = {kappa:.3f}",
              transform=ax_b.transAxes, ha='center', fontsize=12)#, fontweight='bold')
    
    ax_b.set_title(r'$\mathbf{b}$   Observed vs. expected agreement', loc='left', fontsize=14)

    # ================= Panel c: Overall severity composition =================
    ax_d = fig.add_subplot(gs[0, 4:6])
    valid_cats = ['Low', 'Moderate', 'High']
    if 'Sev_final' in df_full.columns:
        counts_d = df_full['Sev_final'].value_counts()
        counts_d = counts_d.reindex(valid_cats, fill_value=0)
        total_d = counts_d.sum()
        props_d = counts_d / total_d * 100 if total_d > 0 else counts_d * 0

        x_pos = np.arange(len(valid_cats)) * BAR_SPACING
        colors = [PALETTE_SEVERITY[cat] for cat in valid_cats]
        vals_w = [props_d[cat] for cat in valid_cats]
        vals_c = [counts_d[cat] for cat in valid_cats]

        bars_d = ax_d.bar(x_pos, vals_w, color=colors, edgecolor='white', width=BAR_W)

        for i, bar in enumerate(bars_d):
            h = bar.get_height()
            annotation_text = f"{h:.0f}%\n(N={vals_c[i]})"
            ax_d.text(bar.get_x() + bar.get_width()/2, h + 1.2, annotation_text,
                      ha='center', va='bottom', color='black',  fontsize=12) #fontweight='bold',

        ax_d.set_xticks(x_pos)
        ax_d.set_xticklabels(valid_cats)

    ax_d.set_ylim(0, 70)
    ax_d.set_ylabel('Proportion of incorrect options (%)')
    ax_d.set_title(r'$\mathbf{c}$   Overall severity composition', loc='left', fontsize=14)

    # ================= Panel d: Severity profile by dataset =================
    ax_f = fig.add_subplot(gs[1, :])
    if 'dataset' in df_full.columns and not df_full.empty and 'Sev_final' in df_full.columns:
        df_pooled = df_full.copy()
        df_pooled['dataset_clean'] = 'Pooled'

        df_split = df_full.copy()
        df_split['dataset_clean'] = df_split['dataset'].replace({
            'Internal_TUM': 'Board-RadQA',
            'Board-RadQA': 'Board-RadQA',
            'Benchmark-RadQA': 'Benchmark-RadQA',
            'RadioRAG': 'Benchmark-RadQA'
        })

        df_combined_f = pd.concat([df_pooled, df_split])
        group_order = ['Pooled', 'Benchmark-RadQA', 'Board-RadQA']
        existing_groups = [g for g in group_order if g in df_combined_f['dataset_clean'].unique()]

        if existing_groups:
            props = df_combined_f.groupby(['dataset_clean', 'Sev_final'], observed=False).size().reset_index(name='count')
            totals = props.groupby('dataset_clean')['count'].transform('sum')
            props['percent'] = (props['count'] / totals) * 100

            sns.barplot(
                data=props, x='dataset_clean', y='percent', hue='Sev_final',
                palette=PALETTE_SEVERITY, hue_order=valid_cats, order=existing_groups,
                ax=ax_f, saturation=0.9,
                width=0.5  
            )
            
            GROUP_COMPRESS = 0.9   
            HUE_GAP = 0.7     
            
            for p in ax_f.patches:
                x = p.get_x() + p.get_width() / 2
                new_x = x * GROUP_COMPRESS
                p.set_x(new_x - p.get_width() / 2)
            
            ticks = np.arange(len(existing_groups)) * GROUP_COMPRESS
            ax_f.set_xticks(ticks)
            ax_f.set_xticklabels(existing_groups)
            
            for p in ax_f.patches:
                w = p.get_width()
                new_w = w * HUE_GAP
                p.set_x(p.get_x() + (w - new_w)/2)
                p.set_width(new_w)

            for c in ax_f.containers:
                ax_f.bar_label(c, fmt='%.0f%%', padding=1, fontsize=12)#, fontweight='bold')

            ax_f.legend(title='', loc='upper right', frameon=True, fontsize=12, bbox_to_anchor=(1, 1))

    ax_f.set_ylabel('Proportion (%)')
    ax_f.set_xlabel('')
    ax_f.set_ylim(0, 70)
    ax_f.set_title(r'$\mathbf{d}$   Severity profile by dataset', loc='left', fontsize=14)

    # ================= Panel e: Per-question agreement distribution =================
    ax_c = fig.add_subplot(gs[2, 0:2])
    data_c = question_stats_df['mean_agreement']

    if not data_c.empty:
        draw_styled_violin_single(ax_c, data_c, pt_color='tab:orange', vn_color='tab:blue')
        ax_c.axhline(P_e, color='black', linestyle='--', label=f'Exp ($P_e$)')

    ax_c.set_ylabel('Mean agreement per question')
    ax_c.set_ylim(0, 1.1)
    ax_c.legend(fontsize=12)
    ax_c.set_title(r'$\mathbf{e}$   Per-question agreement distribution', loc='left', fontsize=14)

    # ================= Panel f: Per-question severity entropy =================
    ax_e = fig.add_subplot(gs[2, 2:6])
    if not question_stats_df.empty:
        sns.histplot(question_stats_df['severity_entropy'], ax=ax_e, element='step',
                     fill=True, color= '#1f77b4' , alpha=0.6, bins=15) #'#999999'
    ax_e.set_xlabel('Severity entropy')
    ax_e.set_ylabel('Number of questions')
    ax_e.set_title(r'$\mathbf{f}$   Per-question severity entropy', loc='left', fontsize=14)

    plt.subplots_adjust(top=0.93, bottom=0.05, left=0.04, right=0.98)

    output_filename = 'Figure6_Analsysis6.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_filename}")
    plt.show()

# ==============================================================================
# --- MAIN ---
# ==============================================================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(script_dir, YOUR_EXCEL_FILE)
    print(f"Loading data from {excel_path}...")
    df_full, df_kappa = load_data(excel_path, YOUR_SHEET_NAME)
    print("Data loaded.")
    plot_combined_analysis6(df_full, df_kappa)
