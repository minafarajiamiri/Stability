"""
Figure 5: Consensus-Robustness Coupling Analysis
=================================================
Requirements: pandas, numpy, matplotlib, seaborn, scipy

Place data files in same directory or specify with --data-dir
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import Rectangle
import os
import sys

# ==========================================
# --- STYLE SETUP ---
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.frameon'] = False

COLOR_CORRECT = 'tab:blue'
COLOR_INCORRECT = 'tab:green'
COLOR_ZEROSHOT = '#d62728'
COLOR_AGENTIC = '#ff7f0e'

# ==========================================
# --- LOAD DATA ---
# ==========================================

def load_data(data_dir='.'):
    """Load all required data files"""
    print(f"Loading data from: {os.path.abspath(data_dir)}")
    
    corr_df = pd.read_csv(os.path.join(data_dir, 'step2_correlations.csv'))
    consensus_df = pd.read_csv(os.path.join(data_dir, 'step3_consensus_correctness.csv'))
    anomalous_df = pd.read_csv(os.path.join(data_dir, 'step4_anomalous_cases.csv'))
    
    xl_file = pd.ExcelFile(os.path.join(data_dir, 'analyse2_final_v2.xlsx'))
    majority_df = pd.read_excel(xl_file, sheet_name='majority')
    
    robustness_tum = pd.read_csv(os.path.join(data_dir, 'robustness_scores_internal_TUM_dataset.csv'))
    robustness_rad = pd.read_csv(os.path.join(data_dir, 'robustness_scores_Radiorag_dataset.csv'))
    
    robustness_tum['dataset'] = 'Internal_TUM'
    robustness_rad['dataset'] = 'RadioRAG'
    robustness_combined = pd.concat([robustness_tum, robustness_rad], ignore_index=True)
    
    majority_df['match_key'] = (majority_df['dataset'].str.replace(' ', '_').str.replace('dataset', '').str.strip() + 
                                '|' + majority_df['question_id'].astype(str) + '|' + majority_df['method'])
    robustness_combined['match_key'] = (robustness_combined['dataset'] + '|' + 
                                       robustness_combined['question_id'].astype(str) + '|' + 
                                       robustness_combined['method'])
    
    merged_df = majority_df.merge(
        robustness_combined[['match_key', 'robustness_score']], 
        on='match_key', 
        how='left'
    )
    
    print("✓ Data loaded successfully\n")
    return corr_df, consensus_df, anomalous_df, merged_df

# ==========================================
# --- HELPER FUNCTIONS ---
# ==========================================

def add_significance_stars(p_val):
    if p_val < 0.001:
        return '***'
    elif p_val < 0.01:
        return '**'
    elif p_val < 0.05:
        return '*'
    else:
        return 'ns'

def loess_smooth(x, y):
    """Simple smoothing"""
    if len(x) < 5:
        return x, y
    sort_idx = np.argsort(x)
    x_sorted = np.array(x)[sort_idx]
    y_sorted = np.array(y)[sort_idx]
    x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 100)
    y_smooth = np.interp(x_smooth, x_sorted, y_sorted)
    return x_smooth, y_smooth

# ==========================================
# --- PANEL A: Schematic ---
# ==========================================

def plot_panel_a_schematic(ax_left, ax_right):
    """Two bar charts showing examples"""
    
    # Example 1: High M, High R
    answers_1 = ['C']*20 + ['A']*3 + ['B']*1 + ['D']*1
    counts_1 = pd.Series(answers_1).value_counts().reindex(['A', 'B', 'C', 'D'], fill_value=0)
    colors_1 = ['#2ca02c' if ans == 'C' else '#cccccc' for ans in counts_1.index]
    
    ax_left.bar(range(4), counts_1.values, color=colors_1, edgecolor='black', linewidth=1.2)
    ax_left.set_xticks(range(4))
    ax_left.set_xticklabels(['A', 'B', 'C', 'D'], fontsize=13)
    ax_left.set_xlabel('Answer option\n(Correct answer: )', fontsize=12, style='italic')
    ax_left.set_ylabel('Number of models', fontsize=14)
    ax_left.set_ylim(0, 25)
    ax_left.set_title('Example 1: High M, High R', fontsize=14, pad=10)
    ax_left.text(0.95, 0.85, 'M = 0.80', transform=ax_left.transAxes, ha='right', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_left.text(0.95, 0.70, 'R = 0.80', transform=ax_left.transAxes, ha='right', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Example 2: High M, Low R
    answers_2 = ['A']*22 + ['C']*2 + ['B']*1
    counts_2 = pd.Series(answers_2).value_counts().reindex(['A', 'B', 'C', 'D'], fill_value=0)
    colors_2 = ['#2ca02c' if ans == 'C' else '#cccccc' for ans in counts_2.index]
    
    ax_right.bar(range(4), counts_2.values, color=colors_2, edgecolor='black', linewidth=1.2)
    ax_right.set_xticks(range(4))
    ax_right.set_xticklabels(['A', 'B', 'C', 'D'], fontsize=13)
    ax_right.set_xlabel('Answer option\n(Correct answer: )', fontsize=12, style='italic')
    ax_right.set_ylabel('Number of models', fontsize=14)
    ax_right.set_ylim(0, 25)
    ax_right.set_title('Example 2: High M, Low R', fontsize=14, pad=10)
    ax_right.text(0.95, 0.85, 'M = 0.88', transform=ax_right.transAxes, ha='right', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_right.text(0.95, 0.70, 'R = 0.08', transform=ax_right.transAxes, ha='right', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

# ==========================================
# --- PANEL B: Scatter facets ---
# ==========================================

def plot_scatter_facet(ax, data, corr_df, method, dataset, dataset_label):
    """Single scatter plot"""
    data_subset = data[
        (data['method'] == method) & 
        (data['dataset'].str.contains(dataset))
    ].copy()
    
    if len(data_subset) == 0:
        return
    
    ax.scatter(data_subset['majority_fraction'], data_subset['robustness_score'],
              s=50, alpha=0.6, color='#555555', edgecolors='none')
    
    valid = data_subset.dropna(subset=['majority_fraction', 'robustness_score'])
    if len(valid) >= 5:
        x_smooth, y_smooth = loess_smooth(valid['majority_fraction'].values, 
                                          valid['robustness_score'].values)
        ax.plot(x_smooth, y_smooth, color='#1f77b4', linewidth=2.5, alpha=0.7)
    
    # No stat labels on plot
    
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Majority fraction (M)', fontsize=14)
    ax.set_ylabel('Robustness (R)', fontsize=14)
    ax.set_title(f'{dataset_label} | {method.capitalize()}', fontsize=15, style='italic')
    # No grid

# Panel C removed as per user request

# ==========================================
# --- PANEL D: Violin facets ---
# ==========================================

def plot_violin_facet(ax, data, consensus_df, method, dataset, dataset_label):
    """Single violin plot with jittered points"""
    data_subset = data[(data['method'] == method) & 
                       (data['dataset'].str.contains(dataset))].copy()
    
    correct = data_subset[data_subset['majority_correct'] == 1]['majority_fraction'].dropna()
    incorrect = data_subset[data_subset['majority_correct'] == 0]['majority_fraction'].dropna()
    
    # Violin plot
    parts = ax.violinplot([correct, incorrect], positions=[1, 2], 
                          showmedians=True, widths=0.6)
    
    for pc, color in zip(parts['bodies'], [COLOR_CORRECT, COLOR_INCORRECT]):
        pc.set_facecolor(color)
        pc.set_alpha(0.3)
        pc.set_edgecolor('none')
    
    for part in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
        parts[part].set_color('black')
        parts[part].set_linewidth(1.5)
    
    # Box plot overlay
    ax.boxplot([correct, incorrect], positions=[1, 2], widths=0.3, showfliers=False,
              patch_artist=False,
              boxprops={'linewidth':1.5},
              whiskerprops={'linewidth':1.5}, capprops={'linewidth':1.5},
              medianprops={'color':'black', 'linewidth':2})
    
    color_points = {1: "tab:orange", 2: "tab:red"}

    # Add jittered points
    np.random.seed(42)
    for pos, vals, key in [(1, correct, 1), (2, incorrect, 2)]:
        color = color_points[key]

        if len(vals) > 0:
            jitter = np.random.normal(0, 0.04, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, 
                      s=20, alpha=0.6, color=color, edgecolors='white', linewidths=0.5, zorder=3)
    
    # No stat labels on plot
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Correct', 'Incorrect'], fontsize=13)
    ax.set_ylabel('Majority fraction', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{dataset_label} | {method.capitalize()}', fontsize=15, style='italic')
    # No grid

# ==========================================
# --- PANEL E: Anomalous cases ---
# ==========================================

def plot_panel_e(ax, merged_df, anomalous_df):
    """Scatter of high consensus cases"""
    high = merged_df[merged_df['majority_fraction'] >= 0.8].copy()
    
    correct = high[high['majority_correct'] == 1]
    incorrect = high[high['majority_correct'] == 0]
    
    ax.scatter(correct['majority_fraction'], correct['robustness_score'],
              s=60, alpha=0.2, color='#999999', label='High M, Correct', zorder=1)
    
    ax.scatter(incorrect['majority_fraction'], incorrect['robustness_score'],
              s=100, alpha=0.8, facecolors=COLOR_INCORRECT, 
              edgecolors='darkred', linewidths=2, label='High M, Incorrect', zorder=3)
    
    ax.axvline(0.8, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.axhline(0.4, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    
    ax.add_patch(Rectangle((0.8, 0), 0.25, 0.4, 
                           facecolor=COLOR_INCORRECT, alpha=0.08, zorder=0))
    
    ax.text(0.935, 0.2, 'Anomalous\nregion\n(M≥0.8, R<0.4)', 
            ha='center', va='center', fontsize=13, style='italic', 
            color=COLOR_INCORRECT, weight='bold')
    
    counts = anomalous_df.groupby('Method').size()
    zero = counts.get('zero-shot', 0)
    agen = counts.get('agentic', 0)
    
    text = f'Anomalous cases:\nZero-shot: {zero}\nAgentic: {agen}'
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=12, va='top', ha='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1.5))
    
    ax.set_xlim(0.75, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Majority fraction (M)', fontsize=14)
    ax.set_ylabel('Robustness (R)', fontsize=14)
    ax.legend(loc='lower left', fontsize=12, frameon=True, edgecolor='gray')
    # No grid

# ==========================================
# --- MAIN FIGURE ---
# ==========================================

def create_figure_5(data_dir='.', output_dir='.'):
    """Create Figure 5"""
    
    corr_df, consensus_df, anomalous_df, merged_df = load_data(data_dir)
    
    print("Creating figure...")
    fig = plt.figure(figsize=(20, 14))
    
    # Create GridSpec: 3 rows
    # Row 1: Panel a (schematic) + Panel b (scatter 2x2)
    # Row 2: Panel c (was correlation, now Panel d - violin 4 columns)
    # Row 3: Panel d (was violin, now Panel e - anomalous cases full width)
    
    gs = gridspec.GridSpec(7, 4, 
                          height_ratios=[0.12, 1.2, 0.12, 1.2, 0.12, 1.2, 0.1],
                          width_ratios=[1, 1, 1, 1],
                          hspace=0.5, wspace=0.4,
                          left=0.08, right=0.96, top=0.97, bottom=0.04)
    
    # Panel a: Schematic (row 0-1, cols 0-1)
    ax_a_title = fig.add_subplot(gs[0, 0:2])
    ax_a_title.axis('off')
    ax_a_title.text(0.05, -0.5, '$\\mathbf{a}$  Metric schematic', fontsize=16, va='center')
    
    ax_a_left = fig.add_subplot(gs[1, 0])
    ax_a_right = fig.add_subplot(gs[1, 1])
    
    plot_panel_a_schematic(ax_a_left, ax_a_right)
    
    # Panel b: Scatter 2x2 (row 0-1, cols 2-3)
    ax_b_title = fig.add_subplot(gs[0, 2:4])
    ax_b_title.axis('off')
    ax_b_title.text(0.02, -0.5, '$\\mathbf{b}$  Consensus–robustness coupling', fontsize=16, va='center')
    
    ax_b1 = fig.add_subplot(gs[1, 2])  # Zero-shot, Benchmark
    ax_b2 = fig.add_subplot(gs[1, 3])  # Zero-shot, Board
    
    plot_scatter_facet(ax_b1, merged_df, corr_df, 'zero-shot', 'RadioRAG', 'Benchmark-RadQA')
    plot_scatter_facet(ax_b2, merged_df, corr_df, 'zero-shot', 'Internal_TUM', 'Board-RadQA')
    
    # Panel c (now d): Violin 1x4 (row 2-3, full width - 4 columns)
    ax_c_title = fig.add_subplot(gs[2, :])
    ax_c_title.axis('off')
    ax_c_title.text(0.02, -0.5, '$\\mathbf{c}$  Consensus by majority correctness', fontsize=16, va='center')
    
    ax_c1 = fig.add_subplot(gs[3, 0])  # Zero-shot, Benchmark
    ax_c2 = fig.add_subplot(gs[3, 1])  # Zero-shot, Board
    ax_c3 = fig.add_subplot(gs[3, 2])  # Agentic, Benchmark
    ax_c4 = fig.add_subplot(gs[3, 3])  # Agentic, Board
    
    plot_violin_facet(ax_c1, merged_df, consensus_df, 'zero-shot', 'RadioRAG', 'Benchmark-RadQA')
    plot_violin_facet(ax_c2, merged_df, consensus_df, 'zero-shot', 'Internal_TUM', 'Board-RadQA')
    plot_violin_facet(ax_c3, merged_df, consensus_df, 'agentic', 'RadioRAG', 'Benchmark-RadQA')
    plot_violin_facet(ax_c4, merged_df, consensus_df, 'agentic', 'Internal_TUM', 'Board-RadQA')
    
    # Panel d (now e): Anomalous (row 4-5, full width)
    ax_d_title = fig.add_subplot(gs[4, :])
    ax_d_title.axis('off')
    ax_d_title.text(0.02, -1.3, '$\\mathbf{d}$  Coordinated but incorrect convergence', 
                   fontsize=16, va='center')
    
    ax_d = fig.add_subplot(gs[5, :])
    plot_panel_e(ax_d, merged_df, anomalous_df)
    
    # Save
    output_path = os.path.join(output_dir, 'Figure5_consensus_robustness.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Figure saved: {os.path.abspath(output_path)}\n")
    
    return fig

# ==========================================
# --- RUN ---
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Figure 5')
    parser.add_argument('--data-dir', default='.', help='Data directory')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Figure 5: Consensus-Robustness Coupling")
    print("="*60 + "\n")
    
    try:
        fig = create_figure_5(data_dir=args.data_dir, output_dir=args.output_dir)
        plt.show()
        print("✓ Complete!\n")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
