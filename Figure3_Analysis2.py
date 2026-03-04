import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import sys
import os


YOUR_EXCEL_FILE = 'analyse2_final_v3.xlsx' 
YOUR_SHEET_NAME = 'Sheet1'       

COL_DATASET_NAME = 'dataset'
COL_M_ZEROSHOT = 'M_zero'
COL_M_AGENTIC = 'M_agentic'
COL_AG_CORRECT = 'C_agentic'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.frameon'] = False

PALETTE_DATASETS = {'Benchmark-RadQA': '#1f77b4', 'Board-RadQA': '#2ca02c'}
PALETTE_CATS_A2 = {
    'Agreement $\\uparrow$ & Correct': '#2ca02c',   # Green
    'Agreement $\\uparrow$ & Incorrect': '#d62728', # Red
    'Agreement $\\downarrow$': '#7f7f7f',         # Grey
    'No change': '#bbbbbb'                        # Light Grey
}

def tukey_outliers(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 4: return np.array([], dtype=float)
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    return v[(v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)]

def add_jitter_points(ax, x_pos, y_vals, rng, jitter=0.06, size=14, alpha=0.6, c=None, ec=None, lw=0.6):
    y = np.asarray(y_vals, dtype=float)
    y = y[~np.isnan(y)]
    x = x_pos + rng.normal(0, jitter, size=y.size)
    ax.scatter(x, y, s=size, alpha=alpha, c=c, edgecolors=ec, linewidths=lw, zorder=2)

def load_and_prep_data_a2(filepath, sheetname):
    if not os.path.exists(filepath): sys.exit(f"Error: File not found at {filepath}")
    try: df = pd.read_excel(filepath, sheet_name=sheetname)
    except Exception as e: sys.exit(f"Error reading Excel: {e}")

    col_map = {
        COL_DATASET_NAME: 'dataset',
        COL_M_ZEROSHOT: 'M_zs',
        COL_M_AGENTIC: 'M_ag',
        COL_AG_CORRECT: 'ag_correct'
    }
    missing = [user_c for user_c, std_c in col_map.items() if user_c not in df.columns]
    if missing: sys.exit(f"Missing columns in {filepath}: {missing}")
    df = df.rename(columns=col_map)
    
    df['M_zs'] = pd.to_numeric(df['M_zs'], errors='coerce')
    df['M_ag'] = pd.to_numeric(df['M_ag'], errors='coerce')
    df['ag_correct'] = pd.to_numeric(df['ag_correct'], errors='coerce')
    df = df.dropna(subset=['dataset', 'M_zs', 'M_ag', 'ag_correct'])

    df['delta_M'] = df['M_ag'] - df['M_zs']

    threshold = 1e-9
    conditions = [
        (df['delta_M'] > threshold) & (df['ag_correct'] == 1),
        (df['delta_M'] > threshold) & (df['ag_correct'] == 0),
        (df['delta_M'] < -threshold)
    ]
    choices = ['Agreement $\\uparrow$ & Correct', 'Agreement $\\uparrow$ & Incorrect', 'Agreement $\\downarrow$']
    df['category'] = np.select(conditions, choices, default='No change')
    
    cat_order = ['Agreement $\\uparrow$ & Correct', 'Agreement $\\uparrow$ & Incorrect', 
                 'Agreement $\\downarrow$', 'No change']
    df['category'] = pd.Categorical(df['category'], categories=cat_order, ordered=True)
    return df


def draw_styled_violin_pair(ax, data_z, data_a):
    """Helper function to draw the specific style on a given axis."""
    data = [data_z, data_a]
    rng = np.random.default_rng(42)
    PT_COLORS = {1: "tab:orange", 2: "tab:red"}
    VN_COLORS = ["tab:blue", "tab:green"]

    add_jitter_points(ax, 1, data_z, rng, size=15, alpha=0.5, c=PT_COLORS[1], ec="none")
    add_jitter_points(ax, 2, data_a, rng, size=15, alpha=0.5, c=PT_COLORS[2], ec="none")

    vp = ax.violinplot(data, positions=[1, 2], showmeans=False, showmedians=True, showextrema=True, widths=0.8)
    for body, c in zip(vp["bodies"], VN_COLORS):
        body.set_facecolor(c)
        body.set_edgecolor('none')
        body.set_alpha(0.25)
        body.set_zorder(1)
    for part in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
        vp[part].set_color('grey')
        vp[part].set_linewidth(1)
        vp[part].set_zorder(3)

    ax.boxplot(data, positions=[1, 2], widths=0.2, showfliers=False, patch_artist=True,
               boxprops={'facecolor':'none', 'linewidth':1.2, 'zorder':4},
               whiskerprops={'linewidth':1.2, 'zorder':4},
               capprops={'linewidth':1.2, 'zorder':4},
               medianprops={'color':'black', 'linewidth':1.5, 'zorder':5})

    for pos, vals, pt_col in zip([1, 2], data, PT_COLORS.values()):
        outs = tukey_outliers(vals)
        if outs.size:
            jitter = rng.normal(0, 0.03, size=outs.size)
            ax.scatter(np.full(outs.size, pos) + jitter, outs, s=45,
                       facecolors=pt_col, edgecolors="black", linewidths=1, alpha=0.9, zorder=6)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Zero-shot", "Agentic"])

def plot_panel_a_faceted_a2(ax_container, df, label):
    """Panel a: Faceted Majority Fraction (M) distribution using custom violin style."""
    gs_inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=ax_container, wspace=0.3)
    axes = [plt.Subplot(ax_container.figure, gs_inner[i]) for i in range(3)]
    for ax in axes: ax_container.figure.add_subplot(ax)
    
    groups = ['Pooled', 'Benchmark-RadQA', 'Board-RadQA']

    ax_container.set_title(f'$\mathbf{{{label}}}$ Majority fraction distribution by method', 
                           loc='left', fontsize=14, y=1.06) #fontweight='bold', 
    
    for i, (ax, group) in enumerate(zip(axes, groups)):
        d_sub = df if group == 'Pooled' else df[df['dataset'] == group]
        
        draw_styled_violin_pair(ax, d_sub['M_zs'], d_sub['M_ag'])
        
        ax.text(0.5, 1.02, group, transform=ax.transAxes, 
                ha='center', va='bottom', fontsize=12)#, fontweight='bold')
        
        ax.set_ylabel("Majority fraction", fontsize=12)
        ax.set_ylim(-0.05, 1.05) # M is bounded 0-1

    ax_container.axis('off')

def plot_panel_b_hist_a2(ax, df, label):
    """Panel b: Histogram of change in majority fraction (Delta M)."""
    sns.histplot(data=df, x='delta_M', hue='dataset', palette=PALETTE_DATASETS,
                 ax=ax, element="step", stat="density", common_norm=False, 
                 alpha=0.3, lw=1.5, zorder=2)
    
    ax.axvline(0, color='black', ls='-', lw=1, zorder=3)
    ax.set_xlabel('ΔM (Agentic − Zero-shot)')
    ax.set_ylabel('Density')
    
    limit = max(abs(df['delta_M'].min()), abs(df['delta_M'].max())) * 1.1 if len(df) > 0 else 0.1
    ax.set_xlim(-limit, limit)
    ymin, ymax = ax.get_ylim()


    y_txt = ymax * 0.85
    # Define arrow dimensions based on y-axis scale for consistency
    arrow_width = y_txt * 0.01 
    arrow_hw = y_txt * 0.04
    arrow_hl = y_txt * 0.01 # Explicitly short head length

    ax.text(limit*0.25, y_txt, 'Agreement\nIncreased', ha='center', color='#555555', fontweight='bold', zorder=4)
    ax.arrow(limit*0.05, y_txt*0.9, limit*0.12, 0, color='#555555', 
             width=arrow_width, head_width=arrow_hw, head_length=arrow_hl, zorder=4)
             
    ax.text(-limit*0.25, y_txt, 'Agreement\nDecreased', ha='center', color='#555555', fontweight='bold', zorder=4)
    ax.arrow(-limit*0.05, y_txt*0.9, -limit*0.12, 0, color='#555555', 
             width=arrow_width, head_width=arrow_hw, head_length=arrow_hl, zorder=4)

    ax.legend_.set_title('')
    ax.set_title(f'$\mathbf{{{label}}}$ Histogram of consensus change (ΔM)', loc='left', fontsize=14)     #fontweight='bold',

def plot_panel_c_bars_a2(ax, df, label):
    """Panel c: Stacked bars using Analysis 2 outcome categories."""
    groups = ['Pooled'] + sorted(df['dataset'].unique())
    props_data = []
    group_counts = [] 

    for group in groups:
        d_sub = df if group == 'Pooled' else df[df['dataset'] == group]
        group_counts.append(len(d_sub))
        
        counts = d_sub['category'].value_counts(normalize=True)
        counts = counts.reindex(PALETTE_CATS_A2.keys(), fill_value=0)
        props_data.append(counts * 100)

    df_props = pd.DataFrame(props_data, index=groups)
    
    colors = [PALETTE_CATS_A2[c] for c in df_props.columns]
    df_props.plot(kind='bar', stacked=True, ax=ax, color=colors, 
                  width=0.6, edgecolor='white')

    ax.set_ylabel('Proportion of questions (%)'); ax.set_ylim(0, 100)
    plt.setp(ax.get_xticklabels(), rotation=0)
    ax.set_xlim(left=-0.4, right=3.5)
    
    for i, count in enumerate(group_counts):
        ax.text(i, 102, f'N={count}', ha='center', va='bottom', 
                fontsize=12,  color='black') #fontweight='bold',

    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels_leg[::-1], title='', loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    for container in ax.containers:
        labels_bar = [f'{v.get_height():.0f}%' if v.get_height() > 4 else '' for v in container]
        ax.bar_label(container, labels=labels_bar, label_type='center', color='white', fontsize=11, fontweight='bold')
        
    ax.set_title(f'$\mathbf{{{label}}}$   Proportions of consensus shift outcomes', loc='left', fontsize=14, y=1.1) #fontweight='bold', 

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(script_dir, YOUR_EXCEL_FILE)
    print(f"Loading data for Analysis 2 from {excel_path}...")
    df = load_and_prep_data_a2(excel_path, YOUR_SHEET_NAME)
    print(f"Data loaded. N={len(df)}")

    fig = plt.figure(figsize=(12, 14)) 
    gs = gridspec.GridSpec(3, 1, 
                           height_ratios=[1.5, 1, 0.8], 
                           hspace=0.4)

    ax_a_container = fig.add_subplot(gs[0, 0]) # Row 1, full width
    ax_b = fig.add_subplot(gs[1, 0])           # Row 2, full width (was split before)
    ax_c = fig.add_subplot(gs[2, 0])           # Row 3, full width

    print("Generating Figure 3 plots...")
    plot_panel_a_faceted_a2(ax_a_container, df, label='a')
    plot_panel_b_hist_a2(ax_b, df, label='b')
    plot_panel_c_bars_a2(ax_c, df, label='c')


    output_filename = os.path.join(script_dir, 'Figure3_Analysis2.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_filename}")

    plt.show()


