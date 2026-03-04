import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import sys
import os

YOUR_EXCEL_FILE = 'analyse1_final_v3.xlsx'
YOUR_SHEET_NAME = 'analysis1_input' 

COL_DATASET_NAME = 'dataset'       
COL_H_ZEROSHOT = 'H_zeroshot'      
COL_H_AGENTIC = 'H_agentic'        

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.frameon'] = False


PALETTE_DATASETS = {'Benchmark-RadQA': '#1f77b4', 'Board-RadQA': '#2ca02c'} 
PALETTE_CATS = {'Improved': '#2ca02c', 'Unchanged': '#bbbbbb', 'Worsened': '#d62728'}

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

def load_and_prep_data(filepath, sheetname):
    if not os.path.exists(filepath): sys.exit(f"Error: File not found at {filepath}")
    try: df = pd.read_excel(filepath, sheet_name=sheetname)
    except Exception as e: sys.exit(f"Error reading Excel: {e}")

    needed = {COL_DATASET_NAME: 'dataset', COL_H_ZEROSHOT: 'H_zeroshot', COL_H_AGENTIC: 'H_agentic'}
    if any(c not in df.columns for c in needed): sys.exit(f"Missing columns. Found: {list(df.columns)}")
    df = df.rename(columns=needed)
    
    df['H_zeroshot'] = pd.to_numeric(df['H_zeroshot'], errors='coerce')
    df['H_agentic'] = pd.to_numeric(df['H_agentic'], errors='coerce')
    df = df.dropna(subset=['H_zeroshot', 'H_agentic'])
    df['delta_H'] = df['H_agentic'] - df['H_zeroshot']

    conditions = [(df['delta_H'] < -0.001), (df['delta_H'] > 0.001)]
    df['change_category'] = np.select(conditions, ['Improved', 'Worsened'], default='Unchanged')
    return df


def draw_styled_violin_pair(ax, data_z, data_a):
    """Helper function to draw the specific style on a given axis."""
    data = [data_z, data_a]
    rng = np.random.default_rng(42)
    PT_COLORS = {1: "tab:orange", 2: "tab:red"}
    VN_COLORS = ["tab:blue", "tab:green"]

    # 1. Jittered Points
    add_jitter_points(ax, 1, data_z, rng, size=15, alpha=0.5, c=PT_COLORS[1], ec="none")
    add_jitter_points(ax, 2, data_a, rng, size=15, alpha=0.5, c=PT_COLORS[2], ec="none")

    # 2. Violin
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

    # 3. Box plot
    ax.boxplot(data, positions=[1, 2], widths=0.2, showfliers=False, patch_artist=True,
               boxprops={'facecolor':'none', 'linewidth':1.2, 'zorder':4},
               whiskerprops={'linewidth':1.2, 'zorder':4},
               capprops={'linewidth':1.2, 'zorder':4},
               medianprops={'color':'black', 'linewidth':1.5, 'zorder':5})

    # 4. Outliers
    for pos, vals, pt_col in zip([1, 2], data, PT_COLORS.values()):
        outs = tukey_outliers(vals)
        if outs.size:
            jitter = rng.normal(0, 0.03, size=outs.size)
            ax.scatter(np.full(outs.size, pos) + jitter, outs, s=45,
                       facecolors=pt_col, edgecolors="black", linewidths=1, alpha=0.9, zorder=6)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Zero-shot", "Agentic"])


def plot_panel_a_faceted(ax_container, df, label):
    """New Panel a: Wide, faceted entropy distribution with custom style."""
    gs_inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=ax_container, wspace=0.25)
    axes = [plt.Subplot(ax_container.figure, gs_inner[i]) for i in range(3)]
    for ax in axes: ax_container.figure.add_subplot(ax)
    
    groups = ['Pooled', 'Benchmark-RadQA', 'Board-RadQA']

    ax_container.set_title(f'$\mathbf{{{label}}}$  Entropy distribution by method', 
                           loc='left',  fontsize=14, y=1.08) #fontweight='bold',
    
    for i, (ax, group) in enumerate(zip(axes, groups)):
        d_sub = df if group == 'Pooled' else df[df['dataset'] == group]
        
        draw_styled_violin_pair(ax, d_sub['H_zeroshot'], d_sub['H_agentic'])
        
        ax.text(0.5, 1.02, group, transform=ax.transAxes, 
                ha='center', va='bottom', fontsize=12)#, fontweight='bold')
        
        ax.set_ylabel("Entropy", fontsize=12)

    ax_container.axis('off') # Hide the container frame lines


def plot_panel_b_scatter(ax, df, label):
    """New Panel b: Paired scatter plot with background shading."""
    max_val = max(df['H_zeroshot'].max(), df['H_agentic'].max()) * 1.1
    
    x_fill = np.linspace(0, max_val, 100)
    ax.fill_between(x_fill, x_fill, max_val, color=PALETTE_CATS['Worsened'], 
                    alpha=0.1, zorder=0, edgecolor='none')
    
    ax.fill_between(x_fill, 0, x_fill, color=PALETTE_CATS['Improved'], 
                    alpha=0.1, zorder=0, edgecolor='none')

    ax.plot([0, max_val], [0, max_val], ls='-', c='black', lw=1, zorder=1)
    
    sns.scatterplot(data=df, x='H_zeroshot', y='H_agentic', hue='dataset', style='dataset',
                    palette=PALETTE_DATASETS, markers=['o', '^'], alpha=0.7, s=50, ax=ax, zorder=2)
    
    ax.set_xlim(0, max_val); ax.set_ylim(0, max_val); ax.set_aspect('equal')
    ax.set_xlabel('Zero-shot Entropy'); ax.set_ylabel('Agentic entropy')
    
    ax.text(max_val*0.75, max_val*0.25, 'Improved\nStability', ha='center', va='center', 
            color=PALETTE_CATS['Improved'], fontweight='bold', zorder=3)
    ax.text(max_val*0.25, max_val*0.75, 'Worsened\nStability', ha='center', va='center', 
            color=PALETTE_CATS['Worsened'], fontweight='bold', zorder=3)
            
    ax.legend(title='', loc='upper left', fontsize=9)
    ax.set_title(f'$\mathbf{{{label}}}$  Paired scatter plot of entropy', loc='left', fontsize=14) #, fontweight='bold'

def plot_panel_c_hist(ax, df, label):
    """New Panel c: Histogram of change with background shading."""
    sns.histplot(data=df, x='delta_H', hue='dataset', palette=PALETTE_DATASETS,
                 ax=ax, element="step", stat="density", common_norm=False, 
                 alpha=0.4, lw=1.5, zorder=2)
    
    ax.axvline(0, color='black', ls='-', lw=1, zorder=3)
    
    ax.set_xlabel('ΔH (Agentic − Zero-shot)')
    ax.set_ylabel('Density')
    
    limit = max(abs(df['delta_H'].min()), abs(df['delta_H'].max())) * 1.1 if len(df) > 0 else 0.1
    ax.set_xlim(-limit, limit)
    ymin, ymax = ax.get_ylim()

    ax.axvspan(-limit, 0, ymin=0, ymax=1, color=PALETTE_CATS['Improved'], 
               alpha=0.1, zorder=0, edgecolor='none')
    
    ax.axvspan(0, limit, ymin=0, ymax=1, color=PALETTE_CATS['Worsened'], 
               alpha=0.1, zorder=0, edgecolor='none')
    
    y_txt = ymax * 0.7
    ax.text(-limit*0.5, y_txt, 'Improved', ha='center', 
            color=PALETTE_CATS['Improved'], fontweight='bold', zorder=4)
    ax.arrow(-limit*0.1, y_txt*0.9, -limit*0.3, 0, color=PALETTE_CATS['Improved'], 
             width=y_txt*0.02, head_width=y_txt*0.06, zorder=4)
             
    ax.text(limit*0.5, y_txt, 'Worsened', ha='center', 
            color=PALETTE_CATS['Worsened'], fontweight='bold', zorder=4)
    ax.arrow(limit*0.1, y_txt*0.9, limit*0.3, 0, color=PALETTE_CATS['Worsened'], 
             width=y_txt*0.02, head_width=y_txt*0.06, zorder=4)

    ax.legend_.set_title('')
    ax.set_title(f'$\mathbf{{{label}}}$  Histogram of entropy change (ΔH)', loc='left',  fontsize=14) # fontweight='bold',

def plot_panel_d_bars(ax, df, label):
    """New Panel d: Stacked bars with total counts (N) above each bar."""
    groups = ['Pooled'] + sorted(df['dataset'].unique())
    props_data = []
    group_counts = [] 

    for group in groups:
        d_sub = df if group == 'Pooled' else df[df['dataset'] == group]
        count = len(d_sub)
        group_counts.append(count)
        
        counts = d_sub['change_category'].value_counts(normalize=True)
        counts = counts.reindex(['Improved', 'Unchanged', 'Worsened'], fill_value=0)
        props_data.append(counts * 100)

    df_props = pd.DataFrame(props_data, index=groups)
    
    df_props.plot(kind='bar', stacked=True, ax=ax, color=[PALETTE_CATS[c] for c in df_props.columns], 
                  width=0.6, edgecolor='white')

    ax.set_ylabel('Proportion of questions (%)'); ax.set_ylim(0, 100)
    plt.setp(ax.get_xticklabels(), rotation=0)

    ax.set_xlim(left=-0.5, right=3.0)
    
    for i, count in enumerate(group_counts):
        ax.text(i, 102, f'N={count}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color='black')

    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels_leg[::-1], title='', loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    for container in ax.containers:
        labels_bar = [f'{v.get_height():.0f}%' if v.get_height() > 3 else '' for v in container]
        ax.bar_label(container, labels=labels_bar, label_type='center', color='white', fontsize=11, fontweight='bold')
        
    ax.set_title(f'$\mathbf{{{label}}}$  Proportions of change categories', loc='left', fontsize=14, y=1.08) #fontweight='bold',
    

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(script_dir, YOUR_EXCEL_FILE)
    print(f"Loading data from {excel_path}...")
    df = load_and_prep_data(excel_path, YOUR_SHEET_NAME)

    fig = plt.figure(figsize=(12, 14)) 
    gs = gridspec.GridSpec(3, 2, 
                           height_ratios=[1, 1, 0.8], 
                           width_ratios=[0.7, 1.3], 
                           hspace=0.4, 
                           wspace=0.2)

    ax_a_container = fig.add_subplot(gs[0, :]) # Row 1, full width
    ax_b = fig.add_subplot(gs[1, 0])           # Row 2, left
    ax_c = fig.add_subplot(gs[1, 1])           # Row 2, right
    ax_d = fig.add_subplot(gs[2, :])           # Row 3, full width

    print("Generating plots...")
    plot_panel_a_faceted(ax_a_container, df, label='a')
    plot_panel_b_scatter(ax_b, df, label='b')
    plot_panel_c_hist(ax_c, df, label='c')
    plot_panel_d_bars(ax_d, df, label='d')

    output_filename = os.path.join(script_dir, 'Figure2_Analysis1.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_filename}")

    plt.show()

