#!/usr/bin/env python3
"""
Task Granularity Analysis: 5-class vs 4-class sleep staging performance

Compares:
- 5-class: Wake, N1, N2, N3, REM
- 4-class v0: Wake, N1, N2+N3, REM  
- 4-class v1: Wake, Light(N1+N2), Deep(N3), REM

Usage:
    python fig_task_granularity.py --csv Plot_Clean/data/all_runs_flat.csv --out Plot_Clean/figures/
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Colors for different configurations
COLORS = {
    '5-class': '#2E8B57',      # Sea green
    '4-class-v0': '#4169E1',   # Royal blue  
    '4-class-v1': '#FF6347',   # Tomato red
    '4-class': '#1f77b4'       # Default blue for undifferentiated 4-class
}

def setup_style():
    """Setup publication-ready matplotlib style."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

def resolve_columns(df):
    """Resolve key columns from CSV."""
    out = df.copy()
    
    # Test kappa (required)
    kappa_cols = ['test_kappa', 'contract.results.test_kappa']
    kappa_col = next((c for c in kappa_cols if c in df.columns), None)
    if not kappa_col:
        raise ValueError("No test_kappa column found")
    out['test_kappa'] = pd.to_numeric(df[kappa_col], errors='coerce')
    
    # Test F1 (required for comparison)  
    f1_cols = ['test_f1', 'contract.results.test_f1']
    f1_col = next((c for c in f1_cols if c in df.columns), None)
    if not f1_col:
        raise ValueError("No test_f1 column found") 
    out['test_f1'] = pd.to_numeric(df[f1_col], errors='coerce')
    
    # Number of classes
    nc_cols = ['num_classes', 'contract.results.num_classes', 'cfg.num_of_classes']
    nc_col = next((c for c in nc_cols if c in df.columns), None)
    if not nc_col:
        raise ValueError("No num_classes column found")
    out['num_classes'] = pd.to_numeric(df[nc_col], errors='coerce')
    
    # Label mapping version (for 4-class variants)
    lv_cols = ['label_mapping_version', 'cfg.label_mapping_version', 'contract.results.label_mapping_version']
    lv_col = next((c for c in lv_cols if c in df.columns), None)
    if lv_col:
        out['label_version'] = df[lv_col].astype(str)
    else:
        out['label_version'] = 'unknown'
    
    # Subject ID
    subj_cols = ['cfg.subject_id', 'cfg.subj_id', 'subject_id', 'sum.subject_id', 'name']
    subj_col = next((c for c in subj_cols if c in df.columns), None)
    if not subj_col:
        raise ValueError("No subject ID column found")
    out['subject'] = df[subj_col].astype(str)
    
    # Validation kappa for best run selection
    val_cols = ['best_val_kappa', 'val_kappa', 'contract.results.val_kappa']
    val_col = next((c for c in val_cols if c in df.columns), None)
    out['val_kappa'] = pd.to_numeric(df[val_col], errors='coerce') if val_col else out['test_kappa']
    
    return out

def create_task_config(row):
    """Create task configuration label."""
    nc = row['num_classes']
    lv = row['label_version']
    
    if nc == 5:
        return '5-class'
    elif nc == 4:
        if 'v0' in str(lv).lower():
            return '4-class-v0'
        elif 'v1' in str(lv).lower():
            return '4-class-v1'
        else:
            return '4-class'
    else:
        return f'{int(nc)}-class'

def select_best_runs(df):
    """Select best run per subject-config combination."""
    def pick_best(group):
        # Use validation kappa if available, else test kappa
        if group['val_kappa'].notna().any():
            return group.loc[group['val_kappa'].idxmax()]
        else:
            return group.loc[group['test_kappa'].idxmax()]
    
    best_runs = (df.groupby(['subject', 'task_config'], as_index=False)
                   .apply(pick_best, include_groups=False)
                   .reset_index(drop=True))
    
    return best_runs

def filter_paired(df_best, prefer="auto"):
    """Filter to subjects who have both 5-class and one 4-class config for paired comparison."""
    # Choose which 4-class variant to compare: "v0", "v1", "auto"
    df4 = df_best[df_best['task_config'].str.startswith('4-class')]
    if prefer in ("v0", "v1"):
        target_config = f'4-class-{prefer}'
        if target_config in df4['task_config'].unique():
            df4 = df4[df4['task_config'] == target_config]
    
    # Keep best 4-class per subject if multiple variants exist
    df4 = df4.sort_values('val_kappa', ascending=False).drop_duplicates('subject')
    
    df5 = df_best[df_best['task_config'] == '5-class']
    paired_subj = set(df4['subject']).intersection(set(df5['subject']))
    
    print(f"\nPaired filtering results:")
    print(f"  Total subjects with 4-class: {len(set(df4['subject']))}")
    print(f"  Total subjects with 5-class: {len(set(df5['subject']))}")
    print(f"  Subjects with both configs: {len(paired_subj)}")
    
    filtered_df = pd.concat([
        df4[df4['subject'].isin(paired_subj)], 
        df5[df5['subject'].isin(paired_subj)]
    ], ignore_index=True)
    
    return filtered_df

def paired_delta_ci(df, metric, four='4-class'):
    """Calculate paired delta with bootstrap CI and directional Wilcoxon tests."""
    # Build paired vectors
    subjects = sorted(set(df[df['task_config'].str.startswith(four)]['subject']).intersection(
                     set(df[df['task_config']=='5-class']['subject'])))
    
    if len(subjects) == 0:
        return None
    
    x = []  # 4-class values
    y = []  # 5-class values
    
    for s in subjects:
        # Handle different 4-class variants
        four_class_data = df[(df['subject']==s) & df['task_config'].str.startswith(four)]
        if len(four_class_data) == 0:
            continue
        # Take best if multiple 4-class variants
        if len(four_class_data) > 1:
            four_class_data = four_class_data.sort_values('val_kappa', ascending=False).iloc[:1]
        
        five_class_data = df[(df['subject']==s) & (df['task_config']=='5-class')]
        if len(five_class_data) == 0:
            continue
            
        x.append(four_class_data[metric].iloc[0])
        y.append(five_class_data[metric].iloc[0])
    
    x = np.array(x)
    y = np.array(y)
    delta = y - x  # 5c - 4c (positive means 5-class better)
    
    if len(delta) < 2:
        return None
    
    # Bootstrap CI of median Δ
    rng = np.random.default_rng(42)
    boots = [np.median(rng.choice(delta, size=len(delta), replace=True)) for _ in range(4000)]
    ci = (np.percentile(boots, 2.5), np.percentile(boots, 97.5))
    med = float(np.median(delta))
    
    # Wilcoxon tests
    try:
        p_two = stats.wilcoxon(x, y, zero_method='pratt', alternative='two-sided').pvalue
        p_less = stats.wilcoxon(x, y, zero_method='pratt', alternative='less').pvalue  # 4c > 5c
        p_greater = stats.wilcoxon(x, y, zero_method='pratt', alternative='greater').pvalue  # 5c > 4c
    except:
        p_two = p_less = p_greater = np.nan
    
    return {
        'subjects': subjects,
        'n_pairs': len(delta),
        'delta': delta,
        'med': med,
        'ci': ci,
        'p_two': p_two,
        'p_less': p_less,
        'p_greater': p_greater,
        'x_values': x,  # 4-class values
        'y_values': y   # 5-class values
    }

def bootstrap_ci(data, n_boot=2000, confidence=0.95):
    """Bootstrap confidence interval for median."""
    if len(data) < 2:
        return np.nan, np.nan
    
    np.random.seed(42)
    boot_medians = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_medians.append(np.median(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(boot_medians, 100 * alpha / 2)
    upper = np.percentile(boot_medians, 100 * (1 - alpha / 2))
    return lower, upper

def wilcoxon_test(x, y):
    """Wilcoxon signed-rank test for paired data."""
    try:
        stat, p = stats.wilcoxon(x, y, alternative='two-sided')
        return p
    except:
        return np.nan

def plot_performance_comparison(df, metric, ax, title):
    """Plot performance comparison across task configurations."""
    configs = ['5-class', '4-class-v0', '4-class-v1', '4-class']
    configs = [c for c in configs if c in df['task_config'].values]
    
    positions = np.arange(len(configs))
    
    # Box plots
    data_by_config = [df[df['task_config'] == config][metric].dropna().values 
                      for config in configs]
    
    bp = ax.boxplot(data_by_config, positions=positions, patch_artist=True,
                    tick_labels=configs, widths=0.6)
    
    # Color the boxes
    for patch, config in zip(bp['boxes'], configs):
        patch.set_facecolor(COLORS.get(config, '#lightgray'))
        patch.set_alpha(0.7)
    
    # Add scatter points
    for i, config in enumerate(configs):
        data = df[df['task_config'] == config][metric].dropna()
        if len(data) > 0:
            y = data.values
            x = np.random.normal(i, 0.04, size=len(y))  # Add jitter
            ax.scatter(x, y, alpha=0.5, s=20, color=COLORS.get(config, 'gray'))
    
    # Add statistical annotations
    medians = []
    cis = []
    ns = []
    
    for config in configs:
        data = df[df['task_config'] == config][metric].dropna()
        if len(data) > 0:
            median = np.median(data)
            ci_low, ci_high = bootstrap_ci(data)
            medians.append(median)
            cis.append((ci_low, ci_high))
            ns.append(len(data))
        else:
            medians.append(np.nan)
            cis.append((np.nan, np.nan))
            ns.append(0)
    
    # Add N and median annotations
    for i, (config, median, (ci_low, ci_high), n) in enumerate(zip(configs, medians, cis, ns)):
        if not np.isnan(median):
            # N annotation at top
            ax.text(i, ax.get_ylim()[1] * 0.95, f'N={n}', 
                   ha='center', va='top', fontsize=9, weight='bold')
            
            # Median annotation
            ax.text(i, median + 0.01, f'{median:.3f}', 
                   ha='center', va='bottom', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=15)
    
    return medians, cis, ns

def plot_paired_comparison(df, ax):
    """Plot paired comparison between configs where subjects have both."""
    # Find subjects with multiple configs
    subject_configs = df.groupby('subject')['task_config'].apply(list).reset_index()
    
    # Focus on 5-class vs 4-class comparisons
    paired_data = []
    for _, row in subject_configs.iterrows():
        configs = row['task_config']
        if '5-class' in configs:
            for four_class in ['4-class-v0', '4-class-v1', '4-class']:
                if four_class in configs:
                    subj = row['subject']
                    kappa_5c = df[(df['subject'] == subj) & (df['task_config'] == '5-class')]['test_kappa'].iloc[0]
                    kappa_4c = df[(df['subject'] == subj) & (df['task_config'] == four_class)]['test_kappa'].iloc[0]
                    
                    f1_5c = df[(df['subject'] == subj) & (df['task_config'] == '5-class')]['test_f1'].iloc[0]
                    f1_4c = df[(df['subject'] == subj) & (df['task_config'] == four_class)]['test_f1'].iloc[0]
                    
                    paired_data.append({
                        'subject': subj,
                        'comparison': f'5-class vs {four_class}',
                        'kappa_5c': kappa_5c,
                        'kappa_4c': kappa_4c,
                        'f1_5c': f1_5c,
                        'f1_4c': f1_4c,
                        'delta_kappa': kappa_5c - kappa_4c,
                        'delta_f1': f1_5c - f1_4c
                    })
    
    if not paired_data:
        ax.text(0.5, 0.5, 'No paired comparisons available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Paired Comparisons')
        return
    
    paired_df = pd.DataFrame(paired_data)
    
    # Group by comparison type
    comparisons = paired_df['comparison'].unique()
    
    x_pos = 0
    colors = ['red', 'blue', 'green']
    
    for i, comparison in enumerate(comparisons):
        comp_data = paired_df[paired_df['comparison'] == comparison]
        
        # Plot lines connecting paired points
        for _, row in comp_data.iterrows():
            ax.plot([x_pos, x_pos + 0.8], [row['kappa_4c'], row['kappa_5c']], 
                   'o-', alpha=0.6, color=colors[i % len(colors)], linewidth=1, markersize=4)
        
        # Add median points
        median_4c = comp_data['kappa_4c'].median()
        median_5c = comp_data['kappa_5c'].median()
        
        ax.plot([x_pos, x_pos + 0.8], [median_4c, median_5c], 
               'o-', color=colors[i % len(colors)], linewidth=3, markersize=8,
               label=comparison)
        
        # Statistical test
        p_val = wilcoxon_test(comp_data['kappa_4c'], comp_data['kappa_5c'])
        
        # Add annotations
        ax.text(x_pos + 0.4, max(median_4c, median_5c) + 0.02, 
               f'N={len(comp_data)}\np={p_val:.3f}' if not np.isnan(p_val) else f'N={len(comp_data)}',
               ha='center', va='bottom', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        x_pos += 1.5
    
    if len(comparisons) > 0:
        tick_positions = []
        tick_labels = []
        for i in range(len(comparisons)):
            tick_positions.extend([i*1.5, i*1.5 + 0.8])
            tick_labels.extend(['4c', '5c'])
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Cohen's κ")
    ax.set_title('Paired Subject Comparisons')
    ax.legend(loc='upper left', fontsize=8)

def create_summary_table(df, is_paired=False):
    """Create summary statistics table with paired analysis results."""
    summary = []
    
    for config in df['task_config'].unique():
        config_data = df[df['task_config'] == config]
        
        kappa_data = config_data['test_kappa'].dropna()
        f1_data = config_data['test_f1'].dropna()
        
        summary.append({
            'Configuration': config,
            'N': len(config_data),
            'Kappa_Median': np.median(kappa_data) if len(kappa_data) > 0 else np.nan,
            'Kappa_IQR': np.percentile(kappa_data, [25, 75]) if len(kappa_data) > 0 else [np.nan, np.nan],
            'F1_Median': np.median(f1_data) if len(f1_data) > 0 else np.nan,
            'F1_IQR': np.percentile(f1_data, [25, 75]) if len(f1_data) > 0 else [np.nan, np.nan]
        })
    
    summary_df = pd.DataFrame(summary)
    
    # Add paired analysis results if applicable
    if is_paired:
        delta_stats = paired_delta_ci(df, 'test_kappa', '4-class')
        if delta_stats:
            summary.append({
                'Configuration': 'Paired Δ (5c-4c)',
                'N': delta_stats['n_pairs'],
                'Kappa_Median': delta_stats['med'],
                'Kappa_IQR': delta_stats['ci'],
                'F1_Median': np.nan,
                'F1_IQR': [np.nan, np.nan]
            })
            summary_df = pd.DataFrame(summary)
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description='Task granularity analysis: 5-class vs 4-class variants')
    parser.add_argument('--csv', required=True, help='Path to CSV file')
    parser.add_argument('--out', default='Plot_Clean/figures/fig5', help='Output directory')
    parser.add_argument('--paired-only', action='store_true', 
                       help='Restrict to subjects with both 5-class and 4-class configurations')
    parser.add_argument('--four-class-variant', choices=['auto', 'v0', 'v1'], default='auto',
                       help='Which 4-class variant to prefer for paired comparison')
    args = parser.parse_args()
    
    # Setup
    setup_style()
    
    print(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows")
    
    # Resolve columns
    df = resolve_columns(df)
    
    # Filter to valid data
    df = df.dropna(subset=['test_kappa', 'test_f1', 'num_classes', 'subject'])
    df = df[df['num_classes'].isin([4, 5])]  # Only 4-class and 5-class
    print(f"After filtering: {len(df)} rows")
    
    # Create task configuration labels
    df['task_config'] = df.apply(create_task_config, axis=1)
    print("\nTask configurations found:")
    print(df['task_config'].value_counts())
    
    # Select best runs per subject-config
    df_best = select_best_runs(df)
    print(f"\nAfter selecting best runs: {len(df_best)} rows")
    
    # Apply paired filtering if requested
    df_plot = df_best.copy()
    if args.paired_only:
        df_plot = filter_paired(df_best, prefer=args.four_class_variant)
        print(f"After paired filtering: {len(df_plot)} rows")
        if len(df_plot) == 0:
            print("ERROR: No paired subjects found. Try without --paired-only flag.")
            return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title_suffix = ' (Paired Subjects Only)' if args.paired_only else ''
    fig.suptitle(f'Task Granularity Analysis: 5-class vs 4-class Sleep Staging{title_suffix}', 
                fontsize=16, fontweight='bold')
    
    # Panel 1: Kappa comparison
    plot_performance_comparison(df_plot, 'test_kappa', axes[0,0], 
                              "Cohen's κ by Task Configuration")
    
    # Panel 2: F1 comparison
    plot_performance_comparison(df_plot, 'test_f1', axes[0,1], 
                              'F1 Score by Task Configuration')
    
    # Panel 3: Paired comparison
    plot_paired_comparison(df_plot, axes[1,0])
    
    # Panel 4: Summary statistics
    axes[1,1].axis('off')  # Turn off axis for text table
    
    summary_df = create_summary_table(df_plot, is_paired=args.paired_only)
    table_text = "Summary Statistics\n\n"
    table_text += f"{'Config':<14} {'N':>3} {'κ Med':>7} {'κ Range':>14}\n"
    table_text += "-" * 42 + "\n"
    
    for _, row in summary_df.iterrows():
        config_name = row['Configuration']
        if config_name == 'Paired Δ (5c-4c)':
            # Special formatting for paired delta
            ci_low, ci_high = row['Kappa_IQR']
            range_str = f"[{ci_low:.3f},{ci_high:.3f}]"
        else:
            # Regular IQR formatting
            if not np.any(np.isnan(row['Kappa_IQR'])):
                range_str = f"[{row['Kappa_IQR'][0]:.3f}-{row['Kappa_IQR'][1]:.3f}]"
            else:
                range_str = "N/A"
        
        table_text += f"{config_name:<14} {row['N']:>3} {row['Kappa_Median']:>7.3f} {range_str:>14}\n"
    
    # Add statistical test summary if paired
    if args.paired_only:
        delta_stats = paired_delta_ci(df_plot, 'test_kappa', '4-class')
        if delta_stats:
            table_text += "\nPaired Tests:\n"
            table_text += f"Two-sided p = {delta_stats['p_two']:.4f}\n"
            table_text += f"5c > 4c p = {delta_stats['p_greater']:.4f}\n"
    
    axes[1,1].text(0.05, 0.95, table_text, transform=axes[1,1].transAxes, 
                   fontsize=9, fontfamily='monospace', va='top')
    
    plt.tight_layout()
    
    # Save outputs
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    suffix = '_paired' if args.paired_only else ''
    fig_path = output_dir / f'fig_task_granularity{suffix}.svg'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {fig_path}")
    
    # Save summary
    summary_path = output_dir / 'task_granularity_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")
    
    # Print key results
    if args.paired_only:
        delta_stats = paired_delta_ci(df_plot, 'test_kappa', '4-class')
        if delta_stats:
            print(f"\n=== PAIRED COMPARISON RESULTS ===")
            print(f"N paired subjects: {delta_stats['n_pairs']}")
            print(f"Median Δ (5c - 4c): {delta_stats['med']:+.4f}")
            print(f"95% CI: [{delta_stats['ci'][0]:.4f}, {delta_stats['ci'][1]:.4f}]")
            print(f"Two-sided p-value: {delta_stats['p_two']:.4f}")
            print(f"5c > 4c p-value: {delta_stats['p_greater']:.4f}")
            
            if delta_stats['p_two'] < 0.05:
                direction = "better" if delta_stats['med'] > 0 else "worse"
                print(f"\n*** SIGNIFICANT: 5-class performs {direction} than 4-class ***")
            else:
                print(f"\n*** NO SIGNIFICANT DIFFERENCE between 5-class and 4-class ***")
    
    plt.show()

if __name__ == '__main__':
    main()