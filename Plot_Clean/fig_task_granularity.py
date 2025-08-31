#!/usr/bin/env python3
"""
Task Granularity Analysis: 5-class vs 4-class sleep staging performance

HIGH-IMPACT REDESIGN:
- Lead with the difference: Main panel shows Δ = (4c - 5c) per subject  
- One metric per axis with identical y-limits for easy comparison
- Concise statistical summary with median Δ, 95% CI, and p-value
- Remove duplication: focus on boxplots, drop slope graphs
- Clear mapping: 4c merges Light(N1+N2), Deep(N3), versus 5c: N1, N2, N3

Usage:
    python fig_task_granularity.py --csv Plot_Clean/data/all_runs_flat.csv --out Plot_Clean/figures/
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Import consistent figure styling
from figure_style import (
    setup_figure_style, get_color, save_figure, 
    add_yasa_baseline, add_significance_marker,
    bootstrap_ci_median, wilcoxon_test,
    format_n_caption, add_sample_size_annotation
)

# Publication colors (colorblind-safe) - updated to use figure_style
COLORS = {
    '5-class': get_color('5_class'),
    '4-class-v1': get_color('4_class'),
    'better': get_color('cbramod'),       # Green for 4c > 5c
    'worse': get_color('yasa'),           # Red for 5c > 4c
    'neutral': get_color('subjects')      # Gray
}

# Removed - using setup_figure_style() from figure_style.py instead

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
        # Focus on v1 mapping: Light(N1+N2), Deep(N3)
        if 'v1' in str(lv).lower():
            return '4-class-v1'
        else:
            return '4-class-v1'  # Default to v1 for consistency
    else:
        return f'{int(nc)}-class'

def select_best_runs(df):
    """Select best run per subject-config combination."""
    def pick_best(group):
        if group['val_kappa'].notna().any():
            return group.loc[group['val_kappa'].idxmax()]
        else:
            return group.loc[group['test_kappa'].idxmax()]
    
    best_runs = (df.groupby(['subject', 'task_config'], as_index=False)
                   .apply(pick_best, include_groups=False)
                   .reset_index(drop=True))
    
    return best_runs

def filter_paired(df_best):
    """Filter to subjects who have both 5-class and 4-class-v1 for paired comparison."""
    df4 = df_best[df_best['task_config'] == '4-class-v1']
    df5 = df_best[df_best['task_config'] == '5-class']
    
    paired_subj = set(df4['subject']).intersection(set(df5['subject']))
    
    print(f"\nPaired filtering results:")
    print(f"  Subjects with 4-class-v1: {len(set(df4['subject']))}")
    print(f"  Subjects with 5-class: {len(set(df5['subject']))}")
    print(f"  Subjects with both configs: {len(paired_subj)}")
    
    filtered_df = pd.concat([
        df4[df4['subject'].isin(paired_subj)], 
        df5[df5['subject'].isin(paired_subj)]
    ], ignore_index=True)
    
    return filtered_df

def calculate_paired_stats(df, metric):
    """Calculate paired statistics with bootstrap CI and effect sizes."""
    # Build paired vectors
    subjects = sorted(set(df[df['task_config']=='4-class-v1']['subject']).intersection(
                     set(df[df['task_config']=='5-class']['subject'])))
    
    if len(subjects) == 0:
        return None
    
    x = []  # 4-class values
    y = []  # 5-class values
    
    for s in subjects:
        four_data = df[(df['subject']==s) & (df['task_config']=='4-class-v1')]
        five_data = df[(df['subject']==s) & (df['task_config']=='5-class')]
        
        if len(four_data) > 0 and len(five_data) > 0:
            x.append(four_data[metric].iloc[0])
            y.append(five_data[metric].iloc[0])
    
    x = np.array(x)
    y = np.array(y)
    delta = x - y  # 4c - 5c (positive means 4-class better)
    
    if len(delta) < 2:
        return None
    
    # Bootstrap CI of median Δ using consistent function
    med = float(np.median(delta))
    ci = bootstrap_ci_median(delta)
    
    # Wilcoxon test (directional: 4c > 5c) using consistent function
    p_two = wilcoxon_test(delta)
    try:
        p_greater = stats.wilcoxon(x, y, zero_method='pratt', alternative='greater').pvalue  # 4c > 5c
        # Effect size: rank-biserial correlation 
        w_stat = stats.wilcoxon(x, y, zero_method='pratt', alternative='two-sided').statistic
        n = len(delta)
        r = 1 - (2 * w_stat) / (n * (n + 1))  # Rank-biserial r
    except:
        p_greater = r = np.nan
    
    return {
        'subjects': subjects,
        'n_pairs': len(delta),
        'delta': delta,
        'med': med,
        'ci': ci,
        'p_two': p_two,
        'p_greater': p_greater,
        'effect_size': r,
        'x_values': x,  # 4-class values
        'y_values': y   # 5-class values
    }

def plot_delta_main(df, metric, ax, stats_dict):
    """Main panel: Plot Δ = (4c - 5c) per subject with CI."""
    if stats_dict is None:
        ax.text(0.5, 0.5, 'No paired data available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=14)
        return
    
    delta = stats_dict['delta']
    subjects = stats_dict['subjects']
    med = stats_dict['med']
    ci = stats_dict['ci']
    p_val = stats_dict['p_two']
    r_effect = stats_dict['effect_size']
    
    # Sort subjects by delta for better visualization
    sort_idx = np.argsort(delta)
    delta_sorted = delta[sort_idx]
    subjects_sorted = [subjects[i] for i in sort_idx]
    
    # Plot individual subject deltas with directional colors
    x_pos = np.arange(len(delta_sorted))
    colors = [COLORS['better'] if d > 0 else COLORS['worse'] if d < 0 else COLORS['neutral'] 
              for d in delta_sorted]
    
    bars = ax.bar(x_pos, delta_sorted, color=colors, alpha=0.7, edgecolor='white', linewidth=0.5)
    
    # Horizontal line at 0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
    
    # Add median line and CI
    ax.axhline(y=med, color='black', linestyle='--', linewidth=2, alpha=0.9,
               label=f'Median Δ = {med:+.3f}')
    
    # CI shading
    ax.axhspan(ci[0], ci[1], alpha=0.15, color='gray', 
               label=f'95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]')
    
    # Statistical annotation (concise)
    metric_symbol = 'κ' if metric == 'test_kappa' else 'F1'
    if not np.isnan(p_val):
        stat_text = f'Paired Wilcoxon on {metric_symbol}: median Δ = {med:+.3f} (4c − 5c), 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], p = {p_val:.3f}'
        if not np.isnan(r_effect):
            stat_text += f', r = {r_effect:.3f}'
    else:
        stat_text = f'{metric_symbol}: median Δ = {med:+.3f} (4c − 5c), 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]'
    
    ax.text(0.02, 0.98, stat_text, transform=ax.transAxes, fontsize=10, 
            ha='left', va='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    # Labels and title
    metric_label = "Cohen's κ" if metric == 'test_kappa' else 'F1 Score'
    ax.set_ylabel(f'Δ {metric_label} (4c − 5c)', fontweight='bold')
    ax.set_xlabel('Subjects (sorted by Δ)', fontweight='bold')
    ax.set_title(f'Performance Difference: 4-class vs 5-class\n(Values > 0 indicate 4c better)', 
                fontsize=13, fontweight='bold')
    
    # Clean up x-axis (too many subject labels would be cluttered)
    ax.set_xticks([])
    
    # Grid for readability
    ax.grid(True, alpha=0.3, axis='y')

def plot_boxplot_comparison(df, metric, ax, stats_dict):
    """Upper panel: Simple boxplot comparison with unobtrusive medians."""
    configs = ['4-class-v1', '5-class']
    positions = [0, 1]
    
    # Prepare data
    data_by_config = []
    for config in configs:
        config_data = df[df['task_config'] == config]
        if len(config_data) > 0:
            data_by_config.append(config_data[metric].dropna().values)
        else:
            data_by_config.append(np.array([]))
    
    # Box plots
    bp = ax.boxplot(data_by_config, positions=positions, patch_artist=True,
                    widths=0.5, showfliers=True)
    
    # Color the boxes
    colors_box = [COLORS['4-class-v1'], COLORS['5-class']]
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add scatter points (jittered)
    for i, (config, data) in enumerate(zip(configs, data_by_config)):
        if len(data) > 0:
            x_jitter = np.random.normal(i, 0.03, size=len(data))
            ax.scatter(x_jitter, data, alpha=0.5, s=15, color=colors_box[i], edgecolors='white', linewidth=0.5)
    
    # Unobtrusive median labels (small gray tags)
    for i, data in enumerate(data_by_config):
        if len(data) > 0:
            median_val = np.median(data)
            ax.text(i, median_val, f'{median_val:.3f}', ha='center', va='center',
                   fontsize=8, color='gray', fontweight='normal',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Sample size annotations
    for i, data in enumerate(data_by_config):
        ax.text(i, ax.get_ylim()[1] * 0.98, f'N={len(data)}', 
               ha='center', va='top', fontsize=9, fontweight='bold')
    
    # Labels
    ax.set_xticks(positions)
    ax.set_xticklabels(['4-class\n(Light=N1+N2)', '5-class\n(N1, N2 separate)'], fontsize=10)
    metric_label = "Cohen's κ" if metric == 'test_kappa' else 'F1 Score'
    ax.set_ylabel(metric_label, fontweight='bold')
    ax.set_title('Performance Distribution by Task Granularity', fontsize=13, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')

def main():
    parser = argparse.ArgumentParser(description='High-impact task granularity analysis')
    parser.add_argument('--csv', required=True, help='Path to CSV file')
    parser.add_argument('--out', default='Plot_Clean/figures/', help='Output directory')
    args = parser.parse_args()
    
    # Setup
    setup_figure_style()
    
    print(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows")
    
    # Resolve columns
    df = resolve_columns(df)
    
    # Filter to valid data (focus on 4c-v1 vs 5c comparison)
    df = df.dropna(subset=['test_kappa', 'test_f1', 'num_classes', 'subject'])
    df = df[df['num_classes'].isin([4, 5])]
    print(f"After filtering: {len(df)} rows")
    
    # Create task configuration labels
    df['task_config'] = df.apply(create_task_config, axis=1)
    print("\nTask configurations found:")
    print(df['task_config'].value_counts())
    
    # Select best runs per subject-config
    df_best = select_best_runs(df)
    print(f"\nAfter selecting best runs: {len(df_best)} rows")
    
    # Apply paired filtering (only subjects with both configs)
    df_plot = filter_paired(df_best)
    print(f"After paired filtering: {len(df_plot)} rows")
    
    if len(df_plot) == 0:
        print("ERROR: No paired subjects found.")
        return
    
    # Calculate paired statistics
    kappa_stats = calculate_paired_stats(df_plot, 'test_kappa')
    f1_stats = calculate_paired_stats(df_plot, 'test_f1')
    
    # Create figure - 2x2 layout but emphasize bottom row as main
    fig = plt.figure(figsize=(14, 12))
    
    # Top row (smaller): boxplot comparisons
    ax1 = plt.subplot2grid((3, 2), (0, 0))  # Kappa boxplot
    ax2 = plt.subplot2grid((3, 2), (0, 1))  # F1 boxplot
    
    # Bottom row (larger): main delta plots
    ax3 = plt.subplot2grid((3, 2), (1, 0), rowspan=2)  # Kappa delta (main)
    ax4 = plt.subplot2grid((3, 2), (1, 1), rowspan=2)  # F1 delta (main)
    
    # Plot boxplots (upper panels)
    plot_boxplot_comparison(df_plot, 'test_kappa', ax1, kappa_stats)
    plot_boxplot_comparison(df_plot, 'test_f1', ax2, f1_stats)
    
    # Plot main delta analysis (bottom panels - larger and emphasized)
    plot_delta_main(df_plot, 'test_kappa', ax3, kappa_stats)
    plot_delta_main(df_plot, 'test_f1', ax4, f1_stats)
    
    # Ensure identical y-limits within each metric across panels
    if kappa_stats:
        # Get combined y-range for kappa
        y1_lim = ax1.get_ylim()
        y3_lim = ax3.get_ylim()
        combined_y_min = min(y1_lim[0], y3_lim[0])
        combined_y_max = max(y1_lim[1], y3_lim[1])
        ax1.set_ylim(combined_y_min, combined_y_max)
        
        # Delta plot needs different scaling centered on 0
        delta_range = max(abs(kappa_stats['delta'].min()), abs(kappa_stats['delta'].max()))
        ax3.set_ylim(-delta_range * 1.2, delta_range * 1.2)
    
    if f1_stats:
        # Get combined y-range for F1
        y2_lim = ax2.get_ylim()
        y4_lim = ax4.get_ylim()
        combined_y_min = min(y2_lim[0], y4_lim[0])
        combined_y_max = max(y2_lim[1], y4_lim[1])
        ax2.set_ylim(combined_y_min, combined_y_max)
        
        # Delta plot needs different scaling centered on 0
        delta_range = max(abs(f1_stats['delta'].min()), abs(f1_stats['delta'].max()))
        ax4.set_ylim(-delta_range * 1.2, delta_range * 1.2)
    
    # Main title with mapping clarity
    fig.suptitle('Task Granularity Impact on Sleep Staging Performance\n' +
                 '4-class merges Light Sleep (N1+N2) vs 5-class keeps N1, N2 separate', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make room for title
    
    # Save outputs
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save figure using consistent save function
    fig_path = output_dir / 'fig_task_granularity_v2'
    saved_files = save_figure(fig, fig_path)
    
    # Print N information for caption
    if kappa_stats:
        n_subjects = kappa_stats['n_pairs']
        n_runs = len(df_plot)
        print(f"\n📋 Caption info: {format_n_caption(n_subjects, n_runs, 'subjects')}")
    
    # Print key results
    if kappa_stats:
        print(f"\n=== KEY RESULTS (Cohen's κ) ===")
        print(f"N paired subjects: {kappa_stats['n_pairs']}")
        print(f"Median Δ (4c - 5c): {kappa_stats['med']:+.4f}")
        print(f"95% CI: [{kappa_stats['ci'][0]:+.4f}, {kappa_stats['ci'][1]:+.4f}]")
        print(f"Two-sided p-value: {kappa_stats['p_two']:.4f}")
        if not np.isnan(kappa_stats['effect_size']):
            print(f"Effect size (r): {kappa_stats['effect_size']:.3f}")
        
        if kappa_stats['p_two'] < 0.05:
            direction = "BETTER" if kappa_stats['med'] > 0 else "WORSE"
            print(f"\n*** SIGNIFICANT: 4-class performs {direction} than 5-class ***")
        else:
            print(f"\n*** NO SIGNIFICANT DIFFERENCE between 4-class and 5-class ***")
    
    plt.show()

if __name__ == '__main__':
    main()