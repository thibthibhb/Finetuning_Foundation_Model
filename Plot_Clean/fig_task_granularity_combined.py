#!/usr/bin/env python3
"""
Combined Task Granularity Analysis: 5-class vs 4-class Sleep Staging
Merges distribution and paired delta analyses into a single publication-ready figure.

Usage:
    python Plot-Clean/fig_task_granularity_combined.py --csv Plot_Clean/data/all_runs_flat.csv --out Plot_Clean/figures --paired-only
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from pathlib import Path
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Import consistent figure styling
from figure_style import (
    setup_figure_style, get_color, save_figure, 
    add_yasa_baseline, add_significance_marker,
    bootstrap_ci_median, wilcoxon_test,
    format_n_caption, add_sample_size_annotation
)

# Removed - using setup_figure_style() from figure_style.py instead

# Color scheme (colorblind-safe) - updated to use figure_style
COLORS = {
    '5-class': get_color('5_class'),
    '4-class': get_color('4_class'),
    'positive': get_color('cbramod'),  # Dark green for positive deltas
    'negative': get_color('yasa')      # Red for negative deltas
}

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV and standardize column names for task granularity analysis.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with standardized columns
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Map actual column names to expected ones (reusing delta script logic)
    actual_cols = {
        'subject_id': 'cfg.subject_id',
        'config': 'label_scheme', 
        'test_kappa': 'sum.test_kappa',
        'test_f1': 'sum.test_f1'
    }
    
    # Check for required columns and use fallbacks
    missing_cols = []
    for expected, actual in actual_cols.items():
        if actual not in df.columns:
            if expected == 'test_kappa' and 'contract.results.test_kappa' in df.columns:
                actual_cols[expected] = 'contract.results.test_kappa'
            elif expected == 'test_f1' and 'contract.results.test_f1' in df.columns:
                actual_cols[expected] = 'contract.results.test_f1'
            else:
                missing_cols.append(actual)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create standardized DataFrame
    df_work = df.copy()
    df_work['subject_id'] = df_work[actual_cols['subject_id']]
    df_work['config'] = df_work[actual_cols['config']]
    
    # Handle metrics with fallback
    if actual_cols['test_kappa'] == 'sum.test_kappa':
        df_work['test_kappa'] = df_work['sum.test_kappa'].fillna(df_work.get('contract.results.test_kappa', np.nan))
        df_work['test_f1'] = df_work['sum.test_f1'].fillna(df_work.get('contract.results.test_f1', np.nan))
    else:
        df_work['test_kappa'] = df_work[actual_cols['test_kappa']]
        df_work['test_f1'] = df_work[actual_cols['test_f1']]
    
    # Filter to finished runs only
    if 'state' in df_work.columns:
        df_work = df_work[df_work['state'] == 'finished'].copy()
    
    # Map config names: treat 4c-v0 and 4c-v1 as "4-class", keep "5c" as "5-class"
    config_mapping = {
        '5c': '5-class',
        '4c-v0': '4-class', 
        '4c-v1': '4-class'
    }
    
    df_work['task_config'] = df_work['config'].map(config_mapping)
    df_work = df_work.dropna(subset=['task_config'])
    
    # Remove rows with missing metrics or subject IDs
    df_clean = df_work.dropna(subset=['test_kappa', 'test_f1', 'subject_id']).copy()
    
    if df_clean.empty:
        raise ValueError("No valid data found after cleaning")
    
    return df_clean[['subject_id', 'task_config', 'test_kappa', 'test_f1']]

def filter_paired(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to subjects who have data in both 5-class and 4-class configurations.
    Handle duplicates by taking the best performance for each subject-config pair.
    
    Args:
        df: DataFrame with all data
        
    Returns:
        DataFrame with only paired subjects
    """
    # Handle duplicates by taking the best performance for each subject-config pair
    df_best = df.groupby(['subject_id', 'task_config']).agg({
        'test_kappa': 'max',
        'test_f1': 'max'
    }).reset_index()
    
    # Find subjects present in both configs
    subjects_4c = set(df_best[df_best['task_config'] == '4-class']['subject_id'])
    subjects_5c = set(df_best[df_best['task_config'] == '5-class']['subject_id'])
    paired_subjects = subjects_4c & subjects_5c
    
    if not paired_subjects:
        raise ValueError("No subjects found in both 4-class and 5-class configurations")
    
    # Return only paired subjects
    df_paired = df_best[df_best['subject_id'].isin(paired_subjects)].copy()
    
    print(f"Found {len(paired_subjects)} paired subjects: {sorted(paired_subjects)}")
    print(f"Data shape after pairing: {df_paired.shape}")
    
    return df_paired

# Removed - using bootstrap_ci_median from figure_style.py instead

def format_float(x: float) -> str:
    """Format float to 3 decimals, stripping trailing zeros."""
    return f"{x:.3f}".rstrip('0').rstrip('.')

def plot_distributions(ax, df: pd.DataFrame, metric: str, palette: Dict[str, str]) -> None:
    """
    Plot side-by-side boxplots with jittered points for distribution comparison.
    
    Args:
        ax: Matplotlib axis
        df: DataFrame with paired subjects
        metric: Column name for the metric ('test_kappa' or 'test_f1')
        palette: Color palette dictionary
    """
    configs = ['5-class', '4-class']
    positions = [0, 1]
    
    # Prepare data
    data_by_config = []
    for config in configs:
        config_data = df[df['task_config'] == config][metric].dropna().values
        data_by_config.append(config_data)
    
    # Create boxplots
    bp = ax.boxplot(data_by_config, positions=positions, patch_artist=True,
                    widths=0.5, showfliers=True, whis=1.5)
    
    # Color the boxes
    colors_box = [palette['5-class'], palette['4-class']]
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)
    
    # Add jittered scatter points
    np.random.seed(42)  # For consistent jitter
    for i, (config, data) in enumerate(zip(configs, data_by_config)):
        if len(data) > 0:
            x_jitter = np.random.normal(i, 0.04, size=len(data))
            ax.scatter(x_jitter, data, alpha=0.6, s=20, color=colors_box[i], 
                      edgecolors='white', linewidth=0.5, zorder=3)
    
    # Add median labels in small rounded boxes
    for i, (config, data) in enumerate(zip(configs, data_by_config)):
        if len(data) > 0:
            median_val = np.median(data)
            ax.text(i, ax.get_ylim()[1] * 0.92, format_float(median_val), 
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                            edgecolor='gray', linewidth=0.5))
    
    # Add sample size annotations
    for i, data in enumerate(data_by_config):
        ax.text(i, ax.get_ylim()[1] * 0.98, f'N = {len(data)}', 
               ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Styling
    ax.set_xticks(positions)
    ax.set_xticklabels(configs)
    
    if metric == 'test_kappa':
        ax.set_ylabel("Test Kappa")
    else:
        ax.set_ylabel("Test F1")
    
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_deltas(ax, df: pd.DataFrame, metric: str, palette: Dict[str, str]) -> None:
    """
    Plot paired subject deltas as sorted bar chart with median CI.
    
    Args:
        ax: Matplotlib axis
        df: DataFrame with paired subjects
        metric: Column name for the metric ('test_kappa' or 'test_f1')
        palette: Color palette dictionary
    """
    # Compute deltas (4-class - 5-class)
    df_pivot = df.pivot(index='subject_id', columns='task_config', values=metric)
    
    if '4-class' not in df_pivot.columns or '5-class' not in df_pivot.columns:
        ax.text(0.5, 0.5, 'No paired data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        return
    
    deltas = (df_pivot['4-class'] - df_pivot['5-class']).dropna()
    
    if len(deltas) == 0:
        ax.text(0.5, 0.5, 'No valid paired data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        return
    
    # Sort subjects by delta for visualization
    deltas_sorted = deltas.sort_values()
    x_pos = np.arange(len(deltas_sorted))
    
    # Color bars by sign
    colors = [palette['positive'] if d >= 0 else palette['negative'] for d in deltas_sorted.values]
    
    # Plot bars
    bars = ax.bar(x_pos, deltas_sorted.values, color=colors, alpha=0.8, 
                 edgecolor='white', linewidth=0.5)
    
    # Add zero line
    ax.axhline(0, color='darkgray', linewidth=1, alpha=0.8)
    
    # Bootstrap median and CI using consistent function
    median_val = np.median(deltas.values)
    ci_low, ci_high = bootstrap_ci_median(deltas.values)
    
    # Add median line and CI band
    ax.axhline(median_val, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axhspan(ci_low, ci_high, alpha=0.2, color='gray')
    
    # Statistical test using consistent function
    p_val = wilcoxon_test(deltas.values)
    try:
        stat, _ = wilcoxon(deltas.values, alternative='two-sided')
        # Rank-biserial effect size
        n = len(deltas)
        r = 1 - (2 * stat) / (n * (n + 1))
        r = abs(r)  # Take absolute value
    except:
        r = np.nan
    
    # Compact subtitle with statistics
    metric_symbol = 'κ' if metric == 'test_kappa' else 'F1'
    p_str = f"{p_val:.3f}" if not np.isnan(p_val) and p_val >= 0.001 else f"{p_val:.2e}" if not np.isnan(p_val) else "N/A"
    
    title_text = f"Δ{metric_symbol} (4c–5c): median = {median_val:+.3f}, 95% CI [{ci_low:+.3f}, {ci_high:+.3f}]\nWilcoxon p = {p_str}"
    if not np.isnan(r):
        title_text += f", r = {r:.2f}"
    title_text += f" (N={len(deltas)})"
    
    ax.set_title(title_text, fontsize=10, pad=10)
    
    # Labels and styling
    if metric == 'test_kappa':
        ax.set_ylabel("Δκ (4c – 5c)")
    else:
        ax.set_ylabel("ΔF1 (4c – 5c)")
    
    ax.set_xlabel("Subjects (sorted by Δ)")
    ax.set_xticks([])  # Hide subject indices
    
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Symmetric y-limits around zero
    y_max = max(abs(deltas_sorted.min()), abs(deltas_sorted.max()))
    y_lim = np.ceil(y_max * 100) / 100  # Round up to nearest 0.01
    ax.set_ylim(-y_lim, y_lim)

def main():
    parser = argparse.ArgumentParser(description='Combined Task Granularity Analysis')
    parser.add_argument('--csv', required=True, help='Path to CSV file with results')
    parser.add_argument('--out', required=True, help='Output directory for figures')
    parser.add_argument('--paired-only', action='store_true', 
                       help='Filter to subjects present in both configurations')
    
    args = parser.parse_args()
    
    print("=== Combined Task Granularity Analysis ===")
    
    # Load data
    print(f"Loading data from {args.csv}...")
    df = load_data(args.csv)
    print(f"Loaded {len(df)} rows after cleaning")
    
    # Apply paired filtering if requested
    if args.paired_only:
        print("\nFiltering to paired subjects only...")
        df = filter_paired(df)
        print(f"After paired filtering: {len(df)} rows")
    
    if len(df) == 0:
        print("ERROR: No data remaining after filtering.")
        return
    
    # Set up consistent styling and create figure with 2x2 grid
    setup_figure_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.5), constrained_layout=True)
    
    # Row 1: Distribution plots
    plot_distributions(axes[0, 0], df, 'test_kappa', COLORS)
    plot_distributions(axes[0, 1], df, 'test_f1', COLORS)
    
    # Row 2: Delta plots (only works with paired data)
    if args.paired_only or len(set(df['subject_id'])) < len(df):
        plot_deltas(axes[1, 0], df, 'test_kappa', COLORS)
        plot_deltas(axes[1, 1], df, 'test_f1', COLORS)
    else:
        # If not paired-only, still try to create deltas if possible
        df_paired_auto = filter_paired(df)
        if len(df_paired_auto) > 0:
            plot_deltas(axes[1, 0], df_paired_auto, 'test_kappa', COLORS)
            plot_deltas(axes[1, 1], df_paired_auto, 'test_f1', COLORS)
    
    # Global title
    title_text = "Task Granularity: 5-class vs 4-class Sleep Staging"
    if args.paired_only:
        title_text += " (Paired Subjects Only)"
    fig.suptitle(title_text, fontsize=14, fontweight='bold')
    
    # Add footnote for delta plots
    fig.text(0.5, 0.02, 'Per-subject paired differences (4c–5c). Line/band: median and bootstrap 95% CI.',
             ha='center', va='bottom', fontsize=9, style='italic')
    
    # Save figures
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    png_path = output_dir / 'fig_task_granularity_combined.png'
    pdf_path = output_dir / 'fig_task_granularity_combined.pdf'
    
    # Use consistent save function
    base_path = output_dir / 'fig_task_granularity_combined'
    saved_files = save_figure(fig, base_path)
    
    # Print N information for caption
    n_subjects = len(set(df['subject_id']))
    n_runs = len(df)
    print(f"\n📋 Caption info: {format_n_caption(n_subjects, n_runs, 'subjects')}")
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    for config in ['5-class', '4-class']:
        config_data = df[df['task_config'] == config]
        if len(config_data) > 0:
            kappa_mean = config_data['test_kappa'].mean()
            kappa_std = config_data['test_kappa'].std()
            f1_mean = config_data['test_f1'].mean()
            f1_std = config_data['test_f1'].std()
            print(f"{config}: N={len(config_data)}, κ={kappa_mean:.3f}±{kappa_std:.3f}, F1={f1_mean:.3f}±{f1_std:.3f}")
    
    plt.show()

if __name__ == '__main__':
    main()