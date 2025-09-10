#!/usr/bin/env python3
"""
Task Granularity Analysis: Paired 4-class vs 5-class Performance
Shows per-subject paired differences (Δ = 4c - 5c) for kappa and F1 metrics.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Import consistent figure styling
import sys; sys.path.append("../style"); from figure_style import (
    setup_figure_style, get_color, save_figure, 
    bootstrap_ci_median, wilcoxon_test
)

def load_and_pair(csv_path: str) -> pd.DataFrame:
    """
    Load CSV and return only subjects present in both 4c and 5c configs.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with paired subjects only
        
    Raises:
        ValueError: If required columns missing or no paired subjects found
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Map actual column names to expected ones
    actual_cols = {
        'subject_id': 'cfg.subject_id',
        'config': 'label_scheme', 
        'kappa': 'sum.test_kappa',  # Use sum.test_kappa as primary, fallback to contract version
        'f1': 'sum.test_f1'         # Use sum.test_f1 as primary, fallback to contract version
    }
    
    # Check for required columns
    missing_cols = []
    for expected, actual in actual_cols.items():
        if actual not in df.columns:
            # Try backup columns for kappa/f1
            if expected == 'kappa' and 'contract.results.test_kappa' in df.columns:
                actual_cols[expected] = 'contract.results.test_kappa'
            elif expected == 'f1' and 'contract.results.test_f1' in df.columns:
                actual_cols[expected] = 'contract.results.test_f1'
            else:
                missing_cols.append(actual)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create standardized column names
    df_work = df.copy()
    df_work['subject_id'] = df_work[actual_cols['subject_id']]
    df_work['config'] = df_work[actual_cols['config']]
    
    # Use sum columns if available, otherwise contract columns
    if actual_cols['kappa'] == 'sum.test_kappa':
        df_work['kappa'] = df_work['sum.test_kappa'].fillna(df_work.get('contract.results.test_kappa', np.nan))
        df_work['f1'] = df_work['sum.test_f1'].fillna(df_work.get('contract.results.test_f1', np.nan))
    else:
        df_work['kappa'] = df_work[actual_cols['kappa']]
        df_work['f1'] = df_work[actual_cols['f1']]
    
    # Filter to finished runs only
    if 'state' in df_work.columns:
        df_work = df_work[df_work['state'] == 'finished'].copy()
    
    # Map config names: treat 4c-v0 and 4c-v1 as "4c", keep "5c"
    config_mapping = {
        '5c': '5c',
        '4c-v0': '4c', 
        '4c-v1': '4c'
    }
    
    df_work['config'] = df_work['config'].map(config_mapping)
    df_work = df_work.dropna(subset=['config'])
    
    # Filter to 4c and 5c configs only
    df_filtered = df_work[df_work['config'].isin(['4c', '5c'])].copy()
    
    if df_filtered.empty:
        raise ValueError("No data found for configs '4c' or '5c'")
    
    # Remove rows with missing metrics
    df_filtered = df_filtered.dropna(subset=['kappa', 'f1', 'subject_id'])
    
    if df_filtered.empty:
        raise ValueError("No valid data (non-null kappa, f1, subject_id) found")
    
    # Find subjects present in both configs
    subjects_4c = set(df_filtered[df_filtered['config'] == '4c']['subject_id'])
    subjects_5c = set(df_filtered[df_filtered['config'] == '5c']['subject_id'])
    paired_subjects = subjects_4c & subjects_5c
    
    if not paired_subjects:
        print(f"Available 4c subjects: {sorted(subjects_4c)}")
        print(f"Available 5c subjects: {sorted(subjects_5c)}")
        raise ValueError("No subjects found in both 4c and 5c configurations")
    
    # Return only paired subjects
    df_paired = df_filtered[df_filtered['subject_id'].isin(paired_subjects)].copy()
    
    # Handle duplicates by taking the best kappa for each subject-config pair
    df_best = df_paired.groupby(['subject_id', 'config']).agg({
        'kappa': 'max',  # Take best kappa
        'f1': 'max'      # Take best f1
    }).reset_index()
    
    print(f"Found {len(paired_subjects)} paired subjects: {sorted(paired_subjects)}")
    print(f"Data shape after deduplication: {df_best.shape}")
    
    return df_best[['subject_id', 'config', 'kappa', 'f1']]

def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-subject paired differences: delta = 4c - 5c.
    
    Args:
        df: DataFrame with paired subjects
        
    Returns:
        DataFrame with columns: subject_id, delta_kappa, delta_f1
    """
    # Pivot to have 4c and 5c as columns
    kappa_pivot = df.pivot(index='subject_id', columns='config', values='kappa')
    f1_pivot = df.pivot(index='subject_id', columns='config', values='f1')
    
    # Compute deltas (4c - 5c)
    delta_kappa = kappa_pivot['4c'] - kappa_pivot['5c']
    delta_f1 = f1_pivot['4c'] - f1_pivot['5c']
    
    # Create result dataframe
    df_deltas = pd.DataFrame({
        'subject_id': delta_kappa.index,
        'delta_kappa': delta_kappa.values,
        'delta_f1': delta_f1.values
    })
    
    return df_deltas

def wilcoxon_ci_median(vec: np.ndarray, n_boot: int = 10000) -> Tuple[float, Tuple[float, float], float, float]:
    """
    Compute median, bootstrap 95% CI, Wilcoxon p-value, and rank-biserial effect size.
    
    Args:
        vec: Array of paired differences
        n_boot: Number of bootstrap samples
        
    Returns:
        Tuple of (median, (ci_low, ci_high), p_value, effect_size)
    """
    median_val = np.median(vec)
    
    # Bootstrap 95% CI for median
    np.random.seed(42)  # For reproducibility
    bootstrap_medians = []
    for _ in range(n_boot):
        boot_sample = np.random.choice(vec, size=len(vec), replace=True)
        bootstrap_medians.append(np.median(boot_sample))
    
    ci_low, ci_high = np.percentile(bootstrap_medians, [2.5, 97.5])
    
    # Wilcoxon signed-rank test
    stat, p_val = wilcoxon(vec, alternative='two-sided')
    
    # Rank-biserial effect size r
    n = len(vec)
    r = 1 - (2 * stat) / (n * (n + 1))
    
    return median_val, (ci_low, ci_high), p_val, abs(r)

def plot_deltas(df_deltas: pd.DataFrame, stats_kappa: Dict[str, Any], stats_f1: Dict[str, Any], outdir: str) -> None:
    """
    Create the paired differences visualization.
    
    Args:
        df_deltas: DataFrame with delta values
        stats_kappa: Statistics dict for kappa deltas
        stats_f1: Statistics dict for F1 deltas
        outdir: Output directory
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    
    # Color scheme (colorblind-safe)
    pos_color = '#2ca25f'  # Green for positive
    neg_color = '#de2d26'  # Red for negative
    
    n_subjects = len(df_deltas)
    
    # Left panel: Delta Kappa
    delta_kappa_sorted = df_deltas.sort_values('delta_kappa')
    colors_kappa = [pos_color if x >= 0 else neg_color for x in delta_kappa_sorted['delta_kappa']]
    
    bars1 = ax1.bar(range(n_subjects), delta_kappa_sorted['delta_kappa'], 
                   color=colors_kappa, alpha=0.8, width=0.8)
    
    # Add zero line
    ax1.axhline(0, color='darkgray', linewidth=0.8, alpha=0.7)
    
    # Add median line and CI band
    median_k, (ci_low_k, ci_high_k), p_k, r_k = stats_kappa['values']
    ax1.axhline(median_k, color='black', linestyle='--', alpha=0.7, linewidth=1.2)
    ax1.axhspan(ci_low_k, ci_high_k, alpha=0.15, color='gray')
    
    # Annotations for top/bottom bars
    neg_indices = np.where(delta_kappa_sorted['delta_kappa'] < 0)[0]
    pos_indices = np.where(delta_kappa_sorted['delta_kappa'] >= 0)[0]
    
    # Add arrows for negative (bottom) bars
    for idx in neg_indices:
        ax1.annotate('▼', (idx, delta_kappa_sorted['delta_kappa'].iloc[idx] - 0.005), 
                    ha='center', va='top', color=neg_color, fontsize=8)
    
    # Add arrows for top 3 positive bars
    if len(pos_indices) > 0:
        top_3_indices = pos_indices[-3:] if len(pos_indices) >= 3 else pos_indices
        for idx in top_3_indices:
            ax1.annotate('▲', (idx, delta_kappa_sorted['delta_kappa'].iloc[idx] + 0.005), 
                        ha='center', va='bottom', color=pos_color, fontsize=8)
    
    # Styling
    ax1.grid(True, axis='y', alpha=0.15)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlabel('Subjects (sorted by Δ)')
    ax1.set_ylabel('Δκ (4c - 5c)')
    ax1.set_xticks([])
    
    # Symmetric y-limits
    y_max_k = max(abs(delta_kappa_sorted['delta_kappa'].min()), abs(delta_kappa_sorted['delta_kappa'].max()))
    y_lim_k = np.ceil(y_max_k * 100) / 100  # Round up to nearest 0.01
    ax1.set_ylim(-y_lim_k, y_lim_k)
    
    # Title with stats
    p_str_k = f"{p_k:.3f}" if p_k >= 0.001 else f"{p_k:.2e}"
    title_k = f"Δκ (4c−5c): median = {median_k:+.3f}, 95% CI [{ci_low_k:+.3f}, {ci_high_k:+.3f}]\nWilcoxon p = {p_str_k}, r = {r_k:.2f} (N={n_subjects})"
    ax1.set_title(title_k, pad=15)
    
    # Right panel: Delta F1
    delta_f1_sorted = df_deltas.sort_values('delta_f1')
    colors_f1 = [pos_color if x >= 0 else neg_color for x in delta_f1_sorted['delta_f1']]
    
    bars2 = ax2.bar(range(n_subjects), delta_f1_sorted['delta_f1'], 
                   color=colors_f1, alpha=0.8, width=0.8)
    
    # Add zero line
    ax2.axhline(0, color='darkgray', linewidth=0.8, alpha=0.7)
    
    # Add median line and CI band
    median_f1, (ci_low_f1, ci_high_f1), p_f1, r_f1 = stats_f1['values']
    ax2.axhline(median_f1, color='black', linestyle='--', alpha=0.7, linewidth=1.2)
    ax2.axhspan(ci_low_f1, ci_high_f1, alpha=0.15, color='gray')
    
    # Annotations for top/bottom bars
    neg_indices_f1 = np.where(delta_f1_sorted['delta_f1'] < 0)[0]
    pos_indices_f1 = np.where(delta_f1_sorted['delta_f1'] >= 0)[0]
    
    # Add arrows for negative bars
    for idx in neg_indices_f1:
        ax2.annotate('▼', (idx, delta_f1_sorted['delta_f1'].iloc[idx] - 0.005), 
                    ha='center', va='top', color=neg_color, fontsize=8)
    
    # Add arrows for top 3 positive bars
    if len(pos_indices_f1) > 0:
        top_3_indices_f1 = pos_indices_f1[-3:] if len(pos_indices_f1) >= 3 else pos_indices_f1
        for idx in top_3_indices_f1:
            ax2.annotate('▲', (idx, delta_f1_sorted['delta_f1'].iloc[idx] + 0.005), 
                        ha='center', va='bottom', color=pos_color, fontsize=8)
    
    # Styling
    ax2.grid(True, axis='y', alpha=0.15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlabel('Subjects (sorted by Δ)')
    ax2.set_ylabel('ΔF1 (4c - 5c)')
    ax2.set_xticks([])
    
    # Symmetric y-limits
    y_max_f1 = max(abs(delta_f1_sorted['delta_f1'].min()), abs(delta_f1_sorted['delta_f1'].max()))
    y_lim_f1 = np.ceil(y_max_f1 * 100) / 100  # Round up to nearest 0.01
    ax2.set_ylim(-y_lim_f1, y_lim_f1)
    
    # Title with stats
    p_str_f1 = f"{p_f1:.3f}" if p_f1 >= 0.001 else f"{p_f1:.2e}"
    title_f1 = f"ΔF1 (4c−5c): median = {median_f1:+.3f}, 95% CI [{ci_low_f1:+.3f}, {ci_high_f1:+.3f}]\nWilcoxon p = {p_str_f1}, r = {r_f1:.2f} (N={n_subjects})"
    ax2.set_title(title_f1, pad=15)
    
    # Global title
    fig.suptitle('Task Granularity Impact (paired): 4-class merges Light (N1+N2) vs 5-class keeps N1, N2 separate', 
                y=0.95, fontsize=14, fontweight='bold')
    
    # Footnote
    fig.text(0.5, 0.02, 'Bars: per-subject paired differences (4c−5c). Lines/band: median and bootstrap 95% CI.',
             ha='center', va='bottom', fontsize=9, style='italic')
    
    # Save figures
    import os
    os.makedirs(outdir, exist_ok=True)
    
    plt.savefig(f"{outdir}/fig_task_granularity_delta.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{outdir}/fig_task_granularity_delta.pdf", bbox_inches='tight')
    
    print(f"Figures saved to {outdir}/fig_task_granularity_delta.[png|pdf]")
    plt.show()

def main():
    """Main function to run the task granularity analysis."""
    parser = argparse.ArgumentParser(description='Task Granularity Analysis: 4-class vs 5-class paired comparison')
    parser.add_argument('--csv', required=True, help='Path to CSV file with results')
    parser.add_argument('--out', required=True, help='Output directory for figures')
    
    args = parser.parse_args()
    
    print("=== Task Granularity Analysis: Paired 4c vs 5c ===")
    
    # Load and pair data
    print(f"Loading data from {args.csv}...")
    df_paired = load_and_pair(args.csv)
    
    # Compute deltas
    print("Computing paired differences (Δ = 4c - 5c)...")
    df_deltas = compute_deltas(df_paired)
    
    # Statistical analysis
    print("Performing statistical analysis...")
    stats_kappa = {
        'values': wilcoxon_ci_median(df_deltas['delta_kappa'].values)
    }
    stats_f1 = {
        'values': wilcoxon_ci_median(df_deltas['delta_f1'].values)
    }
    
    # Print results
    median_k, (ci_low_k, ci_high_k), p_k, r_k = stats_kappa['values']
    median_f1, (ci_low_f1, ci_high_f1), p_f1, r_f1 = stats_f1['values']
    
    print(f"\n--- Results (N = {len(df_deltas)} paired subjects) ---")
    print(f"Δκ (4c-5c): median = {median_k:+.3f}, 95% CI [{ci_low_k:+.3f}, {ci_high_k:+.3f}]")
    print(f"           Wilcoxon p = {p_k:.4f}, effect size r = {r_k:.3f}")
    print(f"ΔF1 (4c-5c): median = {median_f1:+.3f}, 95% CI [{ci_low_f1:+.3f}, {ci_high_f1:+.3f}]")
    print(f"            Wilcoxon p = {p_f1:.4f}, effect size r = {r_f1:.3f}")
    
    # Create visualization
    print(f"\nGenerating visualization...")
    plot_deltas(df_deltas, stats_kappa, stats_f1, args.out)
    
    print("Analysis complete!")

if __name__ == '__main__':
    main()