#!/usr/bin/env python3
"""
Combined Task Granularity Analysis: 5-class vs 4-class (v0/v1) Sleep Staging
Compares performance across different label granularities: 5-class, 4-class-v0, and 4-class-v1.
Merges distribution and paired delta analyses into a single publication-ready figure.

Usage examples:
    # Individual subjects (standard approach)
    python Plot_Clean/plot_task_granularity_combined.py --csv ../data/all_runs_flat.csv --out Plot_Clean/figures --paired-only
    
    # Paired subject groups (increases N for statistical power)
    python Plot_Clean/plot_task_granularity_combined.py --csv ../data/all_runs_flat.csv --out Plot_Clean/figures --paired-only --group-size 2 --pairing-strategy paired_groups
    
    # Cross-validation style analysis (all combinations)
    python Plot_Clean/plot_task_granularity_combined.py --csv ../data/all_runs_flat.csv --out Plot_Clean/figures --paired-only --group-size 2 --pairing-strategy cross_validation
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from pathlib import Path
from typing import Tuple, Dict, Any, List
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Import consistent figure styling
import sys; sys.path.append("../style"); from figure_style import (
    setup_figure_style, get_color, save_figure, 
    add_yasa_baseline, add_significance_marker,
    bootstrap_ci_median, wilcoxon_test,
    format_n_caption, add_sample_size_annotation
)

# Removed - using setup_figure_style() import sys; sys.path.append("../style"); from figure_style.py instead

# Color scheme (colorblind-safe) - updated to use figure_style
COLORS = {
    '5-class': get_color('5_class'),
    '4-class-v0': "#323131",
    '4-class-v1': get_color('cbramod'),  # Use cbramod color for v1
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
    
    print(f"Loaded CSV with {len(df)} rows")
    
    # CRITICAL: Filter out high noise experiments to avoid bias
    if 'noise_level' in df.columns:
        noise_stats = df['noise_level'].value_counts().sort_index()
        print(f"🔊 Noise level distribution: {dict(noise_stats)}")
        
        # Keep only clean data (noise_level <= 0.01 or 1%) 
        df = df[df['noise_level'] <= 0.01].copy()
        print(f"✅ Filtered to clean data: {len(df)} rows remaining (noise ≤ 1%)")
        
        if len(df) == 0:
            raise ValueError("No clean data found after noise filtering.")
    
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
    
    # Map config names: differentiate between 4c-v0 and 4c-v1
    config_mapping = {
        '5c': '5-class',
        '4c-v0': '4-class-v0', 
        '4c-v1': '4-class-v1'
    }
    
    df_work['task_config'] = df_work['config'].map(config_mapping)
    df_work = df_work.dropna(subset=['task_config'])
    
    # Remove rows with missing metrics or subject IDs
    df_clean = df_work.dropna(subset=['test_kappa', 'test_f1', 'subject_id']).copy()
    
    if df_clean.empty:
        raise ValueError("No valid data found after cleaning")
    
    return df_clean[['subject_id', 'task_config', 'test_kappa', 'test_f1']]

def create_subject_groupings(subjects: List[str], group_size: int = 1, pairing_strategy: str = 'individual') -> List[List[str]]:
    """
    Create subject groupings for paired statistical analysis to increase comparison power.
    
    Args:
        subjects: List of subject IDs
        group_size: Size of each subject group (1 for individual subjects)
        pairing_strategy: 'individual', 'paired_groups', or 'cross_validation'
        
    Returns:
        List of subject groups for analysis
    """
    print(f"Creating subject groupings with strategy: {pairing_strategy}, group_size: {group_size}")
    
    if pairing_strategy == 'individual':
        # Each subject is its own group - standard approach
        groupings = [[subject] for subject in subjects]
        
    elif pairing_strategy == 'paired_groups':
        # Create fixed subject pairs/groups for valid paired testing
        if group_size == 1:
            groupings = [[subject] for subject in subjects]
        else:
            groupings = []
            for i in range(0, len(subjects), group_size):
                group = subjects[i:i+group_size]
                if len(group) == group_size:  # Only include complete groups
                    groupings.append(group)
                    
    elif pairing_strategy == 'cross_validation':
        # Create overlapping groups for cross-validation style analysis
        groupings = []
        if group_size == 1:
            groupings = [[subject] for subject in subjects]
        else:
            # Create all possible combinations of group_size
            groupings = [list(combo) for combo in combinations(subjects, group_size)]
            # Limit to reasonable number for computational efficiency
            if len(groupings) > 50:
                print(f"Too many combinations ({len(groupings)}), sampling 50 random groups")
                import random
                random.seed(42)
                groupings = random.sample(groupings, 50)
                
    else:
        raise ValueError(f"Unknown pairing strategy: {pairing_strategy}")
    
    print(f"Created {len(groupings)} subject groups")
    if group_size > 1 and len(groupings) <= 10:
        print(f"Example groups: {groupings[:3]}...")
        
    return groupings

def filter_paired(df: pd.DataFrame, group_size: int = 1, pairing_strategy: str = 'individual') -> pd.DataFrame:
    """
    Filter to subjects/groups who have data in multiple configurations with flexible grouping.
    Handle duplicates by taking the best performance for each subject-config pair.
    
    Args:
        df: DataFrame with all data
        group_size: Size of subject groups (1 for individual subjects)
        pairing_strategy: Strategy for creating subject groups
        
    Returns:
        DataFrame with subjects/groups present in multiple configurations
    """
    print(f"\nApplying paired filtering with grouping strategy: {pairing_strategy}")
    
    # Store original run counts before groupby
    run_counts = df.groupby(['subject_id', 'task_config']).size().reset_index(name='run_count')
    
    # Handle duplicates by taking the best performance for each subject-config pair  
    df_best = df.groupby(['subject_id', 'task_config']).agg({
        'test_kappa': 'max',
        'test_f1': 'max'
    }).reset_index()
    
    # Merge run counts back
    df_best = df_best.merge(run_counts, on=['subject_id', 'task_config'])
    
    # Get all unique subjects
    all_subjects = sorted(df_best['subject_id'].unique())
    available_configs = sorted(df_best['task_config'].unique())
    
    print(f"Total subjects available: {len(all_subjects)}")
    print(f"Available configurations: {available_configs}")
    
    # Find subjects present in each config
    config_subjects = {}
    for config in available_configs:
        config_subjects[config] = set(df_best[df_best['task_config'] == config]['subject_id'])
        print(f"  {config}: {len(config_subjects[config])} subjects")
    
    # Find subjects present in at least two configurations (for individual strategy)
    if pairing_strategy == 'individual':
        paired_subjects = set()
        for subject in all_subjects:
            present_configs = [config for config, subjects in config_subjects.items() if subject in subjects]
            if len(present_configs) >= 2:
                paired_subjects.add(subject)
        
        if not paired_subjects:
            print("Warning: No subjects found in multiple configurations")
            return df_best
        
        # Return only paired subjects
        df_paired = df_best[df_best['subject_id'].isin(paired_subjects)].copy()
        print(f"Found {len(paired_subjects)} subjects with multiple configs")
        
    else:
        # For group-based strategies, we need subjects that appear in multiple configs
        # Find the intersection of subjects across all configs
        subjects_in_all_configs = set.intersection(*config_subjects.values())
        
        if len(subjects_in_all_configs) < group_size:
            print(f"Warning: Only {len(subjects_in_all_configs)} subjects in all configs, but group_size={group_size}")
            # Fall back to subjects in at least 2 configs
            subjects_in_multiple = set()
            for subject in all_subjects:
                present_configs = [config for config, subjects in config_subjects.items() if subject in subjects]
                if len(present_configs) >= 2:
                    subjects_in_multiple.add(subject)
            subjects_for_grouping = list(subjects_in_multiple)
        else:
            subjects_for_grouping = list(subjects_in_all_configs)
        
        print(f"Subjects available for grouping: {len(subjects_for_grouping)}")
        
        if len(subjects_for_grouping) < group_size:
            print("Warning: Not enough subjects for requested group size, falling back to individual")
            df_paired = df_best[df_best['subject_id'].isin(subjects_for_grouping)].copy()
        else:
            # Create subject groupings
            subject_groupings = create_subject_groupings(subjects_for_grouping, group_size, pairing_strategy)
            
            # Expand the dataframe to include group information
            expanded_rows = []
            
            for group_idx, subject_group in enumerate(subject_groupings):
                group_id = f"group_{group_idx:03d}_{'_'.join(sorted(subject_group))}"
                
                # For each configuration, aggregate the group's performance
                for config in available_configs:
                    # Get data for all subjects in this group for this config
                    group_config_data = df_best[
                        (df_best['subject_id'].isin(subject_group)) & 
                        (df_best['task_config'] == config)
                    ]
                    
                    if len(group_config_data) == len(subject_group):  # All subjects have this config
                        # Aggregate performance (mean of the group)
                        agg_kappa = group_config_data['test_kappa'].mean()
                        agg_f1 = group_config_data['test_f1'].mean()
                        total_runs = group_config_data['run_count'].sum()
                        
                        expanded_rows.append({
                            'subject_id': group_id,
                            'task_config': config,
                            'test_kappa': agg_kappa,
                            'test_f1': agg_f1,
                            'run_count': total_runs,
                            'group_size': len(subject_group),
                            'group_composition': ','.join(sorted(subject_group)),
                            'pairing_strategy': pairing_strategy
                        })
            
            if expanded_rows:
                df_paired = pd.DataFrame(expanded_rows)
                print(f"Created {len(subject_groupings)} groups with {len(expanded_rows)} total group-config combinations")
            else:
                print("Warning: No valid groups created, falling back to individual subjects")
                df_paired = df_best
    
    # Final statistics
    if 'group_size' not in df_paired.columns:
        df_paired['group_size'] = 1
        df_paired['group_composition'] = df_paired['subject_id']
        df_paired['pairing_strategy'] = 'individual'
    
    print(f"Final data shape: {df_paired.shape}")
    for config in available_configs:
        n_groups = len(df_paired[df_paired['task_config'] == config])
        print(f"  {config}: {n_groups} groups/subjects")
    
    return df_paired

# Removed - using bootstrap_ci_median import sys; sys.path.append("../style"); from figure_style.py instead

def format_float(x: float) -> str:
    """Format float to 3 decimals, stripping trailing zeros."""
    return f"{x:.3f}".rstrip('0').rstrip('.')

def plot_distributions(ax, df: pd.DataFrame, metric: str, palette: Dict[str, str]) -> None:
    """
    Plot side-by-side boxplots with jittered points for distribution comparison.
    
    Args:
        ax: Matplotlib axis
        df: DataFrame with subjects
        metric: Column name for the metric ('test_kappa' or 'test_f1')
        palette: Color palette dictionary
    """
    # Get all available configs and sort them
    available_configs = sorted(df['task_config'].unique())
    positions = list(range(len(available_configs)))
    
    # Prepare data and run counts
    data_by_config = []
    run_counts_by_config = []
    colors_box = []
    for config in available_configs:
        config_df = df[df['task_config'] == config]
        config_data = config_df[metric].dropna().values
        
        # Sum up run counts for this config (if run_count column exists)
        if 'run_count' in config_df.columns:
            total_runs = config_df['run_count'].sum()
            run_counts_by_config.append(total_runs)
        else:
            run_counts_by_config.append(len(config_data))  # Fallback to subject count
            
        data_by_config.append(config_data)
        colors_box.append(palette.get(config, palette.get('cbramod', '#1f77b4')))  # Fallback color
    
    # Create boxplots
    bp = ax.boxplot(data_by_config, positions=positions, patch_artist=True,
                    widths=0.5, showfliers=True, whis=1.5)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)
    
    # Add jittered scatter points
    np.random.seed(42)  # For consistent jitter
    for i, (config, data) in enumerate(zip(available_configs, data_by_config)):
        if len(data) > 0:
            x_jitter = np.random.normal(i, 0.04, size=len(data))
            ax.scatter(x_jitter, data, alpha=0.6, s=20, color=colors_box[i], 
                      edgecolors='white', linewidth=0.5, zorder=3)
    
    # Add median labels in small rounded boxes
    for i, (config, data) in enumerate(zip(available_configs, data_by_config)):
        if len(data) > 0:
            median_val = np.median(data)
            ax.text(i, ax.get_ylim()[1] * 0.92, format_float(median_val), 
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                            edgecolor='gray', linewidth=0.5))
    
    # Add sample size annotations (show both subjects and runs)
    for i, (data, run_count) in enumerate(zip(data_by_config, run_counts_by_config)):
        if run_count != len(data):  # Show both if different
            ax.text(i, ax.get_ylim()[1] * 0.94, f'N = {len(data)} ({run_count} runs)', 
                   ha='center', va='top', fontsize=9, fontweight='bold')
        else:  # Show just the count if same
            ax.text(i, ax.get_ylim()[1] * 0.94, f'N = {len(data)}', 
                   ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Styling
    ax.set_xticks(positions)
    ax.set_xticklabels(available_configs, rotation=45, ha='right')
    
    if metric == 'test_kappa':
        ax.set_ylabel("Test Kappa")
    else:
        ax.set_ylabel("Test F1")
    
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_deltas_absolute(ax, df: pd.DataFrame, metric: str, palette: Dict[str, str]) -> None:
    """
    Plot absolute values for all available configurations as side-by-side boxplots.
    
    Args:
        ax: Matplotlib axis
        df: DataFrame with subjects
        metric: Column name for the metric ('test_kappa' or 'test_f1')
        palette: Color palette dictionary
    """
    # Get all available configs and sort them
    available_configs = sorted(df['task_config'].unique())
    positions = list(range(len(available_configs)))
    
    # Prepare data and run counts
    data_by_config = []
    run_counts_by_config = []
    colors_box = []
    for config in available_configs:
        config_df = df[df['task_config'] == config]
        config_data = config_df[metric].dropna().values
        
        # Sum up run counts for this config (if run_count column exists)
        if 'run_count' in config_df.columns:
            total_runs = config_df['run_count'].sum()
            run_counts_by_config.append(total_runs)
        else:
            run_counts_by_config.append(len(config_data))  # Fallback to subject count
            
        data_by_config.append(config_data)
        colors_box.append(palette.get(config, palette.get('cbramod', '#1f77b4')))  # Fallback color
    
    # Create boxplots
    bp = ax.boxplot(data_by_config, positions=positions, patch_artist=True,
                    widths=0.5, showfliers=True, whis=1.5)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)
    
    # Add jittered scatter points
    np.random.seed(42)  # For consistent jitter
    for i, (config, data) in enumerate(zip(available_configs, data_by_config)):
        if len(data) > 0:
            x_jitter = np.random.normal(i, 0.04, size=len(data))
            ax.scatter(x_jitter, data, alpha=0.6, s=20, color=colors_box[i], 
                      edgecolors='white', linewidth=0.5, zorder=3)
    
    # Get y-axis limits for proper positioning
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    
    # Add sample size annotations below boxplots (show both subjects and runs)
    for i, (data, run_count) in enumerate(zip(data_by_config, run_counts_by_config)):
        if run_count != len(data):  # Show both if different
            ax.text(i, y_min + 0.02 * y_range, f'N = {len(data)} ({run_count} runs)', 
                   ha='center', va='center', fontsize=9, fontweight='bold')
        else:  # Show just the count if same
            ax.text(i, y_min + 0.02 * y_range, f'N = {len(data)}', 
                   ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add median labels in small rounded boxes below N values
    for i, (config, data) in enumerate(zip(available_configs, data_by_config)):
        if len(data) > 0:
            median_val = np.median(data)
            ax.text(i, y_min + 0.12 * y_range, format_float(median_val), 
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                            edgecolor='gray', linewidth=0.5))
    
    # Set improved y-axis range for better visibility
    all_data = np.concatenate(data_by_config)
    y_min = max(0.3, np.min(all_data) - 0.02)
    y_max = min(0.9, np.max(all_data) + 0.02)
    ax.set_ylim(y_min, y_max)
    
    # Styling
    ax.set_xticks(positions)
    ax.set_xticklabels(available_configs, rotation=45, ha='right')
    
    if metric == 'test_kappa':
        ax.set_ylabel("Test Cohen's κ", fontweight='bold')
        ax.set_title("Sleep Staging Performance Comparison", fontweight='bold', fontsize=12, pad=15)
        # Add YASA baseline
        ax.axhline(y=0.446, color=get_color('yasa'), linestyle='--', linewidth=2, alpha=0.8, 
                   label=f'YASA baseline (κ=0.446)')
    else:
        ax.set_ylabel("Test F1 Score", fontweight='bold')
        ax.set_title("Sleep Staging F1 Score Comparison", fontweight='bold', fontsize=12, pad=15)
    
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
    
    # # Add median line and CI band
    # ax.axhline(median_val, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
    # ax.axhspan(ci_low, ci_high, alpha=0.2, color='gray')
    
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
    parser = argparse.ArgumentParser(description='Combined Task Granularity Analysis with Advanced Subject Grouping')
    parser.add_argument('--csv', required=True, help='Path to CSV file with results')
    parser.add_argument('--out', required=True, help='Output directory for figures')
    parser.add_argument('--paired-only', action='store_true', 
                       help='Filter to subjects present in both configurations')
    parser.add_argument('--group-size', type=int, default=1,
                       help='Size of subject groups for analysis (default: 1 = individual)')
    parser.add_argument('--pairing-strategy', choices=['individual', 'paired_groups', 'cross_validation'], 
                       default='individual', help='Strategy for creating subject groups to increase comparisons')
    
    args = parser.parse_args()
    
    print("=== Combined Task Granularity Analysis with Advanced Subject Grouping ===")
    print(f"Group size: {args.group_size}")
    print(f"Pairing strategy: {args.pairing_strategy}")
    
    # Load data
    print(f"Loading data from {args.csv}...")
    df = load_data(args.csv)
    print(f"Loaded {len(df)} rows after cleaning")
    
    # Apply paired filtering with grouping strategy if requested
    if args.paired_only:
        print("\nApplying advanced paired filtering with subject grouping...")
        print(f"Group size: {args.group_size}, Strategy: {args.pairing_strategy}")
        df = filter_paired(df, args.group_size, args.pairing_strategy)
        print(f"After paired filtering: {len(df)} rows")
        
        # Show grouping statistics
        if 'group_size' in df.columns:
            group_stats = df.groupby(['pairing_strategy', 'group_size']).size().reset_index(name='count')
            print("\nGrouping statistics:")
            for _, row in group_stats.iterrows():
                print(f"  {row['pairing_strategy']} (size {row['group_size']}): {row['count']} group-config pairs")
            
            unique_groups = df['subject_id'].nunique() if args.group_size == 1 else len(df['group_composition'].unique())
            print(f"  Total unique groups: {unique_groups}")
    
    if len(df) == 0:
        print("ERROR: No data remaining after filtering.")
        return
    
    # Set up consistent styling and create figure with 1x2 grid (no histograms)
    setup_figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    
    # Only delta plots (no distribution histograms)
    if args.paired_only or len(set(df['subject_id'])) < len(df):
        plot_deltas_absolute(axes[0], df, 'test_kappa', COLORS)
        plot_deltas_absolute(axes[1], df, 'test_f1', COLORS)
    else:
        # If not paired-only, still try to create deltas if possible with default grouping
        df_paired_auto = filter_paired(df, args.group_size, args.pairing_strategy)
        if len(df_paired_auto) > 0:
            plot_deltas_absolute(axes[0], df_paired_auto, 'test_kappa', COLORS)
            plot_deltas_absolute(axes[1], df_paired_auto, 'test_f1', COLORS)
    
    # Global title with better formatting
    available_configs = sorted(df['task_config'].unique())
    title_text = f"CBraMod Performance Across Sleep Staging Granularities"
    fig.suptitle(title_text, fontsize=16, fontweight='bold')
    
    # Add legend at the bottom - only YASA baseline (config names already in figure)
    handles = []
    labels = []
    
    # Add YASA baseline if kappa is plotted
    if any('test_kappa' in str(ax.get_ylabel()) for ax in axes):
        handles.append(plt.Line2D([0], [0], color=get_color('yasa'), linestyle='--', linewidth=2))
        labels.append('YASA baseline')
    
    # Only show legend if we have items to show
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=len(handles), 
                  bbox_to_anchor=(0.5, 0.02), fontsize=10, frameon=True, 
                  fancybox=True, shadow=True)
    
    
    # Save figures
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    png_path = output_dir / 'fig_task_granularity_combined.png'
    pdf_path = output_dir / 'fig_task_granularity_combined.pdf'
    
    # Use consistent save function
    base_path = output_dir / 'fig_task_granularity_combined'
    saved_files = save_figure(fig, base_path)
    
    # Print N information for caption with grouping details
    if 'group_composition' in df.columns and args.group_size > 1:
        n_groups = len(set(df['subject_id']))
        unique_subjects = set()
        for comp in df['group_composition'].unique():
            if pd.notna(comp) and comp:
                unique_subjects.update(comp.split(','))
        n_unique_subjects = len(unique_subjects)
        n_runs = len(df)
        print(f"\n📋 Caption info: {n_groups} groups ({n_unique_subjects} unique subjects, {n_runs} group-config pairs)")
        print(f"    Grouping: {args.pairing_strategy} with group size {args.group_size}")
    else:
        n_subjects = len(set(df['subject_id']))
        n_runs = len(df)
        print(f"\n📋 Caption info: {format_n_caption(n_subjects, n_runs, 'subjects')}")
    
    # Print summary statistics with grouping information
    print(f"\n=== Summary Statistics ===")
    for config in sorted(df['task_config'].unique()):
        config_data = df[df['task_config'] == config]
        if len(config_data) > 0:
            kappa_mean = config_data['test_kappa'].mean()
            kappa_std = config_data['test_kappa'].std()
            f1_mean = config_data['test_f1'].mean()
            f1_std = config_data['test_f1'].std()
            
            if 'group_size' in config_data.columns and args.group_size > 1:
                avg_group_size = config_data['group_size'].mean()
                print(f"{config}: N={len(config_data)} groups (avg size {avg_group_size:.1f}), κ={kappa_mean:.3f}±{kappa_std:.3f}, F1={f1_mean:.3f}±{f1_std:.3f}")
            else:
                print(f"{config}: N={len(config_data)} subjects, κ={kappa_mean:.3f}±{kappa_std:.3f}, F1={f1_mean:.3f}±{f1_std:.3f}")
    
    # Additional statistics for grouping strategies
    if args.group_size > 1 and 'group_composition' in df.columns:
        print(f"\n=== Grouping Analysis ===")
        print(f"Strategy: {args.pairing_strategy}")
        print(f"Group size: {args.group_size}")
        
        # Count total unique subjects involved
        all_subjects = set()
        for comp in df['group_composition'].unique():
            if pd.notna(comp) and comp:
                all_subjects.update(comp.split(','))
        print(f"Unique subjects in analysis: {len(all_subjects)}")
        print(f"Total group-config combinations: {len(df)}")
        
        # Show some example groups if reasonable number
        example_groups = df['group_composition'].unique()[:5]
        print(f"Example groups: {list(example_groups)}")
    
    plt.show()

if __name__ == '__main__':
    main()