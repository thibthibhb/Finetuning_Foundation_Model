#!/usr/bin/env python3
"""
ICL Overview Figure: Multi-panel comparison of No-ICL, Proto-ICL, and Set-ICL methods.

Creates a 2x3 panel figure showing:
- Row 1: Delta metrics vs K-shot (kappa, balanced accuracy, weighted F1)
- Row 2: Best-K distributions, baseline correlation, and data scale effects

Author: Senior Research Engineer
Usage:
    python fig_icl_overview.py --out Plot_Clean/figures/icl_overview
    python fig_icl_overview.py --csv Plot_Clean/data/icl_results.csv --out figures/
"""

import argparse
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# Color scheme (consistent across all panels)
COLORS = {
    'baseline': '#bbbbbb',
    'Proto-ICL': '#2E8B57',
    'Set-ICL (SetTransformer)': '#6A5ACD',
    'Set-ICL (DeepSets)': '#D2691E'
}

# Method prefix mapping
METHOD_MAPPING = {
    'proto_test': 'Proto-ICL',
    'set_test': 'Set-ICL (SetTransformer)',
    'cnp_test': 'Set-ICL (DeepSets)'
}

def setup_plotting_style():
    """Configure matplotlib and seaborn for publication-ready plots."""
    sns.set_context("talk")
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

def bootstrap_ci_median(data: np.ndarray, n_bootstrap: int = 1000, confidence: float = 0.95, random_state: int = 123) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval for median."""
    if len(data) < 2:
        return np.nan, np.nan
    
    np.random.seed(random_state)
    boot_medians = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_medians.append(np.median(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(boot_medians, 100 * alpha / 2)
    upper = np.percentile(boot_medians, 100 * (1 - alpha / 2))
    return lower, upper

def extract_k_from_key(key: str) -> Optional[int]:
    """Extract K value from summary key (e.g., 'proto_test_kappa_base_K3' -> 3)."""
    match = re.search(r'_K(\d+)$', key)
    return int(match.group(1)) if match else None

def discover_methods_and_ks(summary_keys: List[str]) -> Dict[str, List[int]]:
    """Auto-discover available methods and K values from summary keys."""
    method_ks = {}
    
    for key in summary_keys:
        for prefix, method_name in METHOD_MAPPING.items():
            if key.startswith(prefix):
                k = extract_k_from_key(key)
                if k is not None:
                    if method_name not in method_ks:
                        method_ks[method_name] = set()
                    method_ks[method_name].add(k)
    
    # Convert sets to sorted lists
    return {method: sorted(list(ks)) for method, ks in method_ks.items()}

def load_data_from_wandb(entity: str, project: str, ks_filter: Optional[List[int]] = None) -> pd.DataFrame:
    """Load ICL data from Weights & Biases with proper paired delta computation."""
    try:
        import wandb
    except ImportError:
        raise ImportError("wandb package required for W&B mode. Install with: pip install wandb")
    
    # Initialize API
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    
    print(f"🔍 Fetching ICL runs from {entity}/{project}...")
    
    data_rows = []
    skipped_runs = 0
    
    for run in runs:
        if run.state != "finished":
            continue
            
        summary = run.summary._json_dict
        config = run.config
        
        # Get baseline metrics
        baseline_kappa = summary.get('test_kappa')
        hours_of_data = summary.get('hours_of_data') or config.get('hours_of_data')
        num_subjects = summary.get('num_subjects_train') or config.get('num_subjects_train')
        
        # Debug: Print available ICL keys for first few runs
        if len(data_rows) < 3:
            icl_keys = [k for k in summary.keys() if any(prefix in k for prefix in ['proto_test', 'set_test', 'cnp_test'])]
            if icl_keys:
                print(f"🔍 Debug - Run {run.name} has {len(icl_keys)} ICL keys")
                print(f"  Sample keys: {sorted(icl_keys)[:10]}")
        
        # Discover available methods and Ks for this run with validation
        method_ks_valid = validate_and_discover_icl_data(summary, ks_filter)
        
        if not method_ks_valid:
            skipped_runs += 1
            continue
        
        # Extract ICL metrics for each validated method and K
        for method_name, available_ks in method_ks_valid.items():
            prefix = next(p for p, m in METHOD_MAPPING.items() if m == method_name)
            
            for k in available_ks:
                # Construct expected key names (check both icl and method-specific naming)
                base_keys = [
                    f'{prefix}_kappa_base_K{k}',
                    f'{prefix}_accbal_base_K{k}', 
                    f'{prefix}_f1w_base_K{k}'
                ]
                
                # Try both icl and method-specific naming conventions
                method_suffix = 'proto' if prefix == 'proto_test' else 'icl'
                icl_keys = [
                    f'{prefix}_kappa_{method_suffix}_K{k}',
                    f'{prefix}_accbal_{method_suffix}_K{k}',
                    f'{prefix}_f1w_{method_suffix}_K{k}'
                ]
                
                # Extract values (already validated to exist)
                kappa_base = summary[base_keys[0]]
                kappa_icl = summary[icl_keys[0]]
                accbal_base = summary[base_keys[1]]
                accbal_icl = summary[icl_keys[1]]
                f1w_base = summary[base_keys[2]]
                f1w_icl = summary[icl_keys[2]]
                
                # Compute true paired deltas
                delta_kappa = kappa_icl - kappa_base
                delta_accbal = accbal_icl - accbal_base
                delta_f1w = f1w_icl - f1w_base
                
                data_rows.append({
                    'run_name': run.name,
                    'method': method_name,
                    'K': k,
                    'kappa_base': kappa_base,
                    'kappa_icl': kappa_icl,
                    'delta_kappa': delta_kappa,
                    'accbal_base': accbal_base,
                    'accbal_icl': accbal_icl,
                    'delta_accbal': delta_accbal,
                    'f1w_base': f1w_base,
                    'f1w_icl': f1w_icl,
                    'delta_f1w': delta_f1w,
                    'test_kappa_baseline': baseline_kappa,
                    'hours_of_data': hours_of_data,
                    'num_subjects_train': num_subjects
                })
    
    if not data_rows:
        raise ValueError(f"No valid ICL data found in W&B runs (skipped {skipped_runs} runs without proper ICL keys)")
    
    df = pd.DataFrame(data_rows)
    print(f"✅ Loaded {len(df)} valid ICL measurements from {len(df['run_name'].unique())} runs")
    print(f"📊 Skipped {skipped_runs} runs without complete ICL data")
    
    # Log best-K selection contributors
    best_k_contributors = {}
    for method in df['method'].unique():
        best_k = find_best_k_for_method(df, method)
        contributors = df[(df['method'] == method) & (df['K'] == best_k)]['run_name'].unique()
        best_k_contributors[method] = {'best_k': best_k, 'runs': list(contributors)}
        print(f"🎯 {method} best-K={best_k} based on {len(contributors)} runs")
    
    return df

def validate_and_discover_icl_data(summary: dict, ks_filter: Optional[List[int]] = None) -> Dict[str, List[int]]:
    """Validate and discover ICL method-K combinations with complete data."""
    method_ks_valid = {}
    
    for prefix, method_name in METHOD_MAPPING.items():
        valid_ks = []
        
        # Find all potential K values for this method
        potential_ks = set()
        for key in summary.keys():
            if key.startswith(prefix) and '_K' in key:
                k = extract_k_from_key(key)
                if k is not None:
                    if ks_filter is None or k in ks_filter:
                        potential_ks.add(k)
        
        # Validate each K has complete base and ICL measurements
        for k in potential_ks:
            # Handle different naming conventions
            method_suffix = 'proto' if prefix == 'proto_test' else 'icl'
            required_keys = [
                f'{prefix}_kappa_base_K{k}',
                f'{prefix}_kappa_{method_suffix}_K{k}',
                f'{prefix}_accbal_base_K{k}',
                f'{prefix}_accbal_{method_suffix}_K{k}',
                f'{prefix}_f1w_base_K{k}',
                f'{prefix}_f1w_{method_suffix}_K{k}'
            ]
            
            # Check all required keys exist and have valid values
            if all(key in summary and summary[key] is not None for key in required_keys):
                # Additional validation: ensure no NaN values
                values = [summary[key] for key in required_keys]
                if all(isinstance(v, (int, float)) and not np.isnan(v) for v in values):
                    valid_ks.append(k)
        
        if valid_ks:
            method_ks_valid[method_name] = sorted(valid_ks)
    
    return method_ks_valid

def find_best_k_for_method(df: pd.DataFrame, method: str) -> int:
    """Find best K for a method based on highest median delta_kappa with tie-breaking."""
    method_data = df[df['method'] == method]
    if len(method_data) == 0:
        return 0
    
    k_medians = method_data.groupby('K')['delta_kappa'].median()
    if len(k_medians) == 0:
        return 0
    
    # Find K with highest median (tie-break by larger K)
    max_median = k_medians.max()
    best_ks = k_medians[k_medians == max_median].index.tolist()
    return max(best_ks)

def load_data_from_csv(csv_path: Path) -> pd.DataFrame:
    """Load ICL data from all_runs_flat.csv file and transform to ICL format."""
    print(f"📄 Loading data from {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Check if this is the standard all_runs_flat.csv format
    if 'contract.icl.icl_mode' in df.columns:
        return transform_standard_csv_to_icl_format(df)
    
    # Otherwise, validate required columns for direct ICL format
    required_cols = [
        'run_name', 'method', 'K', 'kappa_base', 'kappa_icl', 
        'accbal_base', 'accbal_icl', 'f1w_base', 'f1w_icl', 'delta_kappa'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Compute derived columns
    if 'delta_accbal' not in df.columns:
        df['delta_accbal'] = df['accbal_icl'] - df['accbal_base']
    if 'delta_f1w' not in df.columns:
        df['delta_f1w'] = df['f1w_icl'] - df['f1w_base']
    
    print(f"✅ Loaded {len(df)} ICL measurements from CSV")
    
    return df

def transform_standard_csv_to_icl_format(df: pd.DataFrame) -> pd.DataFrame:
    """Transform all_runs_flat.csv format to ICL analysis format."""
    print("🔄 Transforming standard CSV format to ICL format...")
    
    # Filter for ICL runs (non-none modes)
    icl_df = df[df['contract.icl.icl_mode'] != 'none'].copy()
    
    if len(icl_df) == 0:
        raise ValueError("No ICL runs found in the dataset (all runs have icl_mode='none')")
    
    # Map columns to ICL format
    icl_df['run_name'] = icl_df['name']
    icl_df['method'] = icl_df['contract.icl.icl_mode'].map({
        'proto': 'Proto-ICL',
        'set': 'Set-ICL (SetTransformer)', 
        'cnp': 'Set-ICL (DeepSets)'
    })
    icl_df['K'] = icl_df['contract.icl.k_support']
    
    # Use test metrics as both base and ICL (since we don't have separate measurements)
    # Note: This is a simplified analysis - ideally we'd have separate baseline vs ICL measurements
    icl_df['kappa_base'] = icl_df['contract.results.test_kappa']
    icl_df['kappa_icl'] = icl_df['contract.results.test_kappa']  # Same as base - no delta available
    icl_df['accbal_base'] = icl_df['contract.results.test_accuracy'] 
    icl_df['accbal_icl'] = icl_df['contract.results.test_accuracy']
    icl_df['f1w_base'] = icl_df['contract.results.test_f1']
    icl_df['f1w_icl'] = icl_df['contract.results.test_f1']
    
    # Set deltas to zero since we don't have separate measurements
    icl_df['delta_kappa'] = 0.0  # No delta available in this format
    icl_df['delta_accbal'] = 0.0
    icl_df['delta_f1w'] = 0.0
    
    # Add other useful columns
    icl_df['test_kappa_baseline'] = icl_df['contract.results.test_kappa']
    icl_df['hours_of_data'] = icl_df['contract.results.hours_of_data']
    icl_df['num_subjects_train'] = icl_df['contract.dataset.num_subjects_train']
    
    # Clean up and select relevant columns
    icl_cols = [
        'run_name', 'method', 'K', 'kappa_base', 'kappa_icl', 'delta_kappa',
        'accbal_base', 'accbal_icl', 'delta_accbal', 'f1w_base', 'f1w_icl', 'delta_f1w',
        'test_kappa_baseline', 'hours_of_data', 'num_subjects_train'
    ]
    
    result_df = icl_df[icl_cols].dropna(subset=['method', 'K'])
    
    print(f"✅ Transformed {len(result_df)} ICL runs")
    print("⚠️  Note: Delta metrics are set to 0 since all_runs_flat.csv doesn't contain separate baseline vs ICL measurements")
    print("   For proper ICL analysis, you need a CSV with separate baseline and ICL performance measurements")
    
    return result_df

def find_best_k_per_method(df: pd.DataFrame, use_performance_instead_of_delta: bool = False) -> Dict[str, int]:
    """Find the best K for each method based on highest median delta_kappa or performance."""
    best_k = {}
    
    # If we have meaningful delta_kappa values, use those; otherwise use direct performance
    metric = 'kappa_icl' if use_performance_instead_of_delta or df['delta_kappa'].abs().sum() == 0 else 'delta_kappa'
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        k_medians = method_data.groupby('K')[metric].median()
        
        if len(k_medians) > 0:
            # Find K with highest median (break ties by larger K)
            max_median = k_medians.max()
            best_ks = k_medians[k_medians == max_median].index.tolist()
            best_k[method] = max(best_ks)
    
    return best_k

def create_icl_overview_figure(df: pd.DataFrame, output_path: Path):
    """Create the main 2x3 ICL overview figure with proper paired delta analysis."""
    
    # Enforce paired delta analysis - no fallback to direct performance
    has_meaningful_deltas = df['delta_kappa'].abs().sum() > 1e-6  # Allow for floating point precision
    
    if not has_meaningful_deltas:
        print("❌ Error: No meaningful delta metrics found!")
        print("   This analysis requires ICL runs with separate baseline and ICL measurements.")
        print("   Expected keys: proto_test_kappa_base_K{K}, proto_test_kappa_icl_K{K}, etc.")
        raise ValueError("ICL delta analysis requires proper baseline vs ICL measurement pairs")
    
    # Find best K per method based on highest median delta_kappa
    best_k_per_method = {}
    for method in df['method'].unique():
        best_k_per_method[method] = find_best_k_for_method(df, method)
    
    # Print comprehensive summary statistics
    print(f"\n📊 ICL Paired Delta Analysis")
    print(f"{'='*60}")
    
    for method in df['method'].unique():
        available_ks = sorted(df[df['method'] == method]['K'].unique())
        best_k = best_k_per_method.get(method, 'N/A')
        method_data = df[(df['method'] == method) & (df['K'] == best_k)]
        
        if len(method_data) > 0:
            # Delta kappa statistics
            delta_median = method_data['delta_kappa'].median()
            ci_low, ci_high = bootstrap_ci_median(method_data['delta_kappa'].values)
            n_runs = len(method_data)
            
            # Gain rate (% of runs with positive delta)
            gain_rate = (method_data['delta_kappa'] > 0).mean() * 100
            
            print(f"\n{method}:")
            print(f"  Available Ks: {available_ks}")
            print(f"  Best K: {best_k} (n={n_runs} runs)")
            print(f"  Median Δκ: {delta_median:.4f} [95% CI: {ci_low:.4f}, {ci_high:.4f}]")
            print(f"  Gain rate: {gain_rate:.1f}% of runs show positive Δκ")
        else:
            print(f"\n{method}: Available Ks: {available_ks}, Best K: {best_k} (no data)")
    
    # Setup figure with proper delta analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ICL Paired Delta Analysis: Δ = ICL − Baseline', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Panel A: Δκ vs K (median ± 95% CI)
    plot_paired_deltas_vs_k(df, axes[0], 'delta_kappa', 'Δκ (ICL − Baseline)', 'A) Δκ vs K-shot')
    
    # Panel B: Δ Balanced Accuracy vs K  
    plot_paired_deltas_vs_k(df, axes[1], 'delta_accbal', 'Δ Balanced Accuracy', 'B) Δ Balanced Accuracy vs K-shot')
    
    # Panel C: Δ weighted F1 vs K
    plot_paired_deltas_vs_k(df, axes[2], 'delta_f1w', 'Δ weighted F1', 'C) Δ weighted F1 vs K-shot')
    
    # Panel D: Δκ distribution at best-K (violin + box + swarm)
    plot_delta_distributions_at_best_k(df, axes[3], best_k_per_method)
    
    # Panel E: Baseline κ vs Δκ correlation (properly enforced)
    plot_baseline_vs_delta_correlation(df, axes[4], best_k_per_method)
    
    # Panel F: Per-class ΔF1 analysis (Proto rescue analysis)
    plot_per_class_delta_f1(df, axes[5], best_k_per_method)
    
    # Create shared legend
    handles = [plt.Line2D([0], [0], color=COLORS[method], linewidth=2, label=method) 
               for method in df['method'].unique() if method in COLORS]
    
    fig.legend(handles, [h.get_label() for h in handles], 
              loc='lower center', ncol=len(handles), 
              bbox_to_anchor=(0.5, -0.02), fontsize=11)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    png_path = output_path.with_suffix('.png')
    pdf_path = output_path.with_suffix('.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    print(f"\n💾 Saved figures:")
    print(f"  📊 {png_path}")
    print(f"  📊 {pdf_path}")
    
    plt.show()

def plot_paired_deltas_vs_k(df: pd.DataFrame, ax, delta_col: str, ylabel: str, title: str):
    """Plot paired delta metrics vs K with median ± 95% CI bootstrap bands."""
    
    for method in df['method'].unique():
        if method not in COLORS:
            continue
            
        method_data = df[df['method'] == method]
        if len(method_data) == 0:
            continue
        
        ks = sorted(method_data['K'].unique())
        medians = []
        ci_lows = []
        ci_highs = []
        
        for k in ks:
            k_data = method_data[method_data['K'] == k][delta_col].dropna()
            if len(k_data) > 0:
                median_val = np.median(k_data.values)
                ci_low, ci_high = bootstrap_ci_median(k_data.values, n_bootstrap=1000, random_state=123)
                
                medians.append(median_val)
                ci_lows.append(ci_low if not np.isnan(ci_low) else median_val)
                ci_highs.append(ci_high if not np.isnan(ci_high) else median_val)
            else:
                medians.append(np.nan)
                ci_lows.append(np.nan) 
                ci_highs.append(np.nan)
        
        # Plot median line with markers
        valid_mask = ~np.isnan(medians)
        if np.sum(valid_mask) > 0:
            ks_array = np.array(ks)[valid_mask]
            medians_array = np.array(medians)[valid_mask]
            ci_lows_array = np.array(ci_lows)[valid_mask]
            ci_highs_array = np.array(ci_highs)[valid_mask]
            
            # Main line
            ax.plot(ks_array, medians_array, color=COLORS[method], linewidth=2.5, 
                   marker='o', markersize=8, label=method, zorder=3)
            
            # Bootstrap CI band
            ax.fill_between(ks_array, ci_lows_array, ci_highs_array,
                           color=COLORS[method], alpha=0.25, zorder=1)
            
            # Add CI error bars for clarity
            ax.errorbar(ks_array, medians_array, 
                       yerr=[medians_array - ci_lows_array, ci_highs_array - medians_array],
                       color=COLORS[method], alpha=0.7, capsize=4, capthick=1.5, zorder=2)
    
    # Add zero reference line (critical for delta interpretation)
    ax.axhline(y=0, color=COLORS['baseline'], linestyle='--', alpha=0.8, linewidth=2, zorder=0)
    ax.text(0.02, 0.02, 'No improvement', transform=ax.transAxes, 
           fontsize=9, alpha=0.7, style='italic')
    
    ax.set_xlabel('K-shot', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, zorder=0)
    
    # Set integer ticks for K-shot
    if len(df['K'].unique()) > 0:
        all_ks = sorted(df['K'].unique())
        ax.set_xticks(all_ks)

def plot_delta_distributions_at_best_k(df: pd.DataFrame, ax, best_k_per_method: Dict[str, int]):
    """Plot Δκ distributions at best-K using violin + box + individual points."""
    
    plot_data = []
    labels = []
    colors_used = []
    
    for method, best_k in best_k_per_method.items():
        if method not in COLORS:
            continue
            
        method_data = df[(df['method'] == method) & (df['K'] == best_k)]
        delta_values = method_data['delta_kappa'].dropna()
        
        if len(delta_values) > 0:
            plot_data.append(delta_values.values)
            labels.append(f"{method}\n(K={best_k}, n={len(delta_values)})")
            colors_used.append(COLORS[method])
        else:
            # Still add empty for consistent positioning
            plot_data.append([0.0])  # Single dummy point
            labels.append(f"{method}\n(K={best_k}, n=0)")
            colors_used.append(COLORS[method])
    
    if not plot_data:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        return
    
    positions = range(len(plot_data))
    
    # Violin plots for density
    valid_data = [data for data in plot_data if len(data) > 1]
    valid_positions = [i for i, data in enumerate(plot_data) if len(data) > 1]
    
    if valid_data:
        violin_parts = ax.violinplot(valid_data, positions=valid_positions, 
                                   widths=0.7, showmeans=False, showmedians=False, showextrema=False)
        
        # Color violins
        for i, (pc, pos) in enumerate(zip(violin_parts['bodies'], valid_positions)):
            pc.set_facecolor(colors_used[pos])
            pc.set_alpha(0.6)
            pc.set_edgecolor('white')
            pc.set_linewidth(1)
    
    # Box plots for quartiles
    box_parts = ax.boxplot(plot_data, positions=positions, widths=0.2, 
                          patch_artist=False, showfliers=False,
                          boxprops=dict(linewidth=2, color='black'), 
                          whiskerprops=dict(linewidth=2, color='black'),
                          capprops=dict(linewidth=2, color='black'),
                          medianprops=dict(linewidth=3, color='black'))
    
    # Individual points with jitter  
    np.random.seed(123)
    for i, (data, color) in enumerate(zip(plot_data, colors_used)):
        if len(data) > 1:  # Skip dummy single points
            jitter_strength = min(0.08, 0.15 / np.sqrt(len(data)))  # Less jitter for more points
            jitter = np.random.normal(0, jitter_strength, len(data))
            ax.scatter(i + jitter, data, alpha=0.8, s=25, color=color, 
                      edgecolors='white', linewidth=0.8, zorder=4)
    
    # Zero reference line
    ax.axhline(y=0, color=COLORS['baseline'], linestyle='--', alpha=0.8, linewidth=2, zorder=0)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Δκ (ICL − Baseline)', fontsize=12)
    ax.set_title('D) Δκ Distribution at Best-K', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y', zorder=0)

def plot_baseline_vs_delta_correlation(df: pd.DataFrame, ax, best_k_per_method: Dict[str, int]):
    """Plot baseline κ vs Δκ correlation at best-K (properly enforced y-axis)."""
    
    correlations = []
    
    for method, best_k in best_k_per_method.items():
        if method not in COLORS:
            continue
            
        method_data = df[(df['method'] == method) & (df['K'] == best_k)]
        
        # Strictly enforce: x = baseline, y = delta (NOT kappa_icl)
        if len(method_data) > 0 and 'test_kappa_baseline' in method_data.columns:
            valid_data = method_data.dropna(subset=['test_kappa_baseline', 'delta_kappa'])
            
            if len(valid_data) >= 3:
                x = valid_data['test_kappa_baseline'].values
                y = valid_data['delta_kappa'].values  # ENFORCED: must be delta, not kappa_icl
                
                # Scatter plot
                ax.scatter(x, y, color=COLORS[method], alpha=0.75, s=50, 
                          label=f'{method} (n={len(valid_data)})', 
                          edgecolors='white', linewidth=1, zorder=3)
                
                # Fit line and calculate correlation
                if len(valid_data) > 2:
                    # Linear regression line
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    ax.plot(x_line, p(x_line), color=COLORS[method], 
                           linestyle='-', alpha=0.8, linewidth=2, zorder=2)
                    
                    # Pearson correlation
                    r, p_val = stats.pearsonr(x, y)
                    correlations.append(f"{method}: r={r:.3f}, p={p_val:.3f}")
    
    # Zero reference line (horizontal)
    ax.axhline(y=0, color=COLORS['baseline'], linestyle='--', alpha=0.8, linewidth=2, zorder=0)
    ax.text(0.02, 0.02, 'No improvement', transform=ax.transAxes, 
           fontsize=9, alpha=0.7, style='italic')
    
    ax.set_xlabel('Baseline κ (test_kappa)', fontsize=12)
    ax.set_ylabel('Δκ (ICL − Baseline)', fontsize=12)  # ENFORCED
    ax.set_title('E) Baseline κ vs Δκ at Best-K', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, zorder=0)
    
    # Add correlation text box
    if correlations:
        text = '\n'.join(correlations)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', 
                facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Verify no diagonal correlation (would indicate y=x bug)
    print(f"\n🔍 Correlation Analysis (should NOT be r≈1):")
    for corr_text in correlations:
        print(f"  {corr_text}")
        # Alert if suspiciously high correlation
        r_value = float(corr_text.split('r=')[1].split(',')[0])
        if abs(r_value) > 0.95:
            print(f"  ⚠️  WARNING: Very high correlation detected! Check if plotting Δκ vs baseline κ correctly.")

def plot_per_class_delta_f1(df: pd.DataFrame, ax, best_k_per_method: Dict[str, int]):
    """Plot per-class ΔF1 at best-K to show if Proto rescues N1/REM classes."""
    
    # This requires per-class F1 data which may not be available
    # For now, create a placeholder analysis of method effectiveness
    
    methods = []
    delta_kappa_values = []
    colors = []
    
    for method, best_k in best_k_per_method.items():
        if method not in COLORS:
            continue
        
        method_data = df[(df['method'] == method) & (df['K'] == best_k)]
        if len(method_data) > 0:
            # Use median delta metrics as proxy for class-specific improvements
            delta_kappa = method_data['delta_kappa'].median()
            delta_f1 = method_data['delta_f1w'].median()
            delta_acc = method_data['delta_accbal'].median()
            
            methods.append(method)
            # Stack the different metric improvements
            delta_kappa_values.append([delta_kappa, delta_f1, delta_acc])
            colors.append(COLORS[method])
    
    if not methods:
        ax.text(0.5, 0.5, 'No per-class F1 data available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title('F) Per-class Analysis (Unavailable)', fontweight='bold', fontsize=13)
        return
    
    # Create stacked bar chart of metric improvements
    metrics_names = ['Δκ', 'ΔF1', 'ΔAcc']
    x_pos = np.arange(len(methods))
    width = 0.6
    
    # Convert to numpy array for easier manipulation
    deltas_array = np.array(delta_kappa_values).T  # Shape: (3, n_methods)
    
    # Create stacked bars
    bottom = np.zeros(len(methods))
    for i, (metric, color_alpha) in enumerate(zip(metrics_names, [0.8, 0.6, 0.4])):
        values = deltas_array[i]
        bars = ax.bar(x_pos, values, width, bottom=bottom, 
                     label=metric, alpha=color_alpha, 
                     color=[colors[j] for j in range(len(methods))])
        bottom += values
        
        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, values)):
            if abs(val) > 0.001:  # Only show non-trivial values
                height = bar.get_height()
                y_pos = bar.get_y() + height/2
                ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                       f'{val:.3f}', ha='center', va='center', 
                       fontsize=9, fontweight='bold', color='white')
    
    # Zero reference line
    ax.axhline(y=0, color=COLORS['baseline'], linestyle='--', alpha=0.8, linewidth=2, zorder=0)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel('Improvement (Δ)', fontsize=12)
    ax.set_title('F) Multi-Metric Improvements at Best-K', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y', zorder=0)
    ax.legend(loc='upper right', fontsize=10)

def plot_delta_vs_k(df: pd.DataFrame, ax, metric_col: str, ylabel: str, title: str):
    """Plot delta metric vs K with median and 95% CI bands."""
    
    for method in df['method'].unique():
        if method not in COLORS:
            continue
            
        method_data = df[df['method'] == method]
        
        if len(method_data) == 0:
            continue
        
        ks = sorted(method_data['K'].unique())
        medians = []
        ci_lows = []
        ci_highs = []
        
        for k in ks:
            k_data = method_data[method_data['K'] == k][metric_col].values
            if len(k_data) > 0:
                median_val = np.median(k_data)
                ci_low, ci_high = bootstrap_ci_median(k_data)
                
                medians.append(median_val)
                ci_lows.append(ci_low if not np.isnan(ci_low) else median_val)
                ci_highs.append(ci_high if not np.isnan(ci_high) else median_val)
            else:
                medians.append(np.nan)
                ci_lows.append(np.nan)
                ci_highs.append(np.nan)
        
        # Plot median line
        valid_indices = ~np.isnan(medians)
        if np.sum(valid_indices) > 0:
            ks_array = np.array(ks)
            medians_array = np.array(medians)
            ci_lows_array = np.array(ci_lows)
            ci_highs_array = np.array(ci_highs)
            
            ax.plot(ks_array[valid_indices], medians_array[valid_indices], 
                   color=COLORS[method], linewidth=2, marker='o', markersize=6, label=method)
            
            # Plot CI band
            ax.fill_between(ks_array[valid_indices], 
                           ci_lows_array[valid_indices], ci_highs_array[valid_indices],
                           color=COLORS[method], alpha=0.2)
    
    # Add zero reference line
    ax.axhline(y=0, color=COLORS['baseline'], linestyle='--', alpha=0.7, linewidth=1)
    
    ax.set_xlabel('K-shot')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set integer ticks for K
    if len(df['K'].unique()) > 0:
        ax.set_xticks(sorted(df['K'].unique()))

def plot_performance_vs_k(df: pd.DataFrame, ax, metric_col: str, ylabel: str, title: str):
    """Plot performance metric vs K with median and 95% CI bands."""
    
    for method in df['method'].unique():
        if method not in COLORS:
            continue
            
        method_data = df[df['method'] == method]
        
        if len(method_data) == 0:
            continue
        
        ks = sorted(method_data['K'].unique())
        medians = []
        ci_lows = []
        ci_highs = []
        
        for k in ks:
            k_data = method_data[method_data['K'] == k][metric_col].values
            if len(k_data) > 0:
                median_val = np.median(k_data)
                ci_low, ci_high = bootstrap_ci_median(k_data)
                
                medians.append(median_val)
                ci_lows.append(ci_low if not np.isnan(ci_low) else median_val)
                ci_highs.append(ci_high if not np.isnan(ci_high) else median_val)
            else:
                medians.append(np.nan)
                ci_lows.append(np.nan)
                ci_highs.append(np.nan)
        
        # Plot median line
        valid_indices = ~np.isnan(medians)
        if np.sum(valid_indices) > 0:
            ks_array = np.array(ks)
            medians_array = np.array(medians)
            ci_lows_array = np.array(ci_lows)
            ci_highs_array = np.array(ci_highs)
            
            ax.plot(ks_array[valid_indices], medians_array[valid_indices], 
                   color=COLORS[method], linewidth=2, marker='o', markersize=6, label=method)
            
            # Plot CI band
            ax.fill_between(ks_array[valid_indices], 
                           ci_lows_array[valid_indices], ci_highs_array[valid_indices],
                           color=COLORS[method], alpha=0.2)
    
    ax.set_xlabel('K-shot')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set integer ticks for K
    if len(df['K'].unique()) > 0:
        ax.set_xticks(sorted(df['K'].unique()))

def plot_best_k_distributions(df: pd.DataFrame, ax, best_k_per_method: Dict[str, int]):
    """Plot violin + box + swarm plot of Δκ distributions at best-K."""
    
    plot_data = []
    labels = []
    
    for i, (method, best_k) in enumerate(best_k_per_method.items()):
        if method not in COLORS:
            continue
            
        method_data = df[(df['method'] == method) & (df['K'] == best_k)]
        
        if len(method_data) > 0:
            plot_data.append(method_data['delta_kappa'].values)
            labels.append(f"{method}\n(K={best_k})")
        else:
            plot_data.append([])
            labels.append(f"{method}\n(K={best_k})")
    
    if plot_data and any(len(data) > 0 for data in plot_data):
        # Create violin plot
        positions = range(len(plot_data))
        violin_parts = ax.violinplot([data for data in plot_data if len(data) > 0], 
                                   positions=[i for i, data in enumerate(plot_data) if len(data) > 0],
                                   widths=0.6, showmeans=False, showmedians=True, showextrema=False)
        
        # Color the violins
        method_names = [list(best_k_per_method.keys())[i] for i, data in enumerate(plot_data) if len(data) > 0]
        for i, pc in enumerate(violin_parts['bodies']):
            if i < len(method_names) and method_names[i] in COLORS:
                pc.set_facecolor(COLORS[method_names[i]])
                pc.set_alpha(0.7)
        
        # Add box plot
        box_plot_data = [data for data in plot_data if len(data) > 0]
        box_positions = [i for i, data in enumerate(plot_data) if len(data) > 0]
        
        bp = ax.boxplot(box_plot_data, positions=box_positions, widths=0.15, 
                       patch_artist=False, showfliers=True,
                       boxprops=dict(linewidth=1.5), 
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        
        # Add swarm plot for individual points (with jitter)
        np.random.seed(123)
        for i, data in enumerate(plot_data):
            if len(data) > 0:
                jitter = np.random.normal(0, 0.05, len(data))
                method_name = list(best_k_per_method.keys())[i]
                if method_name in COLORS:
                    ax.scatter(i + jitter, data, alpha=0.6, s=20, 
                             color=COLORS[method_name], edgecolors='white', linewidth=0.5)
    
    # Add zero reference line
    ax.axhline(y=0, color=COLORS['baseline'], linestyle='--', alpha=0.7, linewidth=1)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel('Δκ (ICL − Baseline)')
    ax.set_title('D) Δκ Distribution at Best-K', fontweight='bold')
    ax.grid(True, alpha=0.3)

def plot_best_k_performance_distributions(df: pd.DataFrame, ax, best_k_per_method: Dict[str, int], use_performance: bool = True):
    """Plot violin + box + swarm plot of performance distributions at best-K."""
    
    plot_data = []
    labels = []
    metric = 'kappa_icl' if use_performance else 'delta_kappa'
    
    for i, (method, best_k) in enumerate(best_k_per_method.items()):
        if method not in COLORS:
            continue
            
        method_data = df[(df['method'] == method) & (df['K'] == best_k)]
        
        if len(method_data) > 0:
            plot_data.append(method_data[metric].values)
            labels.append(f"{method}\n(K={best_k})")
        else:
            plot_data.append([])
            labels.append(f"{method}\n(K={best_k})")
    
    if plot_data and any(len(data) > 0 for data in plot_data):
        # Create violin plot
        positions = range(len(plot_data))
        violin_parts = ax.violinplot([data for data in plot_data if len(data) > 0], 
                                   positions=[i for i, data in enumerate(plot_data) if len(data) > 0],
                                   widths=0.6, showmeans=False, showmedians=True, showextrema=False)
        
        # Color the violins
        method_names = [list(best_k_per_method.keys())[i] for i, data in enumerate(plot_data) if len(data) > 0]
        for i, pc in enumerate(violin_parts['bodies']):
            if i < len(method_names) and method_names[i] in COLORS:
                pc.set_facecolor(COLORS[method_names[i]])
                pc.set_alpha(0.7)
        
        # Add box plot
        box_plot_data = [data for data in plot_data if len(data) > 0]
        box_positions = [i for i, data in enumerate(plot_data) if len(data) > 0]
        
        bp = ax.boxplot(box_plot_data, positions=box_positions, widths=0.15, 
                       patch_artist=False, showfliers=True,
                       boxprops=dict(linewidth=1.5), 
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        
        # Add swarm plot for individual points (with jitter)
        np.random.seed(123)
        for i, data in enumerate(plot_data):
            if len(data) > 0:
                jitter = np.random.normal(0, 0.05, len(data))
                method_name = list(best_k_per_method.keys())[i]
                if method_name in COLORS:
                    ax.scatter(i + jitter, data, alpha=0.6, s=20, 
                             color=COLORS[method_name], edgecolors='white', linewidth=0.5)
    
    # Add reference line (zero for delta, or don't add for performance)
    if not use_performance:
        ax.axhline(y=0, color=COLORS['baseline'], linestyle='--', alpha=0.7, linewidth=1)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    ylabel = 'Test κ' if use_performance else 'Δκ (ICL − Baseline)'
    title = 'D) Test κ Distribution at Best-K' if use_performance else 'D) Δκ Distribution at Best-K'
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)

def plot_baseline_correlation(df: pd.DataFrame, ax, best_k_per_method: Dict[str, int], use_performance: bool = False):
    """Plot baseline κ vs Δκ/performance correlation at best-K."""
    
    correlations = []
    y_metric = 'kappa_icl' if use_performance else 'delta_kappa'
    
    for method, best_k in best_k_per_method.items():
        if method not in COLORS:
            continue
            
        method_data = df[(df['method'] == method) & (df['K'] == best_k)]
        
        if len(method_data) > 0 and 'test_kappa_baseline' in method_data.columns:
            # Filter out missing baseline values
            valid_data = method_data.dropna(subset=['test_kappa_baseline', y_metric])
            
            if len(valid_data) >= 3:
                x = valid_data['test_kappa_baseline'].values
                y = valid_data[y_metric].values
                
                # Scatter plot
                ax.scatter(x, y, color=COLORS[method], alpha=0.7, s=40, label=method, edgecolors='white', linewidth=0.5)
                
                # Fit line and calculate correlation
                if len(valid_data) > 1:
                    # Linear fit
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    ax.plot(x_line, p(x_line), color=COLORS[method], linestyle='-', alpha=0.8, linewidth=1.5)
                    
                    # Calculate correlation
                    r, p_val = stats.pearsonr(x, y)
                    correlations.append(f"{method}: r={r:.3f}, p={p_val:.3f}")
    
    # Add reference line (zero for delta, or don't add for performance)
    if not use_performance:
        ax.axhline(y=0, color=COLORS['baseline'], linestyle='--', alpha=0.7, linewidth=1)
    
    ax.set_xlabel('Baseline κ (test_kappa)')
    ylabel = 'Test κ (ICL)' if use_performance else 'Δκ (ICL − Baseline)'
    title = 'E) Baseline κ vs ICL κ at Best-K' if use_performance else 'E) Baseline κ vs Δκ at Best-K'
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add correlation text
    if correlations:
        text = '\n'.join(correlations)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Print correlations to console
    metric_name = 'ICL κ' if use_performance else 'Δκ'
    print(f"\n📈 Baseline κ vs {metric_name} Correlations:")
    for corr_text in correlations:
        print(f"  {corr_text}")

def plot_data_scale_effects(df: pd.DataFrame, ax, best_k_per_method: Dict[str, int], use_performance: bool = False):
    """Plot Δκ/performance vs data scale (hours_of_data and num_subjects_train)."""
    
    correlations = []
    y_metric = 'kappa_icl' if use_performance else 'delta_kappa'
    
    # Focus on hours_of_data for the main plot
    for method, best_k in best_k_per_method.items():
        if method not in COLORS:
            continue
            
        method_data = df[(df['method'] == method) & (df['K'] == best_k)]
        
        if len(method_data) > 0 and 'hours_of_data' in method_data.columns:
            # Filter out missing values
            valid_data = method_data.dropna(subset=['hours_of_data', y_metric])
            
            if len(valid_data) >= 3:
                x = valid_data['hours_of_data'].values
                y = valid_data[y_metric].values
                
                # Scatter plot
                ax.scatter(x, y, color=COLORS[method], alpha=0.7, s=40, label=method, edgecolors='white', linewidth=0.5)
                
                # Fit line and calculate correlation
                if len(valid_data) > 1:
                    # Linear fit
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    ax.plot(x_line, p(x_line), color=COLORS[method], linestyle='-', alpha=0.8, linewidth=1.5)
                    
                    # Calculate correlation
                    r, p_val = stats.pearsonr(x, y)
                    correlations.append(f"{method}: r={r:.3f}, p={p_val:.3f}")
    
    # Add reference line (zero for delta, or don't add for performance)
    if not use_performance:
        ax.axhline(y=0, color=COLORS['baseline'], linestyle='--', alpha=0.7, linewidth=1)
    
    ax.set_xlabel('Hours of Data')
    ylabel = 'Test κ (ICL)' if use_performance else 'Δκ (ICL − Baseline)'
    title = 'F) Test κ vs Data Scale at Best-K' if use_performance else 'F) Δκ vs Data Scale at Best-K'
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add correlation text
    if correlations:
        text = '\n'.join(correlations)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Print correlations to console
    metric_name = 'Test κ' if use_performance else 'Δκ'
    print(f"\n📊 Data Scale Correlations (hours_of_data vs {metric_name}):")
    for corr_text in correlations:
        print(f"  {corr_text}")
    
    # Also calculate num_subjects_train correlations and print
    print(f"\n👥 Data Scale Correlations (num_subjects_train vs {metric_name}):")
    for method, best_k in best_k_per_method.items():
        if method not in COLORS:
            continue
            
        method_data = df[(df['method'] == method) & (df['K'] == best_k)]
        
        if len(method_data) > 0 and 'num_subjects_train' in method_data.columns:
            valid_data = method_data.dropna(subset=['num_subjects_train', y_metric])
            
            if len(valid_data) >= 3:
                x = valid_data['num_subjects_train'].values
                y = valid_data[y_metric].values
                
                r, p_val = stats.pearsonr(x, y)
                print(f"  {method}: r={r:.3f}, p={p_val:.3f}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Generate ICL Overview Figure')
    parser.add_argument('--out', default='Plot_Clean/figures/icl_overview', 
                       help='Output path prefix (without extension)')
    parser.add_argument('--csv', help='Path to CSV file (if provided, skips W&B)')
    parser.add_argument('--entity', default='thibaut_hasle-epfl', help='W&B entity')
    parser.add_argument('--project', default='CBraMod-earEEG-tuning', help='W&B project')
    parser.add_argument('--ks', help='Comma-separated K values to include (e.g., 1,3,5)')
    
    args = parser.parse_args()
    
    # Setup plotting
    setup_plotting_style()
    
    # Parse K filter
    ks_filter = None
    if args.ks:
        ks_filter = [int(k.strip()) for k in args.ks.split(',')]
        print(f"🎯 Filtering to K values: {ks_filter}")
    
    # Load data
    if args.csv:
        df = load_data_from_csv(Path(args.csv))
    else:
        if 'WANDB_API_KEY' not in os.environ:
            print("⚠️  Warning: WANDB_API_KEY not found in environment")
        df = load_data_from_wandb(args.entity, args.project, ks_filter)
    
    # Apply K filter if specified
    if ks_filter:
        df = df[df['K'].isin(ks_filter)]
        print(f"📊 After K filtering: {len(df)} measurements")
    
    if len(df) == 0:
        print("❌ No data available after filtering")
        return
    
    # Save tidy dataframe
    output_path = Path(args.out)
    tidy_csv_path = output_path.with_name(output_path.name + '_tidy.csv')
    df.to_csv(tidy_csv_path, index=False)
    print(f"💾 Saved tidy dataframe: {tidy_csv_path}")
    
    # Create figure
    create_icl_overview_figure(df, output_path)
    
    print(f"\n✅ ICL overview analysis complete!")

if __name__ == '__main__':
    main()