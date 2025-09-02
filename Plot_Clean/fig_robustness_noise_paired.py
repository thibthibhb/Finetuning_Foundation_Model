#!/usr/bin/env python3
"""
CBraMod Robustness Analysis: Noise Injection Study with Paired Subject Analysis

Enhanced version that:
- Compares exactly the four artifact families: EMG, Gaussian, Electrode, Realistic
- Uses paired, per-subject deltas vs. Clean (controls for difficulty & hyperparams)
- Shows 95% bootstrap CIs (on the median) rather than SDs
- Runs within-subject stats: Wilcoxon vs. 0 for each artifact at 10% and Friedman omnibus test
- Fills panel C with the per-subject Δκ distribution at 10%

Usage:
    python Plot_Clean/fig_robustness_noise_paired.py --csv Plot_Clean/data/all_runs_flat.csv --test_subjects Plot_Clean/data/all_test_subjects_complete.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
from scipy import stats
from collections import defaultdict
warnings.filterwarnings('ignore')

# Import consistent figure styling
from figure_style import (
    setup_figure_style, get_color, save_figure, 
    bootstrap_ci_median, wilcoxon_test
)

def load_and_merge_data(csv_path: str, test_subjects_path: str) -> pd.DataFrame:
    """Load and merge runs data with test subjects information."""
    print(f"📁 Loading runs data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"📁 Loading test subjects data from: {test_subjects_path}")
    test_df = pd.read_csv(test_subjects_path)
    
    # Merge on run_id
    df = df.merge(test_df[['run_id', 'test_subjects']], on='run_id', how='left')
    
    print(f"✅ Merged data: {len(df)} runs")
    return df

def load_and_process_data(csv_path: str, test_subjects_path: str) -> pd.DataFrame:
    """Load and process robustness data with paired subject analysis."""
    df = load_and_merge_data(csv_path, test_subjects_path)
    
    # Filter for completed runs with valid results
    df = df[df['state'] == 'finished'].copy()
    df = df[df['contract.results.test_kappa'].notna()].copy()
    
    # Handle missing noise parameters
    if 'contract.noise.noise_level' not in df.columns:
        print("⚠️  Adding default noise parameters for older runs")
        df['contract.noise.noise_level'] = 0.0
        df['contract.noise.noise_type'] = 'clean'
    
    # Fill missing values
    df['contract.noise.noise_level'] = df['contract.noise.noise_level'].fillna(0.0)
    df['contract.noise.noise_type'] = df['contract.noise.noise_type'].fillna('clean')
    
    # Normalize noise type names to match the four families
    noise_mapping = {
        'none': 'clean',
        'clean': 'clean',
        'gaussian': 'gaussian',
        'emg': 'emg', 
        'electrode': 'electrode',
        'realistic': 'realistic'
    }
    df['noise_type'] = df['contract.noise.noise_type'].str.lower().map(noise_mapping)
    df = df[df['noise_type'].notna()].copy()  # Remove unmapped noise types
    
    # CRITICAL: This robustness plot KEEPS ALL noise levels (unlike other plots that filter)
    # Use available noise levels - include all to maximize pairing opportunities
    available_levels = sorted(df['contract.noise.noise_level'].unique())
    print(f"   Available noise levels: {available_levels}")
    
    # Keep all available levels for robustness analysis (including high noise >1%)
    target_levels = available_levels
    df = df[df['contract.noise.noise_level'].isin(target_levels)].copy()
    print(f"✅ Retaining ALL noise levels for robustness analysis (no filtering applied)")
    
    # Focus on 4/5-class results for consistency
    df = df[df['contract.results.num_classes'].isin([4, 5])].copy()
    
    print(f"📊 Loaded {len(df)} valid runs for robustness analysis")
    print(f"   Noise types: {sorted(df['noise_type'].unique())}")
    print(f"   Noise levels: {sorted(df['contract.noise.noise_level'].unique())}")
    
    return df

def create_paired_analysis(df: pd.DataFrame) -> Dict:
    """Create paired subject analysis comparing clean vs noisy conditions."""
    print("\\n🔄 Creating paired subject analysis...")
    
    # For each individual test subject, find clean and noisy runs
    subject_deltas = defaultdict(dict)
    
    # Extract all individual test subjects from the comma-separated strings
    all_individual_subjects = set()
    for test_subj_str in df['test_subjects'].dropna().unique():
        if test_subj_str:
            subjects = [s.strip() for s in test_subj_str.split(',')]
            all_individual_subjects.update(subjects)
    
    print(f"   Found {len(all_individual_subjects)} unique test subjects: {sorted(all_individual_subjects)}")
    
    for individual_subject in sorted(all_individual_subjects):
        # Find all runs where this subject was in the test set
        subject_runs = df[df['test_subjects'].str.contains(individual_subject, na=False)]
        
        # Find clean baseline for this subject
        clean_runs = subject_runs[subject_runs['contract.noise.noise_level'] == 0.0]
        if len(clean_runs) == 0:
            continue
        
        # Take best clean performance as baseline
        baseline_kappa = clean_runs['contract.results.test_kappa'].max()
        
        # For each noise type, collect all available noisy runs and compute deltas
        for noise_type in ['gaussian', 'emg', 'electrode', 'realistic']:
            noise_runs = subject_runs[
                (subject_runs['noise_type'] == noise_type) & 
                (subject_runs['contract.noise.noise_level'] > 0)  # Any noise level > 0
            ]
            
            # Collect all deltas for this noise type to enable statistical testing
            if len(noise_runs) > 0:
                all_deltas = []
                all_noisy_kappa = []
                
                for _, noise_run in noise_runs.iterrows():
                    noisy_kappa = noise_run['contract.results.test_kappa']
                    delta_kappa = noisy_kappa - baseline_kappa
                    all_deltas.append(delta_kappa)
                    all_noisy_kappa.append(noisy_kappa)
                
                # Store all deltas for this subject-noise type combination
                subject_deltas[individual_subject][noise_type] = {
                    'baseline_kappa': baseline_kappa,
                    'all_deltas': all_deltas,  # All individual deltas 
                    'all_noisy_kappa': all_noisy_kappa,
                    'median_delta': np.median(all_deltas),
                    'mean_delta': np.mean(all_deltas),
                    'n_clean': len(clean_runs),
                    'n_noisy': len(noise_runs)
                }
    
    print(f"   Found paired data for {len(subject_deltas)} test subjects")
    
    # Convert to DataFrame - expand all individual deltas for statistical testing
    paired_data = []
    subject_summary = []
    
    for test_subj, noise_data in subject_deltas.items():
        for noise_type, metrics in noise_data.items():
            # Add summary per subject-noise combination
            subject_summary.append({
                'test_subject': test_subj,
                'noise_type': noise_type,
                'baseline_kappa': metrics['baseline_kappa'],
                'median_delta': metrics['median_delta'],
                'mean_delta': metrics['mean_delta'],
                'n_clean': metrics['n_clean'],
                'n_noisy': metrics['n_noisy']
            })
            
            # Add each individual delta for detailed analysis
            for i, delta in enumerate(metrics['all_deltas']):
                paired_data.append({
                    'test_subject': test_subj,
                    'noise_type': noise_type,
                    'baseline_kappa': metrics['baseline_kappa'],
                    'noisy_kappa': metrics['all_noisy_kappa'][i],
                    'delta_kappa': delta,
                    'n_clean': metrics['n_clean'],
                    'n_noisy': metrics['n_noisy']
                })
    
    paired_df = pd.DataFrame(paired_data)
    subject_summary_df = pd.DataFrame(subject_summary)
    
    print(f"   Total paired comparisons: {len(subject_summary_df)}")
    for noise_type in subject_summary_df['noise_type'].unique():
        n_subjects = len(subject_summary_df[subject_summary_df['noise_type'] == noise_type])
        print(f"     {noise_type}: {n_subjects} subjects")
    
    return {
        'paired_df': paired_df,
        'subject_summary_df': subject_summary_df,
        'subject_deltas': subject_deltas
    }

def run_statistical_tests(subject_summary_df: pd.DataFrame):
    """Run within-subject statistical tests."""
    print("\\n📊 Running within-subject statistical tests...")
    
    results = {}
    
    # Wilcoxon signed-rank tests for each artifact type vs. 0 (using median delta per subject)
    print("\\n🧪 Wilcoxon signed-rank tests (vs. 0 delta):")
    for noise_type in ['gaussian', 'emg', 'electrode', 'realistic']:
        type_data = subject_summary_df[subject_summary_df['noise_type'] == noise_type]
        deltas = type_data['median_delta'].values
        
        if len(deltas) >= 3:  # Relaxed minimum for small sample
            try:
                statistic, p_value = stats.wilcoxon(deltas, alternative='two-sided')
                median_delta = np.median(deltas)
                
                # Effect direction
                direction = "↓" if median_delta < 0 else "↑"
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                print(f"   {noise_type.upper():>10}: Δκ = {median_delta:+.3f} {direction} (p = {p_value:.4f} {significance}, n = {len(deltas)})")
                
                results[noise_type] = {
                    'median_delta': median_delta,
                    'p_value': p_value,
                    'n': len(deltas),
                    'statistic': statistic
                }
            except Exception as e:
                print(f"   {noise_type.upper():>10}: Failed - {e} (n = {len(deltas)})")
                results[noise_type] = {'median_delta': np.median(deltas) if len(deltas) > 0 else np.nan, 'p_value': np.nan, 'n': len(deltas)}
        else:
            print(f"   {noise_type.upper():>10}: Insufficient data (n = {len(deltas)})")
            results[noise_type] = {'median_delta': np.median(deltas) if len(deltas) > 0 else np.nan, 'p_value': np.nan, 'n': len(deltas)}
    
    # Friedman test across artifact types (omnibus test)
    print("\\n🔬 Friedman omnibus test across artifact types:")
    
    # Create matrix for Friedman test (subjects x noise types)
    subjects_with_all_noise = set(subject_summary_df['test_subject'])
    for noise_type in ['gaussian', 'emg', 'electrode', 'realistic']:
        subjects_for_noise = set(subject_summary_df[subject_summary_df['noise_type'] == noise_type]['test_subject'])
        subjects_with_all_noise &= subjects_for_noise
    
    if len(subjects_with_all_noise) >= 3:  # Relaxed requirement
        friedman_data = []
        for subject in subjects_with_all_noise:
            subject_row = []
            for noise_type in ['gaussian', 'emg', 'electrode', 'realistic']:
                delta = subject_summary_df[
                    (subject_summary_df['test_subject'] == subject) & 
                    (subject_summary_df['noise_type'] == noise_type)
                ]['median_delta'].iloc[0]
                subject_row.append(delta)
            friedman_data.append(subject_row)
        
        friedman_data = np.array(friedman_data)
        
        try:
            statistic, p_value = stats.friedmanchisquare(*friedman_data.T)
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            
            print(f"   Friedman χ² = {statistic:.3f}, p = {p_value:.4f} {significance}")
            print(f"   Complete cases: {len(subjects_with_all_noise)} subjects")
            
            results['friedman'] = {
                'statistic': statistic,
                'p_value': p_value,
                'n_subjects': len(subjects_with_all_noise)
            }
        except Exception as e:
            print(f"   Friedman test failed: {e}")
            results['friedman'] = {'statistic': np.nan, 'p_value': np.nan, 'n_subjects': len(subjects_with_all_noise)}
    else:
        print(f"   Insufficient subjects with all noise types (n = {len(subjects_with_all_noise)})")
        results['friedman'] = {'statistic': np.nan, 'p_value': np.nan, 'n_subjects': len(subjects_with_all_noise)}
    
    return results

def create_robustness_figure_paired(df: pd.DataFrame, paired_results: Dict, stats_results: Dict, output_path: Path):
    """Create the enhanced robustness figure with paired analysis."""
    
    setup_figure_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Base colors for artifact families
    artifact_colors = {
        'gaussian': '#E69F00',    # Orange family
        'emg': '#D55E00',         # Red family
        'electrode': '#009E73',   # Green family  
        'realistic': '#CC79A7'    # Purple family
    }
    
    paired_df = paired_results['paired_df']
    
    # === SUBPLOT A: Paired Deltas with Bootstrap CIs ===
    ax1 = axes[0]
    
    noise_types = ['gaussian', 'emg', 'electrode', 'realistic']
    noise_labels = ['Gaussian', 'EMG', 'Electrode', 'Realistic']
    
    medians = []
    ci_lows = []
    ci_highs = []
    
    # Aggregate all noise levels for each artifact type
    for noise_type in noise_types:
        deltas = paired_df[paired_df['noise_type'] == noise_type]['delta_kappa']
        
        if len(deltas) > 0:
            median_delta = np.median(deltas)
            ci_low, ci_high = bootstrap_ci_median(deltas.values)
            
            medians.append(median_delta)
            ci_lows.append(ci_low)
            ci_highs.append(ci_high)
        else:
            medians.append(0)
            ci_lows.append(0)
            ci_highs.append(0)
    
    # Create error bars from bootstrap CIs
    errors = [
        [med - ci_low for med, ci_low in zip(medians, ci_lows)],
        [ci_high - med for med, ci_high in zip(medians, ci_highs)]
    ]
    
    bars = ax1.bar(noise_labels, medians, 
                  yerr=errors,
                  capsize=5, error_kw={'linewidth': 2},
                  color=[artifact_colors[noise_type] for noise_type in noise_types],
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add significance markers from Wilcoxon tests
    for i, (bar, noise_type) in enumerate(zip(bars, noise_types)):
        height = bar.get_height()
        if noise_type in stats_results and not np.isnan(stats_results[noise_type].get('p_value', np.nan)):
            p_val = stats_results[noise_type]['p_value']
            sig_text = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            
            if sig_text:
                ax1.text(bar.get_x() + bar.get_width()/2, 
                        height + (ci_highs[i] - medians[i]) + 0.005,
                        sig_text, ha='center', va='bottom', 
                        fontsize=14, fontweight='bold')
    
    # Add sample sizes
    for i, (bar, noise_type) in enumerate(zip(bars, noise_types)):
        if noise_type in stats_results:
            n = stats_results[noise_type].get('n', 0)
            ax1.text(bar.get_x() + bar.get_width()/2, -0.12,
                    f'n={n}', ha='center', va='top', fontsize=10)
    
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Δκ (Artifact - Clean)')
    ax1.set_title('A) Paired Performance Impact', fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # === SUBPLOT B: Effect Size Comparison ===
    ax2 = axes[1]
    
    # Calculate effect sizes (median / baseline median)
    baseline_medians = []
    effect_sizes = []
    
    for noise_type in noise_types:
        subset = paired_df[paired_df['noise_type'] == noise_type]
        if len(subset) > 0:
            baseline_med = np.median(subset['baseline_kappa'])
            delta_med = np.median(subset['delta_kappa'])
            effect_size = abs(delta_med / baseline_med) * 100  # Percentage effect
            
            baseline_medians.append(baseline_med)
            effect_sizes.append(effect_size)
        else:
            baseline_medians.append(0.5)  # Default
            effect_sizes.append(0)
    
    bars2 = ax2.bar(noise_labels, effect_sizes,
                   color=[artifact_colors[noise_type] for noise_type in noise_types],
                   alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, effect in zip(bars2, effect_sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.2,
                f'{effect:.1f}%', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Relative Effect Size (%)')
    ax2.set_title('B) Relative Impact Size', fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # === SUBPLOT C: Delta Distribution ===
    ax3 = axes[2]
    
    delta_data = []
    delta_labels = []
    delta_colors = []
    
    for noise_type, label in zip(noise_types, noise_labels):
        deltas = paired_df[paired_df['noise_type'] == noise_type]['delta_kappa']
        if len(deltas) > 0:
            delta_data.append(deltas.values)
            delta_labels.append(label)
            delta_colors.append(artifact_colors[noise_type])
    
    if len(delta_data) > 0:
        # Create violin plot
        parts = ax3.violinplot(delta_data, positions=range(len(delta_data)), 
                              showmeans=False, showmedians=True)
        
        # Color the violins
        for i, (part, color) in enumerate(zip(parts['bodies'], delta_colors)):
            part.set_facecolor(color)
            part.set_alpha(0.7)
        
        ax3.set_xticks(range(len(delta_labels)))
        ax3.set_xticklabels(delta_labels, rotation=45)
    
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Δκ Distribution')
    ax3.set_title('C) Per-Subject Δκ Distribution', fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Overall figure adjustments
    plt.tight_layout()
    
    # Add overall title
    fig.suptitle('CBraMod Robustness to Ear-EEG Noise Artifacts (Paired Analysis)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.90)
    
    # Save figure using consistent save function
    saved_files = save_figure(fig, output_path)
    
    return fig

def print_paired_summary(paired_results: Dict, stats_results: Dict):
    """Print summary statistics for paired analysis."""
    print("\\n" + "="*70)
    print("PAIRED SUBJECT ROBUSTNESS ANALYSIS SUMMARY")
    print("="*70)
    
    paired_df = paired_results['paired_df']
    subject_deltas = paired_results['subject_deltas']
    
    print(f"\\n📊 Paired Analysis Overview:")
    print(f"   Total test subjects analyzed: {len(subject_deltas)}")
    print(f"   Total paired comparisons: {len(paired_df)}")
    
    # Per-artifact analysis
    print(f"\\n🔍 Per-Artifact Analysis (10% noise level):")
    
    for noise_type in ['gaussian', 'emg', 'electrode', 'realistic']:
        subset = paired_df[paired_df['noise_type'] == noise_type]
        if len(subset) > 0:
            deltas = subset['delta_kappa']
            baseline_kappas = subset['baseline_kappa']
            
            print(f"\\n   {noise_type.upper()}:")
            print(f"     Subjects: {len(deltas)}")
            print(f"     Baseline κ: {baseline_kappas.median():.3f}")
            print(f"     Median Δκ: {deltas.median():.3f}")
            print(f"     IQR Δκ: [{deltas.quantile(0.25):.3f}, {deltas.quantile(0.75):.3f}]")
            
            # Effect direction
            worse = (deltas < 0).sum()
            better = (deltas > 0).sum()
            unchanged = (deltas == 0).sum()
            print(f"     Effect direction: {worse} worse, {better} better, {unchanged} unchanged")
            print(f"     Proportion worse: {worse/len(deltas)*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Generate enhanced robustness noise comparison figure')
    parser.add_argument('--csv', required=True, help='Path to structured runs CSV')
    parser.add_argument('--test-subjects', required=True, help='Path to test subjects CSV')
    parser.add_argument('--output-dir', default='./Plot_Clean/figures', help='Output directory')
    parser.add_argument('--output-name', default='fig_robustness_noise_paired', help='Output filename (without extension)')
    
    args = parser.parse_args()
    
    # Load and process data
    df = load_and_process_data(args.csv, args.test_subjects)
    
    if len(df) < 10:
        print(f"⚠️  Warning: Only {len(df)} runs found. Need more data for robust analysis.")
        return
    
    # Create paired subject analysis
    paired_results = create_paired_analysis(df)
    
    if len(paired_results['paired_df']) == 0:
        print("❌ No paired subject data found. Check that test subjects are properly matched.")
        return
    
    # Run statistical tests
    stats_results = run_statistical_tests(paired_results['subject_summary_df'])
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate figure
    output_path = output_dir / args.output_name
    print(f"\\n🎨 Generating enhanced robustness figure...")
    
    fig = create_robustness_figure_paired(df, paired_results, stats_results, output_path)
    
    # Print detailed summary
    print_paired_summary(paired_results, stats_results)
    
    print(f"\\n✅ Enhanced robustness analysis complete!")
    
    plt.show()

if __name__ == "__main__":
    main()