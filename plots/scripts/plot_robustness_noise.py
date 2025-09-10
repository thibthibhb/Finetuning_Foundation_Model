#!/usr/bin/env python3
"""
CBraMod Robustness Analysis: Noise Injection Study

Creates a publication-ready figure comparing model performance across
different noise conditions for ear-EEG sleep staging robustness evaluation.

Usage:
    python Plot_Clean/plot_robustness_noise.py --csv ../data/all_runs_flat.csv

Author: CBraMod Research Team
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
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

def load_and_process_data(csv_path: str) -> pd.DataFrame:
    """Load and process robustness data."""
    df = pd.read_csv(csv_path)
    
    # Filter for completed runs with valid results
    df = df[df['state'] == 'finished'].copy()
    df = df[df['contract.results.test_kappa'].notna()].copy()
    
    # Handle missing noise parameters (backward compatibility)
    if 'contract.noise.noise_level' not in df.columns:
        print("⚠️  Adding default noise parameters for older runs")
        df['contract.noise.noise_level'] = 0.0
        df['contract.noise.noise_type'] = 'clean'
    
    # Fill missing values
    df['contract.noise.noise_level'] = df['contract.noise.noise_level'].fillna(0.0)
    df['contract.noise.noise_type'] = df['contract.noise.noise_type'].fillna('clean')
    
    # Create clean labels for plotting
    df['noise_label'] = df.apply(lambda row: 
        'Clean' if row['contract.noise.noise_level'] == 0.0 
        else f"{row['contract.noise.noise_type'].title()} {int(row['contract.noise.noise_level']*100)}%", 
        axis=1)
    
    # Focus on specific noise levels for clear comparison
    target_levels = [0.0, 0.05, 0.10, 0.20]  # Clean, 5%, 10%, 20%
    df = df[df['contract.noise.noise_level'].isin(target_levels)].copy()
    
    # Focus on 4/5-class results for consistency
    df = df[df['contract.results.num_classes'].isin([4, 5])].copy()
    
    print(f"📊 Loaded {len(df)} valid runs for robustness analysis")
    
    return df

def create_robustness_figure(df: pd.DataFrame, output_path: Path):
    """Create the main robustness comparison figure."""
    
    # Set up consistent styling and figure with 3 subplots in a row
    setup_figure_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Color scheme - professional and colorblind-friendly using figure_style
    colors = {
        'Clean': get_color('cbramod'),     # Clean baseline
        'Gaussian': get_color('4_class'),  # Gaussian noise
        'EMG': get_color('yasa'),          # EMG artifacts
        'Movement': get_color('5_class'),  # Movement artifacts
        'Electrode': get_color('orp'),     # Electrode artifacts
        'Realistic': get_color('combined') # Realistic combined artifacts
    }
    
    # === SUBPLOT A: Performance vs Noise Level (Line Plot) ===
    ax1 = axes[0]
    
    # Group by noise type and level
    summary_stats = []
    for (noise_type, noise_level), group in df.groupby(['contract.noise.noise_type', 'contract.noise.noise_level']):
        kappas = group['contract.results.test_kappa']
        summary_stats.append({
            'noise_type': noise_type,
            'noise_level': noise_level,
            'noise_pct': int(noise_level * 100),
            'mean_kappa': kappas.mean(),
            'std_kappa': kappas.std(),
            'count': len(kappas)
        })
    
    stats_df = pd.DataFrame(summary_stats)
    
    # Plot each noise type
    noise_types = ['clean', 'gaussian', 'emg', 'movement', 'electrode', 'realistic']
    for noise_type in noise_types:
        type_data = stats_df[stats_df['noise_type'] == noise_type].sort_values('noise_level')
        if len(type_data) > 0:
            color_key = noise_type.title() if noise_type != 'clean' else 'Clean'
            ax1.errorbar(type_data['noise_pct'], type_data['mean_kappa'], 
                        yerr=type_data['std_kappa'], 
                        marker='o', linewidth=2, capsize=4,
                        label=color_key, color=colors.get(color_key, '#333333'),
                        markersize=6)
    
    ax1.set_xlabel('Noise Level (%)')
    ax1.set_ylabel("Cohen's κ")
    ax1.set_title('A) Robustness Curves', fontweight='bold', pad=15)
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.3, 0.8)  # Focus on relevant range
    
    # === SUBPLOT B: Performance Drop at 10% Level (Bar Chart) ===
    ax2 = axes[1]
    
    # Calculate performance drops relative to clean baseline
    baseline_kappa = stats_df[stats_df['noise_level'] == 0.0]['mean_kappa'].mean()
    noise_10pct = stats_df[stats_df['noise_level'] == 0.10].copy()
    
    if len(noise_10pct) > 0:
        noise_10pct['performance_drop'] = baseline_kappa - noise_10pct['mean_kappa']
        noise_10pct = noise_10pct.sort_values('performance_drop')
        
        bar_colors = [colors.get(nt.title(), '#333333') for nt in noise_10pct['noise_type']]
        bars = ax2.bar(range(len(noise_10pct)), noise_10pct['performance_drop'], 
                      color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax2.set_xticks(range(len(noise_10pct)))
        ax2.set_xticklabels([nt.title() for nt in noise_10pct['noise_type']], rotation=45)
        ax2.set_ylabel('Performance Drop (Δκ)')
        ax2.set_title('B) Artifact Impact (10% Level)', fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, drop in zip(bars, noise_10pct['performance_drop']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{drop:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # === SUBPLOT C: Distribution Comparison (Box Plot) ===
    ax3 = axes[2]
    
    # Prepare data for box plot - focus on clean vs most realistic conditions
    plot_conditions = []
    plot_data = []
    plot_colors = []
    
    # Clean baseline
    clean_data = df[df['contract.noise.noise_level'] == 0.0]['contract.results.test_kappa']
    if len(clean_data) > 0:
        plot_conditions.append('Clean')
        plot_data.append(clean_data.values)
        plot_colors.append(colors['Clean'])
    
    # 10% realistic noise (most representative)
    realistic_10 = df[(df['contract.noise.noise_level'] == 0.10) & 
                     (df['contract.noise.noise_type'] == 'realistic')]['contract.results.test_kappa']
    if len(realistic_10) > 0:
        plot_conditions.append('Realistic 10%')
        plot_data.append(realistic_10.values)
        plot_colors.append(colors['Realistic'])
    
    # 20% realistic noise (challenging condition)
    realistic_20 = df[(df['contract.noise.noise_level'] == 0.20) & 
                     (df['contract.noise.noise_type'] == 'realistic')]['contract.results.test_kappa']
    if len(realistic_20) > 0:
        plot_conditions.append('Realistic 20%')
        plot_data.append(realistic_20.values)
        plot_colors.append('#8B0000')  # Darker red for higher noise
    
    if len(plot_data) >= 2:
        box_plot = ax3.boxplot(plot_data, labels=plot_conditions, patch_artist=True,
                              boxprops=dict(linewidth=1), medianprops=dict(linewidth=2))
        
        # Color the boxes
        for patch, color in zip(box_plot['boxes'], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax3.set_ylabel("Cohen's κ")
    ax3.set_title('C) Performance Distribution', fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=45)
    
    # Overall figure adjustments
    plt.tight_layout()
    
    # Add overall title
    fig.suptitle('CBraMod Robustness to Ear-EEG Noise Artifacts', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.90)
    
    # Save figure using consistent save function
    saved_files = save_figure(fig, output_path)
    
    return fig

def print_robustness_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("ROBUSTNESS ANALYSIS SUMMARY")
    print("="*60)
    
    # Baseline performance
    clean_data = df[df['contract.noise.noise_level'] == 0.0]['contract.results.test_kappa']
    if len(clean_data) > 0:
        print(f"\n📊 Baseline (Clean): κ = {clean_data.mean():.4f} ± {clean_data.std():.4f} (n={len(clean_data)})")
    
    # Performance by noise type at 10% level
    print(f"\n🔊 Performance at 10% Noise Level:")
    noise_10 = df[df['contract.noise.noise_level'] == 0.10]
    baseline_mean = clean_data.mean()
    
    for noise_type in sorted(noise_10['contract.noise.noise_type'].unique()):
        subset = noise_10[noise_10['contract.noise.noise_type'] == noise_type]['contract.results.test_kappa']
        if len(subset) > 0:
            mean_perf = subset.mean()
            drop = baseline_mean - mean_perf
            drop_pct = drop / baseline_mean * 100
            print(f"   {noise_type.title():>10}: κ = {mean_perf:.4f} (drop: {drop:.4f}, {drop_pct:5.1f}%, n={len(subset)})")
    
    # Data availability
    print(f"\n📈 Data Summary:")
    noise_summary = df.groupby(['contract.noise.noise_level', 'contract.noise.noise_type']).size().unstack(fill_value=0)
    print(noise_summary)

def main():
    parser = argparse.ArgumentParser(description='Generate robustness noise comparison figure')
    parser.add_argument('--csv', required=True, help='Path to structured runs CSV')
    parser.add_argument('--output-dir', default='./Plot_Clean/figures', help='Output directory')
    parser.add_argument('--output-name', default='robustness_noise_analysis', help='Output filename (without extension)')
    
    args = parser.parse_args()
    
    # Load and process data
    df = load_and_process_data(args.csv)
    
    if len(df) < 10:
        print(f"⚠️  Warning: Only {len(df)} runs found. Need more data for robust analysis.")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate figure
    output_path = output_dir / args.output_name
    print(f"\n🎨 Generating robustness figure...")
    
    fig = create_robustness_figure(df, output_path)
    
    # Print summary statistics
    print_robustness_summary(df)
    
    print(f"\n✅ Robustness analysis complete!")
    
    # Print N information for caption
    n_subjects = df['contract.dataset.num_subjects_train'].nunique() if 'contract.dataset.num_subjects_train' in df.columns else len(df)
    n_runs = len(df)
    print(f"\n📋 Caption info: {format_n_caption(n_subjects, n_runs, 'runs')}")
    
    plt.show()

if __name__ == "__main__":
    main()