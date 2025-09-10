#!/usr/bin/env python3
# fig_freeze_comparison.py
"""
Freeze vs Unfreeze Backbone Comparison from CSV

Compares performance between frozen backbone (feature extractor only) 
and unfrozen backbone (full fine-tuning) training approaches.

Shows:
- Box plots comparing distributions
- Individual data points
- Statistical significance testing
- Performance metrics summary

CSV requirements:
  - Produced by your loader "flat" step (e.g., all_runs_flat.csv)
  - Should contain:
      * metrics: 'sum.test_kappa' OR 'contract.results.test_kappa'
      * backbone state: 'contract.model.frozen'
      * subject id: cfg.subject_id / cfg.subj_id / cfg.subject / sum.subject_id / name

Usage:
  python Plot_Clean/plot_freeze_comparison.py --csv ../data/all_runs_flat.csv --out ../figures/
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

# Import consistent figure styling
import sys; sys.path.append("../style"); from figure_style import (
    setup_figure_style, get_color, save_figure, 
    bootstrap_ci_median, wilcoxon_test,
    OKABE_ITO
)

# Named roles for key elements using consistent colors
CB_COLORS = {
    "frozen": get_color("yasa"),      # orange - for frozen backbone
    "unfrozen": get_color("cbramod"), # blue - for unfrozen backbone
    "significance": get_color("t_star"), # green - for significance indicators
}


def setup_plotting_style():
    setup_figure_style()

# Remove duplicate bootstrap_ci_median function - using the one import sys; sys.path.append("../style"); from figure_style.py


def derive_subject_id(row: pd.Series) -> str:
    for c in ["cfg.subject_id", "cfg.subj_id", "cfg.subject", "sum.subject_id", "subject_id", "name"]:
        if c in row and pd.notna(row[c]) and str(row[c]).strip() != "":
            return str(row[c])
    return str(row.get("name", "unknown"))


def load_and_prepare(csv_path: Path, num_classes: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    
    # Check if this is the pre-processed backbone comparison data
    if "backbone_frozen" in df.columns:
        print("Using pre-processed backbone comparison data")
        # Data already has the right format
        if "subject" not in df.columns:
            df["subject"] = df["subject_id"] if "subject_id" in df.columns else df["name"]
    else:
        # Normalize metric names - prefer sum.test_kappa, fallback to contract.results.test_kappa
        if "test_kappa" not in df.columns:
            if "sum.test_kappa" in df.columns:
                df["test_kappa"] = df["sum.test_kappa"]
            elif "contract.results.test_kappa" in df.columns:
                df["test_kappa"] = df["contract.results.test_kappa"]
        
        if "num_classes" not in df.columns:
            alt = "contract.results.num_classes"
            if alt in df.columns:
                df = df.rename(columns={alt: "num_classes"})
            else:
                alt2 = "cfg.num_of_classes"
                if alt2 in df.columns:
                    df = df.rename(columns={alt2: "num_classes"})
        
        # Add subject ID and backbone frozen state
        df["subject"] = df.apply(derive_subject_id, axis=1)
        
        # Get frozen state - ensure it exists
        if "contract.model.frozen" not in df.columns:
            print("Warning: contract.model.frozen column not found, assuming all unfrozen")
            df["backbone_frozen"] = False
        else:
            df["backbone_frozen"] = df["contract.model.frozen"].fillna(False)
    
    print(f"Available columns: {list(df.columns)}")
    print(f"test_kappa available for {df['test_kappa'].notna().sum()} rows")
    print(f"backbone_frozen distribution:\n{df['backbone_frozen'].value_counts(dropna=False)}")
    
    if 'num_classes' in df.columns:
        print(f"num_classes distribution:\n{df['num_classes'].value_counts(dropna=False)}")
    
    # Filter basic criteria
    df = df[df["test_kappa"].notna() & df["subject"].notna()]
    
    if num_classes is not None and 'num_classes' in df.columns:
        df = df[df["num_classes"] == num_classes]
        print(f"After filtering to {num_classes} classes: {len(df)} rows")
    
    print(f"After filtering (valid kappa, subject): {len(df)} rows")
    
    # Select top 10 runs from each group (frozen vs unfrozen)
    print(f"\nSelecting top 10 runs from each backbone training strategy...")
    
    frozen_df = df[df["backbone_frozen"] == True]
    unfrozen_df = df[df["backbone_frozen"] == False]
    
    print(f"Available runs - Frozen: {len(frozen_df)}, Unfrozen: {len(unfrozen_df)}")
    
    # Take top 10 (or all if fewer than 10) from each group
    if len(frozen_df) > 10:
        frozen_top = frozen_df.nlargest(10, 'test_kappa')
        print(f"Selected top 10 frozen backbone runs (from {len(frozen_df)} total)")
    else:
        frozen_top = frozen_df
        print(f"Using all {len(frozen_df)} frozen backbone runs")
    
    if len(unfrozen_df) > 10:
        unfrozen_top = unfrozen_df.nlargest(10, 'test_kappa')
        print(f"Selected top 10 unfrozen backbone runs (from {len(unfrozen_df)} total)")
    else:
        unfrozen_top = unfrozen_df
        print(f"Using all {len(unfrozen_df)} unfrozen backbone runs")
    
    # Combine the top runs
    df_filtered = pd.concat([frozen_top, unfrozen_top], ignore_index=True)
    
    print(f"Final dataset: {len(df_filtered)} runs ({len(frozen_top)} frozen + {len(unfrozen_top)} unfrozen)")
    
    if len(frozen_top) > 0:
        print(f"  Frozen backbone - κ range: {frozen_top['test_kappa'].min():.3f} to {frozen_top['test_kappa'].max():.3f}")
    if len(unfrozen_top) > 0:
        print(f"  Unfrozen backbone - κ range: {unfrozen_top['test_kappa'].min():.3f} to {unfrozen_top['test_kappa'].max():.3f}")
    
    return df_filtered


def statistical_comparison(frozen_data: np.ndarray, unfrozen_data: np.ndarray) -> Dict:
    """Perform statistical comparison between frozen and unfrozen groups."""
    
    frozen_data = frozen_data[~np.isnan(frozen_data)]
    unfrozen_data = unfrozen_data[~np.isnan(unfrozen_data)]
    
    results = {
        'frozen_n': len(frozen_data),
        'unfrozen_n': len(unfrozen_data),
        'frozen_mean': np.mean(frozen_data) if len(frozen_data) > 0 else np.nan,
        'unfrozen_mean': np.mean(unfrozen_data) if len(unfrozen_data) > 0 else np.nan,
        'frozen_std': np.std(frozen_data) if len(frozen_data) > 0 else np.nan,
        'unfrozen_std': np.std(unfrozen_data) if len(unfrozen_data) > 0 else np.nan,
        'frozen_median': np.median(frozen_data) if len(frozen_data) > 0 else np.nan,
        'unfrozen_median': np.median(unfrozen_data) if len(unfrozen_data) > 0 else np.nan,
    }
    
    # Statistical tests
    if len(frozen_data) >= 1 and len(unfrozen_data) >= 1:
        # Mann-Whitney U test (non-parametric)
        try:
            u_stat, p_value = stats.mannwhitneyu(unfrozen_data, frozen_data, alternative='two-sided')
            results['mann_whitney_u'] = u_stat
            results['mann_whitney_p'] = p_value
        except Exception:
            results['mann_whitney_u'] = np.nan
            results['mann_whitney_p'] = np.nan
        
        # Effect size (Cohen's d)
        if len(frozen_data) >= 2 and len(unfrozen_data) >= 2:
            pooled_std = np.sqrt(((len(unfrozen_data) - 1) * np.var(unfrozen_data, ddof=1) + 
                                 (len(frozen_data) - 1) * np.var(frozen_data, ddof=1)) / 
                                (len(unfrozen_data) + len(frozen_data) - 2))
            if pooled_std > 0:
                cohens_d = (results['unfrozen_mean'] - results['frozen_mean']) / pooled_std
                results['cohens_d'] = cohens_d
            else:
                results['cohens_d'] = np.nan
        else:
            results['cohens_d'] = np.nan
    else:
        results['mann_whitney_u'] = np.nan
        results['mann_whitney_p'] = np.nan
        results['cohens_d'] = np.nan
    
    return results


def plot_comparison(df: pd.DataFrame, outdir: Path, num_classes: Optional[int] = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Separate data
    frozen_data = df[df["backbone_frozen"] == True]["test_kappa"].values
    unfrozen_data = df[df["backbone_frozen"] == False]["test_kappa"].values
    
    # Statistical comparison
    stats_results = statistical_comparison(frozen_data, unfrozen_data)
    
    # Plot 1: Box plot comparison
    ax1.set_title("Performance Comparison: Frozen vs Unfrozen Backbone\n(Top 10 runs per strategy)", fontweight="bold", pad=20)
    
    box_data = [frozen_data, unfrozen_data]
    box_labels = [f"Frozen\n(n={stats_results['frozen_n']})", 
                  f"Unfrozen\n(n={stats_results['unfrozen_n']})"]
    
    bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True,
                     boxprops=dict(facecolor='lightgray', alpha=0.7),
                     medianprops=dict(color='black', linewidth=2))
    
    # Color the boxes
    bp['boxes'][0].set_facecolor(CB_COLORS["frozen"])
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(CB_COLORS["unfrozen"])
    bp['boxes'][1].set_alpha(0.7)
    
    # Add individual points with jitter
    for i, data in enumerate(box_data):
        if len(data) > 0:
            x_pos = i + 1
            jitter = np.random.normal(0, 0.05, len(data))
            ax1.scatter(x_pos + jitter, data, alpha=0.6, s=50, 
                       color=CB_COLORS["frozen"] if i == 0 else CB_COLORS["unfrozen"])
    
    ax1.set_ylabel("Cohen's κ", fontweight="bold")
    ax1.set_xlabel("Backbone Training Mode", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    
    # Add statistical significance annotation
    if not np.isnan(stats_results['mann_whitney_p']):
        y_max = max(np.max(frozen_data) if len(frozen_data) > 0 else 0,
                    np.max(unfrozen_data) if len(unfrozen_data) > 0 else 0)
        y_pos = y_max + 0.02
        
        if stats_results['mann_whitney_p'] < 0.001:
            sig_text = "***"
        elif stats_results['mann_whitney_p'] < 0.01:
            sig_text = "**"
        elif stats_results['mann_whitney_p'] < 0.05:
            sig_text = "*"
        else:
            sig_text = "ns"
        
        ax1.plot([1, 2], [y_pos, y_pos], 'k-', linewidth=1)
        ax1.text(1.5, y_pos + 0.01, sig_text, ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Detailed statistics table
    ax2.axis('off')
    ax2.set_title("Statistical Summary", fontweight="bold", pad=20)
    
    # Create statistics table
    table_data = [
        ["Metric", "Frozen Backbone", "Unfrozen Backbone"],
        ["N", f"{stats_results['frozen_n']}", f"{stats_results['unfrozen_n']}"],
        ["Mean κ", f"{stats_results['frozen_mean']:.3f}" if not np.isnan(stats_results['frozen_mean']) else "N/A", 
         f"{stats_results['unfrozen_mean']:.3f}" if not np.isnan(stats_results['unfrozen_mean']) else "N/A"],
        ["Median κ", f"{stats_results['frozen_median']:.3f}" if not np.isnan(stats_results['frozen_median']) else "N/A",
         f"{stats_results['unfrozen_median']:.3f}" if not np.isnan(stats_results['unfrozen_median']) else "N/A"],
        ["Std Dev", f"{stats_results['frozen_std']:.3f}" if not np.isnan(stats_results['frozen_std']) else "N/A",
         f"{stats_results['unfrozen_std']:.3f}" if not np.isnan(stats_results['unfrozen_std']) else "N/A"],
        ["", "", ""],
        ["Statistical Test", "Value", "Interpretation"],
        ["Mann-Whitney U", f"{stats_results['mann_whitney_u']:.1f}" if not np.isnan(stats_results['mann_whitney_u']) else "N/A", ""],
        ["p-value", f"{stats_results['mann_whitney_p']:.4f}" if not np.isnan(stats_results['mann_whitney_p']) else "N/A", 
         "Significant" if not np.isnan(stats_results['mann_whitney_p']) and stats_results['mann_whitney_p'] < 0.05 else "Not Significant"],
        ["Cohen's d", f"{stats_results['cohens_d']:.3f}" if not np.isnan(stats_results['cohens_d']) else "N/A",
         "Large Effect" if not np.isnan(stats_results['cohens_d']) and abs(stats_results['cohens_d']) > 0.8 
         else "Medium Effect" if not np.isnan(stats_results['cohens_d']) and abs(stats_results['cohens_d']) > 0.5 
         else "Small Effect" if not np.isnan(stats_results['cohens_d']) and abs(stats_results['cohens_d']) > 0.2 
         else "Negligible"],
    ]
    
    # Create table
    table = ax2.table(cellText=table_data[1:], colLabels=table_data[0], 
                      cellLoc='center', loc='center',
                      colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#E8E8E8')
        elif i == 6:  # Separator row for statistical tests
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#F0F0F0')
    
    # Overall title
    title = (f"Backbone Training Strategy Comparison - Top 10 Runs ({num_classes}-class)"
             if num_classes is not None else
             "Backbone Training Strategy Comparison - Top 10 Runs (Mixed Classes)")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.95)
    
    plt.tight_layout()
    
    # Save plot
    outdir.mkdir(parents=True, exist_ok=True)
    fp_svg = outdir / "freeze_unfreeze_comparison.svg"
    plt.savefig(fp_svg)
    print(f"Saved: {fp_svg}")
    
    # Print summary
    print("\n" + "="*60)
    print("BACKBONE TRAINING STRATEGY COMPARISON (TOP 10 RUNS)")
    print("="*60)
    print(f"Frozen Backbone:   {stats_results['frozen_n']} runs, κ = {stats_results['frozen_mean']:.3f} ± {stats_results['frozen_std']:.3f}")
    print(f"Unfrozen Backbone: {stats_results['unfrozen_n']} runs, κ = {stats_results['unfrozen_mean']:.3f} ± {stats_results['unfrozen_std']:.3f}")
    
    if not np.isnan(stats_results['mann_whitney_p']):
        improvement = stats_results['unfrozen_mean'] - stats_results['frozen_mean']
        print(f"\nPerformance Difference: {improvement:+.3f} κ (unfrozen - frozen)")
        print(f"Statistical Significance: p = {stats_results['mann_whitney_p']:.4f}")
        print(f"Effect Size (Cohen's d): {stats_results['cohens_d']:.3f}")
        
        if stats_results['mann_whitney_p'] < 0.05:
            direction = "significantly better" if improvement > 0 else "significantly worse"
            print(f"\n🎯 CONCLUSION: Unfrozen backbone training performs {direction} than frozen backbone training")
        else:
            print(f"\n📊 CONCLUSION: No significant difference found between training strategies")
    else:
        print("\n⚠️ Insufficient data for statistical comparison")
    
    return fig, stats_results


def main():
    ap = argparse.ArgumentParser(description='Generate freeze vs unfreeze backbone comparison plot')
    ap.add_argument("--csv", required=True, help="Path to flattened CSV (e.g., all_runs_flat.csv)")
    ap.add_argument("--out", default="../figures/", help="Output directory")
    ap.add_argument("--classes", type=int, choices=[4, 5], help="Filter by number of classes")
    args = ap.parse_args()

    setup_plotting_style()
    
    print("Loading data and comparing frozen vs unfrozen backbone training...")
    df = load_and_prepare(Path(args.csv), num_classes=args.classes)
    
    if df.empty:
        print("No valid data found. Check your CSV file and filtering criteria.")
        return
    
    # Check if we have both frozen and unfrozen data
    frozen_count = (df["backbone_frozen"] == True).sum()
    unfrozen_count = (df["backbone_frozen"] == False).sum()
    
    if frozen_count == 0 and unfrozen_count == 0:
        print("No backbone training data found.")
        return
    elif frozen_count == 0:
        print("⚠️ Warning: No frozen backbone data found, showing unfrozen data only")
    elif unfrozen_count == 0:
        print("⚠️ Warning: No unfrozen backbone data found, showing frozen data only")
    
    plot_comparison(df, Path(args.out), num_classes=args.classes)

if __name__ == "__main__":
    main()