#!/usr/bin/env python3
"""
Figure 4: Subjects vs Minutes per Subject - What Matters More?

Shows how spreading calibration data across subjects affects performance.
Addresses the practical question: "Is it better to have more subjects with less data each,
or fewer subjects with more data each?"

Uses real W&B data: hours_of_data / num_subjects_train to calculate minutes per subject.

Usage:
  python fig4_subjects_vs_minutes.py --csv Plot_Clean/data/all_runs_flat.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import bootstrap
from scipy.interpolate import griddata
import argparse
import warnings
from matplotlib import cycler

warnings.filterwarnings("ignore")

# Colorblind-friendly palette (Okabe–Ito)
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00", 
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}

def setup_plotting_style():
    """Configure matplotlib for publication-ready plots."""
    plt.style.use("default")
    plt.rcParams.update({
        "figure.figsize": (16, 10),
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.prop_cycle": cycler(color=list(OKABE_ITO.values())),
        # Key for overlap management:
        "figure.constrained_layout.use": True,
    })


def bootstrap_ci_median(arr: np.ndarray, confidence: float = 0.95) -> tuple:
    """Calculate bootstrap CI for median."""
    if len(arr) < 3:
        return np.nan, np.nan
    
    try:
        def median_stat(x):
            return np.median(x)
        
        rng = np.random.default_rng(42)
        res = bootstrap((arr,), median_stat, n_resamples=1000, 
                       confidence_level=confidence, random_state=rng)
        return res.confidence_interval.low, res.confidence_interval.high
    except:
        return np.nan, np.nan

def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV data and calculate minutes per subject."""
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} rows")
    
    # Create convenience aliases
    if "test_kappa" not in df.columns:
        if "contract.results.test_kappa" in df.columns:
            df["test_kappa"] = df["contract.results.test_kappa"]
        elif "sum.test_kappa" in df.columns:
            df["test_kappa"] = df["sum.test_kappa"]
    
    # Get hours of data
    if "sum.hours_of_data" in df.columns:
        df["hours_of_data"] = df["sum.hours_of_data"]
    elif "contract.results.hours_of_data" in df.columns:
        df["hours_of_data"] = df["contract.results.hours_of_data"]
    
    # Get number of subjects
    if "contract.dataset.num_subjects_train" in df.columns:
        df["num_subjects"] = df["contract.dataset.num_subjects_train"]
    
    # Filter to valid data
    valid_mask = (
        df["test_kappa"].notna() & 
        df["hours_of_data"].notna() & 
        df["num_subjects"].notna() & 
        (df["num_subjects"] > 0) & 
        (df["hours_of_data"] > 0) &
        (df["test_kappa"] >= 0.3)  # Remove poor performers
    )
    
    # Filter to 5-class if available
    if "contract.results.num_classes" in df.columns:
        valid_mask = valid_mask & (df["contract.results.num_classes"] == 5)
    elif "cfg.num_of_classes" in df.columns:
        valid_mask = valid_mask & (df["cfg.num_of_classes"] == 5)
    
    df = df[valid_mask]
    print(f"After filtering to valid 5-class data (κ≥0.3): {len(df)} runs")
    
    # Calculate derived metrics
    df["total_hours"] = df["hours_of_data"]
    df["hours_per_subject"] = df["hours_of_data"] / df["num_subjects"]
    df["minutes_per_subject"] = df["hours_per_subject"] * 60
    
    print(f"\nData summary:")
    print(f"Number of subjects range: {df['num_subjects'].min()}-{df['num_subjects'].max()}")
    print(f"Total hours range: {df['total_hours'].min():.1f}-{df['total_hours'].max():.1f}")
    print(f"Hours per subject range: {df['hours_per_subject'].min():.1f}-{df['hours_per_subject'].max():.1f}")
    print(f"Minutes per subject range: {df['minutes_per_subject'].min():.1f}-{df['minutes_per_subject'].max():.1f}")
    
    return df

def select_top_runs_per_bin(df: pd.DataFrame, column: str, bin_size: float, n_top: int = 10) -> pd.DataFrame:
    """Select top N runs per bin based on test_kappa."""
    
    # Create bins
    min_val = df[column].min()
    max_val = df[column].max()
    bin_edges = np.arange(min_val, max_val + bin_size, bin_size)
    
    df['bin'] = pd.cut(df[column], bins=bin_edges, include_lowest=True)
    
    # Select top runs per bin
    top_runs_list = []
    
    for bin_val, group in df.groupby('bin', observed=True):
        if len(group) == 0:
            continue
        
        # Take top n_top runs by test_kappa
        if len(group) > n_top:
            top_runs = group.nlargest(n_top, 'test_kappa')
        else:
            top_runs = group
        
        top_runs_list.append(top_runs)
        print(f"Bin {bin_val}: {len(group)} runs → selected top {len(top_runs)}")
    
    if top_runs_list:
        result = pd.concat(top_runs_list, ignore_index=True)
        # Remove the temporary bin column
        result = result.drop('bin', axis=1, errors='ignore')
        return result
    else:
        return pd.DataFrame()

def create_subjects_minutes_heatmap(df: pd.DataFrame, yasa_kappa: float = 0.446):
    """Create heatmap data for subjects vs minutes per subject."""
    
    # Define grid ranges based on actual data
    subjects_range = np.arange(1, int(df['num_subjects'].max()) + 1)
    minutes_range = np.arange(5, int(df['minutes_per_subject'].max()) + 5, 5)
    
    print(f"Creating heatmap grid: {len(subjects_range)} subjects × {len(minutes_range)} minutes/subject")
    
    # Create pivot table with median delta kappa
    df['delta_kappa'] = df['test_kappa'] - yasa_kappa
    
    # Bin the continuous data
    df['subjects_bin'] = pd.cut(df['num_subjects'], bins=np.arange(0.5, df['num_subjects'].max() + 1.5, 1))
    df['minutes_bin'] = pd.cut(df['minutes_per_subject'], bins=np.arange(0, df['minutes_per_subject'].max() + 10, 10))
    
    # Calculate median delta kappa for each bin combination
    heatmap_data = df.groupby(['subjects_bin', 'minutes_bin']).agg({
        'delta_kappa': ['median', 'count'],
        'test_kappa': ['median']
    }).reset_index()
    
    # Flatten column names
    heatmap_data.columns = ['subjects_bin', 'minutes_bin', 'delta_median', 'count', 'kappa_median']
    
    # Extract bin centers
    heatmap_data['subjects'] = heatmap_data['subjects_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan)
    heatmap_data['minutes'] = heatmap_data['minutes_bin'].apply(lambda x: x.mid if pd.notna(x) else np.nan)
    
    # Filter to bins with sufficient data
    heatmap_data = heatmap_data[heatmap_data['count'] >= 3]
    
    return heatmap_data

def create_figure_4(df: pd.DataFrame, output_dir: Path):
    """Create Figure 4: Subjects vs Minutes per Subject analysis."""
    
    yasa_kappa = 0.446
    delta_threshold = 0.05
    
    print("Creating subjects vs minutes per subject analysis...")
    
    if df.empty:
        print("ERROR: No valid data found!")
        return None
    
    # Calculate delta kappa
    df['delta_kappa'] = df['test_kappa'] - yasa_kappa
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1])

    fig.suptitle(
        'Figure 4: Subjects vs Minutes per Subject - What Matters More?\n'
        'Analysis of Calibration Data Distribution Strategy',
        fontweight='bold', fontsize=16
    )
    # Main heatmap - top spanning both columns
    ax_main = fig.add_subplot(gs[0, :])

    # Create heatmap data
    heatmap_data = create_subjects_minutes_heatmap(df, yasa_kappa)

    if not heatmap_data.empty:
        # Pivot to matrix
        pivot_data = heatmap_data.pivot(index='subjects', columns='minutes', values='delta_median')

        # Decide whether to annotate based on grid size
        n_cells = pivot_data.shape[0] * pivot_data.shape[1]
        do_annot = n_cells <= 120  # annotate only if not too dense

        hm = sns.heatmap(
            pivot_data,
            cmap='RdYlBu_r',
            center=0,
            annot=do_annot,
            fmt='.2f' if do_annot else '',
            annot_kws={"size": 9} if do_annot else None,
            cbar_kws={'label': 'Δκ vs YASA', 'shrink': 0.85, 'pad': 0.02},
            linewidths=0.5,
            linecolor=(0,0,0,0.08),
            ax=ax_main,
            rasterized=True
        )

        # Ticks: rotate x, make y nice integers
        ax_main.set_xlabel('Minutes per Subject', fontweight='bold')
        ax_main.set_ylabel('Number of Subjects', fontweight='bold')
        ax_main.set_title('A: Performance Heatmap - Subjects vs Minutes per Subject', fontweight='bold', pad=10)
        ax_main.set_xticklabels(ax_main.get_xticklabels(), rotation=35, ha='right')

        # If subjects are nearly integers, force as int labels
        yticklabs = []
        for t in ax_main.get_yticks():
            try:
                yticklabs.append(f"{int(round(float(t)))}")
            except Exception:
                yticklabs.append(str(t))
        ax_main.set_yticklabels(yticklabs)

        # Optional contours (only if matrix is big enough and mostly finite)
        Z = pivot_data.values
        if pivot_data.shape[0] >= 3 and pivot_data.shape[1] >= 3 and np.isfinite(Z).sum() >= 9:
            try:
                X, Y = np.meshgrid(pivot_data.columns.astype(float), pivot_data.index.astype(float))
                mask = np.isfinite(Z)
                if mask.sum() >= 9:
                    ax_main.contour(X, Y, Z, levels=[0], colors=[OKABE_ITO['vermillion']], linestyles='--', linewidths=2)
                    if np.nanmax(Z) > delta_threshold:
                        ax_main.contour(X, Y, Z, levels=[delta_threshold], colors=[OKABE_ITO['green']], linestyles='-', linewidths=2.5)
            except Exception:
                pass

        # Add contour lines for key thresholds if we have enough data
        if pivot_data.shape[0] > 2 and pivot_data.shape[1] > 2:
            try:
                X, Y = np.meshgrid(pivot_data.columns, pivot_data.index)
                Z = pivot_data.values
                
                # Remove NaN for contouring
                mask = ~np.isnan(Z)
                if mask.sum() > 4:  # Need at least some valid points
                    # Δκ = 0 (tie with YASA) - vermillion  
                    contour_zero = ax_main.contour(X, Y, Z, levels=[0], 
                                                 colors=[OKABE_ITO['vermillion']], 
                                                 linestyles='--', linewidths=2)
                    
                    # Δκ = threshold (target) - green
                    if np.nanmax(Z) > delta_threshold:
                        contour_target = ax_main.contour(X, Y, Z, levels=[delta_threshold], 
                                                       colors=[OKABE_ITO['green']], 
                                                       linestyles='-', linewidths=3)
            except:
                print("Could not add contour lines to heatmap")
    
    # Panel B: Scatter plot - Performance vs Total Data Hours (binned, top 10 per 100h bin)
    ax1 = fig.add_subplot(gs[1, 0])
    
    # Select top 10 runs per 100-hour bin
    df_total_hours = select_top_runs_per_bin(df.copy(), 'total_hours', 100, n_top=10)
    print(f"Panel B: {len(df_total_hours)} top runs selected from total hours bins")
    
    if not df_total_hours.empty:
        # slight jitter function to reduce vertical stacking
        rng = np.random.default_rng(42)
        jitter = (rng.standard_normal(len(df_total_hours)) * 0.005)

        scatter = ax1.scatter(
            df_total_hours['total_hours'],
            df_total_hours['delta_kappa'] + jitter,
            c=df_total_hours['num_subjects'],
            s=50, alpha=0.75,  # Larger, more visible points
            cmap='viridis',
            edgecolors='black', linewidth=0.6
        )

        cbar1 = plt.colorbar(scatter, ax=ax1, pad=0.01, shrink=0.9)
        cbar1.set_label('Number of Subjects', fontweight='bold')

    ax1.axhline(y=0, color=OKABE_ITO['vermillion'], linestyle='--', linewidth=2, label='YASA baseline')
    ax1.axhline(y=delta_threshold, color=OKABE_ITO['green'], linestyle='-', linewidth=2, label=f'Target (+{delta_threshold})')

    ax1.set_xlabel('Total Hours of Data', fontweight='bold')
    ax1.set_ylabel('Δκ vs YASA', fontweight='bold')
    ax1.set_title('B: Performance vs Total Data\n(Top 10 per 100h bin)', fontweight='bold')
    ax1.legend(handlelength=1.8, borderpad=0.4)
    ax1.grid(True, alpha=0.3)
    ax1.margins(x=0.03, y=0.1)

    
    # Panel C: Minutes per subject vs Performance (binned, top 10 per 100min bin)
    ax2 = fig.add_subplot(gs[1, 1])
    
    # Select top 10 runs per 100-minute bin
    df_minutes = select_top_runs_per_bin(df.copy(), 'minutes_per_subject', 100, n_top=10)
    print(f"Panel C: {len(df_minutes)} top runs selected from minutes per subject bins")
    
    if not df_minutes.empty:
        jitter = (rng.standard_normal(len(df_minutes)) * 0.005)
        ax2.scatter(
            df_minutes['minutes_per_subject'],
            df_minutes['delta_kappa'] + jitter,
            c=OKABE_ITO['blue'],
            s=50, alpha=0.75,  # Larger, more visible points
            edgecolors='black', linewidth=0.6
        )

    ax2.axhline(y=0, color=OKABE_ITO['vermillion'], linestyle='--', linewidth=2)
    ax2.axhline(y=delta_threshold, color=OKABE_ITO['green'], linestyle='-', linewidth=2)

    ax2.set_xlabel('Minutes per Subject', fontweight='bold')
    ax2.set_ylabel('Δκ vs YASA', fontweight='bold')
    ax2.set_title('C: Performance vs Minutes per Subject\n(Top 10 per 100min bin)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.margins(x=0.03, y=0.1)

    
    # Panel D: Number of subjects vs Performance (binned, top 10 per 3-subject bin)
    ax3 = fig.add_subplot(gs[2, 0])
    
    # Select top 10 runs per 3-subject bin
    df_subjects = select_top_runs_per_bin(df.copy(), 'num_subjects', 3, n_top=10)
    print(f"Panel D: {len(df_subjects)} top runs selected from subject count bins")
    
    if not df_subjects.empty:
        jitter = (rng.standard_normal(len(df_subjects)) * 0.005)
        ax3.scatter(
            df_subjects['num_subjects'],
            df_subjects['delta_kappa'] + jitter,
            c=OKABE_ITO['orange'],
            s=50, alpha=0.75,  # Larger, more visible points
            edgecolors='black', linewidth=0.6
        )

    ax3.axhline(y=0, color=OKABE_ITO['vermillion'], linestyle='--', linewidth=2)
    ax3.axhline(y=delta_threshold, color=OKABE_ITO['green'], linestyle='-', linewidth=2)

    ax3.set_xlabel('Number of Subjects', fontweight='bold')
    ax3.set_ylabel('Δκ vs YASA', fontweight='bold')
    ax3.set_title('D: Performance vs Number of Subjects\n(Top 10 per 3-subject bin)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.margins(x=0.03, y=0.1)

    
    # Panel E: Box plots by subject count strategy (using binned top performers)
    ax4 = fig.add_subplot(gs[2, 1])

    # Use the same binned data for consistent analysis
    if not df_subjects.empty:
        subject_counts = sorted(df_subjects['num_subjects'].unique())
        
        if len(subject_counts) > 1:
            if len(subject_counts) > 8:
                cats = pd.cut(df_subjects['num_subjects'], bins=min(8, len(subject_counts)))
                df_subjects['subject_category'] = cats
                categories = cats.cat.categories
                box_data = [df_subjects.loc[df_subjects['subject_category'] == cat, 'delta_kappa'].values for cat in categories]
                labels = [f'{int(c.left)}–{int(c.right)}' for c in categories]
            else:
                box_data = [df_subjects.loc[df_subjects['num_subjects'] == c, 'delta_kappa'].values for c in subject_counts]
                labels = [f'{int(c)}' for c in subject_counts]

            # Filter out empty boxes
            box_data_filtered = []
            labels_filtered = []
            for data, label in zip(box_data, labels):
                if len(data) > 0:
                    box_data_filtered.append(data)
                    labels_filtered.append(label)
            
            if box_data_filtered:
                bp = ax4.boxplot(box_data_filtered, labels=labels_filtered, patch_artist=True)

                palette = [OKABE_ITO['blue'], OKABE_ITO['orange'], OKABE_ITO['green'],
                        OKABE_ITO['purple'], OKABE_ITO['sky'], OKABE_ITO['yellow']]
                for i, patch in enumerate(bp['boxes']):
                    patch.set_facecolor(palette[i % len(palette)])
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('black')

                ax4.axhline(y=0, color=OKABE_ITO['vermillion'], linestyle='--', linewidth=2)
                ax4.axhline(y=delta_threshold, color=OKABE_ITO['green'], linestyle='-', linewidth=2)

                ax4.set_xlabel('Number of Subjects', fontweight='bold')
                ax4.set_ylabel('Δκ vs YASA', fontweight='bold')
                ax4.set_title('E: Top Performers by\nSubject Count', fontweight='bold')
                ax4.grid(True, alpha=0.3)

                ax4.tick_params(axis='x', labelrotation=30)
                ax4.margins(x=0.02, y=0.1)

                # Rotate labels if needed
                if len(labels_filtered) > 6:
                    ax4.tick_params(axis='x', rotation=45)
    
    #plt.tight_layout()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_svg = output_dir / 'figure_4_subjects_vs_minutes.svg'
    plt.savefig(fig_svg)
    print(f"\nSaved: {fig_svg}")
    
    # Analysis summary
    print(f"\nSubjects vs Minutes Analysis Summary (Top Performers Only):")
    print(f"Original runs (κ≥0.3): {len(df)}")
    print(f"Panel B top runs: {len(df_total_hours) if 'df_total_hours' in locals() and not df_total_hours.empty else 0}")
    print(f"Panel C top runs: {len(df_minutes) if 'df_minutes' in locals() and not df_minutes.empty else 0}")
    print(f"Panel D top runs: {len(df_subjects) if 'df_subjects' in locals() and not df_subjects.empty else 0}")
    
    # Find optimal strategies from the filtered top performers
    # Combine all binned datasets for comprehensive analysis
    all_top_performers = []
    if 'df_total_hours' in locals() and not df_total_hours.empty:
        all_top_performers.append(df_total_hours)
    if 'df_minutes' in locals() and not df_minutes.empty:
        all_top_performers.append(df_minutes)
    if 'df_subjects' in locals() and not df_subjects.empty:
        all_top_performers.append(df_subjects)
    
    if all_top_performers:
        # Remove duplicates by run_id if available, otherwise by all columns
        combined_top = pd.concat(all_top_performers, ignore_index=True)
        if 'run_id' in combined_top.columns:
            combined_top = combined_top.drop_duplicates(subset='run_id')
        else:
            combined_top = combined_top.drop_duplicates()
        
        beating_yasa = combined_top[combined_top['delta_kappa'] > 0]
        target_achieved = combined_top[combined_top['delta_kappa'] >= delta_threshold]
        
        analysis_df = combined_top
        print(f"\nCombined top performers analysis: {len(combined_top)} unique runs")
    else:
        beating_yasa = df[df['delta_kappa'] > 0]
        target_achieved = df[df['delta_kappa'] >= delta_threshold]
        analysis_df = df
        print(f"\nFallback to full dataset analysis: {len(df)} runs")
    
    if not beating_yasa.empty:
        print(f"\nStrategies beating YASA: {len(beating_yasa)}/{len(df)} runs ({len(beating_yasa)/len(df)*100:.1f}%)")
        
        # Best overall performance from top performers
        best_idx = analysis_df['delta_kappa'].idxmax()
        best_run = analysis_df.loc[best_idx]
        print(f"\nBest performance:")
        print(f"  {best_run['num_subjects']:.0f} subjects × {best_run['minutes_per_subject']:.1f} min/subject")
        print(f"  Total: {best_run['total_hours']:.1f} hours")
        print(f"  Performance: Δκ = +{best_run['delta_kappa']:.3f}")
        
        # Most efficient (least total time while beating YASA)
        most_efficient_idx = beating_yasa['total_hours'].idxmin()
        efficient_run = beating_yasa.loc[most_efficient_idx]
        print(f"\nMost efficient (minimal total time beating YASA):")
        print(f"  {efficient_run['num_subjects']:.0f} subjects × {efficient_run['minutes_per_subject']:.1f} min/subject")
        print(f"  Total: {efficient_run['total_hours']:.1f} hours")
        print(f"  Performance: Δκ = +{efficient_run['delta_kappa']:.3f}")
        
        if not target_achieved.empty:
            target_efficient_idx = target_achieved['total_hours'].idxmin()
            target_run = target_achieved.loc[target_efficient_idx]
            print(f"\nMinimal time achieving target (+{delta_threshold}):")
            print(f"  {target_run['num_subjects']:.0f} subjects × {target_run['minutes_per_subject']:.1f} min/subject")
            print(f"  Total: {target_run['total_hours']:.1f} hours")
            print(f"  Performance: Δκ = +{target_run['delta_kappa']:.3f}")
    
    # Correlation analysis using top performers
    if 'analysis_df' in locals() and not analysis_df.empty:
        corr_subjects = np.corrcoef(analysis_df['num_subjects'], analysis_df['delta_kappa'])[0,1]
        corr_minutes_per_subject = np.corrcoef(analysis_df['minutes_per_subject'], analysis_df['delta_kappa'])[0,1]
        corr_total = np.corrcoef(analysis_df['total_hours'], analysis_df['delta_kappa'])[0,1]
    else:
        corr_subjects = np.corrcoef(df['num_subjects'], df['delta_kappa'])[0,1]
        corr_minutes_per_subject = np.corrcoef(df['minutes_per_subject'], df['delta_kappa'])[0,1]
        corr_total = np.corrcoef(df['total_hours'], df['delta_kappa'])[0,1]
    
    print(f"\nCorrelations with performance improvement:")
    print(f"  Number of subjects: {corr_subjects:.3f}")
    print(f"  Minutes per subject: {corr_minutes_per_subject:.3f}")
    print(f"  Total hours: {corr_total:.3f}")
    
    # Key finding
    if abs(corr_subjects) > abs(corr_minutes_per_subject):
        print(f"\n🔍 Key finding: Number of subjects matters more than minutes per subject")
        if corr_subjects > 0:
            print("   → More subjects (even with less data each) tends to improve performance")
        else:
            print("   → Fewer subjects (with more data each) tends to improve performance")
    elif abs(corr_minutes_per_subject) > abs(corr_subjects):
        print(f"\n🔍 Key finding: Minutes per subject matters more than number of subjects")
        if corr_minutes_per_subject > 0:
            print("   → More data per subject (even with fewer subjects) tends to improve performance")
        else:
            print("   → Less data per subject tends to improve performance (unlikely - check data)")
    else:
        print(f"\n🔍 Key finding: Both subjects and minutes per subject matter equally")
    
    # Practical recommendation
    if not target_achieved.empty:
        avg_subjects = target_achieved['num_subjects'].median()
        avg_minutes = target_achieved['minutes_per_subject'].median()
        avg_hours = target_achieved['total_hours'].median()
        
        print(f"\n✅ Practical recommendation (based on runs achieving target):")
        print(f"   {avg_subjects:.0f} subjects × {avg_minutes:.0f} minutes per subject")
        print(f"   = {avg_hours:.1f} total hours of recording")
        print(f"   This strategy consistently beats YASA by ≥{delta_threshold}")
    else:
        print(f"\n⚠️  No runs consistently achieve target improvement of +{delta_threshold}")
        if not beating_yasa.empty:
            print(f"   However, {len(beating_yasa)} runs do beat YASA baseline")
    
    plt.show()
    return fig

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Generate Figure 4: Subjects vs Minutes Analysis')
    parser.add_argument("--csv", required=True, help="Path to flattened CSV")
    parser.add_argument("--out", default="Plot_Clean/figures/fig4", help="Output directory")
    args = parser.parse_args()

    setup_plotting_style()
    
    # Load and prepare data
    df = load_and_prepare_data(Path(args.csv))
    
    if df.empty:
        print("No valid data found")
        return
    
    # Create figure
    create_figure_4(df, Path(args.out))

if __name__ == "__main__":
    main()