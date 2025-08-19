#!/usr/bin/env python3
"""
Figure 2: Win Rate & Δκ Distribution by Training Data Size

Upper panel: Win rate = fraction of subjects with κ(CBraMod) > κ(YASA) vs training subjects
Lower panel: Violin plots of per-subject Δκ distributions with Wilcoxon p-values

Shows individual-level improvements, not just average performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import bootstrap
import argparse
import warnings

warnings.filterwarnings("ignore")

def setup_plotting_style():
    """Configure matplotlib for publication-ready plots."""
    plt.style.use("default")
    plt.rcParams.update({
        "figure.figsize": (12, 10),
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
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
    })

def bootstrap_ci_proportion(successes: int, total: int, confidence: float = 0.95) -> tuple:
    """Calculate bootstrap CI for a proportion."""
    if total < 3:
        return np.nan, np.nan
    
    try:
        # Create binary array
        data = np.array([1] * successes + [0] * (total - successes))
        
        def prop_stat(x):
            return np.mean(x)
        
        rng = np.random.default_rng(42)
        res = bootstrap((data,), prop_stat, n_resamples=1000, 
                       confidence_level=confidence, random_state=rng)
        return res.confidence_interval.low, res.confidence_interval.high
    except:
        # Fallback: Wilson score interval
        p = successes / total
        z = 1.96  # 95% CI
        n = total
        denominator = 1 + z**2/n
        center = (p + z**2/(2*n)) / denominator
        margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
        return max(0, center - margin), min(1, center + margin)

def wilcoxon_test(delta_values: np.ndarray) -> float:
    """Wilcoxon signed-rank test against zero."""
    try:
        deltas = np.asarray(delta_values, dtype=float)
        deltas = deltas[~np.isnan(deltas)]
        if len(deltas) < 3:
            return np.nan
        stat, p = stats.wilcoxon(deltas, zero_method="wilcox", alternative="two-sided")
        return p
    except:
        return np.nan

def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    """Load and prepare data for win rate analysis."""
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} rows")
    
    # Create convenience aliases
    if "test_kappa" not in df.columns:
        if "contract.results.test_kappa" in df.columns:
            df["test_kappa"] = df["contract.results.test_kappa"]
        elif "sum.test_kappa" in df.columns:
            df["test_kappa"] = df["sum.test_kappa"]
    
    if "num_classes" not in df.columns:
        if "contract.results.num_classes" in df.columns:
            df["num_classes"] = df["contract.results.num_classes"]
        elif "cfg.num_of_classes" in df.columns:
            df["num_classes"] = df["cfg.num_of_classes"]
    
    # Create num_subjects column
    for col in ["contract.dataset.num_subjects_train", "cfg.num_subjects_train", "num_subjects_train"]:
        if col in df.columns:
            df['num_subjects'] = df[col]
            break
    
    # Create subject ID column
    for col in ["cfg.subject_id", "subject_id", "name"]:
        if col in df.columns:
            df['subject'] = df[col]
            break
    
    # Filter and clean
    df = df[(df["num_classes"] == 5) & df["test_kappa"].notna() & (df['num_subjects'] > 0)]
    df = df.dropna(subset=['num_subjects', 'subject'])
    
    print(f"After filtering: {len(df)} runs")
    print(f"Subject counts: {sorted(df['num_subjects'].unique())}")
    
    return df

def create_figure_2(df: pd.DataFrame, output_dir: Path):
    """Create Figure 2: Win Rate & Delta-kappa Distribution."""
    import numpy as np
    
    # Fixed parameters
    yasa_kappa = 0.446
    
    # Filter out bottom 30% and select top runs per subject count
    print("Processing data with bottom 30% filtering...")
    
    subject_data = []
    
    for subj_count in sorted(df['num_subjects'].unique()):
        subj_df = df[df['num_subjects'] == subj_count].copy()
        
        # Filter out bottom 30% performers
        if len(subj_df) > 3:
            kappa_30th = subj_df['test_kappa'].quantile(0.30)
            subj_df = subj_df[subj_df['test_kappa'] >= kappa_30th]
        
        # Take top 10 runs or all if fewer
        if len(subj_df) > 10:
            subj_df = subj_df.nlargest(10, 'test_kappa')
        
        if len(subj_df) == 0:
            continue
        
        print(f"Processing {subj_count} subjects: {len(subj_df)} runs")
        
        # For each subject, calculate delta-kappa
        for _, run in subj_df.iterrows():
            delta_kappa = run['test_kappa'] - yasa_kappa
            wins_yasa = 1 if run['test_kappa'] > yasa_kappa else 0
            
            subject_data.append({
                'num_subjects': int(subj_count),
                'subject': run['subject'],
                'test_kappa': run['test_kappa'],
                'delta_kappa': delta_kappa,
                'wins_yasa': wins_yasa
            })
    
    subject_df = pd.DataFrame(subject_data)
    
    if subject_df.empty:
        print("No data to plot")
        return None
    
    # Calculate win rates per subject count
    win_rate_data = []
    violin_data = []
    
    for subj_count in sorted(subject_df['num_subjects'].unique()):
        subj_data = subject_df[subject_df['num_subjects'] == subj_count]
        
        # Win rate calculation
        total_runs = len(subj_data)
        wins = subj_data['wins_yasa'].sum()
        win_rate = wins / total_runs
        
        # Bootstrap CI for win rate
        ci_low, ci_high = bootstrap_ci_proportion(wins, total_runs)
        
        # Wilcoxon test
        delta_values = subj_data['delta_kappa'].values
        p_wilcoxon = wilcoxon_test(delta_values)
        
        win_rate_data.append({
            'num_subjects': subj_count,
            'win_rate': win_rate,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n_runs': total_runs,
            'n_wins': wins,
            'p_wilcoxon': p_wilcoxon
        })
        
        # Add data for violin plots
        for _, row in subj_data.iterrows():
            violin_data.append({
                'num_subjects': subj_count,
                'delta_kappa': row['delta_kappa']
            })
    
    win_rate_df = pd.DataFrame(win_rate_data)
    violin_df = pd.DataFrame(violin_data)
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1.2])
    
    # Upper panel: Win rate
    valid_ci = ~(np.isnan(win_rate_df['ci_low']) | np.isnan(win_rate_df['ci_high']))
    
    if valid_ci.any():
        ax1.errorbar(
            win_rate_df.loc[valid_ci, 'num_subjects'],
            win_rate_df.loc[valid_ci, 'win_rate'],
            yerr=[
                win_rate_df.loc[valid_ci, 'win_rate'] - win_rate_df.loc[valid_ci, 'ci_low'],
                win_rate_df.loc[valid_ci, 'ci_high'] - win_rate_df.loc[valid_ci, 'win_rate']
            ],
            fmt='o-', color='#2E86AB', linewidth=2.5, markersize=6,
            capsize=4, capthick=2, label='Win Rate ± 95% CI'
        )
    
    # Points without CI
    missing_ci = ~valid_ci
    if missing_ci.any():
        ax1.plot(
            win_rate_df.loc[missing_ci, 'num_subjects'],
            win_rate_df.loc[missing_ci, 'win_rate'],
            'o-', color='#2E86AB', linewidth=2.5, markersize=6
        )
    
    # 50% line (random chance)
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Random chance (50%)')
    
    ax1.set_ylabel('Win Rate\n(Fraction > YASA)', fontweight='bold')
    ax1.set_title('Figure 2A: Win Rate vs Training Data Size', fontweight='bold', pad=15)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Set x-axis ticks every 5 subjects
    max_subjects = max(win_rate_df['num_subjects'])
    x_ticks = np.arange(0, max_subjects + 5, 5)
    ax1.set_xticks(x_ticks)
    ax1.set_xlim(0, max_subjects + 2)
    
    # Lower panel: Violin plots of delta-kappa
    subject_counts = sorted(violin_df['num_subjects'].unique())
    
    # Create violin plot
    violin_parts = ax2.violinplot(
        [violin_df[violin_df['num_subjects'] == sc]['delta_kappa'].values 
         for sc in subject_counts],
        positions=subject_counts,
        widths=2.0,
        showmeans=False,
        showmedians=True,
        showextrema=False
    )
    
    # Color the violins
    for pc in violin_parts['bodies']:
        pc.set_facecolor('#87CEEB')
        pc.set_alpha(0.7)
    
    # Add median line and p-value annotations
    for i, subj_count in enumerate(subject_counts):
        data = violin_df[violin_df['num_subjects'] == subj_count]['delta_kappa'].values
        if len(data) > 0:
            median_val = np.median(data)
            
            # Get p-value for this subject count
            p_val = win_rate_df[win_rate_df['num_subjects'] == subj_count]['p_wilcoxon'].iloc[0]
            
            # Add p-value annotation
            y_pos = max(data) + 0.02
            if not np.isnan(p_val):
                if p_val < 0.001:
                    p_text = 'p<0.001***'
                elif p_val < 0.01:
                    p_text = f'p={p_val:.3f}**'
                elif p_val < 0.05:
                    p_text = f'p={p_val:.3f}*'
                else:
                    p_text = f'p={p_val:.3f}'
            else:
                p_text = 'p=n.s.'
            
            ax2.text(subj_count, y_pos, p_text, ha='center', va='bottom', 
                    fontsize=9, alpha=0.8)
    
    # Zero line (no improvement over YASA)
    ax2.axhline(y=0, color='#A23B72', linestyle='--', linewidth=2,
                label='YASA baseline (Δκ=0)')
    
    ax2.set_xlabel('Number of Training Subjects', fontweight='bold')
    ax2.set_ylabel('Δκ vs YASA', fontweight='bold')
    ax2.set_title('Figure 2B: Per-Subject Performance Improvement Distribution', 
                  fontweight='bold', pad=15)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Set x-axis to match upper panel
    ax2.set_xticks(x_ticks)
    ax2.set_xlim(0, max_subjects + 2)
    
    # Find T* (first subject count where win rate > 80% or similar threshold)
    t_star = None
    for _, row in win_rate_df.iterrows():
        if row['win_rate'] >= 0.8:  # 80% win rate threshold
            t_star = row['num_subjects']
            break
    
    if t_star is not None:
        # Add vertical line at T*
        ax1.axvline(x=t_star, color='red', linestyle=':', alpha=0.8, linewidth=2)
        ax2.axvline(x=t_star, color='red', linestyle=':', alpha=0.8, linewidth=2,
                    label=f'T* = {t_star} subjects (≥80% win rate)')
        
        # Update legend
        ax2.legend(loc='upper right')
        
        # Report T* win rate
        t_star_win_rate = win_rate_df[win_rate_df['num_subjects'] == t_star]['win_rate'].iloc[0]
        print(f"T* = {t_star} subjects with {t_star_win_rate:.1%} win rate")
    
    plt.tight_layout()
    
    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_svg = output_dir / 'figure_2_win_rate_distribution.svg'
    fig_pdf = output_dir / 'figure_2_win_rate_distribution.pdf'
    
    plt.savefig(fig_svg)
    plt.savefig(fig_pdf)
    
    print(f"Saved: {fig_svg}")
    print(f"Saved: {fig_pdf}")
    
    # Summary statistics
    print(f"\nSummary:")
    max_win_rate = win_rate_df['win_rate'].max()
    best_subj_count = win_rate_df.loc[win_rate_df['win_rate'].idxmax(), 'num_subjects']
    print(f"Best win rate: {max_win_rate:.1%} at {best_subj_count} training subjects")
    
    if t_star is not None:
        print(f"T*: {t_star} subjects (≥80% subjects improved over YASA)")
    else:
        print("T*: Not reached (no subject count achieves ≥80% win rate)")
    
    plt.show()
    return fig

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Generate Figure 2: Win Rate & Delta-kappa Distribution')
    parser.add_argument("--csv", required=True, help="Path to flattened CSV")
    parser.add_argument("--out", default="artifacts/results/figures/paper", help="Output directory")
    args = parser.parse_args()

    setup_plotting_style()
    
    # Load and prepare data
    df = load_and_prepare_data(Path(args.csv))
    
    if df.empty:
        print("No valid data found.")
        return
    
    # Create figure
    create_figure_2(df, Path(args.out))

if __name__ == "__main__":
    main()