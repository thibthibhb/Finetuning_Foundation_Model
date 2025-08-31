#!/usr/bin/env python3
"""
Combined Figure 1 & 2: Calibration Dose-Response and Win Rate Analysis

Two subplots:
- Top: Figure 1 - Calibration dose-response (subject count vs performance)
- Bottom: Figure 2 - Violin plots of per-subject Δκ distributions with Wilcoxon p-values

Shows comprehensive analysis of CBraMod performance scaling with training data.
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
from itertools import cycle
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")

# Import consistent figure styling
from figure_style import (
    setup_figure_style, get_color, save_figure, 
    add_yasa_baseline, add_significance_marker,
    bootstrap_ci_median, wilcoxon_test,
    format_n_caption, add_sample_size_annotation
)

def setup_combined_figure_style():
    """Setup style specifically for combined figures."""
    setup_figure_style()
    plt.rcParams.update({
        "figure.figsize": (12, 12),  # Larger for combined plots
    })

def jeffreys_interval(successes: int, total: int, confidence: float = 0.95) -> tuple:
    """Calculate Jeffreys interval (Beta-Bernoulli) for proportion."""
    if total == 0:
        return 0.0, 0.0, 1.0
    
    # Jeffreys prior: Beta(0.5, 0.5)
    alpha = 0.5 + successes
    beta_param = 0.5 + total - successes
    
    from scipy.stats import beta as beta_dist
    
    alpha_level = (1 - confidence) / 2
    
    # Calculate credible interval
    ci_low = beta_dist.ppf(alpha_level, alpha, beta_param)
    ci_high = beta_dist.ppf(1 - alpha_level, alpha, beta_param)
    
    # Point estimate (posterior mean)
    point_est = alpha / (alpha + beta_param)
    
    return point_est, ci_low, ci_high

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

# ---------- EXACT SAME FUNCTIONS FROM fig1_from_csv.py ----------

def pick_first(df: pd.DataFrame, candidates: list) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def derive_subject_id(row: pd.Series) -> str:
    for c in ["cfg.subject_id", "cfg.subj_id", "cfg.subject", "sum.subject_id", "subject_id", "name"]:
        if c in row and pd.notna(row[c]) and str(row[c]).strip() != "":
            return str(row[c])
    # fall back to run name if present; else an index
    return str(row.get("name", "unknown"))

def derive_nights(row: pd.Series) -> float:
    # preferred explicit nights
    for c in ["cfg.calib_nights", "cfg.nights_training", "cfg.nights_calibration"]:
        if c in row and pd.notna(row[c]):
            try:
                return float(row[c])
            except Exception:
                pass
    # minutes → nights
    for c in ["cfg.calib_minutes", "cfg.minutes_calib"]:
        if c in row and pd.notna(row[c]):
            try:
                return float(row[c]) / 480.0  # 8h * 60
            except Exception:
                pass
    # hours per subject → nights
    for c in ["sum.hours_of_data_per_subject", "sum.hours_of_data"]:
        if c in row and pd.notna(row[c]):
            try:
                return float(row[c]) / 8.0
            except Exception:
                pass
    return np.nan

def derive_minutes(row: pd.Series) -> float:
    for c in ["cfg.calib_minutes", "cfg.minutes_calib"]:
        if c in row and pd.notna(row[c]):
            try: return float(row[c])
            except Exception: pass
    # nights → minutes
    n = derive_nights(row)
    if pd.notna(n):
        return float(n) * 480.0
    # epochs * epoch_len
    if "cfg.n_calib_epochs" in row and pd.notna(row["cfg.n_calib_epochs"]):
        ep = float(row["cfg.n_calib_epochs"])
        epoch_len = 30.0
        if "cfg.epoch_len" in row and pd.notna(row["cfg.epoch_len"]):
            try: epoch_len = float(row["cfg.epoch_len"])
            except Exception: pass
        return ep * (epoch_len / 60.0)
    return np.nan

def load_and_prepare(csv_path: Path, x_mode: str, num_subjects=None) -> pd.DataFrame:
    """EXACT SAME as fig1_from_csv.py"""
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    
    # normalize metric names
    if "test_kappa" not in df.columns:
        alt = "contract.results.test_kappa"
        if alt in df.columns:
            df = df.rename(columns={alt: "test_kappa"})
    if "test_f1" not in df.columns:
        alt = "contract.results.test_f1"
        if alt in df.columns:
            df = df.rename(columns={alt: "test_f1"})
    if "num_classes" not in df.columns:
        alt = "contract.results.num_classes"
        if alt in df.columns:
            df = df.rename(columns={alt: "num_classes"})
    
    print(f"Available columns: {list(df.columns)}")
    
    # derive subject
    df["subject"] = df.apply(derive_subject_id, axis=1)
    
    # derive x variable
    if x_mode == "nights":
        df["xvalue"] = df.apply(derive_nights, axis=1)
    else:
        df["xvalue"] = df.apply(derive_minutes, axis=1)
    
    print(f"X-variable ({x_mode}) derived for {df['xvalue'].notna().sum()} rows")
    print(f"test_kappa available for {df['test_kappa'].notna().sum()} rows")
    print(f"num_classes distribution:\n{df['num_classes'].value_counts(dropna=False)}")
    
    # Add number of subjects column for filtering BEFORE we filter columns
    num_subj_col = None
    for col in ["contract.dataset.num_subjects_train", "cfg.num_subjects_train", "num_subjects_train"]:
        if col in df.columns:
            num_subj_col = col
            df['num_subjects'] = df[col]  # Create standardized column
            break
    
    # Add dataset composition column
    dataset_col = None
    for col in ["contract.dataset.datasets", "cfg.datasets", "datasets"]:
        if col in df.columns:
            dataset_col = col
            df['dataset_composition'] = df[col]
            break
    
    if dataset_col is None:
        print("Warning: Could not find datasets column, using default")
        df['dataset_composition'] = 'Unknown'
    
    # keep essentials (including num_subjects and dataset_composition)
    keep = ["subject", "xvalue", "test_kappa", "test_f1", "num_classes", "name", "num_subjects", "dataset_composition"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    
    # filter basic criteria first
    df = df[(df["num_classes"] == 5) & df["test_kappa"].notna() & df["xvalue"].notna()]
    print(f"After filtering (5-class, valid kappa, valid x): {len(df)} rows")
    
    # Remove outliers: if single dataset (num_datasets=1) but subject_count >11, discard
    num_datasets_col = None
    for col in ["contract.dataset.num_datasets", "cfg.num_datasets", "num_datasets"]:
        if col in df.columns:
            num_datasets_col = col
            break
    
    # Remove data corruption: ORP alone cannot have >11 subjects
    if 'dataset_composition' in df.columns and 'num_subjects' in df.columns:
        corrupt_entries = df[(df['dataset_composition'] == 'ORP') & (df['num_subjects'] > 11)]
        if len(corrupt_entries) > 0:
            print(f"🚨 Removing {len(corrupt_entries)} CORRUPT entries (ORP dataset with >11 subjects):")
            for _, row in corrupt_entries.iterrows():
                print(f"  {row['name']}: {row['num_subjects']} subjects, dataset={row['dataset_composition']} - IMPOSSIBLE!")
            df = df[~((df['dataset_composition'] == 'ORP') & (df['num_subjects'] > 11))]
            print(f"After removing corrupt ORP entries: {len(df)} rows")
    
    # Remove entries with spaces in dataset composition (formatting issue)
    if 'dataset_composition' in df.columns:
        space_entries = df[df['dataset_composition'].str.contains(', ', na=False)]
        if len(space_entries) > 0:
            print(f"🚨 Removing {len(space_entries)} entries with spaces in dataset composition:")
            for _, row in space_entries.iterrows():
                print(f"  {row['name']}: dataset='{row['dataset_composition']}' - HAS SPACES!")
            df = df[~df['dataset_composition'].str.contains(', ', na=False)]
            print(f"After removing spaced dataset entries: {len(df)} rows")
    
    if num_datasets_col and 'num_subjects' in df.columns:
        outliers = df[(df[num_datasets_col] == 1) & (df['num_subjects'] > 11)]
        if len(outliers) > 0:
            print(f"Removing {len(outliers)} impossible outliers (single dataset with >11 subjects):")
            for _, row in outliers.iterrows():
                datasets = row.get('dataset_composition', 'unknown')
                print(f"  {row['name']}: {row['num_subjects']} subjects, datasets={datasets}")
            df = df[~((df[num_datasets_col] == 1) & (df['num_subjects'] > 11))]
            print(f"After removing impossible outliers: {len(df)} rows")

    
    # Check if we successfully created num_subjects column
    if 'num_subjects' not in df.columns:
        print(f"Warning: Could not find num_subjects column for filtering")
        if num_subjects is not None:
            return pd.DataFrame()  # Return empty if specific subject count requested but no column found
    else:
        print(f"num_subjects distribution:\n{df['num_subjects'].value_counts(dropna=False).sort_index()}")
    
    # Filter by number of subjects if specified
    if num_subjects is not None:
        df = df[df['num_subjects'] == num_subjects]
        print(f"After filtering to {num_subjects} subjects: {len(df)} rows")
        
        # Filter out bottom 30% performers and then take top 10
        if len(df) > 3:  # Only filter if we have enough data
            kappa_30th = df['test_kappa'].quantile(0.30)
            df = df[df['test_kappa'] >= kappa_30th]
            print(f"Filtered out bottom 30% (κ<{kappa_30th:.3f}) for {num_subjects} subjects")
        
        if len(df) > 10:
            top_10_df = df.nlargest(10, 'test_kappa')
            print(f"Selected top 10 runs for {num_subjects} subjects by test_kappa:")
            for _, row in top_10_df.iterrows():
                print(f"  {row['name']}: κ={row['test_kappa']:.3f}, x={row['xvalue']:.2f}")
            df = top_10_df
        else:
            print(f"Using all {len(df)} available runs for {num_subjects} subjects after filtering")
    else:
        # If no specific subject count, take top 10 runs PER subject count
        print("Taking top 10 runs per subject count (filtering out bottom 30%)...")
        df_list = []
        
        if 'num_subjects' in df.columns:
            for subj_count in sorted(df['num_subjects'].unique()):
                subj_df = df[df['num_subjects'] == subj_count]
                
                # Filter out bottom 30% performers to remove outliers
                if len(subj_df) > 3:  # Only filter if we have enough data
                    kappa_30th = subj_df['test_kappa'].quantile(0.30)
                    subj_df = subj_df[subj_df['test_kappa'] >= kappa_30th]
                    print(f"  {subj_count} subjects: filtered out bottom 30% (κ<{kappa_30th:.3f})")
                
                if len(subj_df) > 10:
                    top_10_subj = subj_df.nlargest(10, 'test_kappa')
                    print(f"  {subj_count} subjects: selected top 10 from {len(subj_df)} runs after filtering")
                else:
                    top_10_subj = subj_df
                    print(f"  {subj_count} subjects: using all {len(subj_df)} runs after filtering")
                df_list.append(top_10_subj)
            
            if df_list:
                df = pd.concat(df_list, ignore_index=True)
                print(f"Total runs after selecting top 10 per subject count: {len(df)}")
            else:
                df = pd.DataFrame()
        else:
            # Fallback to overall top 10 if no subject column
            if len(df) > 10:
                df = df.nlargest(10, 'test_kappa')
                print(f"Selected overall top 10 runs by test_kappa (no subject column found)")
    
    return df

def create_combined_figure(df: pd.DataFrame, output_dir: Path):
    """Create combined Figure 1 & 2 - keeping exact same Figure 1 from original."""
    yasa_kappa = 0.446
    yasa_ci = 0.05
    delta_threshold = 0.05
    
    print("Creating combined calibration and win rate analysis...")
    
    if df.empty:
        print("No data to plot")
        return None
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[1, 1])
    
    # ===== FIGURE 1: EXACT SAME AS ORIGINAL (Top Panel) =====
    # Use the exact same logic from fig1_from_csv.py create_comprehensive_plot function
    
    df_fig1 = df[df['num_subjects'] > 0].dropna(subset=['num_subjects'])
    if df_fig1.empty:
        print("No valid subject count data found for Figure 1")
        return None

    unique_datasets = sorted(df_fig1['dataset_composition'].unique())
    print(f"Found dataset compositions: {unique_datasets}")

    # Use different colors for each dataset - this is crucial for differentiation!
    dataset_colors = {}
    color_cycle = cycle(OKABE_ITO)
    for ds in unique_datasets:
        dataset_colors[ds] = next(color_cycle)

    subject_count_data = []
    for subj_count in sorted(df_fig1['num_subjects'].unique()):
        subj_df = df_fig1[df_fig1['num_subjects'] == subj_count]
        if len(subj_df) == 0:
            continue

        for dataset_comp in subj_df['dataset_composition'].unique():
            dataset_subj_df = subj_df[subj_df['dataset_composition'] == dataset_comp]
            if len(dataset_subj_df) == 0:
                continue

            median_kappa = dataset_subj_df['test_kappa'].median()
            kappa_values = dataset_subj_df['test_kappa'].values
            ci_low, ci_high = bootstrap_ci_median(kappa_values)
            delta_median = median_kappa - yasa_kappa
            ci_delta_low = ci_low - yasa_kappa if not np.isnan(ci_low) else np.nan

            subject_count_data.append({
                'num_subjects': int(subj_count),
                'dataset_composition': dataset_comp,
                'median_kappa': median_kappa,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'delta_median': delta_median,
                'ci_delta_low': ci_delta_low,
                'n_runs': len(dataset_subj_df),
                'color': dataset_colors[dataset_comp]
            })

    if not subject_count_data:
        print("No data to plot for Figure 1")
        return None

    stats_df = pd.DataFrame(subject_count_data)

    for dataset_comp in unique_datasets:
        dataset_data = stats_df[stats_df['dataset_composition'] == dataset_comp]
        if dataset_data.empty:
            continue

        dataset_color = dataset_data['color'].iloc[0]
        valid_ci = ~(np.isnan(dataset_data['ci_low']) | np.isnan(dataset_data['ci_high']))

        if valid_ci.any():
            ax1.errorbar(
                dataset_data.loc[valid_ci, 'num_subjects'],
                dataset_data.loc[valid_ci, 'median_kappa'],
                yerr=[
                    dataset_data.loc[valid_ci, 'median_kappa'] - dataset_data.loc[valid_ci, 'ci_low'],
                    dataset_data.loc[valid_ci, 'ci_high'] - dataset_data.loc[valid_ci, 'median_kappa']
                ],
                fmt='o', color=dataset_color, markersize=8,
                markerfacecolor=dataset_color,
                markeredgecolor='white',
                markeredgewidth=1.5,
                capsize=3, capthick=1.8, elinewidth=1.8,
                label=dataset_comp, zorder=3
            )

        missing_ci = ~valid_ci
        if missing_ci.any():
            ax1.plot(
                dataset_data.loc[missing_ci, 'num_subjects'],
                dataset_data.loc[missing_ci, 'median_kappa'],
                'o', color=dataset_color, markersize=8,
                markerfacecolor=dataset_color,
                markeredgecolor='white',
                markeredgewidth=1.5,
                label=dataset_comp if not valid_ci.any() else "", zorder=3
            )

    # Get max_subjects first
    max_subjects = max(stats_df['num_subjects'])
    
    # YASA baseline as simple horizontal line
    ax1.axhline(y=yasa_kappa, color=get_color("yasa"), linestyle='--', linewidth=1.8, 
               label=f'YASA baseline (κ={yasa_kappa:.3f})', zorder=2)

    # T* marker (first crossing of YASA baseline)
    crossing_points = [int(r['num_subjects']) for _, r in stats_df.iterrows() if r['median_kappa'] > yasa_kappa]
    if crossing_points:
        first_crossing = min(crossing_points)
        ax1.axvline(x=first_crossing, color=get_color("t_star"), linestyle='-', alpha=0.8, linewidth=2,
                   label=f'T* = {first_crossing}')
        ax1.text(first_crossing + 0.5, ax1.get_ylim()[1] - 0.02, 'T*', 
                fontweight='bold', fontsize=12, ha='left', va='top', color=get_color("t_star"))

    ax1.set_xlabel('Number of training subjects (calibration cohort size)', fontweight='bold')
    ax1.set_ylabel("Cohen's κ", fontweight='bold')
    ax1.set_title('CBraMod Calibration: Performance vs Training Data Size\n(Top 10 runs per subject count. T* = first crossing of YASA baseline.)',
                 fontweight='bold', fontsize=14, pad=20)
    # Create comprehensive legend with all datasets
    legend_elements = []
    # Add dataset legend entries
    for dataset_comp in unique_datasets:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color=dataset_colors[dataset_comp], 
                                         linewidth=0, markersize=8, markerfacecolor=dataset_colors[dataset_comp], 
                                         markeredgecolor='white', markeredgewidth=1.5, label=dataset_comp))
    # Add other elements
    if crossing_points:
        legend_elements.append(plt.Line2D([0], [0], color=get_color("t_star"), linewidth=2, label=f'T* = {first_crossing}'))
    legend_elements.append(plt.Line2D([0], [0], color=get_color("yasa"), linestyle='--', linewidth=1.8, label=f'YASA baseline (κ={yasa_kappa:.3f})'))
    ax1.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(1.0, 0.0))
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_axisbelow(True)

    ax1.set_xlim(0, max_subjects + 2)
    x_ticks = np.arange(0, max_subjects + 5, 5)
    ax1.set_xticks(x_ticks)

    y_min = min(stats_df['median_kappa'].min() - 0.05, yasa_kappa - 0.05)
    y_max = max(stats_df['median_kappa'].max() + 0.10, yasa_kappa + 0.10)
    ax1.set_ylim(y_min, y_max)
    
    # ===== FIGURE 2: Reliability Curve Analysis (Bottom Panel) =====
    print("Processing data for Figure 2: Reliability curve analysis...")
    
    # Build subject-level reliability data for each subject count
    reliability_data = []
    
    for subj_count in sorted(df['num_subjects'].unique()):
        subj_df = df[df['num_subjects'] == subj_count].copy()
        
        if len(subj_df) == 0:
            continue
        
        # Group by subject and get their runs
        subject_success_data = []
        subjects = subj_df['subject'].unique()
        n_subjects = len(subjects)
        
        if n_subjects < 5:  # Skip if too few subjects
            print(f"  T={int(subj_count)}: Skipping (n_subjects={n_subjects} < 5)")
            continue
            
        print(f"  T={int(subj_count)}: Processing {n_subjects} subjects...")
        
        for subject in subjects:
            subj_runs = subj_df[subj_df['subject'] == subject].copy()
            if len(subj_runs) == 0:
                continue
                
            # Sort by test_kappa descending
            subj_runs = subj_runs.sort_values('test_kappa', ascending=False)
            
            # Best-of-1: single best run beats YASA
            best_1_kappa = subj_runs['test_kappa'].iloc[0]
            success_1 = 1 if best_1_kappa > yasa_kappa else 0
            
            # Best-of-3: best of up to 3 runs beats YASA
            best_3_runs = subj_runs.head(3)
            best_3_kappa = best_3_runs['test_kappa'].max()
            success_3 = 1 if best_3_kappa > yasa_kappa else 0
            
            subject_success_data.append({
                'subject': subject,
                'success_1': success_1,
                'success_3': success_3,
                'n_runs': len(subj_runs)
            })
        
        if not subject_success_data:
            continue
            
        success_df = pd.DataFrame(subject_success_data)
        
        # Calculate reliability statistics
        n_subjects = len(success_df)
        successes_1 = success_df['success_1'].sum()
        successes_3 = success_df['success_3'].sum()
        
        # Use Jeffreys intervals for reliability estimates
        r1_est, r1_low, r1_high = jeffreys_interval(successes_1, n_subjects)
        r3_est, r3_low, r3_high = jeffreys_interval(successes_3, n_subjects)
        
        reliability_data.append({
            'T': int(subj_count),
            'n': n_subjects,
            'succ1': successes_1,
            'r1': r1_est,
            'r1_low': r1_low, 
            'r1_high': r1_high,
            'succ3': successes_3,
            'r3': r3_est,
            'r3_low': r3_low,
            'r3_high': r3_high
        })
        
        print(f"  T={subj_count}: r(1)={r1_est:.3f} [{r1_low:.3f},{r1_high:.3f}], r(3)={r3_est:.3f} [{r3_low:.3f},{r3_high:.3f}] ({successes_3}/{n_subjects})")
    
    if not reliability_data:
        print("No reliability data for Figure 2")
        return None
    
    reliability_df = pd.DataFrame(reliability_data)
    
    # Print table
    print(f"\n{'T':>3} {'n':>3} {'succ1':>5} {'r1':>5} {'r1_low':>6} {'r1_high':>7} {'succ3':>5} {'r3':>5} {'r3_low':>6} {'r3_high':>7}")
    print("-" * 65)
    for _, row in reliability_df.iterrows():
        print(f"{row['T']:>3} {row['n']:>3} {row['succ1']:>5} {row['r1']:>5.3f} {row['r1_low']:>6.3f} "
              f"{row['r1_high']:>7.3f} {row['succ3']:>5} {row['r3']:>5.3f} {row['r3_low']:>6.3f} {row['r3_high']:>7.3f}")
    
    subject_counts = reliability_df['T'].values
    
    # Plot reliability curves with Jeffreys intervals
    # Best-of-1 curve (blue, circles)
    ax2.errorbar(subject_counts, reliability_df['r1'], 
                yerr=[reliability_df['r1'] - reliability_df['r1_low'],
                      reliability_df['r1_high'] - reliability_df['r1']],
                marker='o', markersize=7, linewidth=2.5, capsize=3, capthick=1.5,
                color='#1f77b4', markerfacecolor='#1f77b4', markeredgecolor='white', 
                markeredgewidth=1.5, label='Best-of-1', zorder=3)
    
    # Best-of-3 curve (magenta, squares)
    ax2.errorbar(subject_counts, reliability_df['r3'],
                yerr=[reliability_df['r3'] - reliability_df['r3_low'],
                      reliability_df['r3_high'] - reliability_df['r3']], 
                marker='s', markersize=7, linewidth=2.5, capsize=3, capthick=1.5,
                color='#b03a9c', markerfacecolor='#b03a9c', markeredgecolor='white',
                markeredgewidth=1.5, label='Best-of-3', zorder=3)
    
    # Add n(T) annotations above each point
    for _, row in reliability_df.iterrows():
        T = int(row['T'])
        n_subjects = row['n']
        max_r = max(row['r1_high'], row['r3_high'])
        
        ax2.text(T, max_r + 0.03, f'n={n_subjects}', 
               ha='center', va='bottom', fontsize=8, color='gray', alpha=0.8)
    
    # Random chance baseline (dotted line at 0.5)
    ax2.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, 
               label='Random chance (50%)')
    
    # Styling
    ax2.set_xlabel('Training subjects T', fontweight='bold')
    ax2.set_ylabel('Reliability r(T)', fontweight='bold') 
    ax2.set_title('Probability of beating YASA baseline — T* = none', 
                  fontweight='bold', pad=15)
    
    # Set y-axis to probability range [0,1]
    ax2.set_ylim(0, 1)
    ax2.set_yticks(np.arange(0, 1.1, 0.2))
    ax2.set_yticklabels([f'{y:.1f}' for y in np.arange(0, 1.1, 0.2)])
    
    # Match x-axis with Panel A
    ax2.set_xticks(x_ticks)
    ax2.set_xlim(0, max_subjects + 2)
    
    # Clean grid behind data
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_axisbelow(True)
    
    # Legend positioned outside bottom-right to never overlap
    ax2.legend(bbox_to_anchor=(1.0, -0.05), loc="upper right", 
               frameon=True, fancybox=True, shadow=False, framealpha=0.95)
    
    # Layout adjustment for legend
    plt.subplots_adjust(right=0.95, bottom=0.12)
    
    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / 'figure_1_2_combined_calibration_analysis'
    
    plt.tight_layout()
    save_figure(fig, base_path)
    
    # Summary statistics
    print(f"\nSummary:")
    best_kappa = stats_df['median_kappa'].max()
    best_subj_count = stats_df.loc[stats_df['median_kappa'].idxmax(), 'num_subjects']
    print(f"Best performance: κ={best_kappa:.3f} at {best_subj_count} training subjects")
    
    if crossing_points:
        print(f"CBraMod first surpasses YASA at {first_crossing} training subjects")
    else:
        print("CBraMod does not surpass YASA baseline in this data")
    
    plt.show()
    return fig

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Generate Combined Figure 1 & 2: Calibration and Win Rate Analysis')
    parser.add_argument("--csv", required=True, help="Path to flattened CSV")
    parser.add_argument("--out", default="Plot_Clean/figures/fig1_2_combined", help="Output directory")
    args = parser.parse_args()

    setup_combined_figure_style()
    
    # Load and prepare data EXACTLY like fig1_from_csv.py
    print("Loading data and selecting top 10 runs per subject count...")
    df = load_and_prepare(Path(args.csv), x_mode="nights", num_subjects=None)  # Load all subjects
    
    if df.empty:
        print("No valid data found. Check your CSV file.")
        return
    
    # Create combined figure
    create_combined_figure(df, Path(args.out))

if __name__ == "__main__":
    main()