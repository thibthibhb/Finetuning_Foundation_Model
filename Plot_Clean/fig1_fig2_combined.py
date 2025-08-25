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

# Okabe–Ito colorblind-friendly palette
OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange  
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

CB_COLORS = {
    "cbramod": "#0072B2",      # teal/blue for CBraMod
    "yasa": "#E69F00",         # warm orange for YASA
    "t_star": "#009E73",      # green
    "subjects": "#8C8C8C",    # neutral grey
}

def setup_plotting_style():
    """Configure matplotlib for publication-ready plots."""
    plt.style.use("default")
    plt.rcParams.update({
        "figure.figsize": (12, 12),
        "font.family": "sans-serif", 
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
    })

def bootstrap_ci_median(arr: np.ndarray, confidence: float = 0.95) -> tuple:
    """Calculate bootstrap CI for median."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 3:
        return np.nan, np.nan
    
    try:
        rng = np.random.default_rng(42)
        res = bootstrap((arr,), np.median, n_resamples=1000,
                       confidence_level=confidence, random_state=rng)
        return float(res.confidence_interval.low), float(res.confidence_interval.high)
    except:
        return np.nan, np.nan

def bootstrap_ci_proportion(successes: int, total: int, confidence: float = 0.95) -> tuple:
    """Calculate bootstrap CI for a proportion."""
    if total < 3:
        return np.nan, np.nan
    
    try:
        data = np.array([1] * successes + [0] * (total - successes))
        def prop_stat(x):
            return np.mean(x)
        
        rng = np.random.default_rng(42)
        res = bootstrap((data,), prop_stat, n_resamples=1000,
                       confidence_level=confidence, random_state=rng)
        return res.confidence_interval.low, res.confidence_interval.high
    except:
        # Wilson score interval fallback
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
    ax1.axhline(y=yasa_kappa, color=CB_COLORS["yasa"], linestyle='--', linewidth=1.8, 
               label=f'YASA baseline (κ={yasa_kappa:.3f})', zorder=2)

    # T* marker (first crossing of YASA baseline)
    crossing_points = [int(r['num_subjects']) for _, r in stats_df.iterrows() if r['median_kappa'] > yasa_kappa]
    if crossing_points:
        first_crossing = min(crossing_points)
        ax1.axvline(x=first_crossing, color=CB_COLORS["t_star"], linestyle='-', alpha=0.8, linewidth=2,
                   label=f'T* = {first_crossing}')
        ax1.text(first_crossing + 0.5, ax1.get_ylim()[1] - 0.02, 'T*', 
                fontweight='bold', fontsize=12, ha='left', va='top', color=CB_COLORS["t_star"])

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
        legend_elements.append(plt.Line2D([0], [0], color=CB_COLORS["t_star"], linewidth=2, label=f'T* = {first_crossing}'))
    legend_elements.append(plt.Line2D([0], [0], color=CB_COLORS["yasa"], linestyle='--', linewidth=1.8, label=f'YASA baseline (κ={yasa_kappa:.3f})'))
    ax1.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(1.0, 0.0))
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_axisbelow(True)

    ax1.set_xlim(0, max_subjects + 2)
    x_ticks = np.arange(0, max_subjects + 5, 5)
    ax1.set_xticks(x_ticks)

    y_min = min(stats_df['median_kappa'].min() - 0.05, yasa_kappa - 0.05)
    y_max = max(stats_df['median_kappa'].max() + 0.10, yasa_kappa + 0.10)
    ax1.set_ylim(y_min, y_max)
    
    # ===== FIGURE 2: Top-Performer Distribution Analysis (Bottom Panel) =====
    print("Processing data for Figure 2: Top-performer distributions per subject count...")
    
    # Configuration
    SHOW_SUCCESS_RATE = False  # Set to True to show success rate annotations
    
    # Build best-run-per-subject data for each subject count
    top_performer_data = []
    
    for subj_count in sorted(df['num_subjects'].unique()):
        subj_df = df[df['num_subjects'] == subj_count].copy()
        
        if len(subj_df) == 0:
            continue
        
        # Get top-10 runs at this subject count for reliability analysis
        top_10_runs = subj_df.nlargest(10, 'test_kappa') if len(subj_df) >= 10 else subj_df
        
        # Calculate success rate (how many of top-10 beat YASA)
        n_beats_yasa = sum(1 for _, run in top_10_runs.iterrows() if run['test_kappa'] > yasa_kappa)
        success_rate = n_beats_yasa / len(top_10_runs)
        
        # Get best run per subject within the top-10
        best_per_subject = []
        for subject_id in top_10_runs['subject'].unique():
            subj_runs = top_10_runs[top_10_runs['subject'] == subject_id]
            best_run = subj_runs.loc[subj_runs['test_kappa'].idxmax()]
            best_per_subject.append({
                'num_subjects': int(subj_count),
                'subject': subject_id,
                'test_kappa': best_run['test_kappa'],
                'delta_kappa': best_run['test_kappa'] - yasa_kappa,
                'success_rate': success_rate,
                'n_top10': len(top_10_runs),
                'n_beats_yasa': n_beats_yasa
            })
        
        top_performer_data.extend(best_per_subject)
        print(f"Figure 2: S={subj_count}: {len(best_per_subject)} best-per-subject, {n_beats_yasa}/{len(top_10_runs)} beat YASA")
    
    if not top_performer_data:
        print("No top-performer data for Figure 2")
        return None
    
    top_df = pd.DataFrame(top_performer_data)
    subject_counts = sorted(top_df['num_subjects'].unique())
    
    # Create beeswarm/violin hybrid plot for best-per-subject distributions
    violin_data_dict = {}
    overlay_stats = {}
    
    for subj_count in subject_counts:
        subj_data = top_df[top_df['num_subjects'] == subj_count]
        delta_values = subj_data['delta_kappa'].values
        
        violin_data_dict[subj_count] = delta_values
        
        # Calculate overlay statistics (upper-tail focus)
        top3_mean = np.mean(np.partition(delta_values, -min(3, len(delta_values)))[-min(3, len(delta_values)):])
        single_best = np.max(delta_values)
        success_rate = subj_data['success_rate'].iloc[0]
        n_beats_yasa = subj_data['n_beats_yasa'].iloc[0] 
        n_top10 = subj_data['n_top10'].iloc[0]
        
        overlay_stats[subj_count] = {
            'top3_mean': top3_mean,
            'single_best': single_best,
            'success_rate': success_rate,
            'n_beats_yasa': n_beats_yasa,
            'n_top10': n_top10
        }
    
    # Create violin plot for best-per-subject distributions
    violin_parts = ax2.violinplot(
        [violin_data_dict[sc] for sc in subject_counts],
        positions=subject_counts,
        widths=1.8,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    
    # Color the violins with champion-focused color (reduced visual dominance)
    for pc in violin_parts['bodies']:
        pc.set_facecolor('#4CAF50')  # Green for champions
        pc.set_alpha(0.3)  # Reduced alpha for less dominance
        pc.set_edgecolor('black')
        pc.set_linewidth(0.6)
        pc.set_zorder(1)  # Behind the points
    
    # Add overlay markers and annotations
    for subj_count in subject_counts:
        stats = overlay_stats[subj_count]
        
        # Diamond for top-3 mean (smaller, professional)
        ax2.scatter(subj_count, stats['top3_mean'], marker='D', s=25, 
                   color='#FF6B35', edgecolor='black', linewidth=0.5, 
                   alpha=0.9, zorder=3, label='Top-3 mean' if subj_count == subject_counts[0] else "")
        
        # Triangle for single best (replaced star with small triangle)
        ax2.scatter(subj_count, stats['single_best'], marker='^', s=30, 
                   color='#FFD700', edgecolor='black', linewidth=0.5, 
                   alpha=0.9, zorder=3, label='Single best' if subj_count == subject_counts[0] else "")
        
        # Success rate annotation (gated behind flag)
        if SHOW_SUCCESS_RATE:
            success_text = f"{stats['n_beats_yasa']}/{stats['n_top10']}"
            ax2.text(subj_count, -0.12, success_text, ha='center', va='center',
                    fontsize=10, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.8))
    
    # Zero line (no improvement over YASA) - thin baseline
    ax2.axhline(y=0, color=CB_COLORS['yasa'], linestyle='--', linewidth=1.2, 
               alpha=0.8, zorder=0.5)
    
    ax2.set_xlabel('Number of training subjects (calibration cohort size)', fontweight='bold')
    ax2.set_ylabel('Δκ vs YASA', fontweight='bold')
    ax2.set_title('Panel B: Top-Performer Reliability Analysis\n(Best run per subject from top-10)', 
                  fontweight='bold', pad=15)
    
    # Create legend with all elements (updated to match new styling)
    legend_elements = [
        plt.Line2D([0], [0], color=CB_COLORS['yasa'], linestyle='--', linewidth=1.2, 
                  alpha=0.8, label='YASA baseline (Δκ=0)'),
        plt.Line2D([0], [0], marker='D', color='#FF6B35', linewidth=0, markersize=5, 
                  markerfacecolor='#FF6B35', markeredgecolor='black', markeredgewidth=0.5, 
                  alpha=0.9, label='Top-3 mean'),
        plt.Line2D([0], [0], marker='^', color='#FFD700', linewidth=0, markersize=6, 
                  markerfacecolor='#FFD700', markeredgecolor='black', markeredgewidth=0.5, 
                  alpha=0.9, label='Single best'),
        plt.Rectangle((0,0),1,1, facecolor='#4CAF50', alpha=0.3, edgecolor='black', 
                     linewidth=0.6, label='Best-per-subject distribution')
    ]
    
    # Position legend outside plot area to prevent overlap
    leg = ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1), 
                    borderaxespad=0, frameon=False)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_axisbelow(True)
    
    # Set x-axis to match upper panel with margins for clean layout
    ax2.set_xticks(x_ticks)
    ax2.set_xlim(0, max_subjects + 2)
    ax2.margins(x=0.05)  # Prevent x-tick labels from colliding with figure border
    
    # Set y-axis limits (adjust based on whether annotations are shown)
    if top_df['delta_kappa'].size > 0:
        if SHOW_SUCCESS_RATE:
            y_min_panel2 = min(-0.18, top_df['delta_kappa'].min() - 0.05)
        else:
            y_min_panel2 = top_df['delta_kappa'].min() - 0.05
        y_max_panel2 = top_df['delta_kappa'].max() + 0.05
        ax2.set_ylim(y_min_panel2, y_max_panel2)
    
    # Adjust layout to prevent legend overlap and provide space for future notes
    plt.subplots_adjust(right=0.82, bottom=0.18)
    
    # Optional: Add explanatory footer only if success rate is shown
    if SHOW_SUCCESS_RATE:
        fig.text(0.5, 0.02, 'Numbers below violins show success rate: top-10 runs beating YASA baseline at each subject count', 
                 ha='center', va='bottom', fontsize=10, style='italic', alpha=0.8)
    
    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_svg = output_dir / 'figure_1_2_combined_calibration_analysis.svg'
    
    plt.savefig(fig_svg)
    print(f"Saved: {fig_svg}")
    
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

    setup_plotting_style()
    
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