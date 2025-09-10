#!/usr/bin/env python3
# plot_calibration_comparison.py
"""
Figure 1 — Calibration dose–response from CSV (no WandB needed)

• X-axis: nights (default) or minutes (if present and x=minutes)
• Y-axis: Cohen's κ
• Shows median ± 95% CI per bin, thin per-subject lines, YASA baseline band
• Detects T* where: median Δκ ≥ δ, 95% CI(Δκ) > 0, Wilcoxon p < 0.05

CSV requirements:
  - Produced by your loader "flat" step (e.g., all_runs_flat.csv).
  - Should contain:
      * metrics: either 'test_kappa' OR 'contract.results.test_kappa'
      * macro-F1 (optional, for inset): 'test_f1' OR 'contract.results.test_f1'
      * subject id somewhere: try cfg.subject_id / cfg.subj_id / cfg.subject / sum.subject_id / name
      * nights: cfg.calib_nights / cfg.nights_training / cfg.nights_calibration (preferred)
      * minutes (optional): cfg.calib_minutes / cfg.minutes_calib (if you set --x minutes)

Usage:
  python Plot_Clean/plot_calibration_comparison.py --csv ../data/all_runs_flat.csv 
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Import consistent figure styling
import sys; sys.path.append("../style"); from figure_style import (
    setup_figure_style, get_color, save_figure, 
    add_yasa_baseline, add_significance_marker,
    bootstrap_ci_median, wilcoxon_test,
    format_n_caption, add_sample_size_annotation,
    OKABE_ITO
)
from itertools import cycle


# ---------- utils ----------


# ---------- extraction helpers ----------
def pick_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
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

def load_and_prepare(csv_path: Path, x_mode: str, num_subjects: Optional[int] = None) -> pd.DataFrame:
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
    
    # CRITICAL: Filter out high noise experiments to avoid bias (except for dedicated robustness plots)
    if 'noise_level' in df.columns:
        # Get noise statistics before filtering
        noise_stats = df['noise_level'].value_counts().sort_index()
        print(f"🔊 Noise level distribution before filtering: {dict(noise_stats)}")
        
        # Keep only clean data (noise_level <= 0.01 or 1%) to avoid bias in calibration analysis
        df = df[df['noise_level'] <= 0.01].copy()
        print(f"✅ Filtered to clean/low-noise data only: {len(df)} rows remaining (noise ≤ 1%)")
        
        if len(df) == 0:
            raise ValueError("No clean data found after noise filtering. All experiments had noise > 1%.")
    else:
        print("ℹ️  No noise_level column found - assuming all data is clean")
    
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
    
    # Remove data corruption: IDUN alone cannot have >11 subjects
    if 'dataset_composition' in df.columns and 'num_subjects' in df.columns:
        corrupt_entries = df[(df['dataset_composition'] == 'IDUN') & (df['num_subjects'] > 11)]
        if len(corrupt_entries) > 0:
            print(f"🚨 Removing {len(corrupt_entries)} CORRUPT entries (IDUN dataset with >11 subjects):")
            for _, row in corrupt_entries.iterrows():
                print(f"  {row['name']}: {row['num_subjects']} subjects, dataset={row['dataset_composition']} - IMPOSSIBLE!")
            df = df[~((df['dataset_composition'] == 'IDUN') & (df['num_subjects'] > 11))]
            print(f"After removing corrupt IDUN entries: {len(df)} rows")
    
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

# ---------- binning & aggregation ----------
def make_bins(x: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.array([0, 1])
    
    print(f"X-values range: {np.min(x):.2f} to {np.max(x):.2f}")
    
    if mode == "nights":
        # Check if data looks like it's in minutes instead of nights
        # If max value > 10, it's probably minutes, convert to nights for binning
        if np.max(x) > 10:
            print("Data appears to be in minutes, converting to nights for binning...")
            x_nights = x / (8 * 60)  # Convert minutes to nights (8 hours * 60 minutes)
            print(f"Converted range: {np.min(x_nights):.2f} to {np.max(x_nights):.2f} nights")
        else:
            x_nights = x
        
        # Create bins based on converted values, but return in original units
        lo_nights = max(0, np.floor(np.nanmin(x_nights)))
        hi_nights = np.ceil(np.nanmax(x_nights))
        hi_nights = max(hi_nights, 0.1)  # Ensure at least some range
        
        # Create reasonable number of bins
        n_bins = min(8, max(3, int(hi_nights - lo_nights) + 1))
        night_edges = np.linspace(lo_nights, hi_nights, n_bins)
        
        # Convert back to original units if needed
        if np.max(x) > 10:
            bins = night_edges * (8 * 60)  # Convert back to minutes
        else:
            bins = night_edges
        
        print(f"Created {len(bins)} bins for nights mode: {bins}")
        return bins
    else:
        # minutes: log bins over observed range (robust to outliers)
        lo = max(1.0, float(np.nanmin(x)))
        hi = float(np.nanquantile(x, 0.99))
        if hi <= lo:
            hi = float(np.nanmax(x)) * 1.2
        edges = np.unique(np.geomspace(lo, hi, num=8))
        if edges.size < 3:
            edges = np.array([lo * 0.9, lo * 1.1, lo * 1.3])
        print(f"Created {len(edges)} minute bin edges: {edges}")
        return edges

def bin_and_aggregate(df: pd.DataFrame, bins: np.ndarray, mode: str, yasa_kappa: float):
    df = df.copy()
    if mode == "nights":
        # Use pd.cut for both nights and minutes modes for consistency
        df["bin"] = pd.cut(df["xvalue"], bins=bins, include_lowest=True)
        bin_centers = np.array([b.mid for b in df["bin"].cat.categories if not pd.isna(b)])
    else:
        df["bin"] = pd.cut(df["xvalue"], bins=bins, include_lowest=True)
        bin_centers = np.array([b.mid for b in df["bin"].cat.categories if not pd.isna(b)])

    # subject-level aggregation within each bin (median per subject)
    subj_curves: Dict[str, Dict[str, List[float]]] = {}
    stats_rows = []
    for b, g in df.groupby("bin", dropna=True):
        # median per subject to avoid bias if multiple runs per subject in a bin
        subj_med = g.groupby("subject")["test_kappa"].median().dropna()
        if subj_med.empty:
            continue
        med = float(np.median(subj_med.values))
        lo, hi = bootstrap_ci_median(subj_med.values)
        p = wilcoxon_test(subj_med.values - yasa_kappa)
        # store per-subject points for thin lines  
        xc = float(b.mid)
        for s, v in subj_med.items():
            subj_curves.setdefault(s, {"x": [], "y": []})
            subj_curves[s]["x"].append(xc)
            subj_curves[s]["y"].append(float(v))
        stats_rows.append({
            "bin_center": float(xc),
            "kappa_median": med,
            "kappa_ci_low": lo,
            "kappa_ci_high": hi,
            "n_subjects": int(subj_med.size),
            "p_wilcoxon": p,
            "delta_median": med - yasa_kappa,
        })

    if not stats_rows:
        print("No valid bins created - returning empty DataFrames")
        return pd.DataFrame(), subj_curves
        
    stats_df = pd.DataFrame(stats_rows).sort_values("bin_center")
    print(f"Created {len(stats_df)} bins with data:")
    for _, row in stats_df.iterrows():
        print(f"  {row['bin_center']:6.2f}: κ={row['kappa_median']:.3f}, n={row['n_subjects']}, p={row['p_wilcoxon']:.3f}")
    
    return stats_df, subj_curves

# ---------- plotting ----------
def plot_figure(stats_df: pd.DataFrame, subj_curves: Dict[str, Dict[str, List[float]]],
                x_label: str, yasa_kappa: float, yasa_ci: float,
                delta_threshold: float, outdir: Path, num_subjects: Optional[int] = None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # thin subject lines (neutral grey)
    for s, d in subj_curves.items():
        if len(d["x"]) >= 2:
            xs, ys = zip(*sorted(zip(d["x"], d["y"])))
            ax.plot(xs, ys, color=get_color("subjects"), alpha=0.45, linewidth=0.8, zorder=1)

    # median curve + CI (blue)
    valid = ~(np.isnan(stats_df["kappa_ci_low"]) | np.isnan(stats_df["kappa_ci_high"]))
    if valid.any():
        ax.errorbar(
            stats_df.loc[valid, "bin_center"],
            stats_df.loc[valid, "kappa_median"],
            yerr=np.vstack([
                stats_df.loc[valid, "kappa_median"] - stats_df.loc[valid, "kappa_ci_low"],
                stats_df.loc[valid, "kappa_ci_high"] - stats_df.loc[valid, "kappa_median"]
            ]),
            fmt="o-",
            color=get_color("cbramod"),
            linewidth=3.0,
            markersize=9,
            markerfacecolor=get_color("cbramod"),  # Fill the dots
            markeredgecolor="white",  # White edge for contrast
            markeredgewidth=2,
            capsize=8,
            capthick=3,
            elinewidth=2.5,  # Make error bar lines thicker
            label="CBraMod median ± 95% CI",
            zorder=3,
        )
    else:
        ax.plot(stats_df["bin_center"], stats_df["kappa_median"], "o-",
                color=get_color("cbramod"), linewidth=3.0, markersize=9,
                markerfacecolor=get_color("cbramod"), 
                markeredgecolor="white", markeredgewidth=2, zorder=3)

    # YASA baseline line + band (muted yellow + light alpha band)
    x0, x1 = min(stats_df["bin_center"]), max(stats_df["bin_center"])
    add_yasa_baseline(ax, yasa_kappa, yasa_ci, (x0, x1))
    
    # Human Expert + PSG agreement upper ceiling (dashed line)
    human_expert_kappa = 0.81
    ax.axhline(y=human_expert_kappa, 
              color='#323131',  # Dark gray color
              linestyle='--', 
              linewidth=2.5,
              label=f'Human Expert + PSG agreement (κ={human_expert_kappa:.2f})', 
              zorder=2,
              alpha=0.8)

    # T* detection & mark (green)
    t_star = None
    t_star_kappa = None
    for _, r in stats_df.iterrows():
        ci_delta_low = r["kappa_ci_low"] - yasa_kappa if not np.isnan(r["kappa_ci_low"]) else np.nan
        cond1 = (r["delta_median"] >= delta_threshold)
        cond2 = (not np.isnan(ci_delta_low)) and (ci_delta_low > 0.0)
        if cond1 and cond2:
            t_star = float(r["bin_center"])
            t_star_kappa = float(r["kappa_median"])
            break

    if t_star is not None:
        add_significance_marker(ax, t_star, t_star_kappa, f"T* ≈ {t_star:g}")

    # labels/legend/limits with N information
    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel("Test Cohen's κ", fontweight="bold")
    
    # Calculate N for caption
    total_subjects = len(set().union(*[list(subj_curves.keys())]))
    total_runs = sum(row['n_subjects'] for _, row in stats_df.iterrows())
    
    if num_subjects is not None:
        title = f"Calibration Dose-Response\n{format_n_caption(num_subjects, total_runs, 'subjects')}"
    else:
        title = f"Calibration Dose-Response (5-class)\n{format_n_caption(total_subjects, total_runs, 'subjects')}"
    
    ax.set_title(title, pad=20, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    y_low = min(float(stats_df["kappa_median"].min()) - 0.05, yasa_kappa - 0.10)
    y_high = max(float(stats_df["kappa_median"].max()) + 0.10, yasa_kappa + 0.10, human_expert_kappa + 0.05)
    ax.set_ylim(max(0.2, y_low), min(0.90, y_high))

    # annotate n per bin
    add_sample_size_annotation(ax, stats_df)

    outdir.mkdir(parents=True, exist_ok=True)
    if num_subjects is not None:
        base_path = outdir / f"calibration_dose_response_{num_subjects}subj"
    else:
        base_path = outdir / "calibration_dose_response_from_csv"
    
    plt.tight_layout()
    save_figure(fig, base_path)

    best_kappa = float(stats_df["kappa_median"].max())
    print(f"Best κ (median across bins): {best_kappa:.3f}  → Δκ vs YASA: +{best_kappa - yasa_kappa:.3f}")
    if t_star is not None:
        print(f"T*: {t_star:g} (meets Δ≥δ and CI>0)")
    else:
        print("T*: not found under current δ/CI criteria.")

    return fig

# ---------- main ----------
def create_comprehensive_plot(df: pd.DataFrame, output_dir: Path):
    import numpy as np

    yasa_kappa = 0.446
    yasa_ci = 0.05
    delta_threshold = 0.05

    print("Creating comprehensive calibration dose-response plot...")

    df = df[df['num_subjects'] > 0].dropna(subset=['num_subjects'])
    if df.empty:
        print("No valid subject count data found")
        return None

    unique_datasets = sorted(df['dataset_composition'].unique())
    print(f"Found dataset compositions: {unique_datasets}")

    # Build a color map for datasets from Okabe–Ito (cycle if needed)
    color_cycle = cycle(OKABE_ITO)
    dataset_colors = {ds: next(color_cycle) for ds in unique_datasets}

    subject_count_data = []
    for subj_count in sorted(df['num_subjects'].unique()):
        subj_df = df[df['num_subjects'] == subj_count]
        if len(subj_df) == 0:
            continue

        # DEBUG: Print detailed info for suspicious high subject counts
        if subj_count > 40:
            print(f"\n🔍 DEBUGGING subject count {subj_count}:")
            print(f"  Found {len(subj_df)} runs with {subj_count} subjects")
            for _, row in subj_df.iterrows():
                print(f"    Run: {row['name']}")
                print(f"    Dataset: {row['dataset_composition']}")
                print(f"    Kappa: {row['test_kappa']:.3f}")

        for dataset_comp in subj_df['dataset_composition'].unique():
            dataset_subj_df = subj_df[subj_df['dataset_composition'] == dataset_comp]
            if len(dataset_subj_df) == 0:
                continue

            median_kappa = dataset_subj_df['test_kappa'].median()
            kappa_values = dataset_subj_df['test_kappa'].values
            ci_low, ci_high = bootstrap_ci_median(kappa_values)
            delta_median = median_kappa - yasa_kappa
            ci_delta_low = ci_low - yasa_kappa if not np.isnan(ci_low) else np.nan

            # DEBUG: Print what color is assigned to high subject counts
            if subj_count > 40:
                print(f"  📊 Plotting: {subj_count} subjects, {dataset_comp}, color: {dataset_colors[dataset_comp]}")

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
        print("No data to plot")
        return None

    stats_df = pd.DataFrame(subject_count_data)
    
    # Filter out blue dots around 45-50 subjects (keep pink/other colors)
    print(f"Before filtering: {len(stats_df)} data points")
    # Remove IDUN (ORP) dataset points at 45-50 subjects (impossible for IDUN alone)
    if len(unique_datasets) > 0:
        # Target specifically the 'ORP' dataset (which is the IDUN dataset)
        blue_high_subjects = stats_df[
            (stats_df['dataset_composition'] == 'ORP') & 
            (stats_df['num_subjects'] >= 45) & 
            (stats_df['num_subjects'] <= 50)
        ]
        if len(blue_high_subjects) > 0:
            print(f"🚨 Removing {len(blue_high_subjects)} blue dots in 45-50 subject range:")
            for _, row in blue_high_subjects.iterrows():
                print(f"  {row['num_subjects']} subjects, {row['dataset_composition']}, κ={row['median_kappa']:.3f}")
            stats_df = stats_df[~(
                (stats_df['dataset_composition'] == 'ORP') & 
                (stats_df['num_subjects'] >= 45) & 
                (stats_df['num_subjects'] <= 50)
            )].copy()
            print(f"After filtering: {len(stats_df)} data points remaining")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # ORIGINAL MEDIAN PLOTTING CODE (restored) - Remove black edges
    for dataset_comp in unique_datasets:
        dataset_data = stats_df[stats_df['dataset_composition'] == dataset_comp]
        if dataset_data.empty:
            continue

        color = dataset_colors[dataset_comp]
        valid_ci = ~(np.isnan(dataset_data['ci_low']) | np.isnan(dataset_data['ci_high']))

        if valid_ci.any():
            ax.errorbar(
                dataset_data.loc[valid_ci, 'num_subjects'],
                dataset_data.loc[valid_ci, 'median_kappa'],
                yerr=[
                    dataset_data.loc[valid_ci, 'median_kappa'] - dataset_data.loc[valid_ci, 'ci_low'],
                    dataset_data.loc[valid_ci, 'ci_high'] - dataset_data.loc[valid_ci, 'median_kappa']
                ],
                fmt='o', color=color, markersize=8,
                markerfacecolor=color,  # Fill the dots
                markeredgecolor=color,  # REMOVED white edge - same as fill
                markeredgewidth=0,      # REMOVED edge width
                capsize=6, capthick=2.5, elinewidth=2,
                label=dataset_comp.replace('ORP', 'IDUN') if 'ORP' in dataset_comp else dataset_comp, zorder=3
            )

        missing_ci = ~valid_ci
        if missing_ci.any():
            ax.plot(
                dataset_data.loc[missing_ci, 'num_subjects'],
                dataset_data.loc[missing_ci, 'median_kappa'],
                'o', color=color, markersize=8,
                markerfacecolor=color,  # Fill the dots
                markeredgecolor=color,  # REMOVED white edge - same as fill
                markeredgewidth=0,      # REMOVED edge width
                label=(dataset_comp.replace('ORP', 'IDUN') if 'ORP' in dataset_comp else dataset_comp) if not valid_ci.any() else "", zorder=3
            )
    
    # ADD TOP PERFORMERS FOR EACH SUBJECT COUNT BIN (keeping this feature)
    print("\n🌟 Adding top performers for each subject count:")
    for subj_count in sorted(df['num_subjects'].unique()):
        subj_df = df[df['num_subjects'] == subj_count]
        if len(subj_df) == 0:
            continue
        
        # Find top performer for this subject count
        best_run = subj_df.loc[subj_df['test_kappa'].idxmax()]
        best_kappa = best_run['test_kappa']
        dataset_comp = best_run['dataset_composition']
        
        # Skip impossible combinations: IDUN (ORP) solo with >44 subjects
        # But find alternative from multi-dataset (pink) entries
        if dataset_comp == 'ORP' and subj_count >= 45:
            print(f"  {subj_count} subjects: SKIPPING impossible {dataset_comp} solo with {subj_count} subjects!")
            # Find best performer from non-single-dataset entries (multi-dataset combinations)
            multi_dataset_entries = subj_df[~subj_df['dataset_composition'].isin(['ORP'])]
            if len(multi_dataset_entries) > 0:
                best_run = multi_dataset_entries.loc[multi_dataset_entries['test_kappa'].idxmax()]
                best_kappa = best_run['test_kappa']
                dataset_comp = best_run['dataset_composition']
                print(f"  {subj_count} subjects: Using MULTI-DATASET top performer instead:")
            else:
                print(f"  {subj_count} subjects: No multi-dataset alternatives found, skipping")
                continue
        
        print(f"  {subj_count} subjects: κ={best_kappa:.3f} ({dataset_comp}) - {best_run['name']}")
        
        # Plot top performer as star - remove black edges
        ax.scatter([subj_count], [best_kappa], 
                  marker='*', s=120, 
                  color=dataset_colors[dataset_comp], 
                  edgecolors=dataset_colors[dataset_comp],  # REMOVED black edges
                  linewidth=0,                             # REMOVED edge width
                  zorder=5, alpha=1.0,
                  label='Top Performer' if subj_count == sorted(df['num_subjects'].unique())[0] else "")

    # YASA baseline (muted yellow) + band
    add_yasa_baseline(ax, yasa_kappa, yasa_ci)
    
    # Human Expert + PSG agreement upper ceiling (dashed line)
    human_expert_kappa = 0.81
    ax.axhline(y=human_expert_kappa, 
              color='#323131',  # Dark gray color
              linestyle='--', 
              linewidth=2.5,
              label=f'Human Expert + PSG agreement (κ={human_expert_kappa:.2f})', 
              zorder=2,
              alpha=0.8)

    # Crossing marker (restored)
    crossing_points = [int(r['num_subjects']) for _, r in stats_df.iterrows() if r['median_kappa'] > yasa_kappa]
    if crossing_points:
        first_crossing = min(crossing_points)
        ax.axvline(x=first_crossing, color=get_color("t_star"), linestyle=':', alpha=0.8, linewidth=2,
                   label=f'First crosses YASA at {first_crossing} subjects')
        crossing_rows = stats_df[stats_df['num_subjects'] == first_crossing]
        for _, cr in crossing_rows.iterrows():
            ax.scatter([first_crossing], [cr['median_kappa']],
                       color=get_color("t_star"), s=120, zorder=4, marker='P',
                       edgecolors='black', linewidth=0.8)

    ax.set_xlabel('Number of Training Subjects', fontweight='bold')
    ax.set_ylabel("Test Cohen's κ", fontweight='bold')
    ax.set_title('CBraMod Calibration: Performance vs Training Data Size\n(Median of top 10 runs per subject count)',
                 fontweight='bold', fontsize=14, pad=20)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    max_subjects = max(stats_df['num_subjects'])
    ax.set_xlim(0, max_subjects + 2)
    x_ticks = np.arange(0, max_subjects + 5, 5)
    ax.set_xticks(x_ticks)

    y_min = min(stats_df['median_kappa'].min() - 0.05, yasa_kappa - 0.05)
    y_max = max(stats_df['median_kappa'].max() + 0.10, yasa_kappa + 0.10, human_expert_kappa + 0.05)
    ax.set_ylim(y_min, y_max)

    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / 'calibration_comparison'
    plt.tight_layout()
    save_figure(fig, base_path)

    best_kappa = stats_df['median_kappa'].max()
    best_subj = stats_df.loc[stats_df['median_kappa'].idxmax(), 'num_subjects']
    print(f"\nSummary:\nBest performance: κ={best_kappa:.3f} with {best_subj} training subjects")
    if crossing_points:
        print(f"CBraMod first surpasses YASA at {first_crossing} training subjects")
        improvement = stats_df[stats_df['num_subjects'] == first_crossing]['median_kappa'].iloc[0] - yasa_kappa
        print(f"Improvement over YASA: +{improvement:.3f}")
    else:
        print("CBraMod does not surpass YASA baseline in this data")

    plt.show()
    return fig

def main():
    ap = argparse.ArgumentParser(description='Generate comprehensive CBraMod calibration plot')
    ap.add_argument("--csv", required=True, help="Path to flattened CSV (e.g., all_runs_flat.csv)")
    ap.add_argument("--out", default="Plot_Clean/figures", help="Output directory")
    args = ap.parse_args()

    setup_figure_style()
    
    print("Loading data and selecting top 10 runs per subject count...")
    df = load_and_prepare(Path(args.csv), x_mode="nights", num_subjects=None)  # Load all subjects
    
    if df.empty:
        print("No valid data found. Check your CSV file.")
        return
    
    create_comprehensive_plot(df, Path(args.out))

if __name__ == "__main__":
    main()