#!/usr/bin/env python3
# fig1_from_csv.py
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
  python fig1_from_csv.py --csv Plot_Clean/data/all_runs_flat.csv --x nights --yasa-kappa 0.446 --delta 0.05
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---- add near imports ----
from itertools import cycle

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

# Named roles for key elements
CB_COLORS = {
    "median_curve": "#0072B2",   # blue
    "yasa":         "#D55E00",   # vermillion
    "t_star":       "#009E73",   # green
    "subjects":     "#8C8C8C",   # neutral grey for per-subject thin lines
}


# ---------- utils ----------
def bootstrap_ci_median(arr: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 3:
        return (np.nan, np.nan)
    try:
        from scipy.stats import bootstrap
        rng = np.random.default_rng(42)
        res = bootstrap((arr,), np.median, n_resamples=2000,
                        confidence_level=confidence, random_state=rng, method="BCa")
        return float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception:
        rng = np.random.default_rng(42)
        meds = [np.median(rng.choice(arr, size=arr.size, replace=True)) for _ in range(2000)]
        lo = np.percentile(meds, (1 - confidence) / 2 * 100)
        hi = np.percentile(meds, (1 + confidence) / 2 * 100)
        return float(lo), float(hi)

def wilcoxon_signed_rank(delta_vec: np.ndarray) -> Optional[float]:
    try:
        from scipy.stats import wilcoxon
        deltas = np.asarray(delta_vec, dtype=float)
        deltas = deltas[~np.isnan(deltas)]
        if deltas.size < 1:  # Relaxed from 1 for small samples
            return np.nan
        # zero_method='wilcox' drops zeros; use alternative two-sided
        stat, p = wilcoxon(deltas, zero_method="wilcox", alternative="two-sided", correction=False, mode="auto")
        return float(p)
    except Exception:
        return np.nan
    
def setup_plotting_style():
    plt.style.use("default")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        # Use colorblind-friendly cycle by default
        "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
    })


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
        p = wilcoxon_signed_rank(subj_med.values - yasa_kappa)
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
            ax.plot(xs, ys, color=CB_COLORS["subjects"], alpha=0.45, linewidth=0.8, zorder=1)

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
            color=CB_COLORS["median_curve"],
            linewidth=2.5,
            markersize=7,
            markerfacecolor="white",
            markeredgecolor=CB_COLORS["median_curve"],
            capsize=5,
            capthick=2,
            label="CBraMod median ± 95% CI",
            zorder=3,
        )
    else:
        ax.plot(stats_df["bin_center"], stats_df["kappa_median"], "o-",
                color=CB_COLORS["median_curve"], linewidth=2.5, markersize=7,
                markerfacecolor="white", markeredgecolor=CB_COLORS["median_curve"], zorder=3)

    # YASA baseline line + band (vermillion + light alpha band)
    ax.axhline(y=yasa_kappa, color=CB_COLORS["yasa"], linestyle="--", linewidth=2.5,
               label=f"YASA baseline (κ={yasa_kappa:.3f})", zorder=2)
    x0, x1 = min(stats_df["bin_center"]), max(stats_df["bin_center"])
    ax.fill_between([x0, x1], yasa_kappa - yasa_ci, yasa_kappa + yasa_ci,
                    color=CB_COLORS["yasa"], alpha=0.15, zorder=1)

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
        ax.scatter([t_star], [t_star_kappa], color=CB_COLORS["t_star"], s=110, zorder=4,
                   marker="P", edgecolors="black", linewidth=0.8)
        ax.annotate(
            f"T* ≈ {t_star:g}",
            xy=(t_star, t_star_kappa),
            xytext=(t_star * 1.15, t_star_kappa + 0.05),
            arrowprops=dict(arrowstyle="->", color=CB_COLORS["t_star"], lw=2),
            fontsize=12, fontweight="bold", color=CB_COLORS["t_star"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
        )

    # labels/legend/limits
    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel("Cohen's κ", fontweight="bold")
    title = (f"Figure 1: Calibration Dose-Response ({num_subjects} subjects, Top 10 Runs)"
             if num_subjects is not None else
             "Figure 1: Calibration Dose-Response (5-class, Top 10 Runs)")
    ax.set_title(title, pad=16, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    y_low = min(float(stats_df["kappa_median"].min()) - 0.05, yasa_kappa - 0.10)
    y_high = max(float(stats_df["kappa_median"].max()) + 0.10, yasa_kappa + 0.10)
    ax.set_ylim(max(0.2, y_low), min(0.95, y_high))

    # annotate n per bin
    ymin, ymax = ax.get_ylim()
    y_pos = ymin + 0.02 * (ymax - ymin)
    for _, r in stats_df.iterrows():
        ax.text(r["bin_center"], y_pos, f"n={int(r['n_subjects'])}",
                ha="center", va="bottom", fontsize=9, alpha=0.85,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75))

    outdir.mkdir(parents=True, exist_ok=True)
    if num_subjects is not None:
        fp_svg = outdir / f"figure_1_calibration_dose_response_{num_subjects}subj.svg"
    else:
        fp_svg = outdir / "figure_1_calibration_dose_response_from_csv.svg"
    plt.tight_layout()
    plt.savefig(fp_svg)
    print(f"Saved: {fp_svg}")

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
        print("No data to plot")
        return None

    stats_df = pd.DataFrame(subject_count_data)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

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
                fmt='o', color=color, markersize=7,
                markerfacecolor="white", markeredgecolor=color,
                capsize=4, capthick=1.5, label=dataset_comp, zorder=3
            )

        missing_ci = ~valid_ci
        if missing_ci.any():
            ax.plot(
                dataset_data.loc[missing_ci, 'num_subjects'],
                dataset_data.loc[missing_ci, 'median_kappa'],
                'o', color=color, markersize=7,
                markerfacecolor="white", markeredgecolor=color,
                label=dataset_comp if not valid_ci.any() else "", zorder=3
            )

    # YASA baseline (vermillion) + band
    ax.axhline(y=yasa_kappa, color=CB_COLORS["yasa"], linestyle='--', linewidth=2.5,
               label=f'YASA baseline (κ={yasa_kappa:.3f})', zorder=2)

    # Crossing marker (use green)
    crossing_points = [int(r['num_subjects']) for _, r in stats_df.iterrows() if r['median_kappa'] > yasa_kappa]
    if crossing_points:
        first_crossing = min(crossing_points)
        ax.axvline(x=first_crossing, color=CB_COLORS["t_star"], linestyle=':', alpha=0.8, linewidth=2,
                   label=f'First crosses YASA at {first_crossing} subjects')
        crossing_rows = stats_df[stats_df['num_subjects'] == first_crossing]
        for _, cr in crossing_rows.iterrows():
            ax.scatter([first_crossing], [cr['median_kappa']],
                       color=CB_COLORS["t_star"], s=120, zorder=4, marker='P',
                       edgecolors='black', linewidth=0.8)

    ax.set_xlabel('Number of Training Subjects', fontweight='bold')
    ax.set_ylabel("Cohen's κ", fontweight='bold')
    ax.set_title('CBraMod Calibration: Performance vs Training Data Size\n(Top 10 runs per subject count)',
                 fontweight='bold', fontsize=14, pad=20)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    max_subjects = max(stats_df['num_subjects'])
    ax.set_xlim(0, max_subjects + 2)
    x_ticks = np.arange(0, max_subjects + 5, 5)
    ax.set_xticks(x_ticks)

    y_min = min(stats_df['median_kappa'].min() - 0.05, yasa_kappa - 0.05)
    y_max = max(stats_df['median_kappa'].max() + 0.10, yasa_kappa + 0.10)
    ax.set_ylim(y_min, y_max)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_svg = output_dir / 'figure_1_calibration_comprehensive.svg'
    plt.tight_layout()
    plt.savefig(fig_svg)
    print(f"Saved: {fig_svg}")

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
    ap.add_argument("--out", default="artifacts/results/figures/paper", help="Output directory")
    args = ap.parse_args()

    setup_plotting_style()
    
    print("Loading data and selecting top 10 runs per subject count...")
    df = load_and_prepare(Path(args.csv), x_mode="nights", num_subjects=None)  # Load all subjects
    
    if df.empty:
        print("No valid data found. Check your CSV file.")
        return
    
    create_comprehensive_plot(df, Path(args.out))

if __name__ == "__main__":
    main()