#!/usr/bin/env python3
"""
Publication-ready multi-panel figure analyzing task granularity effects:
How does 4-class vs 5-class sleep staging affect performance on single-channel ear-EEG?

Creates 4 panels:
- Panel A: Paired slope plot per subject
- Panel B: Dose-response curves by task
- Panel C: Distribution of Δκ
- Panel D: Mixed-effects analysis (optional)

Usage:
    python fig_task_granularity.py --csv Plot_Clean/data/all_runs_flat.csv --out Plot_Clean/figures/fig1
"""

import argparse
import warnings
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon
import json

# Optional imports
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Style constants
COLORS = {
    '4c': '#1f77b4',  # Blue
    '5c': '#A23B72',  # Magenta
    'delta': '#666666'  # Gray for delta plots
}

FIGURE_SIZE = (16, 12)  # 4-panel figure
DPI = 300

def setup_style():
    """Setup matplotlib style for publication-ready figures."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

def load_flat_csv(path):
    """Load and return flattened CSV data."""
    if not Path(path).exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    
    return df

def resolve_columns(df):
    """
    Resolve columns and derive consistent calibration units.
    Returns df with standard cols: test_kappa, test_f1, num_classes, label_scheme,
    subject, hours_data (true hours), nights, minutes.
    """
    df_resolved = df.copy()

    # --- test_kappa ---
    kappa_cols = ['test_kappa', 'contract.results.test_kappa']
    kappa_cols += [c for c in df.columns if 'test' in c.lower() and 'kappa' in c.lower()]
    kappa_col = next((c for c in kappa_cols if c in df.columns and not df[c].isna().all()), None)
    if not kappa_col: raise ValueError("Could not find test_kappa column")
    df_resolved['test_kappa'] = pd.to_numeric(df[kappa_col], errors='coerce')

    # --- test_f1 (optional) ---
    f1_cols = ['test_f1', 'contract.results.test_f1']
    f1_cols += [c for c in df.columns if 'test' in c.lower() and 'f1' in c.lower()]
    f1_col = next((c for c in f1_cols if c in df.columns and not df[c].isna().all()), None)
    df_resolved['test_f1'] = pd.to_numeric(df[f1_col], errors='coerce') if f1_col else np.nan

    # --- val_kappa (for model selection) ---
    val_cols = ['val_kappa','contract.results.val_kappa']
    val_cols += [c for c in df.columns if 'val' in c.lower() and 'kappa' in c.lower()]
    val_col = next((c for c in val_cols if c in df.columns and not df[c].isna().all()), None)
    df_resolved['val_kappa'] = pd.to_numeric(df[val_col], errors='coerce') if val_col else np.nan

    # --- num_classes ---
    nc_cols = ['num_classes', 'contract.results.num_classes', 'cfg.num_of_classes']
    nc_col = next((c for c in nc_cols if c in df.columns and not df[c].isna().all()), None)
    if not nc_col: raise ValueError("Could not find num_classes column")
    df_resolved['num_classes'] = pd.to_numeric(df[nc_col], errors='coerce')

    # --- label_scheme ---
    ls_cols = ['label_scheme', 'contract.results.label_scheme']
    ls_col = next((c for c in ls_cols if c in df.columns and not df[c].isna().all()), None)
    if ls_col:
        df_resolved['label_scheme'] = df[ls_col].astype(str)
    else:
        df_resolved['label_scheme'] = df_resolved['num_classes'].map({5:'5c',4:'4c'}).fillna('4c-unknown')
        print("Warning: Inferred label_scheme from num_classes")

    # --- subject ---
    subject_cols = ['cfg.subject_id','cfg.subj_id','cfg.subject','sum.subject_id','subject_id','name']
    subj_col = next((c for c in subject_cols if c in df.columns and df[c].notna().any()), None)
    if not subj_col: raise ValueError("Could not find subject identifier column")
    df_resolved['subject'] = df[subj_col].astype(str)

    # ---- Derive calibration robustly: prefer explicit nights/minutes, then hours ----
    # nights (preferred if present)
    nights_series = None
    for c in ['cfg.calib_nights','cfg.nights_training','cfg.nights_calibration']:
        if c in df.columns and df[c].notna().any():
            nights_series = pd.to_numeric(df[c], errors='coerce'); break

    # minutes → nights
    minutes_series = None
    for c in ['cfg.calib_minutes','cfg.minutes_calib']:
        if c in df.columns and df[c].notna().any():
            minutes_series = pd.to_numeric(df[c], errors='coerce'); break

    # hours (but many tables store MINUTES here; detect and fix)
    hours_series = None
    for c in ['sum.hours_of_data','contract.results.hours_of_data','hours_of_data']:
        if c in df.columns and df[c].notna().any():
            hours_series = pd.to_numeric(df[c], errors='coerce'); break

    # Normalize to true hours
    true_hours = pd.Series(np.nan, index=df.index, dtype=float)
    if nights_series is not None:
        true_hours = nights_series * 8.0
    elif minutes_series is not None:
        true_hours = minutes_series / 60.0
    elif hours_series is not None:
        # Heuristic: if "hours" looks like minutes (median > 48), convert
        med = np.nanmedian(hours_series)
        if med > 48:   # >2 nights worth → likely minutes
            print("Detected hours_of_data appears to be in MINUTES; converting /60.")
            true_hours = hours_series / 60.0
        else:
            true_hours = hours_series
    else:
        raise ValueError("No calibration fields found (nights/minutes/hours)")

    df_resolved['hours_data'] = true_hours
    df_resolved['nights'] = df_resolved['hours_data'] / 8.0
    df_resolved['minutes'] = df_resolved['hours_data'] * 60.0

    print("Column resolution complete:")
    print(f"  - κ: {kappa_col}")
    print(f"  - F1: {f1_col}")
    print(f"  - val_κ: {val_col}")
    print(f"  - num_classes: {nc_col}")
    print(f"  - label_scheme: {ls_col}")
    print(f"  - subject: {subj_col}")
    print(f"  - derived nights range: {df_resolved['nights'].min():.3f}–{df_resolved['nights'].max():.3f}")

    return df_resolved


def apply_filters(df, args):
    """Apply command-line filters to the dataframe."""
    df_filtered = df.copy()
    n_original = len(df_filtered)
    
    # CRITICAL: Filter to only rows with valid subject IDs
    # This is essential for paired comparisons
    df_filtered = df_filtered[df_filtered['subject'].notna()]
    print(f"Filtered to rows with valid subject IDs: {len(df_filtered)}/{n_original}")
    
    if len(df_filtered) == 0:
        raise ValueError("No valid subject IDs found in data!")
    
    # Filter by frozen status
    if args.only_unfrozen and 'contract.model.frozen' in df.columns:
        df_filtered = df_filtered[df_filtered['contract.model.frozen'] == False]
        print(f"Filtered to only unfrozen models: {len(df_filtered)}")
    
    # Filter by pretrained weights usage
    if args.use_pretrained_only and 'cfg.use_pretrained_weights' in df.columns:
        df_filtered = df_filtered[df_filtered['cfg.use_pretrained_weights'] == True]
        print(f"Filtered to only pretrained models: {len(df_filtered)}")
    
    # Filter by nights range
    if args.min_nights:
        df_filtered = df_filtered[df_filtered['nights'] >= args.min_nights]
        print(f"Filtered by min_nights >= {args.min_nights}: {len(df_filtered)}")
    
    if args.max_nights:
        df_filtered = df_filtered[df_filtered['nights'] <= args.max_nights]
        print(f"Filtered by max_nights <= {args.max_nights}: {len(df_filtered)}")
    
    # Filter to only 4-class and 5-class models
    df_filtered = df_filtered[df_filtered['num_classes'].isin([4, 5])]
    print(f"Filtered to 4-class and 5-class models: {len(df_filtered)}")
    
    # Handle 4c scheme variants
    if not args.include_4c_v0:
        df_filtered = df_filtered[~df_filtered['label_scheme'].str.contains('v0', na=False)]
        print(f"Excluded 4c-v0 schemes: {len(df_filtered)}")
    
    if not args.include_4c_v1:
        df_filtered = df_filtered[~df_filtered['label_scheme'].str.contains('v1', na=False)]
        print(f"Excluded 4c-v1 schemes: {len(df_filtered)}")
    
    print(f"Final filtered data: {len(df_filtered)} rows, {df_filtered['subject'].nunique()} unique subjects")
    
    return df_filtered

def choose_panel_bin(paired_df, requested, min_pairs=5):
    """Return a sensible bin for Panels A/C.
    - If requested is numeric: pick the closest bin with >= min_pairs pairs,
      else fall back to the bin with the most pairs.
    """
    if paired_df.empty:
        return None
    # counts per bin (paired subjects)
    counts = (paired_df.groupby('calib_bin')['subject']
              .nunique().sort_index())
    if counts.empty:
        return None
    # try closest with enough pairs
    try:
        req = float(requested)
        bins = counts.index.to_numpy(dtype=float)
        order = np.argsort(np.abs(bins - req))
        for i in order:
            b = float(bins[i])
            if counts.iloc[i] >= min_pairs:
                return b
        # fallback: most pairs
        return float(counts.idxmax())
    except Exception:
        # if requested is "auto" or invalid, pick the best bin
        return float(counts.idxmax())

def bin_calibration(df, mode='nights', max_bins=10):
    print(f"Data range before binning: nights {df['nights'].min():.2f}-{df['nights'].max():.2f}, "
          f"minutes {df['minutes'].min():.0f}-{df['minutes'].max():.0f}")

    if mode == 'nights':
        # integer bins (robust + interpretable)
        lo = int(np.floor(df['nights'].min()))
        hi = int(np.ceil(df['nights'].max()))
        hi = min(hi, 10)  # keep tidy
        centers = np.arange(lo, hi + 1).astype(float)
        bins = np.concatenate([[centers[0]-0.5], (centers[:-1]+centers[1:])/2, [centers[-1]+0.5]])
        df_binned = df.copy()
        df_binned['calib_bin'] = pd.cut(df['nights'], bins=bins, labels=centers, include_lowest=True)
        df_binned['calib_bin'] = pd.to_numeric(df_binned['calib_bin'], errors='coerce')
    else:
        # minutes: log bins
        lo = max(1.0, float(df['minutes'].min()))
        hi = float(df['minutes'].quantile(0.99))
        if hi <= lo: hi = float(df['minutes'].max()) * 1.2
        edges = np.unique(np.geomspace(lo, hi, num=max_bins+1))
        centers = np.sqrt(edges[:-1]*edges[1:])
        df_binned = df.copy()
        df_binned['calib_bin'] = pd.cut(df['minutes'], bins=edges, labels=centers, include_lowest=True)
        df_binned['calib_bin'] = pd.to_numeric(df_binned['calib_bin'], errors='coerce')
        bins = edges

    df_binned = df_binned.dropna(subset=['calib_bin'])
    print(f"Created {len(centers)} bins: {centers}")
    return df_binned, bins, centers


def aggregate_per_subject_bin(df_binned, strategy='best_val'):
    """
    Return one row per (subject, calib_bin, task) using:
      - 'best_val' : pick row with max val_kappa (fall back to test_kappa)
      - 'best_test': pick row with max test_kappa
      - 'median'   : median of test_kappa (old behavior)
    """
    def pick_best(g):
        g = g.copy()
        g['task'] = g['num_classes'].map({4:'4c',5:'5c'})
        if strategy == 'median' or (g['val_kappa'].isna().all() and strategy=='best_val'):
            # robust median across repeats
            out = {
                'test_kappa': g['test_kappa'].median(),
                'test_f1': g['test_f1'].median(),
                'label_scheme': g['label_scheme'].iloc[0],
                'hours_data': g['hours_data'].median(),
                'nights': g['nights'].median(),
                'minutes': g['minutes'].median(),
                'task': g['task'].iloc[0],
            }
            return pd.Series(out)
        # choose best row index
        if strategy == 'best_val':
            idx = g['val_kappa'].astype(float).idxmax()
        else:  # best_test
            idx = g['test_kappa'].astype(float).idxmax()
        row = g.loc[idx]
        out = {
            'test_kappa': float(row['test_kappa']),
            'test_f1': float(row['test_f1']) if pd.notna(row['test_f1']) else np.nan,
            'label_scheme': row['label_scheme'],
            'hours_data': float(row['hours_data']),
            'nights': float(row['nights']),
            'minutes': float(row['minutes']),
            'task': row['task'],
        }
        return pd.Series(out)

    grouped = df_binned.groupby(['subject','calib_bin','num_classes'], as_index=False)
    tidy = grouped.apply(pick_best).reset_index()
    # drop the added groupby index columns that pandas inserts
    tidy = tidy[['subject','calib_bin','num_classes','task','test_kappa','test_f1',
                 'label_scheme','hours_data','nights','minutes']]
    print(f"[aggregate_per_subject_bin] strategy='{strategy}', produced {len(tidy)} rows")
    return tidy

def make_pairs(tidy):
    """
    Create paired comparisons between 4c and 5c for same subject/bin.
    
    Returns DataFrame with κ_4c, κ_5c, Δκ per subject/bin.
    """
    # Pivot to get 4c and 5c side by side
    pivot = tidy.pivot_table(
        index=['subject', 'calib_bin'], 
        columns='task',
        values=['test_kappa', 'test_f1'],
        aggfunc='first'
    ).reset_index()
    
    # Flatten column names
    pivot.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] 
                     for col in pivot.columns]
    
    # Keep only rows where both 4c and 5c exist
    required_cols = ['4c_test_kappa', '5c_test_kappa']
    paired = pivot.dropna(subset=required_cols).copy()
    
    # Calculate delta kappa
    paired['delta_kappa'] = paired['5c_test_kappa'] - paired['4c_test_kappa']
    
    # Calculate delta f1 if available
    if '4c_test_f1' in paired.columns and '5c_test_f1' in paired.columns:
        paired['delta_f1'] = paired['5c_test_f1'] - paired['4c_test_f1']
    
    print(f"Created {len(paired)} subject/bin pairs")
    print(f"Subjects with pairs: {len(paired['subject'].unique())}")
    print(f"Bins with pairs: {sorted(paired['calib_bin'].unique())}")
    
    return paired

def bootstrap_ci_median(values, n_resamples=2000, seed=42):
    """
    Bootstrap confidence interval for median.
    
    Returns (lower_bound, upper_bound) for 95% CI.
    """
    if len(values) < 2:
        return np.nan, np.nan
    
    np.random.seed(seed)
    values = np.array(values)
    
    bootstrap_medians = []
    for _ in range(n_resamples):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_medians.append(np.median(sample))
    
    return np.percentile(bootstrap_medians, [2.5, 97.5])

def wilcoxon_p(delta_vec):
    """Wilcoxon signed-rank test p-value."""
    delta_clean = np.array(delta_vec).astype(float)
    delta_clean = delta_clean[~np.isnan(delta_clean)]
    
    if len(delta_clean) < 5:
        return np.nan
    
    try:
        statistic, p_value = wilcoxon(delta_clean)
        return p_value
    except:
        return np.nan

def cliffs_delta(delta_vec):
    """
    Cliff's delta effect size for delta vector.
    
    Returns value in [-1, 1] where:
    - 0: no effect
    - ±0.147: small effect
    - ±0.33: medium effect  
    - ±0.474: large effect
    """
    delta_clean = np.array(delta_vec).astype(float)
    delta_clean = delta_clean[~np.isnan(delta_clean)]
    
    if len(delta_clean) < 2:
        return np.nan
    
    # Cliff's delta compares each value to zero (no change)
    n_positive = np.sum(delta_clean > 0)
    n_negative = np.sum(delta_clean < 0)
    n_total = len(delta_clean)
    
    if n_total == 0:
        return np.nan
    
    return (n_positive - n_negative) / n_total

def compute_bin_stats(paired):
    """
    Compute statistics per calibration bin.
    
    Returns DataFrame with per-bin statistics.
    """
    stats_list = []
    
    for bin_val in sorted(paired['calib_bin'].unique()):
        bin_data = paired[paired['calib_bin'] == bin_val]
        
        if len(bin_data) < 2:
            continue
        
        # 4c stats
        kappa_4c = bin_data['4c_test_kappa'].values
        med_4c = np.median(kappa_4c)
        ci_4c = bootstrap_ci_median(kappa_4c)
        
        # 5c stats
        kappa_5c = bin_data['5c_test_kappa'].values
        med_5c = np.median(kappa_5c)
        ci_5c = bootstrap_ci_median(kappa_5c)
        
        # Delta stats
        delta_kappa = bin_data['delta_kappa'].values
        med_delta = np.median(delta_kappa)
        ci_delta = bootstrap_ci_median(delta_kappa)
        p_wilcoxon = wilcoxon_p(delta_kappa)
        cliff_d = cliffs_delta(delta_kappa)
        win_rate = np.mean(delta_kappa > 0)
        
        stats_list.append({
            'calib_bin': bin_val,
            'n_subjects': len(bin_data),
            'median_4c': med_4c,
            'ci_4c_low': ci_4c[0],
            'ci_4c_high': ci_4c[1],
            'median_5c': med_5c,
            'ci_5c_low': ci_5c[0],
            'ci_5c_high': ci_5c[1],
            'median_delta': med_delta,
            'ci_delta_low': ci_delta[0],
            'ci_delta_high': ci_delta[1],
            'p_wilcoxon': p_wilcoxon,
            'cliffs_delta': cliff_d,
            'win_rate': win_rate
        })
    
    return pd.DataFrame(stats_list)

def create_panels(paired, bin_stats, args):
    """Create all 4 panels of the figure."""
    fig = plt.figure(figsize=FIGURE_SIZE)
    
    # Panel A: Paired slope plot
    ax_a = plt.subplot(2, 2, 1)
    panel_a(paired, bin_stats, args, ax_a)
    
    # Panel B: Dose-response curves  
    ax_b = plt.subplot(2, 2, 2)
    panel_b(bin_stats, args, ax_b)
    
    # Panel C: Distribution of Δκ
    ax_c = plt.subplot(2, 2, 3)
    panel_c(paired, bin_stats, args, ax_c)
    
    # Panel D: Mixed-effects (if available)
    ax_d = plt.subplot(2, 2, 4)
    panel_d(paired, args, ax_d)
    
    plt.tight_layout()
    return fig

def panel_a(paired, bin_stats, args, ax):
    if paired.empty:
        ax.set_title('Panel A: Paired Comparisons'); return
    target_bin = choose_panel_bin(paired, args.panelA_bin, min_pairs=5)
    if target_bin is None:
        ax.text(0.5,0.5,'No paired data', transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Panel A: Paired Comparisons'); return
    bin_data = paired[np.isclose(paired['calib_bin'].astype(float), float(target_bin))]
    if bin_data.empty:
        ax.text(0.5,0.5,f'No data near bin {args.panelA_bin}', transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Panel A: Paired Comparisons'); return

    # draw one line per subject
    for _, r in bin_data.iterrows():
        ax.plot([0,1], [r['4c_test_kappa'], r['5c_test_kappa']],
                'o-', color='lightgray', alpha=0.6, linewidth=1)

    # medians ±CI for each task (compute directly so we're not dependent on bin_stats filtering)
    k4 = bin_data['4c_test_kappa'].to_numpy(float)
    k5 = bin_data['5c_test_kappa'].to_numpy(float)
    d  = bin_data['delta_kappa'].to_numpy(float)
    ci4_lo, ci4_hi = bootstrap_ci_median(k4)
    ci5_lo, ci5_hi = bootstrap_ci_median(k5)

    ax.errorbar(0, np.median(k4), yerr=[[np.median(k4)-ci4_lo],[ci4_hi-np.median(k4)]],
                fmt='o', color=COLORS['4c'], markersize=10, capsize=5, linewidth=3, label='4-class')
    ax.errorbar(1, np.median(k5), yerr=[[np.median(k5)-ci5_lo],[ci5_hi-np.median(k5)]],
                fmt='o', color=COLORS['5c'], markersize=10, capsize=5, linewidth=3, label='5-class')

    ax.set_xticks([0,1]); ax.set_xticklabels(['4-class','5-class'])
    ax.set_ylabel("Cohen's κ")
    ax.set_title(f'Panel A: Paired Comparisons (Bin {target_bin:g})\nEach line = 1 subject\'s best runs at ~{target_bin:.1f} nights calibration')

    p = wilcoxon_p(d)
    d_ci_lo, d_ci_hi = bootstrap_ci_median(d) if len(d) >= 2 else (np.nan, np.nan)
    
    # Enhanced stats with effect interpretation
    effect_size = "large" if abs(np.median(d)) > 0.05 else "medium" if abs(np.median(d)) > 0.02 else "small"
    direction = "4c > 5c" if np.median(d) < 0 else "5c > 4c"
    
    stats_text = f"N = {bin_data['subject'].nunique()} subjects\n"
    stats_text += f"Δκ = {np.median(d):.3f} [{d_ci_lo:.3f}, {d_ci_hi:.3f}]\n"
    stats_text += f"Effect: {effect_size} ({direction})\n"
    stats_text += f"p = {p:.3f}" if not np.isnan(p) else "p = n.s."
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va='top',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    # Add median values as text
    ax.text(0, np.median(k4) + 0.01, f'κ={np.median(k4):.3f}', 
            ha='center', va='bottom', fontsize=9, weight='bold', color=COLORS['4c'])
    ax.text(1, np.median(k5) + 0.01, f'κ={np.median(k5):.3f}', 
            ha='center', va='bottom', fontsize=9, weight='bold', color=COLORS['5c'])
    
    ax.legend(loc='upper right')

def panel_b(bin_stats, args, ax):
    """Panel B: Dose-response curves by task."""
    if len(bin_stats) == 0:
        ax.text(0.5, 0.5, 'No bin statistics available', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Panel B: Dose-Response')
        return
    
    x_vals = bin_stats['calib_bin'].values
    
    # 4c curve
    ax.errorbar(x_vals, bin_stats['median_4c'],
                yerr=[bin_stats['median_4c'] - bin_stats['ci_4c_low'],
                      bin_stats['ci_4c_high'] - bin_stats['median_4c']],
                fmt='o-', color=COLORS['4c'], label='4-class',
                capsize=3, linewidth=2, markersize=6)
    
    # 5c curve  
    ax.errorbar(x_vals, bin_stats['median_5c'],
                yerr=[bin_stats['median_5c'] - bin_stats['ci_5c_low'],
                      bin_stats['ci_5c_high'] - bin_stats['median_5c']],
                fmt='o-', color=COLORS['5c'], label='5-class', 
                capsize=3, linewidth=2, markersize=6)
    
    # Annotations for N subjects per bin
    for _, row in bin_stats.iterrows():
        ax.annotate(f"N={row['n_subjects']}", 
                   (row['calib_bin'], max(row['median_4c'], row['median_5c']) + 0.02),
                   ha='center', va='bottom', fontsize=8, alpha=0.7)
    
    # Optional: Wilcoxon p-values per bin
    for _, row in bin_stats.iterrows():
        if not np.isnan(row['p_wilcoxon']):
            p_text = f"p={row['p_wilcoxon']:.3f}" if row['p_wilcoxon'] >= 0.001 else "p<0.001"
            ax.annotate(p_text,
                       (row['calib_bin'], min(row['median_4c'], row['median_5c']) - 0.05),
                       ha='center', va='top', fontsize=7, alpha=0.6)
    
    ax.set_xlabel('Calibration Nights' if args.x == 'nights' else 'Calibration Minutes')
    ax.set_ylabel('Cohen\'s κ')
    title = f'Panel B: Performance vs Calibration Amount\n'
    title += f'Medians ± 95% CI across subjects (best runs per subject/task)'
    ax.set_title(title)
    ax.legend()
    
    # Add overall trend annotation
    if len(bin_stats) >= 2:
        x_vals = bin_stats['calib_bin'].values
        # Simple linear trend
        trend_4c = np.polyfit(x_vals, bin_stats['median_4c'], 1)[0]
        trend_5c = np.polyfit(x_vals, bin_stats['median_5c'], 1)[0]
        trend_text = f"Trends: 4c {trend_4c:+.3f}/night, 5c {trend_5c:+.3f}/night"
        ax.text(0.02, 0.02, trend_text, transform=ax.transAxes, fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    if args.x == 'minutes':
        ax.set_xscale('log')

def panel_c(paired, bin_stats, args, ax):
    if paired.empty:
        ax.set_title('Panel C: Δκ Distribution'); return
    target_bin = choose_panel_bin(paired, args.panelA_bin, min_pairs=5)
    if target_bin is None:
        ax.text(0.5,0.5,'No paired data', transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Panel C: Δκ Distribution'); return
    bin_data = paired[np.isclose(paired['calib_bin'].astype(float), float(target_bin))]
    if bin_data.empty:
        ax.text(0.5,0.5,f'No data near bin {args.panelA_bin}', transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Panel C: Δκ Distribution'); return

    d = bin_data['delta_kappa'].to_numpy(float)

    parts = ax.violinplot([d], positions=[0], widths=0.6, showmeans=False, showmedians=False)
    for pc in parts['bodies']: pc.set_facecolor(COLORS['delta']); pc.set_alpha(0.7)
    bp = ax.boxplot([d], positions=[0], widths=0.3, patch_artist=True)
    bp['boxes'][0].set_facecolor('white'); bp['boxes'][0].set_edgecolor(COLORS['delta'])

    rng = np.random.default_rng(42); ax.scatter(rng.normal(0,0.05,len(d)), d, s=18, alpha=0.6, color=COLORS['delta'])
    ax.axhline(0, color='gray', ls='--', alpha=0.6)

    d_ci_lo, d_ci_hi = bootstrap_ci_median(d) if len(d)>=2 else (np.nan,np.nan)
    p = wilcoxon_p(d); cd = cliffs_delta(d); win = np.mean(d>0)
    
    # Effect size interpretation
    effect = "large" if abs(cd) > 0.5 else "medium" if abs(cd) > 0.3 else "small"
    direction = "favors 4c" if np.median(d) < 0 else "favors 5c"
    
    ax.set_title(f"Panel C: Δκ Distribution (Bin {target_bin:g})\n"
                 f"Δκ = {np.median(d):.3f} [{d_ci_lo:.3f}, {d_ci_hi:.3f}] | "
                 f"{'p='+format(p,'.3f') if not np.isnan(p) else 'p=n.s.'} | "
                 f"Effect: {effect} {direction} | Win rate: {win:.1%}",
                 fontsize=10)
    
    # Add interpretation text
    interpret_text = f"Each dot = 1 subject's (5c best - 4c best)\n"
    interpret_text += f"Negative = 4-class outperforms 5-class"
    ax.text(0.98, 0.02, interpret_text, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    ax.set_ylabel('Δκ (5c − 4c)'); ax.set_xticks([]); ax.set_xlabel('')

def panel_d(paired, args, ax):
    # build long table
    rows = []
    for _, r in paired.iterrows():
        rows += [
            {'subject': r['subject'], 'calib_bin': float(r['calib_bin']), 'task': '4c', 'kappa': r['4c_test_kappa']},
            {'subject': r['subject'], 'calib_bin': float(r['calib_bin']), 'task': '5c', 'kappa': r['5c_test_kappa']},
        ]
    dfL = pd.DataFrame(rows).dropna(subset=['kappa','calib_bin'])
    if dfL.empty:
        ax.text(0.5,0.5,'No data for mixed-effects', transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Panel D: Mixed Effects'); return

    dfL['task'] = pd.Categorical(dfL['task'], categories=['4c','5c'])
    dfL['calib_z'] = (dfL['calib_bin'] - dfL['calib_bin'].mean()) / (dfL['calib_bin'].std() + 1e-8)

    coef_rows = []; model_note = ''
    if HAS_STATSMODELS:
        try:
            md = smf.mixedlm("kappa ~ C(task) + calib_z", dfL, groups=dfL["subject"], re_formula="1")
            mdf = md.fit(method='lbfgs', reml=False, maxiter=200, disp=False)
            params, ci = mdf.params, mdf.conf_int()
            model_note = f"AIC: {getattr(mdf,'aic',np.nan):.1f}\nN obs: {len(dfL)}\nN subjects: {dfL['subject'].nunique()}"
        except Exception:
            ols = smf.ols("kappa ~ C(task) + calib_z", data=dfL).fit(
                cov_type='cluster', cov_kwds={'groups': dfL['subject']})
            params, ci = ols.params, ols.conf_int()
            model_note = f"OLS (cluster-robust)\nN obs: {len(dfL)}\nN subjects: {dfL['subject'].nunique()}"
    else:
        ax.text(0.5,0.5,'statsmodels not installed', transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Panel D: Mixed Effects'); return

    for name,label,color in [
        ('Intercept','Intercept','black'),
        ('C(task)[T.5c]','5c vs 4c', COLORS['5c']),
        ('calib_z','Calibration (z)', 'black')
    ]:
        if name in params.index:
            coef = params[name]; lo, hi = ci.loc[name]
            coef_rows.append((label, coef, lo, hi, color))

    # forest plot
    ax.axvline(0, color='gray', ls='--', alpha=0.5)
    y = np.arange(len(coef_rows))
    for i,(label,coef,lo,hi,color) in enumerate(coef_rows):
        ax.errorbar(coef, i, xerr=[[coef-lo],[hi-coef]], fmt='o', color=color, capsize=5, markersize=7)
        ax.text(hi + 0.01, i, f"{coef:.3f}", va='center', fontsize=9)
        ax.text(ax.get_xlim()[0], i, label, va='center', fontsize=10)
    ax.set_yticks([]); ax.set_xlabel('Coefficient (95% CI)')
    ax.set_title('Panel D: Mixed Effects Model\nκ ~ task + calibration + (1|subject)')
    ax.text(0.02, 0.98, model_note, transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85), fontsize=8)


def save_figure(fig, output_dir):
    """Save figure in multiple formats."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_name = 'figure_task_granularity'
    
    # Save as PDF
    pdf_path = output_path / f'{base_name}.pdf'
    fig.savefig(pdf_path, format='pdf', dpi=DPI, bbox_inches='tight')
    print(f"Saved: {pdf_path}")
    
    # Save as SVG
    svg_path = output_path / f'{base_name}.svg'
    fig.savefig(svg_path, format='svg', dpi=DPI, bbox_inches='tight')
    print(f"Saved: {svg_path}")
    
    # Save as PNG for preview
    png_path = output_path / f'{base_name}.png'
    fig.savefig(png_path, format='png', dpi=DPI, bbox_inches='tight')
    print(f"Saved: {png_path}")

def create_summary_stats(paired, bin_stats, output_dir):
    """Create and save summary statistics."""
    stats_summary = {
        'total_subjects': int(paired['subject'].nunique()),
        'total_pairs': int(len(paired)),
        'bins_analyzed': sorted([float(b) for b in bin_stats['calib_bin']]),
        'overall_median_delta_kappa': float(np.median(paired['delta_kappa'])),
        'overall_win_rate': float(np.mean(paired['delta_kappa'] > 0)),
        'per_bin_stats': bin_stats.to_dict('records')
    }
    
    # Save as JSON
    stats_path = Path(output_dir) / 'task_granularity_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats_summary, f, indent=2, default=str)
    
    print(f"Summary statistics saved: {stats_path}")
    
    return stats_summary

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Generate task granularity analysis figure',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--csv', required=True,
                       help='Path to flattened CSV file')
    parser.add_argument('--out', default='Plot_Clean/figures/fig1',
                       help='Output directory for figures')
    
    # Analysis options
    parser.add_argument('--x', choices=['nights', 'minutes'], default='nights',
                       help='X-axis calibration units')
    parser.add_argument('--panelA-bin', default='auto',
                       help='Calibration bin for Panels A/C (number or "auto")')
    parser.add_argument('--subject-agg',
                       choices=['median','best_val','best_test'],
                       default='best_val',
                       help='How to aggregate multiple runs per (subject,bin,task)')
    
    # Filtering options
    parser.add_argument('--only-unfrozen', action='store_true',
                       help='Filter to only unfrozen models')
    parser.add_argument('--use-pretrained-only', action='store_true',
                       help='Filter to only pretrained models')
    parser.add_argument('--min-nights', type=float,
                       help='Minimum nights for inclusion')
    parser.add_argument('--max-nights', type=float,
                       help='Maximum nights for inclusion')
    
    # 4c scheme handling
    parser.add_argument('--include-4c-v0', action='store_true', default=True,
                       help='Include 4c-v0 schemes')
    parser.add_argument('--include-4c-v1', action='store_true', default=True,
                       help='Include 4c-v1 schemes')
    parser.add_argument('--facet-4c-schemes', action='store_true',
                       help='Facet by 4c scheme variants')
    
    args = parser.parse_args()
    
    # Setup
    setup_style()
    
    print("=" * 60)
    print("TASK GRANULARITY ANALYSIS")
    print("=" * 60)
    
    try:
        # Load and process data
        print("\n1. Loading data...")
        df = load_flat_csv(args.csv)
        
        print("\n2. Resolving columns...")
        df = resolve_columns(df)
        
        print("\n3. Applying filters...")
        df = apply_filters(df, args)
        
        print("\n4. Creating calibration bins...")
        df_binned, bins, centers = bin_calibration(df, args.x)
        
        print("\n5. Aggregating per subject/bin...")
        tidy = aggregate_per_subject_bin(df_binned, strategy=args.subject_agg)
        
        # Debug: Show what aggregation did for a few examples
        print("\n=== AGGREGATION VERIFICATION ===")
        sample_groups = df_binned.groupby(['subject','calib_bin','num_classes']).size()
        multi_runs = sample_groups[sample_groups > 1].head(3)
        
        for (subj, bin_val, num_classes), count in multi_runs.items():
            group_data = df_binned[(df_binned['subject'] == subj) & 
                                  (df_binned['calib_bin'] == bin_val) & 
                                  (df_binned['num_classes'] == num_classes)]
            kappas = group_data['test_kappa'].sort_values(ascending=False)
            tidy_result = tidy[(tidy['subject'] == subj) & 
                              (tidy['calib_bin'] == bin_val) & 
                              (tidy['num_classes'] == num_classes)]
            
            print(f"Subject {subj}, Bin {bin_val}, {num_classes}c:")
            print(f"  {count} runs with κ: {kappas.values}")
            print(f"  Median: {kappas.median():.3f}, Best: {kappas.max():.3f}")
            if len(tidy_result) > 0:
                selected = tidy_result.iloc[0]['test_kappa']
                print(f"  Selected ({args.subject_agg}): {selected:.3f}")
                if args.subject_agg == 'best_test':
                    improvement = (selected - kappas.median()) / kappas.median() * 100
                    print(f"  Improvement: {improvement:.1f}%")
            print()

        print("\n6. Making pairs...")
        paired = make_pairs(tidy)
        
        if len(paired) == 0:
            print("ERROR: No paired comparisons found!")
            return
        
        print("\nPAIRS PER BIN (subjects):")
        print(paired.groupby('calib_bin')['subject'].nunique().to_string())
        
        # How many repeats were collapsed per (subject,bin,task)?
        rep_counts = df_binned.groupby(['subject','calib_bin','num_classes']).size()
        print("\nCollapsed repeats per (subject,bin,task):")
        print(rep_counts.describe())
        
        print("\n7. Computing bin statistics...")
        bin_stats = compute_bin_stats(paired)
        
        print("\n8. Creating figure...")
        fig = create_panels(paired, bin_stats, args)
        
        # Label the figure so readers know aggregation method
        fig.suptitle(f"Aggregation: {args.subject_agg.replace('_','-')} per subject/bin",
                     fontsize=10, y=0.99)
        
        print("\n9. Saving outputs...")
        save_figure(fig, args.out)
        summary = create_summary_stats(paired, bin_stats, args.out)
        
        print(f"\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Subjects analyzed: {summary['total_subjects']}")
        print(f"Total pairs: {summary['total_pairs']}")
        print(f"Overall Δκ: {summary['overall_median_delta_kappa']:.3f}")
        print(f"Win rate (5c > 4c): {summary['overall_win_rate']:.1%}")
        print(f"Outputs saved to: {args.out}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        raise

if __name__ == '__main__':
    main()