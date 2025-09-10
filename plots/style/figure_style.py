#!/usr/bin/env python3
"""
Consistent style configuration for all CBraMod figures.

Global style requirements:
- Use consistent typeface, font sizes, and color palette
- YASA = muted yellow (#F0E442)
- CBraMod = saturated blue (#0072B2)
- 4c vs 5c = two hues of the Set2 palette
- Increased axis label font for print legibility
- State N (subjects and runs) in every caption
- Specify whether CIs are across subjects or runs
- Show thin gray dots/violins for all runs and overlay filled dots for top-10
- Output both .pdf and .png formats
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path

# =========== COLOR PALETTES ===========

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

# Set2 palette for class comparisons (4c vs 5c)
SET2_COLORS = [
    "#66C2A5",  # teal
    "#FC8D62",  # orange
    "#8DA0CB",  # purple
    "#E78AC3",  # pink
    "#A6D854",  # lime
    "#FFD92F",  # yellow
    "#E5C494",  # tan
    "#B3B3B3",  # gray
]

# Main color assignments
FIGURE_COLORS = {
    # Primary methods
    "yasa": "#060606",           # orange (from Okabe-Ito)
    "cbramod": "#0072B2",        # saturated blue (from Okabe-Ito)
    
    # Class comparisons  
    "4_class": SET2_COLORS[0],   # teal
    "5_class": SET2_COLORS[1],   # orange
    
    # Utility colors
    "median_curve": "#0072B2",   # same as CBraMod
    "t_star": "#009E73",         # green for significance markers
    "subjects": "#8C8C8C",       # neutral grey for per-subject lines
    "all_runs": "#CCCCCC",       # light gray for background runs
    "top_10": "#0072B2",         # blue for highlighted top runs
    "ci_band": "#0072B2",        # blue for confidence intervals
    
    # Dataset compositions
    "orp": "#D55E00",           # vermillion
    "multi_dataset": "#009E73",  # green
    "combined": "#CC79A7",       # reddish purple
}

# =========== FONT AND STYLE SETTINGS ===========

def setup_figure_style():
    """
    Set up consistent plotting style for all figures.
    Increased font sizes for print legibility.
    """
    plt.style.use("default")
    
    # Font settings - larger for print legibility
    plt.rcParams.update({
        # Figure settings
        "figure.figsize": (10, 6),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.format": "pdf",  # Default to PDF
        
        # Font settings - increased for print
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 14,           # Base font size (was 12)
        "axes.titlesize": 18,      # Title size (was 15)
        "axes.labelsize": 16,      # Axis label size (was 13) - INCREASED FOR PRINT
        "xtick.labelsize": 14,     # X-tick size (was 11)
        "ytick.labelsize": 14,     # Y-tick size (was 11)
        "legend.fontsize": 13,     # Legend size (was 11)
        
        # Axes settings
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
        
        # Line and marker settings
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
        
        # Color cycle
        "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
    })

def get_color(name: str) -> str:
    """Get color by name from the figure color palette."""
    return FIGURE_COLORS.get(name, OKABE_ITO[0])

def save_figure(fig, base_path: Path, formats: list = None):
    """
    Save figure in multiple formats (.pdf and .png by default).
    
    Args:
        fig: matplotlib figure
        base_path: Path without extension
        formats: List of formats to save (default: ['pdf', 'png'])
    """
    if formats is None:
        formats = ['pdf', 'png']
    
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    for fmt in formats:
        file_path = base_path.with_suffix(f'.{fmt}')
        fig.savefig(file_path, format=fmt, bbox_inches='tight', dpi=300)
        saved_files.append(file_path)
        print(f"Saved: {file_path}")
    
    return saved_files

# =========== VISUALIZATION HELPERS ===========

def plot_with_background_and_top10(ax, x_all, y_all, x_top10, y_top10, 
                                   label_all="All runs", label_top10="Top 10",
                                   alpha_all=0.3, alpha_top10=1.0):
    """
    Plot thin gray dots/violins for all runs with overlay of filled dots for top-10.
    Addresses the "top-10 only" ambiguity requirement.
    """
    # Background: all runs (thin gray)
    ax.scatter(x_all, y_all, 
              color=get_color("all_runs"), 
              alpha=alpha_all, 
              s=20, 
              label=label_all,
              zorder=1)
    
    # Foreground: top 10 runs (filled, prominent)
    ax.scatter(x_top10, y_top10, 
              color=get_color("top_10"), 
              alpha=alpha_top10, 
              s=60, 
              edgecolors='white',
              linewidth=1.5,
              label=label_top10,
              zorder=3)

def add_yasa_baseline(ax, yasa_kappa: float, yasa_ci: float = 0.05, 
                     x_range: tuple = None, label: str = None):
    """Add YASA baseline line without confidence band."""
    if label is None:
        label = f"YASA baseline (κ={yasa_kappa:.3f})"
    
    # Baseline line
    ax.axhline(y=yasa_kappa, 
              color=get_color("yasa"), 
              linestyle="--", 
              linewidth=2.5,
              label=label, 
              zorder=2)

def add_significance_marker(ax, x: float, y: float, label: str = "T*", 
                           offset_x: float = 0.15, offset_y: float = 0.05):
    """Add significance marker (T*) with annotation."""
    ax.scatter([x], [y], 
              color=get_color("t_star"), 
              s=120, 
              zorder=4,
              marker="P", 
              edgecolors="black", 
              linewidth=1.0)
    
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(x * (1 + offset_x), y + offset_y),
        arrowprops=dict(arrowstyle="->", color=get_color("t_star"), lw=2),
        fontsize=14, 
        fontweight="bold", 
        color=get_color("t_star"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
    )

def format_n_caption(n_subjects: int, n_runs: int, ci_type: str = "subjects") -> str:
    """
    Format N information for captions.
    
    Args:
        n_subjects: Number of subjects
        n_runs: Number of runs
        ci_type: "subjects" or "runs" - what the CIs are computed across
    
    Returns:
        Formatted string for caption
    """
    return f"N={n_subjects} subjects, {n_runs} runs; 95% CIs across {ci_type}"

def add_sample_size_annotation(ax, stats_df, y_offset_frac: float = 0.02):
    """Add sample size annotations to plot bins."""
    ymin, ymax = ax.get_ylim()
    y_pos = ymin + y_offset_frac * (ymax - ymin)
    
    for _, row in stats_df.iterrows():
        if 'bin_center' in row and 'n_subjects' in row:
            ax.text(row["bin_center"], y_pos, f"n={int(row['n_subjects'])}",
                   ha="center", va="bottom", fontsize=12, alpha=0.85,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75))

# =========== STATISTICAL HELPERS ===========

def bootstrap_ci_median(arr: np.ndarray, confidence: float = 0.95) -> tuple:
    """
    Compute bootstrap confidence interval for median.
    
    Returns:
        (lower_bound, upper_bound)
    """
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
        # Fallback to percentile method
        rng = np.random.default_rng(42)
        meds = [np.median(rng.choice(arr, size=arr.size, replace=True)) for _ in range(2000)]
        lo = np.percentile(meds, (1 - confidence) / 2 * 100)
        hi = np.percentile(meds, (1 + confidence) / 2 * 100)
        return float(lo), float(hi)

def bootstrap_ci_mean(arr: np.ndarray, confidence: float = 0.95) -> tuple:
    """
    Compute symmetric confidence interval for mean using standard error.
    This ensures error bars are centered around the mean.
    
    Returns:
        (lower_bound, upper_bound)
    """
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return (np.nan, np.nan)
    
    mean_val = float(np.mean(arr))
    
    if arr.size < 3:
        # For very small samples, use simple range
        std_err = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
        margin = 1.96 * std_err  # 95% CI approximation
    else:
        # Use t-distribution for small samples
        try:
            from scipy.stats import t
            std_err = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
            t_value = t.ppf((1 + confidence) / 2, df=arr.size - 1)
            margin = t_value * std_err
        except ImportError:
            # Fallback to normal approximation
            std_err = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
            margin = 1.96 * std_err
    
    return (mean_val - margin, mean_val + margin)

def wilcoxon_test(delta_vec: np.ndarray) -> float:
    """Wilcoxon signed-rank test p-value."""
    try:
        from scipy.stats import wilcoxon
        deltas = np.asarray(delta_vec, dtype=float)
        deltas = deltas[~np.isnan(deltas)]
        if deltas.size < 1:
            return np.nan
        stat, p = wilcoxon(deltas, zero_method="wilcox", alternative="two-sided", 
                          correction=False, mode="auto")
        return float(p)
    except Exception:
        return np.nan