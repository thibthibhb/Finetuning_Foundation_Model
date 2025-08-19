#!/usr/bin/env python3
"""
CBraMod Paper – Figure 1: Calibration Dose–Response (5-class)

What it fixes vs your previous version
--------------------------------------
• Reads calibration minutes from many possible config keys (minutes OR nights OR epochs).
• Reads κ/F1 from many possible summary keys (e.g., "test/kappa", "eval_kappa", etc.).
• Builds log-spaced bins directly from observed data (no hard-coded x-range).
• Draws the YASA CI band across the visible x-range.
• Provides sanity prints (counts, ranges) so you can see what's loaded.

Usage
-----
python paper_fig1_calibration.py \
  --project thibaut_hasle-epfl/CBraMod-earEEG-tuning \
  --output-dir artifacts/results/figures/paper
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ----------------------------- Plot style ---------------------------------- #
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
    })


# ------------------- Robust extraction helpers (keys vary) ----------------- #
def _pick(d: Dict, candidates: List[str], default=np.nan):
    """Return the first present key from candidates; also fuzzy-match variants."""
    # 1) exact matches
    for k in candidates:
        if k in d and d[k] is not None:
            return d[k]
    # 2) soft/fuzzy: normalize keys
    norm = {str(k).lower().replace("/", "_").replace("-", "_"): k for k in d.keys()}
    for want in candidates:
        w = want.lower().replace("/", "_").replace("-", "_")
        for k_norm, k_orig in norm.items():
            if w in k_norm:
                val = d.get(k_orig, None)
                if val is not None:
                    return val
    return default


def ensure_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        # strings like "0.45±0.02" etc → try to split
        try:
            return float(str(x).split("±")[0])
        except Exception:
            return np.nan


def extract_calibration_minutes(config: Dict, summary: Dict) -> float:
    """
    Prefer explicit per-subject calibration minutes. Otherwise nights*480,
    otherwise epochs*(epoch_len/60).
    """
    # 1) direct minutes in config or summary
    cal_min = _pick(config, [
        "calib_minutes", "minutes_calib", "subject_calib_minutes", "calibration_minutes"
    ])
    if pd.isna(cal_min):
        cal_min = _pick(summary, [
            "calib_minutes", "minutes_calib", "subject_calib_minutes", "calibration_minutes"
        ])
    cal_min = ensure_float(cal_min)

    # 2) nights → minutes (8h/night)
    if pd.isna(cal_min):
        cal_nights = _pick(config, ["calib_nights", "nights_calibration", "nights_training"])
        if pd.isna(cal_nights):
            cal_nights = _pick(summary, ["calib_nights", "nights_calibration", "nights_training"])
        cal_nights = ensure_float(cal_nights)
        if not pd.isna(cal_nights):
            return float(cal_nights) * 8.0 * 60.0

    # 3) epochs → minutes (epoch_len defaults to 30 s)
    if pd.isna(cal_min):
        cal_epochs = _pick(config, ["calib_epochs", "n_calib_epochs", "n_labeled_epochs"])
        if pd.isna(cal_epochs):
            cal_epochs = _pick(summary, ["calib_epochs", "n_calib_epochs", "n_labeled_epochs"])
        cal_epochs = ensure_float(cal_epochs)

        epoch_len = _pick(config, ["epoch_len", "epoch_length_s"])
        if pd.isna(epoch_len):
            epoch_len = _pick(summary, ["epoch_len", "epoch_length_s"])
        epoch_len = ensure_float(epoch_len)
        if pd.isna(epoch_len):
            epoch_len = 30.0

        if not pd.isna(cal_epochs):
            return float(cal_epochs) * (float(epoch_len) / 60.0)

    return cal_min  # may be NaN


def extract_metric(summary: Dict, names: List[str], fallback_contains: Optional[List[str]] = None) -> float:
    """
    Try several explicit names; if missing, find a key whose normalized name
    contains all tokens in fallback_contains (e.g., ["test","kappa"]).
    """
    v = _pick(summary, names)
    v = ensure_float(v)
    if not pd.isna(v):
        return v

    if fallback_contains:
        tokens = [t.lower() for t in fallback_contains]
        for k, val in summary.items():
            kk = str(k).lower()
            if all(t in kk for t in tokens):
                vv = ensure_float(val)
                if not pd.isna(vv):
                    return vv
    return np.nan


# ------------------------------ W&B loader --------------------------------- #
def load_wandb_data(project: str) -> pd.DataFrame:
    try:
        import wandb
    except ImportError as e:
        raise ImportError("wandb package required. Install with: pip install wandb") from e

    api = wandb.Api()
    print(f"Loading all runs from {project} ...")
    runs = list(api.runs(project, per_page=100))
    print(f"Found {len(runs)} total runs")

    rows = []
    for i, run in enumerate(runs, 1):
        try:
            if run.state != "finished":
                continue

            config = dict(run.config or {})
            summary = dict(run.summary or {})

            # Core identifiers
            subject_id = _pick(config, ["subject_id", "subj_id", "subject"], default=run.name)

            # Core model settings
            num_classes = _pick(config, ["num_of_class", "num_of_classes", "n_classes"], default=5)
            num_classes = int(ensure_float(num_classes)) if not pd.isna(num_classes) else 5

            # Calibration
            calib_minutes = extract_calibration_minutes(config, summary)

            # Metrics
            test_kappa = extract_metric(
                summary,
                names=["test_kappa", "test/κ", "test/kappa", "kappa_test", "eval_kappa", "kappa"],
                fallback_contains=["test", "kappa"]
            )
            test_f1_macro = extract_metric(
                summary,
                names=["test_f1", "test_f1_macro", "test/f1_macro", "f1_macro_test", "eval_f1_macro"],
                fallback_contains=["test", "f1", "macro"]
            )

            rows.append({
                "run_id": run.id,
                "run_name": run.name,
                "subject_id": subject_id,
                "num_classes": num_classes,
                "calibration_minutes": ensure_float(calib_minutes),
                "test_kappa": ensure_float(test_kappa),
                "test_f1_macro": ensure_float(test_f1_macro),
            })

            if i % 100 == 0:
                print(f"Processed {i}/{len(runs)} runs ...")

        except Exception as e:
            print(f"[warn] skipping run {getattr(run, 'id', '?')}: {e}")

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} finished runs")

    # Basic sanity prints
    if not df.empty:
        print("non-NaN test_kappa:", df["test_kappa"].notna().sum())
        print("non-NaN calibration_minutes:", df["calibration_minutes"].notna().sum())
        print("num_classes counts:\n", df["num_classes"].value_counts(dropna=False))
        print(df[["num_classes", "calibration_minutes", "test_kappa"]].describe())

    return df


# --------------------------- Bootstrap for medians -------------------------- #
def bootstrap_ci(arr: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    arr = np.asarray(arr)
    arr = arr[~np.isnan(arr)]
    if arr.size < 3:
        return (np.nan, np.nan)
    try:
        from scipy.stats import bootstrap
        rng = np.random.default_rng(42)

        def stat(x):
            return np.median(x)

        res = bootstrap((arr,), stat, n_resamples=1000, confidence_level=confidence, random_state=rng)
        return float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception:
        # Percentile fallback
        rng = np.random.default_rng(42)
        meds = []
        for _ in range(1000):
            meds.append(np.median(rng.choice(arr, size=arr.size, replace=True)))
        alpha = 1 - confidence
        return (np.percentile(meds, 100 * alpha / 2), np.percentile(meds, 100 * (1 - alpha / 2)))


# -------------------------- Figure 1 construction --------------------------- #
def create_figure_1_calibration(df: pd.DataFrame, output_dir: Path):
    # Filter to 5-class runs
    df5 = df[(df["num_classes"] == 5)].copy()
    df5 = df5.dropna(subset=["calibration_minutes", "test_kappa"])
    print(f"Using {len(df5)} runs for 5-class calibration analysis")

    if df5.empty:
        print("No 5-class runs with both calibration_minutes and test_kappa.")
        return None

    # Clamp non-positive minutes (log-scale safety)
    df5.loc[df5["calibration_minutes"] <= 0, "calibration_minutes"] = np.nan
    df5 = df5.dropna(subset=["calibration_minutes"])

    mins = df5["calibration_minutes"].astype(float)
    kappas = df5["test_kappa"].astype(float)

    # Build data-driven, log-spaced bins
    low = max(1.0, float(np.nanmin(mins)))
    high = float(np.nanquantile(mins, 0.99))  # robust to outliers
    if high <= low:
        high = float(np.nanmax(mins))

    # Ensure at least 6 bins across range
    num_bins = 8
    edges = np.unique(np.geomspace(low, high, num=num_bins)).tolist()
    if len(edges) < 3:  # degenerate case
        edges = [low * 0.9, low * 1.1, low * 1.3]

    df5["calib_bin"] = pd.cut(df5["calibration_minutes"], bins=edges, include_lowest=True)
    bin_counts = df5["calib_bin"].value_counts().sort_index()
    print("Runs per bin:\n", bin_counts)

    # Aggregate medians and CIs per bin
    stats_rows = []
    subject_curves: Dict[str, Dict[str, list]] = {}
    for b, g in df5.groupby("calib_bin"):
        if pd.isna(b) or g["test_kappa"].notna().sum() == 0:
            continue
        center = float(b.mid)
        k_vals = g["test_kappa"].astype(float).values
        median_k = float(np.median(k_vals))
        lo, hi = bootstrap_ci(k_vals, confidence=0.95)
        stats_rows.append({
            "bin_center": center,
            "kappa_median": median_k,
            "kappa_ci_low": lo,
            "kappa_ci_high": hi,
            "n_runs": int(np.sum(~np.isnan(k_vals))),
        })
        # store subject curves (thin grey lines)
        for _, r in g.iterrows():
            sid = str(r["subject_id"])
            subject_curves.setdefault(sid, {"minutes": [], "kappa": []})
            subject_curves[sid]["minutes"].append(center)
            subject_curves[sid]["kappa"].append(float(r["test_kappa"]))

    stats_df = pd.DataFrame(stats_rows).sort_values("bin_center")
    if stats_df.empty:
        print("No valid bins to plot.")
        return None

    # YASA baseline (ear-EEG)
    yasa_kappa = 0.446
    yasa_ci = 0.05  # visual band width ~±0.05

    # ---- Plot ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Subject thin lines
    for sid, d in subject_curves.items():
        if len(d["minutes"]) >= 2:
            data = sorted(zip(d["minutes"], d["kappa"]))
            xs, ys = zip(*data)
            ax.plot(xs, ys, color="lightgray", alpha=0.4, linewidth=0.8, zorder=1)

    # Median curve with asymmetric CI
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
            color="#2E86AB",
            linewidth=2.5,
            markersize=7,
            capsize=5,
            capthick=2,
            label="CBraMod median ± 95% CI",
            zorder=3,
        )

    # Points without CI (too few samples)
    missing = ~valid
    if missing.any():
        ax.plot(
            stats_df.loc[missing, "bin_center"],
            stats_df.loc[missing, "kappa_median"],
            "o-",
            color="#2E86AB",
            linewidth=2.5,
            markersize=7,
            zorder=3,
        )

    # Dynamic x-limits (log)
    ax.set_xscale("log")
    x_min = float(np.min(stats_df["bin_center"])) / 1.15
    x_max = float(np.max(stats_df["bin_center"])) * 1.15
    ax.set_xlim(x_min, x_max)

    # YASA baseline line + band across visible x
    ax.axhline(y=yasa_kappa, color="#A23B72", linestyle="--", linewidth=2.5,
               label=f"YASA baseline (κ={yasa_kappa:.3f})", zorder=2)
    x0, x1 = ax.get_xlim()
    ax.fill_between([x0, x1], yasa_kappa - yasa_ci, yasa_kappa + yasa_ci,
                    color="#A23B72", alpha=0.2, zorder=1)

    # T* (first bin where median CI entirely above YASA and Δκ ≥ δ)
    t_star = None
    t_star_kappa = None
    delta_threshold = 0.05
    for _, r in stats_df.iterrows():
        if np.isnan(r["kappa_ci_low"]) or np.isnan(r["kappa_median"]):
            continue
        if (r["kappa_median"] - yasa_kappa) >= delta_threshold and r["kappa_ci_low"] > yasa_kappa:
            t_star = float(r["bin_center"])
            t_star_kappa = float(r["kappa_median"])
            break

    if t_star is not None:
        ax.scatter([t_star], [t_star_kappa], color="red", s=110, zorder=4,
                   marker="*", edgecolors="darkred", linewidth=1)
        ax.annotate(
            f"T* ≈ {t_star:.0f} min",
            xy=(t_star, t_star_kappa),
            xytext=(t_star * 1.4, t_star_kappa + 0.05),
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
            fontsize=12, fontweight="bold", color="red",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85)
        )

    # Labels, legend, limits
    ax.set_xlabel("Calibration amount (minutes per subject)", fontweight="bold")
    ax.set_ylabel("Cohen's κ", fontweight="bold")
    ax.set_title("Figure 1: Calibration Dose-Response for 5-Class Sleep Staging", pad=16, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    # y-limits from data and YASA
    y_low = min(float(stats_df["kappa_median"].min()) - 0.05, yasa_kappa - 0.10)
    y_high = max(float(stats_df["kappa_median"].max()) + 0.08, yasa_kappa + 0.08)
    ax.set_ylim(max(0.2, y_low), min(0.95, y_high))

    # Sample-size annotations at bottom of each bin center
    ymin, ymax = ax.get_ylim()
    y_pos = ymin + 0.02 * (ymax - ymin)
    for _, r in stats_df.iterrows():
        ax.text(
            r["bin_center"], y_pos, f"n={int(r['n_runs'])}",
            ha="center", va="bottom", fontsize=9, alpha=0.75,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.65)
        )

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path_svg = output_dir / "figure_1_calibration_dose_response.svg"
    fig_path_pdf = output_dir / "figure_1_calibration_dose_response.pdf"
    plt.savefig(fig_path_svg)
    plt.savefig(fig_path_pdf)
    print(f"Saved: {fig_path_svg}")
    print(f"Saved: {fig_path_pdf}")

    # Console summary
    best_kappa = float(stats_df["kappa_median"].max())
    delta_best = best_kappa - yasa_kappa
    if t_star is not None:
        print(f"T* (minimum effective calibration): ~{t_star:.0f} minutes")
    print(f"Best κ (median across bins): {best_kappa:.3f}  → Δκ vs YASA: +{delta_best:.3f}")

    return fig


# ----------------------------------- CLI ----------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Generate CBraMod Figure 1: Calibration Dose–Response")
    parser.add_argument("--project", required=True, help="Weights & Biases project (e.g., user/proj)")
    parser.add_argument("--output-dir", default="artifacts/results/figures/paper", help="Output directory")
    args = parser.parse_args()

    setup_plotting_style()
    df = load_wandb_data(args.project)

    # Guard
    if df.empty:
        print("No runs loaded.")
        return

    fig = create_figure_1_calibration(df, Path(args.output_dir))
    if fig is None:
        print("Nothing to plot (check key names and filters).")
    else:
        print("✅ Figure 1 generated.")


if __name__ == "__main__":
    main()
