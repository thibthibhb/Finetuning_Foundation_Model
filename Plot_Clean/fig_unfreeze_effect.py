#!/usr/bin/env python3
"""
Full vs Two-phase unfreezing (simple, CSV-only)

Panel A: Test Cohen's κ by training mode
Panel B: Overfitting proxy per mode: (best Val κ or Val κ) − Test κ

Usage:
  python unfreeze_modes_quick.py \
    --csv Plot_Clean/data/all_runs_flat.csv \
    --classes 5 \
    --out Plot_Clean/figures/unfreeze_modes_quick
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- style ----------
def setup_style(dpi=300):
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 15,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

# ---------- utilities ----------
def to_bool(x):
    if isinstance(x, bool): return x
    if isinstance(x, (int, float)) and not np.isnan(x): return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true","yes","y","t","1"}: return True
        if s in {"false","no","n","f","0"}: return False
    return np.nan

def bootstrap_median_ci(x: pd.Series, n=2000, seed=42):
    arr = x.dropna().to_numpy(dtype=float)
    if arr.size < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    med = np.median(arr)
    boots = [np.median(rng.choice(arr, size=arr.size, replace=True)) for _ in range(n)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return med, lo, hi

# ---------- column resolution ----------
def resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # --- Test κ (required)
    for c in ['test_kappa','contract.results.test_kappa']:
        if c in out: out['test_kappa'] = pd.to_numeric(out[c], errors='coerce')
    if 'test_kappa' not in out:
        raise ValueError("No test_kappa column found in CSV.")

    # --- Validation κ (final) and Best validation κ (max)
    candidates_val = [
        'val_kappa','contract.results.val_kappa','sum.val_kappa'
    ]
    candidates_best = [
        'best_val_kappa','sum.best_val_kappa','contract.results.best_val_kappa',
        'val_kappa_best','best_val_kappa_overall','val_kappa_max'
    ]
    for c in candidates_val:
        if c in out and 'val_kappa' not in out:
            out['val_kappa'] = pd.to_numeric(out[c], errors='coerce')
    if 'val_kappa' not in out:
        out['val_kappa'] = np.nan

    for c in candidates_best:
        if c in out and 'best_val_kappa' not in out:
            out['best_val_kappa'] = pd.to_numeric(out[c], errors='coerce')
    if 'best_val_kappa' not in out:
        out['best_val_kappa'] = np.nan

    # --- Subject ID
    for c in ['cfg.subject_id','cfg.subj_id','subject_id','sum.subject_id','name']:
        if c in out and 'subject' not in out:
            out['subject'] = out[c].astype(str)
    if 'subject' not in out:
        raise ValueError("No subject identifier column resolved.")

    # --- Num classes (optional filter)
    for c in ['num_classes','contract.results.num_classes','cfg.num_of_classes']:
        if c in out and 'num_classes' not in out:
            out['num_classes'] = pd.to_numeric(out[c], errors='coerce')

    # --- Unfreeze / Phase-1 length
    for c in ['cfg.unfreeze_epoch','unfreeze_epoch','contract.training.unfreeze_epoch']:
        if c in out and 'unfreeze_epoch' not in out:
            out['unfreeze_epoch'] = pd.to_numeric(out[c], errors='coerce')
    for c in ['cfg.phase1_epochs','phase1_epochs','contract.training.phase1_epochs']:
        if c in out and 'phase1_epochs' not in out:
            out['phase1_epochs'] = pd.to_numeric(out[c], errors='coerce')
    if 'unfreeze_epoch' not in out: out['unfreeze_epoch'] = np.nan
    if 'phase1_epochs' not in out:  out['phase1_epochs'] = np.nan

    # --- Total epochs (optional)
    for c in ['cfg.epochs','epochs','contract.training.epochs']:
        if c in out and 'epochs_total' not in out:
            out['epochs_total'] = pd.to_numeric(out[c], errors='coerce')
    if 'epochs_total' not in out: out['epochs_total'] = np.nan

    # --- two_phase_training flag (the arg you mentioned)
    for c in ['cfg.two_phase_training','two_phase_training','contract.training.two_phase_training']:
        if c in out and 'two_phase_training' not in out:
            out['two_phase_training'] = out[c].apply(to_bool)
    if 'two_phase_training' not in out:
        out['two_phase_training'] = np.nan  # unknown

    # --- "frozen" flag (helps detect truly frozen-all)
    for c in ['cfg.frozen','contract.model.frozen','frozen']:
        if c in out and 'frozen_flag' not in out:
            out['frozen_flag'] = out[c].apply(to_bool)
    if 'frozen_flag' not in out: out['frozen_flag'] = np.nan

    # --- Build unfreeze_group robustly
    def infer_unfreeze_epoch(row):
        # Prefer explicit unfreeze_epoch, else phase1_epochs (freeze K then unfreeze at K)
        ue = row.get('unfreeze_epoch', np.nan)
        if pd.isna(ue) and not pd.isna(row.get('phase1_epochs', np.nan)):
            return row['phase1_epochs']
        return ue

    out['unfreeze_epoch_inferred'] = out.apply(infer_unfreeze_epoch, axis=1)

    def to_group(row):
        ue = row['unfreeze_epoch_inferred']
        te = row['epochs_total']
        two_phase = row['two_phase_training']
        frozen = row['frozen_flag']

        # If explicitly two-phase, treat as K>0 even if ue missing (fallback K=1)
        if isinstance(two_phase, bool) and two_phase:
            k = 1 if pd.isna(ue) else int(max(1, round(float(ue))))
            return f'unfreeze@{k}'

        # If ue is known:
        if not pd.isna(ue):
            k = int(round(float(ue)))
            if not pd.isna(te) and k >= int(te):
                return 'frozen-all'
            return f'unfreeze@{k if k>0 else 0}'

        # No ue info: if we see an explicit frozen flag and not two-phase, call it frozen-all
        if isinstance(frozen, bool) and frozen and (not isinstance(two_phase, bool) or not two_phase):
            return 'frozen-all'

        # Unknown → assume full finetune (most training defaults)
        return 'unfreeze@0'

    out['unfreeze_group'] = out.apply(to_group, axis=1)

    # --- High-level training mode
    def to_mode(g):
        if g == 'frozen-all': return 'Frozen-all'
        if g == 'unfreeze@0': return 'Full finetune'
        return 'Two-phase (K>0)'
    out['training_mode'] = out['unfreeze_group'].map(to_mode)

    # --- Overfitting proxy: best available validation κ − test κ
    # Prefer best_val_kappa, else val_kappa
    best_or_final_val = out['best_val_kappa'].where(out['best_val_kappa'].notna(), out['val_kappa'])
    out['optimism_gap'] = best_or_final_val - out['test_kappa']

    return out

# ---------- aggregation ----------
def pick_best_per_subject_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Within (subject × training_mode), keep run with highest best_val_kappa, else val_kappa, else test_kappa."""
    def choose(g):
        g = g.copy()
        if 'best_val_kappa' in g and g['best_val_kappa'].notna().any():
            return g.loc[g['best_val_kappa'].idxmax()]
        if g['val_kappa'].notna().any():
            return g.loc[g['val_kappa'].idxmax()]
        return g.loc[g['test_kappa'].idxmax()]
    return (df.groupby(['subject','training_mode'], as_index=False)
              .apply(choose)
              .reset_index(drop=True))

# ---------- plotting ----------
def plot_modes(tidy: pd.DataFrame, outdir: Path, classes_label: str):
    # mode order
    modes = ['Full finetune', 'Two-phase (K>0)']
    if 'Frozen-all' in tidy['training_mode'].unique():
        modes.append('Frozen-all')
    x = np.arange(len(modes))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A — Test κ
    for i, m in enumerate(modes):
        vals = tidy.loc[tidy['training_mode']==m, 'test_kappa']
        if vals.empty: continue
        bp = ax1.boxplot([vals], positions=[i], widths=0.55, patch_artist=True,
                         medianprops=dict(linewidth=2),
                         boxprops=dict(facecolor='white', edgecolor='black'),
                         whiskerprops=dict(color='black'), capprops=dict(color='black'))
        jitter = np.random.default_rng(0).normal(0, 0.06, size=len(vals))
        ax1.scatter(i + jitter, vals, s=18, alpha=0.35)
        med, lo, hi = bootstrap_median_ci(vals)
        ax1.text(i, vals.max() + 0.01, f"N={vals.count()}\nmedian={med:.3f}",
                 ha='center', va='bottom', fontsize=9)

    ax1.set_xticks(x); ax1.set_xticklabels(modes, rotation=20, ha='right')
    ax1.set_ylabel("Test Cohen's κ")
    ax1.set_title(f"Test κ by training mode  (classes={classes_label})")

    # Panel B — Overfitting proxy
    have_val = tidy['optimism_gap'].notna().sum()
    if have_val == 0:
        ax2.text(0.5, 0.5,
                 "No validation κ found in CSV\n(cannot compute overfitting proxy)",
                 ha='center', va='center', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax2.set_axis_off()
    else:
        for i, m in enumerate(modes):
            vals = tidy.loc[tidy['training_mode']==m, 'optimism_gap'].dropna()
            if vals.empty: continue
            ax2.boxplot([vals], positions=[i], widths=0.55, patch_artist=True,
                        medianprops=dict(linewidth=2),
                        boxprops=dict(facecolor='white', edgecolor='black'),
                        whiskerprops=dict(color='black'), capprops=dict(color='black'))
            jitter = np.random.default_rng(1).normal(0, 0.06, size=len(vals))
            ax2.scatter(i + jitter, vals, s=18, alpha=0.35)
        ax2.axhline(0, color='gray', ls='--', alpha=0.6)
        ax2.set_xticks(x); ax2.set_xticklabels(modes, rotation=20, ha='right')
        ax2.set_ylabel("(best Val κ or Val κ) − Test κ\n(higher = more overfitting)")
        ax2.set_title("Overfitting by training mode")

    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in ['pdf','png','svg']:
        fig.savefig(outdir / f'unfreeze_modes_quick.{fmt}')
    print("Saved:", outdir / 'unfreeze_modes_quick.[pdf|png|svg]')

# ---------- main ----------
def main():
    setup_style()
    ap = argparse.ArgumentParser(description="Full vs Two-phase unfreezing (simple CSV analysis)")
    ap.add_argument('--csv', required=True, help='Flat CSV path (e.g., Plot_Clean/data/all_runs_flat.csv)')
    ap.add_argument('--classes', choices=['4','5','both'], default='5', help='Filter by number of classes')
    ap.add_argument('--out', default='Plot_Clean/figures/unfreeze_modes_quick', help='Output directory')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = resolve_columns(df)

    if args.classes != 'both' and 'num_classes' in df.columns:
        df = df[df['num_classes'] == int(args.classes)]
        classes_label = args.classes
    else:
        classes_label = 'mixed'

    # Keep rows with subject, training mode, and test κ
    df = df.dropna(subset=['subject','training_mode','test_kappa'])
    if df.empty:
        raise SystemExit("No usable rows after cleaning. Check CSV/columns.")

    # Best run per (subject × training_mode)
    tidy = pick_best_per_subject_mode(df)

    # Console sanity checks
    print("\nTraining mode counts (after best-run selection):")
    print(tidy['training_mode'].value_counts().sort_index())
    print("\nUnfreeze group breakdown:")
    if 'unfreeze_group' in tidy.columns:
        print(tidy['unfreeze_group'].value_counts().sort_index())
    print("\nValidation availability:")
    print("  best_val_kappa non-NaN:", int(tidy['best_val_kappa'].notna().sum()))
    print("  val_kappa      non-NaN:", int(tidy['val_kappa'].notna().sum()))

    # Plot
    plot_modes(tidy, Path(args.out), classes_label)

    # Save tidy data for reproducibility
    tidy.to_csv(Path(args.out) / 'unfreeze_modes_quick_data.csv', index=False)
    print("Saved:", Path(args.out) / 'unfreeze_modes_quick_data.csv')

if __name__ == '__main__':
    main()
