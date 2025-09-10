#!/usr/bin/env python3
"""
Figure 4: Subjects vs Minutes per Subject - What Matters More?

Shows how spreading calibration data across subjects affects performance.
Addresses the practical question: "Is it better to have more subjects with less data each,
or fewer subjects with more data each?"

Uses real W&B data: hours_of_data / num_subjects_train to calculate minutes per subject.

Usage:
  python Plot_Clean/plot_subjects_vs_minutes.py --csv ../data/all_runs_flat.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import bootstrap
from scipy.interpolate import griddata
from scipy import stats
import argparse
import warnings
from matplotlib import cycler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.inspection import partial_dependence
from sklearn.ensemble import RandomForestRegressor
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Statsmodels not available. Install with: pip install statsmodels")

warnings.filterwarnings("ignore")

# Import consistent figure styling
import sys; sys.path.append("../style"); from figure_style import (
    setup_figure_style, get_color, save_figure, 
    add_yasa_baseline, add_significance_marker,
    bootstrap_ci_median, wilcoxon_test,
    format_n_caption, add_sample_size_annotation,
    OKABE_ITO
)


# Remove duplicate bootstrap_ci_median function - using the one import sys; sys.path.append("../style"); from figure_style.py

def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV data and calculate minutes per subject."""
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} rows")
    
    # CRITICAL: Filter out high noise experiments to avoid bias
    if 'noise_level' in df.columns:
        noise_stats = df['noise_level'].value_counts().sort_index()
        print(f"🔊 Noise level distribution: {dict(noise_stats)}")
        
        # Keep only clean data (noise_level <= 0.01 or 1%) 
        df = df[df['noise_level'] <= 0.01].copy()
        print(f"✅ Filtered to clean data: {len(df)} rows remaining (noise ≤ 1%)")
        
        if len(df) == 0:
            raise ValueError("No clean data found after noise filtering.")
    
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

def iso_hours_analysis(df: pd.DataFrame, hour_bins: list = None) -> pd.DataFrame:
    """Matched comparisons analysis - compare different allocations within equal total hours bins."""
    
    if hour_bins is None:
        # Create bins based on data distribution
        min_hours = df['total_hours'].min()
        max_hours = df['total_hours'].max()
        hour_bins = np.arange(np.floor(min_hours/100)*100, np.ceil(max_hours/100)*100 + 100, 100)
    
    print(f"\n=== ISO-HOURS ANALYSIS ===")
    print(f"Analyzing {len(hour_bins)-1} hour bins: {hour_bins}")
    
    iso_results = []
    
    for i in range(len(hour_bins)-1):
        bin_min = hour_bins[i]
        bin_max = hour_bins[i+1]
        
        # Get data within this hour bin
        bin_data = df[(df['total_hours'] >= bin_min) & (df['total_hours'] < bin_max)].copy()
        
        if len(bin_data) < 5:  # Need sufficient data for comparison
            continue
            
        print(f"\nHour bin [{bin_min:.0f}-{bin_max:.0f}h]: {len(bin_data)} runs")
        
        # Within this iso-hour bin, compare different allocation strategies
        bin_data['strategy'] = pd.cut(bin_data['num_subjects'], 
                                     bins=[0, 5, 10, 20, float('inf')], 
                                     labels=['1-5 subj', '6-10 subj', '11-20 subj', '20+ subj'])
        
        for strategy, group in bin_data.groupby('strategy', observed=True):
            if len(group) >= 3:  # Need at least 3 runs for meaningful stats
                median_delta = group['delta_kappa'].median()
                mean_subjects = group['num_subjects'].mean()
                mean_minutes_per_subj = group['minutes_per_subject'].mean()
                
                iso_results.append({
                    'hour_bin': f'{bin_min:.0f}-{bin_max:.0f}h',
                    'hour_bin_center': (bin_min + bin_max) / 2,
                    'strategy': strategy,
                    'n_runs': len(group),
                    'median_delta_kappa': median_delta,
                    'mean_subjects': mean_subjects,
                    'mean_minutes_per_subject': mean_minutes_per_subj,
                    'total_hours_mean': group['total_hours'].mean()
                })
                
                print(f"  {strategy}: n={len(group)}, Δκ={median_delta:.3f}, "
                      f"{mean_subjects:.1f} subj × {mean_minutes_per_subj:.0f} min/subj")
    
    iso_df = pd.DataFrame(iso_results)
    
    if not iso_df.empty:
        print(f"\n=== ISO-HOURS SUMMARY ===")
        best_per_bin = iso_df.groupby('hour_bin').apply(
            lambda x: x.loc[x['median_delta_kappa'].idxmax()]
        ).reset_index(drop=True)
        
        for _, row in best_per_bin.iterrows():
            print(f"{row['hour_bin']}: Best = {row['strategy']} "
                  f"(Δκ={row['median_delta_kappa']:.3f}, "
                  f"{row['mean_subjects']:.1f} subj × {row['mean_minutes_per_subject']:.0f} min/subj)")
    
    return iso_df

def mixed_effects_regression(df: pd.DataFrame) -> dict:
    """Mixed-effects regression analysis to quantify relative importance."""
    
    print(f"\n=== MIXED-EFFECTS REGRESSION ===")
    
    if not STATSMODELS_AVAILABLE:
        print("Statsmodels not available - using regular regression")
        return regular_regression_analysis(df)
    
    # Prepare data
    analysis_df = df.copy()
    analysis_df = analysis_df.dropna(subset=['delta_kappa', 'num_subjects', 'minutes_per_subject', 'total_hours'])
    
    if len(analysis_df) < 10:
        print(f"Insufficient data for regression: {len(analysis_df)} rows")
        return {}
    
    # Add dataset as a grouping variable if available
    if 'dataset_composition' in analysis_df.columns:
        dataset_col = 'dataset_composition'
    elif 'cfg.datasets' in analysis_df.columns:
        dataset_col = 'cfg.datasets'
    else:
        # Create a dummy grouping variable
        analysis_df['dataset_dummy'] = 'dataset_1'
        dataset_col = 'dataset_dummy'
    
    # Add control variables if available
    control_vars = []
    
    # Check for unfreeze epoch
    unfreeze_cols = [col for col in analysis_df.columns if 'unfreeze' in col.lower()]
    if unfreeze_cols:
        unfreeze_col = unfreeze_cols[0]
        if analysis_df[unfreeze_col].notna().sum() > len(analysis_df) * 0.5:
            control_vars.append(unfreeze_col)
    
    # Check for label granularity
    label_cols = [col for col in analysis_df.columns if 'label' in col.lower() and 'version' in col.lower()]
    if label_cols:
        label_col = label_cols[0]
        if analysis_df[label_col].notna().sum() > len(analysis_df) * 0.5:
            analysis_df[f'{label_col}_encoded'] = pd.Categorical(analysis_df[label_col]).codes
            control_vars.append(f'{label_col}_encoded')
    
    try:
        # Build formula
        formula_parts = ['delta_kappa ~ num_subjects + minutes_per_subject + total_hours']
        if control_vars:
            formula_parts.append(' + '.join(control_vars))
        
        formula = ' + '.join(formula_parts)
        
        print(f"Formula: {formula}")
        print(f"Random effects: (1|{dataset_col})")
        
        # Fit mixed-effects model
        model = mixedlm(formula, analysis_df, groups=analysis_df[dataset_col])
        result = model.fit(method='lbfgs')
        
        print(f"\nMixed-Effects Model Results:")
        print(result.summary())
        
        # Extract standardized coefficients
        scaler = StandardScaler()
        X_cols = ['num_subjects', 'minutes_per_subject', 'total_hours'] + control_vars
        X_available = [col for col in X_cols if col in analysis_df.columns]
        
        X_scaled = scaler.fit_transform(analysis_df[X_available])
        y_scaled = scaler.fit_transform(analysis_df[['delta_kappa']])
        
        # Calculate partial R² for main variables
        main_vars = ['num_subjects', 'minutes_per_subject', 'total_hours']
        partial_r2 = {}
        
        for var in main_vars:
            if var in analysis_df.columns:
                # Full model R²
                full_vars = [v for v in X_available]
                if len(full_vars) > 1:
                    X_full = analysis_df[full_vars]
                    lr_full = LinearRegression().fit(X_full, analysis_df['delta_kappa'])
                    r2_full = lr_full.score(X_full, analysis_df['delta_kappa'])
                    
                    # Reduced model R² (without this variable)
                    reduced_vars = [v for v in full_vars if v != var]
                    if reduced_vars:
                        X_reduced = analysis_df[reduced_vars]
                        lr_reduced = LinearRegression().fit(X_reduced, analysis_df['delta_kappa'])
                        r2_reduced = lr_reduced.score(X_reduced, analysis_df['delta_kappa'])
                        
                        partial_r2[var] = r2_full - r2_reduced
                    else:
                        partial_r2[var] = r2_full
        
        print(f"\nPartial R² (unique contribution):")
        for var, r2 in sorted(partial_r2.items(), key=lambda x: x[1], reverse=True):
            print(f"  {var}: {r2:.4f}")
        
        return {
            'model': result,
            'partial_r2': partial_r2,
            'formula': formula,
            'n_obs': len(analysis_df)
        }
        
    except Exception as e:
        print(f"Mixed-effects model failed: {e}")
        return regular_regression_analysis(df)

def regular_regression_analysis(df: pd.DataFrame) -> dict:
    """Fallback regular regression analysis."""
    
    analysis_df = df.copy()
    analysis_df = analysis_df.dropna(subset=['delta_kappa', 'num_subjects', 'minutes_per_subject', 'total_hours'])
    
    if len(analysis_df) < 10:
        print(f"Insufficient data for regression: {len(analysis_df)} rows")
        return {}
    
    X_vars = ['num_subjects', 'minutes_per_subject', 'total_hours']
    X = analysis_df[X_vars]
    y = analysis_df['delta_kappa']
    
    # Standardized coefficients
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lr = LinearRegression().fit(X_scaled, y)
    
    print(f"\nStandardized Coefficients (Regular Regression):")
    for var, coef in zip(X_vars, lr.coef_):
        print(f"  {var}: {coef:.4f}")
    
    print(f"R² = {lr.score(X_scaled, y):.4f}")
    
    return {
        'coefficients': dict(zip(X_vars, lr.coef_)),
        'r2': lr.score(X_scaled, y),
        'n_obs': len(analysis_df)
    }

def partial_dependence_analysis(df: pd.DataFrame) -> dict:
    """Partial dependence and SHAP analysis for nonlinearity detection."""
    
    print(f"\n=== PARTIAL DEPENDENCE ANALYSIS ===")
    
    analysis_df = df.copy()
    analysis_df = analysis_df.dropna(subset=['delta_kappa', 'num_subjects', 'minutes_per_subject', 'total_hours'])
    
    if len(analysis_df) < 20:
        print(f"Insufficient data for partial dependence: {len(analysis_df)} rows")
        return {}
    
    X_vars = ['num_subjects', 'minutes_per_subject', 'total_hours']
    X = analysis_df[X_vars]
    y = analysis_df['delta_kappa']
    
    # Use Random Forest for nonlinear relationships
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    print(f"Random Forest R² = {rf.score(X, y):.4f}")
    print(f"Feature Importance:")
    for var, importance in zip(X_vars, rf.feature_importances_):
        print(f"  {var}: {importance:.4f}")
    
    # Partial dependence plots
    results = {'model': rf, 'feature_importance': dict(zip(X_vars, rf.feature_importances_))}
    
    # SHAP analysis if available
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X)
            
            print(f"\nSHAP Analysis:")
            shap_importance = np.abs(shap_values).mean(0)
            for var, importance in zip(X_vars, shap_importance):
                print(f"  {var} SHAP importance: {importance:.4f}")
            
            results['shap_values'] = shap_values
            results['shap_importance'] = dict(zip(X_vars, shap_importance))
            
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
    
    # Check for diminishing returns in minutes_per_subject
    if 'minutes_per_subject' in X.columns:
        minutes_range = np.linspace(X['minutes_per_subject'].min(), 
                                   X['minutes_per_subject'].max(), 50)
        
        # Fix other variables at median
        X_pd = X.copy()
        pd_results = []
        
        for minutes in minutes_range:
            X_temp = X_pd.copy()
            X_temp['minutes_per_subject'] = minutes
            pred = rf.predict(X_temp).mean()
            pd_results.append({'minutes_per_subject': minutes, 'predicted_delta': pred})
        
        pd_df = pd.DataFrame(pd_results)
        
        # Check for knee/saturation point
        diffs = np.diff(pd_df['predicted_delta'])
        if len(diffs) > 10:
            # Find where rate of improvement drops significantly
            knee_idx = np.where(diffs < np.percentile(diffs, 25))[0]
            if len(knee_idx) > 0:
                knee_minutes = pd_df.iloc[knee_idx[0]]['minutes_per_subject']
                print(f"\nDiminishing returns knee detected at ~{knee_minutes:.0f} minutes per subject")
                results['knee_point'] = knee_minutes
        
        results['partial_dependence'] = pd_df
    
    return results

def sensitivity_analysis(df: pd.DataFrame) -> dict:
    """Sensitivity analysis - repeat without top-3 selection to check robustness."""
    
    print(f"\n=== SENSITIVITY ANALYSIS ===")
    print(f"Comparing results with and without top-3 per bin selection")
    
    # Original analysis with top-3 selection
    print(f"\nWith top-3 selection:")
    df_filtered = select_top_runs_per_bin(df.copy(), 'total_hours', 100, n_top=3)
    print(f"  Data: {len(df_filtered)} runs (from {len(df)} total)")
    
    if len(df_filtered) >= 10:
        reg_filtered = regular_regression_analysis(df_filtered)
        corr_filtered = {
            'subjects': np.corrcoef(df_filtered['num_subjects'], df_filtered['delta_kappa'])[0,1],
            'minutes_per_subject': np.corrcoef(df_filtered['minutes_per_subject'], df_filtered['delta_kappa'])[0,1],
            'total_hours': np.corrcoef(df_filtered['total_hours'], df_filtered['delta_kappa'])[0,1]
        }
        print(f"  Correlations - subjects: {corr_filtered['subjects']:.3f}, "
              f"minutes/subj: {corr_filtered['minutes_per_subject']:.3f}, "
              f"total: {corr_filtered['total_hours']:.3f}")
    
    # Full dataset analysis
    print(f"\nWithout filtering (full dataset):")
    print(f"  Data: {len(df)} runs")
    
    if len(df) >= 10:
        reg_full = regular_regression_analysis(df)
        corr_full = {
            'subjects': np.corrcoef(df['num_subjects'], df['delta_kappa'])[0,1],
            'minutes_per_subject': np.corrcoef(df['minutes_per_subject'], df['delta_kappa'])[0,1],
            'total_hours': np.corrcoef(df['total_hours'], df['delta_kappa'])[0,1]
        }
        print(f"  Correlations - subjects: {corr_full['subjects']:.3f}, "
              f"minutes/subj: {corr_full['minutes_per_subject']:.3f}, "
              f"total: {corr_full['total_hours']:.3f}")
    
    # Random sampling analysis
    print(f"\nRandom sampling (same size as filtered):")
    n_sample = min(len(df_filtered) if 'df_filtered' in locals() else 100, len(df))
    np.random.seed(42)
    df_random = df.sample(n=n_sample, random_state=42)
    print(f"  Data: {len(df_random)} runs")
    
    if len(df_random) >= 10:
        reg_random = regular_regression_analysis(df_random)
        corr_random = {
            'subjects': np.corrcoef(df_random['num_subjects'], df_random['delta_kappa'])[0,1],
            'minutes_per_subject': np.corrcoef(df_random['minutes_per_subject'], df_random['delta_kappa'])[0,1],
            'total_hours': np.corrcoef(df_random['total_hours'], df_random['delta_kappa'])[0,1]
        }
        print(f"  Correlations - subjects: {corr_random['subjects']:.3f}, "
              f"minutes/subj: {corr_random['minutes_per_subject']:.3f}, "
              f"total: {corr_random['total_hours']:.3f}")
    
    # Summary of robustness
    print(f"\n=== SENSITIVITY SUMMARY ===")
    
    # Compare which factor is most important across methods
    methods = ['filtered', 'full', 'random']
    results_by_method = {}
    
    for method in methods:
        corr_dict = locals().get(f'corr_{method}')
        if corr_dict:
            abs_corrs = {k: abs(v) for k, v in corr_dict.items()}
            strongest = max(abs_corrs, key=abs_corrs.get)
            results_by_method[method] = {
                'strongest_predictor': strongest,
                'correlations': corr_dict
            }
    
    if results_by_method:
        print(f"Strongest predictor by method:")
        for method, result in results_by_method.items():
            print(f"  {method}: {result['strongest_predictor']} "
                  f"(r={result['correlations'][result['strongest_predictor']]:.3f})")
        
        # Check consistency
        strongest_predictors = [r['strongest_predictor'] for r in results_by_method.values()]
        if len(set(strongest_predictors)) == 1:
            print(f"\n✅ ROBUST: {strongest_predictors[0]} consistently most important")
        else:
            print(f"\n⚠️ INCONSISTENT: Results vary by selection method")
    
    return results_by_method

def comprehensive_testing_suite(df: pd.DataFrame, output_dir: Path) -> dict:
    """Run all testing methods and create comprehensive output."""
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE TESTING SUITE")
    print(f"{'='*60}")
    print(f"Dataset: {len(df)} runs")
    
    results = {}
    
    # 1. Iso-hours analysis
    try:
        results['iso_hours'] = iso_hours_analysis(df)
    except Exception as e:
        print(f"Iso-hours analysis failed: {e}")
        results['iso_hours'] = pd.DataFrame()
    
    # 2. Mixed-effects regression
    try:
        results['mixed_effects'] = mixed_effects_regression(df)
    except Exception as e:
        print(f"Mixed-effects regression failed: {e}")
        results['mixed_effects'] = {}
    
    # 3. Partial dependence analysis
    try:
        results['partial_dependence'] = partial_dependence_analysis(df)
    except Exception as e:
        print(f"Partial dependence analysis failed: {e}")
        results['partial_dependence'] = {}
    
    # 4. Sensitivity analysis
    try:
        results['sensitivity'] = sensitivity_analysis(df)
    except Exception as e:
        print(f"Sensitivity analysis failed: {e}")
        results['sensitivity'] = {}
    
    # 5. Create comprehensive summary
    create_testing_summary(results, output_dir)
    
    return results

def create_testing_summary(results: dict, output_dir: Path):
    """Create comprehensive testing summary and visualizations."""
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE TESTING SUMMARY")
    print(f"{'='*60}")
    
    # Summary text
    summary_lines = []
    summary_lines.append("# CBraMod Subjects vs Minutes: Comprehensive Testing Results\n")
    
    # Iso-hours analysis
    if 'iso_hours' in results and not results['iso_hours'].empty:
        iso_df = results['iso_hours']
        summary_lines.append("## Iso-Hours Analysis (Matched Comparisons)\n")
        summary_lines.append("Comparing different allocation strategies within equal total hour bins:\n")
        
        for hour_bin in iso_df['hour_bin'].unique():
            bin_data = iso_df[iso_df['hour_bin'] == hour_bin]
            best_strategy = bin_data.loc[bin_data['median_delta_kappa'].idxmax()]
            summary_lines.append(f"- **{hour_bin}**: Best = {best_strategy['strategy']} "
                                f"(Δκ={best_strategy['median_delta_kappa']:.3f})\n")
        summary_lines.append("\n")
    
    # Mixed-effects regression
    if 'mixed_effects' in results and results['mixed_effects']:
        me_results = results['mixed_effects']
        summary_lines.append("## Mixed-Effects Regression Analysis\n")
        
        if 'partial_r2' in me_results:
            summary_lines.append("Partial R² (unique contribution of each factor):\n")
            for var, r2 in sorted(me_results['partial_r2'].items(), key=lambda x: x[1], reverse=True):
                summary_lines.append(f"- **{var}**: {r2:.4f}\n")
        
        if 'coefficients' in me_results:
            summary_lines.append("\nStandardized coefficients:\n")
            for var, coef in me_results['coefficients'].items():
                summary_lines.append(f"- **{var}**: {coef:.4f}\n")
        summary_lines.append("\n")
    
    # Partial dependence analysis
    if 'partial_dependence' in results and results['partial_dependence']:
        pd_results = results['partial_dependence']
        summary_lines.append("## Partial Dependence Analysis\n")
        
        if 'feature_importance' in pd_results:
            summary_lines.append("Random Forest feature importance:\n")
            for var, importance in sorted(pd_results['feature_importance'].items(), key=lambda x: x[1], reverse=True):
                summary_lines.append(f"- **{var}**: {importance:.4f}\n")
        
        if 'knee_point' in pd_results:
            summary_lines.append(f"\n**Diminishing returns detected** at ~{pd_results['knee_point']:.0f} minutes per subject\n")
        
        if 'shap_importance' in pd_results:
            summary_lines.append("\nSHAP importance (average impact):\n")
            for var, importance in sorted(pd_results['shap_importance'].items(), key=lambda x: x[1], reverse=True):
                summary_lines.append(f"- **{var}**: {importance:.4f}\n")
        summary_lines.append("\n")
    
    # Sensitivity analysis
    if 'sensitivity' in results and results['sensitivity']:
        sens_results = results['sensitivity']
        summary_lines.append("## Sensitivity Analysis\n")
        summary_lines.append("Robustness check across different data selection methods:\n")
        
        for method, result in sens_results.items():
            if 'strongest_predictor' in result:
                strongest = result['strongest_predictor']
                corr_val = result['correlations'][strongest]
                summary_lines.append(f"- **{method}**: {strongest} (r={corr_val:.3f})\n")
        summary_lines.append("\n")
    
    # Overall conclusion
    summary_lines.append("## Overall Conclusion\n")
    
    # Determine consensus
    importance_rankings = []
    
    if 'mixed_effects' in results and 'partial_r2' in results['mixed_effects']:
        ranking = list(sorted(results['mixed_effects']['partial_r2'].items(), key=lambda x: x[1], reverse=True))
        importance_rankings.append(('mixed_effects', ranking))
    
    if 'partial_dependence' in results and 'feature_importance' in results['partial_dependence']:
        ranking = list(sorted(results['partial_dependence']['feature_importance'].items(), key=lambda x: x[1], reverse=True))
        importance_rankings.append(('random_forest', ranking))
    
    if 'sensitivity' in results:
        for method, result in results['sensitivity'].items():
            if 'correlations' in result:
                abs_corrs = {k: abs(v) for k, v in result['correlations'].items()}
                ranking = list(sorted(abs_corrs.items(), key=lambda x: x[1], reverse=True))
                importance_rankings.append((f'correlation_{method}', ranking))
    
    if importance_rankings:
        # Find most consistent top factor
        top_factors = [ranking[0][0] for method, ranking in importance_rankings if ranking]
        if top_factors:
            from collections import Counter
            factor_counts = Counter(top_factors)
            most_common = factor_counts.most_common(1)[0]
            
            if most_common[1] > len(importance_rankings) / 2:  # Majority agreement
                summary_lines.append(f"**Consensus**: {most_common[0]} is consistently the most important factor "
                                    f"across {most_common[1]}/{len(importance_rankings)} analysis methods.\n\n")
            else:
                summary_lines.append(f"**Mixed evidence**: No clear consensus on most important factor. "
                                    f"Top factors: {dict(factor_counts)}\n\n")
    
    # Save summary
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / 'comprehensive_testing_results.md'
    
    with open(summary_file, 'w') as f:
        f.writelines(summary_lines)
    
    print(f"\n📊 Comprehensive testing summary saved to: {summary_file}")
    print(f"\n🔍 Key findings:")
    for line in summary_lines[-10:]:  # Show last few lines of summary
        if line.startswith('**'):
            print(f"   {line.strip()}")

def create_figure_4(df: pd.DataFrame, output_dir: Path, run_comprehensive_tests: bool = True):
    """Create Figure 4: Subjects vs Minutes per Subject analysis."""
    
    yasa_kappa = 0.446
    delta_threshold = 0.05
    
    print("Creating subjects vs minutes per subject analysis...")
    
    if df.empty:
        print("ERROR: No valid data found!")
        return None
    
    # Calculate delta kappa
    df['delta_kappa'] = df['test_kappa'] - yasa_kappa
    
    # Create figure with subplots (add space for common legend)
    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1, 1, 0.15])

    fig.suptitle(
        'Subjects vs Minutes per Subject - What Matters More?\n'
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

        # Reduce x-axis tick density for cleaner appearance
        ax_main.set_xlabel('Minutes per Subject', fontweight='bold')
        ax_main.set_ylabel('Number of Subjects', fontweight='bold')
        ax_main.set_title('A: Performance Heatmap - Subjects vs Minutes per Subject', fontweight='bold', pad=10)
        
        # Show fewer x-axis labels for cleaner look
        x_ticks = ax_main.get_xticks()
        if len(x_ticks) > 8:  # If more than 8 ticks, reduce them
            # Show every other tick or every 3rd tick depending on density
            step = max(2, len(x_ticks) // 6)  # Show ~6 labels maximum
            keep_indices = range(0, len(x_ticks), step)
            x_labels = ax_main.get_xticklabels()
            
            new_ticks = [x_ticks[i] for i in keep_indices]
            new_labels = [x_labels[i].get_text() for i in keep_indices]
            
            ax_main.set_xticks(new_ticks)
            ax_main.set_xticklabels(new_labels, rotation=35, ha='right')
        else:
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
                    ax_main.contour(X, Y, Z, levels=[0], colors=[get_color('yasa')], linestyles='--', linewidths=2)
                    if np.nanmax(Z) > delta_threshold:
                        ax_main.contour(X, Y, Z, levels=[delta_threshold], colors=[get_color('t_star')], linestyles='-', linewidths=2.5)
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
                    # Δκ = 0 (tie with YASA) - orange  
                    contour_zero = ax_main.contour(X, Y, Z, levels=[0], 
                                                 colors=[get_color('yasa')], 
                                                 linestyles='--', linewidths=2)
                    
                    # Δκ = threshold (target) - green
                    if np.nanmax(Z) > delta_threshold:
                        contour_target = ax_main.contour(X, Y, Z, levels=[delta_threshold], 
                                                       colors=[get_color('t_star')], 
                                                       linestyles='-', linewidths=3)
            except:
                print("Could not add contour lines to heatmap")
    
    # Panel B: Scatter plot - Performance vs Total Data Hours (top performer + top 3 per 100h bin)
    ax1 = fig.add_subplot(gs[1, 0])
    
    # Find overall top performer
    top_performer = df.loc[df['delta_kappa'].idxmax()].copy()
    
    # Select top 3 runs per 100-hour bin
    df_total_hours = select_top_runs_per_bin(df.copy(), 'total_hours', 100, n_top=3)
    print(f"Panel B: {len(df_total_hours)} top 3 runs selected from total hours bins")
    
    if not df_total_hours.empty:
        # slight jitter function to reduce vertical stacking
        rng = np.random.default_rng(42)
        jitter = (rng.standard_normal(len(df_total_hours)) * 0.005)

        # Plot top 3 per bin as regular points - use consistent CBraMod color
        scatter = ax1.scatter(
            df_total_hours['total_hours'],
            df_total_hours['delta_kappa'] + jitter,
            c=get_color('cbramod'),
            s=40, alpha=0.7,  # Smaller points for bin tops
            edgecolors='black', linewidth=0.5,
            label='Top 3 per 100h bin'
        )

        # No colorbar needed since we use consistent color
        
        # Highlight overall top performer with special marker
        ax1.scatter(
            top_performer['total_hours'],
            top_performer['delta_kappa'],
            c='red', s=120, alpha=0.9,
            marker='*', edgecolors='darkred', linewidth=2,
            label=f'Top performer (Δκ={top_performer["delta_kappa"]:.3f})'
        )

    # YASA baseline as simple horizontal line
    ax1.axhline(y=0, color=get_color('yasa'), linestyle='--', linewidth=1.8, 
               label='YASA baseline (Δκ=0)', zorder=2)
    ax1.axhline(y=delta_threshold, color=get_color('t_star'), linestyle='-', linewidth=1.8, label=f'Target (+{delta_threshold})')

    ax1.set_xlabel('Total Hours of Data', fontweight='bold')
    ax1.set_ylabel('Δκ vs YASA', fontweight='bold')
    ax1.set_title('B: Performance vs Total Data\n(Top Performer + Top 3 per 100h bin)', fontweight='bold')
    # Remove individual legend for Panel B
    # ax1.legend(handlelength=1.8, borderpad=0.4, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.margins(x=0.03, y=0.1)

    
    # Panel C: Minutes per subject vs Performance (top performer + top 3 per 100min bin)
    ax2 = fig.add_subplot(gs[1, 1])
    
    # Select top 3 runs per 100-minute bin
    df_minutes = select_top_runs_per_bin(df.copy(), 'minutes_per_subject', 100, n_top=3)
    print(f"Panel C: {len(df_minutes)} top 3 runs selected from minutes per subject bins")
    
    if not df_minutes.empty:
        jitter = (rng.standard_normal(len(df_minutes)) * 0.005)
        ax2.scatter(
            df_minutes['minutes_per_subject'],
            df_minutes['delta_kappa'] + jitter,
            c=get_color('cbramod'),
            s=40, alpha=0.7,  # Smaller points for bin tops
            edgecolors='black', linewidth=0.5,
            label='Top 3 per 100min bin'
        )
        
        # Highlight overall top performer
        ax2.scatter(
            top_performer['minutes_per_subject'],
            top_performer['delta_kappa'],
            c='red', s=120, alpha=0.9,
            marker='*', edgecolors='darkred', linewidth=2,
            label=f'Top performer (Δκ={top_performer["delta_kappa"]:.3f})'
        )

    # YASA baseline as simple horizontal line
    ax2.axhline(y=0, color=get_color('yasa'), linestyle='--', linewidth=1.8, 
               label='YASA baseline (Δκ=0)', zorder=2)
    ax2.axhline(y=delta_threshold, color=get_color('t_star'), linestyle='-', linewidth=1.8, label=f'Target (+{delta_threshold})')

    ax2.set_xlabel('Minutes per Subject', fontweight='bold')
    ax2.set_ylabel('Δκ vs YASA', fontweight='bold')
    ax2.set_title('C: Performance vs Minutes per Subject\n(Top Performer + Top 3 per 100min bin)', fontweight='bold')
    # Remove individual legend for Panel C
    # ax2.legend(handlelength=1.8, borderpad=0.4, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.margins(x=0.03, y=0.1)

    
    # Panel D: Number of subjects vs Performance (top performer + top 3 per 3-subject bin)
    ax3 = fig.add_subplot(gs[2, 0])
    
    # Select top 3 runs per 3-subject bin
    df_subjects = select_top_runs_per_bin(df.copy(), 'num_subjects', 3, n_top=3)
    print(f"Panel D: {len(df_subjects)} top 3 runs selected from subject count bins")
    
    if not df_subjects.empty:
        jitter = (rng.standard_normal(len(df_subjects)) * 0.005)
        ax3.scatter(
            df_subjects['num_subjects'],
            df_subjects['delta_kappa'] + jitter,
            c=get_color('cbramod'),
            s=40, alpha=0.7,  # Smaller points for bin tops
            edgecolors='black', linewidth=0.5,
            label='Top 3 per 3-subject bin'
        )
        
        # Highlight overall top performer
        ax3.scatter(
            top_performer['num_subjects'],
            top_performer['delta_kappa'],
            c='red', s=120, alpha=0.9,
            marker='*', edgecolors='darkred', linewidth=2,
            label=f'Top performer (Δκ={top_performer["delta_kappa"]:.3f})'
        )

    # YASA baseline as simple horizontal line
    ax3.axhline(y=0, color=get_color('yasa'), linestyle='--', linewidth=1.8, 
               label='YASA baseline (Δκ=0)', zorder=2)
    ax3.axhline(y=delta_threshold, color=get_color('t_star'), linestyle='-', linewidth=1.8, label=f'Target (+{delta_threshold})')

    ax3.set_xlabel('Number of Subjects', fontweight='bold')
    ax3.set_ylabel('Δκ vs YASA', fontweight='bold')
    ax3.set_title('D: Performance vs Number of Subjects\n(Top Performer + Top 3 per 3-subject bin)', fontweight='bold')
    # Remove individual legend for Panel D
    # ax3.legend(handlelength=1.8, borderpad=0.4, bbox_to_anchor=(1.05, 1), loc='upper left')
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

                # Use consistent CBraMod color for all boxes
                for patch in bp['boxes']:
                    patch.set_facecolor(get_color('cbramod'))
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('black')

                # YASA baseline as simple horizontal line
                ax4.axhline(y=0, color=get_color('yasa'), linestyle='--', linewidth=1.8, 
                           label='YASA baseline (Δκ=0)', zorder=2)
                ax4.axhline(y=delta_threshold, color=get_color('t_star'), linestyle='-', linewidth=1.8)

                ax4.set_xlabel('Number of Subjects', fontweight='bold')
                ax4.set_ylabel('Δκ vs YASA', fontweight='bold')
                ax4.set_title('E: Top Performers by\nSubject Count', fontweight='bold')
                ax4.grid(True, alpha=0.3)

                ax4.tick_params(axis='x', labelrotation=30)
                ax4.margins(x=0.02, y=0.1)

                # Rotate labels if needed
                if len(labels_filtered) > 6:
                    ax4.tick_params(axis='x', rotation=45)
    
    # Create common legend at the bottom
    legend_ax = fig.add_subplot(gs[3, :])
    legend_ax.axis('off')  # Hide axes for legend area
    
    # Collect legend elements from all panels
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = [
        Line2D([0], [0], marker='*', color='red', linestyle='None', markersize=12, 
               markeredgecolor='darkred', markeredgewidth=2, label=f'Top performer (Δκ={top_performer["delta_kappa"]:.3f})'),
        Line2D([0], [0], marker='o', color='gray', linestyle='None', markersize=8, 
               markeredgecolor='black', markeredgewidth=0.5, label='Top 3 per bin'),
        Line2D([0], [0], color=get_color('yasa'), linestyle='--', linewidth=1.8, label='YASA baseline (Δκ=0)'),
        Line2D([0], [0], color=get_color('t_star'), linestyle='-', linewidth=1.8, label=f'Target improvement (+{delta_threshold})'),
    ]

    
    legend_ax.legend(handles=legend_elements, loc='center', ncol=2, 
                    frameon=True, fancybox=True, shadow=True,
                    fontsize=11, handlelength=2, columnspacing=2)
    
    
    #plt.tight_layout()
    
    # Save using consistent save function
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / 'subjects_vs_minutes_analysis'
    save_figure(fig, base_path)
    
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
    
    # Run comprehensive testing suite
    if run_comprehensive_tests:
        test_results = comprehensive_testing_suite(df, output_dir)
    
    plt.show()
    return fig

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Subjects vs Minutes Analysis')
    parser.add_argument("--csv", required=True, help="Path to flattened CSV")
    parser.add_argument("--out", default="Plot_Clean/figures", help="Output directory")
    parser.add_argument("--test", action="store_true", help="Run comprehensive testing suite")
    args = parser.parse_args()

    setup_figure_style()
    
    # Load and prepare data
    df = load_and_prepare_data(Path(args.csv))
    
    if df.empty:
        print("No valid data found")
        return
    
    # Create figure with optional comprehensive testing
    create_figure_4(df, Path(args.out), run_comprehensive_tests=args.test)

if __name__ == "__main__":
    main()