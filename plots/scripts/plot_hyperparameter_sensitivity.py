#!/usr/bin/env python3
"""
fig_hparam_sensitivity_focused.py

Focused hyperparameter sensitivity analysis for CBraMod with proper filtering,
grouping, and publication-ready visualization.

Usage:
    python Plot_Clean/plot_hyperparameter_sensitivity.py --csv ../data/all_runs_flat.csv --out ../figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Import consistent figure styling
import sys; sys.path.append("../style"); from figure_style import (
    setup_figure_style, get_color, save_figure, 
    add_yasa_baseline, add_significance_marker,
    bootstrap_ci_median, wilcoxon_test,
    format_n_caption, add_sample_size_annotation
)

class FocusedHPAnalyzer:
    """Focused hyperparameter sensitivity analysis for CBraMod."""
    
    def __init__(self):
        # Semantic hyperparameter mapping - ONLY optimization and adaptation parameters
        self.hp_mapping = {
            # Optimization hyperparameters
            'contract.training.lr': ('Learning Rate', 'Optimization'),
            'contract.training.head_lr': ('Head LR', 'Optimization'), 
            'contract.training.backbone_lr': ('Backbone LR', 'Optimization'),
            'contract.training.weight_decay': ('Weight Decay', 'Optimization'),
            'contract.training.batch_size': ('Batch Size', 'Optimization'),
            'contract.training.scheduler': ('LR Schedule', 'Optimization'),
            'contract.training.use_amp': ('Mixed Precision', 'Optimization'),
            'contract.training.optimizer': ('Optimizer', 'Optimization'),
            'contract.training.clip_value': ('Gradient Clip', 'Optimization'),
            'contract.training.focal_gamma': ('Focal Gamma', 'Optimization'),
            'contract.training.use_focal_loss': ('Focal Loss', 'Optimization'),
            'contract.training.use_class_weights': ('Class Weights', 'Optimization'),
            'contract.training.use_weighted_sampler': ('Weighted Sampler', 'Optimization'),
            
            # Adaptation hyperparameters (training strategy)
            'contract.model.frozen': ('Frozen Backbone', 'Adaptation'),
            'contract.training.unfreeze_epoch': ('Unfreeze Epoch', 'Adaptation'),
            'contract.training.label_smoothing': ('Label Smoothing', 'Adaptation'),
            'contract.training.two_phase_training': ('Two Phase Training', 'Adaptation'),
            'contract.training.phase1_epochs': ('Phase 1 Epochs', 'Adaptation'),
            'contract.training.dropout': ('Dropout', 'Adaptation'),
            'contract.training.head_type': ('Head Type', 'Adaptation'),
            'contract.dataset.data_ORP': ('ORP Data Fraction', 'Adaptation'),
            'contract.training.label_mapping_version': ('Label Mapping', 'Adaptation'),
            'contract.training.preprocess': ('Preprocessing', 'Adaptation'),
            
        }
        
        # Columns to exclude (architectural, data info, results, and non-hyperparameters)
        self.exclude_patterns = {
            # Results and metadata
            'cfg.subject_id', 'cfg.seed', 'cfg.data_splits_test', 'run_id', 'name', 'state',
            'duration_seconds', 'tags', 'label_scheme', 'contract.dataset.name', 
            'contract.dataset.dataset_names', 'contract.dataset.datasets', 'cfg.datasets',
            'contract.results.num_classes', 'contract.results.test_kappa', 
            'contract.results.test_f1', 'contract.results.test_accuracy',
            'contract.results.hours_of_data', 'sum.test_kappa', 'sum.test_accuracy',
            'sum.test_f1', 'sum.hours_of_data', 'sum._runtime', 'cfg.num_of_classes',
            'contract.training.epochs', 'cfg.epochs', 'cfg.training_epochs',
            'contract.dataset.data_fraction', 'contract.dataset.num_subjects_train',
            
            # Architecture parameters (exclude per user request)
            'contract.model.model_name', 'contract.model.model_size', 'contract.model.layers',
            'contract.model.heads', 'contract.model.embedding_dim', 'contract.model.sample_rate',
            'cfg.model_name', 'cfg.model_size', 'cfg.layers', 'cfg.heads', 'cfg.embedding_dim',
            
            # ICL parameters (not optimization/adaptation)
            'contract.icl.icl_mode', 'contract.icl.k_support', 'contract.icl.icl_layers',
            'contract.icl.icl_hidden', 'contract.icl.proto_temp', 'contract.icl.icl_eval_Ks',
            'contract.icl.icl_loss_weight', 'contract.icl.icl_contrastive_weight',
        }
        
        # Publication-quality colors for semantic groups (colorblind-safe palette)
        self.group_colors = {
            'Optimization': get_color('cbramod'),     # Blue (learning dynamics)
            'Adaptation': get_color('4_class'),       # Pink (model adaptation)
            'Other': get_color('subjects')            # Neutral gray
        }
    
    def load_and_clean_data(self, csv_path: str) -> pd.DataFrame:
        """Load and clean experimental data."""
        print(f"📊 Loading data: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"   Original: {len(df)} rows, {len(df.columns)} columns")
        
        # CRITICAL: Filter out high noise experiments to avoid bias
        if 'noise_level' in df.columns:
            noise_stats = df['noise_level'].value_counts().sort_index()
            print(f"🔊 Noise level distribution: {dict(noise_stats)}")
            
            # Keep only clean data (noise_level <= 0.01 or 1%) 
            df = df[df['noise_level'] <= 0.01].copy()
            print(f"✅ Filtered to clean data: {len(df)} rows remaining (noise ≤ 1%)")
            
            if len(df) == 0:
                raise ValueError("No clean data found after noise filtering.")
        
        # Remove rows with missing target
        target = 'contract.results.test_kappa'
        df_clean = df.dropna(subset=[target]).copy()
        print(f"   After target filter: {len(df_clean)} rows")
        
        return df_clean
    
    def identify_hyperparameters(self, df: pd.DataFrame) -> List[str]:
        """Identify true hyperparameter columns."""
        all_cols = set(df.columns)
        
        # Start with mapped hyperparameters that exist
        hp_cols = [col for col in self.hp_mapping.keys() if col in all_cols]
        
        # Add other potential HPs not in exclude list
        for col in all_cols:
            if col not in self.exclude_patterns and col not in hp_cols:
                # Include training/model config columns
                if any(pattern in col.lower() for pattern in ['training.', 'model.', 'icl.']):
                    hp_cols.append(col)
        
        print(f"🔍 Identified {len(hp_cols)} hyperparameter columns:")
        for col in sorted(hp_cols):
            label, group = self.hp_mapping.get(col, (col, 'Other'))
            print(f"   • {col} → {label} ({group})")
        
        return hp_cols
    
    def deduplicate_columns(self, df: pd.DataFrame, hp_cols: List[str]) -> List[str]:
        """Remove duplicate columns (e.g., cfg.epochs vs contract.training.epochs)."""
        dedup_cols = []
        seen_values = {}
        
        for col in hp_cols:
            if col not in df.columns:
                continue
                
            # Get non-null values as a tuple for comparison
            values = tuple(sorted(df[col].dropna().unique()))
            
            # Check if we've seen identical values
            duplicate_found = False
            for seen_col, seen_vals in seen_values.items():
                if values == seen_vals:
                    print(f"   Duplicate: {col} ≡ {seen_col} (keeping {seen_col})")
                    duplicate_found = True
                    break
            
            if not duplicate_found:
                seen_values[col] = values
                dedup_cols.append(col)
        
        print(f"🔄 Deduplicated: {len(hp_cols)} → {len(dedup_cols)} columns")
        return dedup_cols
    
    def preprocess_features(self, df: pd.DataFrame, hp_cols: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Preprocess hyperparameters for ML model."""
        X_features = []
        feature_names = []
        
        for col in hp_cols:
            if col not in df.columns:
                continue
                
            series = df[col].copy()
            
            # Skip if too many missing values
            if series.isnull().sum() > len(series) * 0.5:
                print(f"   Skipping {col}: >50% missing")
                continue
            
            # Fill missing values
            if series.dtype in ['object', 'str']:
                series = series.fillna('unknown')
            else:
                series = series.fillna(series.median())
            
            # Handle categorical variables
            if series.dtype in ['object', 'str'] or series.nunique() <= 10:
                # One-hot encode if reasonable number of categories
                unique_vals = series.unique()
                if len(unique_vals) > 15:
                    print(f"   Skipping {col}: too many categories ({len(unique_vals)})")
                    continue
                
                # Create one-hot features
                for val in unique_vals:
                    feature_names.append(f"{col}___{val}")
                    X_features.append((series == val).astype(int).values)
            else:
                # Numeric feature
                feature_names.append(col)
                X_features.append(series.values)
        
        X = np.column_stack(X_features) if X_features else np.empty((len(df), 0))
        print(f"🔧 Preprocessed to {X.shape[1]} features from {len(hp_cols)} hyperparameters")
        
        return X, feature_names
    
    def group_feature_importance(self, importances: Dict[str, float]) -> Dict[str, Tuple[float, str]]:
        """Group one-hot encoded features back to parent hyperparameters."""
        grouped = {}
        
        for feature, importance in importances.items():
            # Extract base hyperparameter name
            if '___' in feature:  # One-hot encoded
                base_hp = feature.split('___')[0]
            else:
                base_hp = feature
            
            # Get semantic label and group
            label, group = self.hp_mapping.get(base_hp, (base_hp.split('.')[-1], 'Other'))
            
            # Aggregate importance for grouped features
            if label not in grouped:
                grouped[label] = (0.0, group)
            
            current_imp, current_group = grouped[label]
            grouped[label] = (current_imp + importance, current_group)
        
        return grouped
    
    def calculate_importance_with_ci(self, X: np.ndarray, y: np.ndarray, 
                                   feature_names: List[str], n_bootstrap: int = 50) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
        """Calculate permutation importance with bootstrap confidence intervals."""
        print(f"🤖 Training model and calculating importance (bootstrap n={n_bootstrap})...")
        
        # Fit primary model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        # Calculate cross-validation score (for validation only)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        print(f"   Model validation: R² = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Primary permutation importance
        perm_result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        
        # Map to feature importance dict
        feature_importance = {}
        for i, fname in enumerate(feature_names):
            feature_importance[fname] = perm_result.importances_mean[i]
        
        # Bootstrap confidence intervals
        bootstrap_results = {fname: [] for fname in feature_names}
        
        print("   Computing bootstrap confidence intervals...")
        for b in range(n_bootstrap):
            if b % 10 == 0:
                print(f"     Bootstrap {b+1}/{n_bootstrap}")
            
            # Bootstrap sample
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot, y_boot = X[indices], y[indices]
            
            try:
                # Fit model on bootstrap sample
                model_boot = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
                model_boot.fit(X_boot, y_boot)
                
                # Calculate importance
                perm_boot = permutation_importance(model_boot, X_boot, y_boot, n_repeats=5, random_state=42, n_jobs=1)
                
                # Store results
                for i, fname in enumerate(feature_names):
                    bootstrap_results[fname].append(perm_boot.importances_mean[i])
                    
            except Exception:
                # If bootstrap fails, append zero
                for fname in feature_names:
                    bootstrap_results[fname].append(0.0)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for fname in feature_names:
            boot_scores = np.array(bootstrap_results[fname])
            if len(boot_scores) > 0:
                lower = np.percentile(boot_scores, 2.5)
                upper = np.percentile(boot_scores, 97.5)
                confidence_intervals[fname] = (lower, upper)
            else:
                confidence_intervals[fname] = (0.0, 0.0)
        
        return feature_importance, confidence_intervals
    
    def create_sensitivity_plot(self, grouped_importance: Dict[str, Tuple[float, str]], 
                              grouped_ci: Dict[str, Tuple[float, float]], 
                              n_runs: int, output_path: str):
        """Create publication-ready horizontal bar plot with enhanced styling."""
        print(f"🎨 Creating publication-quality sensitivity plot: {output_path}")
        
        # Set up consistent figure style
        setup_figure_style()
        
        # Filter and sort by importance
        items = [(label, imp, group, grouped_ci.get(label, (0, 0))) 
                for label, (imp, group) in grouped_importance.items()]
        
        # Filter with more sophisticated criteria for publication quality
        filtered_items = []
        for label, imp, group, (ci_low, ci_high) in items:
            # Keep if: (1) practically significant OR (2) statistically significant
            if imp >= 0.005 or ci_low > 0:  # Meaningful effect threshold raised to 0.005
                filtered_items.append((label, imp, group, (ci_low, ci_high)))
        
        # Sort by importance and take top 12
        filtered_items.sort(key=lambda x: x[1], reverse=True)
        top_items = filtered_items[:12]
        
        if not top_items:
            print("❌ No significant hyperparameters found!")
            return
        
        # Extract data for plotting
        labels = [item[0] for item in top_items]
        importances = [item[1] for item in top_items]
        groups = [item[2] for item in top_items]
        ci_lows = [item[3][0] for item in top_items]
        ci_highs = [item[3][1] for item in top_items]
        
        # Make all histograms blue
        colors = [get_color('cbramod') for group in groups]  # All blue
        
        # Calculate optimal figure dimensions
        n_bars = len(labels)
        fig_height = max(7, n_bars * 0.55 + 2.5)  # More space per bar + room for title/legend
        fig_width = 12  # Wider for better proportions
        
        # Create figure with publication quality
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.patch.set_facecolor('white')
        
        # Horizontal bars with enhanced styling - all blue
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, importances, height=0.7, color=colors, alpha=0.85, 
                      edgecolor='white', linewidth=1.5)
        
        # Enhanced error bars with asymmetric confidence intervals
        xerr_lower = [max(0, imp - ci_low) for imp, ci_low in zip(importances, ci_lows)]
        xerr_upper = [max(0, ci_high - imp) for imp, ci_high in zip(importances, ci_highs)]
        xerr = [xerr_lower, xerr_upper]
        ax.errorbar(importances, y_pos, xerr=xerr, fmt='none', 
                   color='#2C3E50', capsize=3, capthick=1.0, 
                   elinewidth=1.0, alpha=0.7, zorder=10)
        
        # Enhanced axis formatting - use consistent styling
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=12, fontweight='normal')
        ax.set_xlabel('Feature Importance Score (Permutation-based)', fontsize=16, fontweight='bold')
        
        # Enhanced title with model insights - use consistent styling
        title_text = 'CBraMod Hyperparameter Impact Analysis'
        ax.set_title(title_text, fontsize=18, fontweight='bold', pad=25)
        
        # Subtitle with key insights - use consistent styling
        subtitle = f'Impact of Randomizing Hyperparameters on Model Performance | N = {n_runs:,} experiments'
        ax.text(0.5, 0.98, subtitle, transform=ax.transAxes, ha='center', 
               fontsize=12, style='italic')
        
        # Add significance markers at the very end of bars (stars only, no dots or legend)
        for i, (imp, ci_low, ci_high) in enumerate(zip(importances, ci_lows, ci_highs)):
            if ci_low > 0:  # Statistically significant (CI doesn't include 0)
                marker = '***' if imp > 0.02 else '**' if imp > 0.01 else '*'
                # Add significance level text at the end of the bar
                ax.text(imp + max(imp * 0.02, 0.002), i, marker, fontsize=12, fontweight='bold',
                       color='red', ha='left', va='center')
        
        # Use consistent grid styling
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)  # Grid behind bars
        
        # Add practical significance threshold line - use consistent colors
        ax.axvline(x=0.01, color=get_color('t_star'), linestyle=':', alpha=0.6, linewidth=2)
        ax.text(0.01, len(labels) * 0.95, 'Practical\nSignificance\nThreshold', rotation=90,
               ha='right', va='top', fontsize=9, color=get_color('t_star'), alpha=0.8)
        
        # Tighten layout without bottom legend space
        plt.subplots_adjust(left=0.25, right=0.95, top=0.92, bottom=0.10)
        
        # Save using consistent save function
        base_path = Path(output_path).with_suffix('')
        saved_files = save_figure(fig, base_path, formats=['png', 'svg', 'pdf'])
        
        # Print N information for caption
        print(f"\n📋 Caption info: {format_n_caption(n_runs, n_runs, 'runs')}")
        
        plt.close()
    
    def run_analysis(self, csv_path: str, output_dir: str):
        """Run complete focused hyperparameter sensitivity analysis."""
        print("🚀 CBraMod Focused Hyperparameter Sensitivity Analysis")
        print("=" * 60)
        
        # Set up consistent styling and create output directory
        setup_figure_style()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load and clean data
        df = self.load_and_clean_data(csv_path)
        
        # Identify true hyperparameters
        hp_cols = self.identify_hyperparameters(df)
        
        # Deduplicate
        hp_cols = self.deduplicate_columns(df, hp_cols)
        
        if len(hp_cols) == 0:
            print("❌ No hyperparameters found!")
            return
        
        # Preprocess features
        X, feature_names = self.preprocess_features(df, hp_cols)
        y = df['contract.results.test_kappa'].values
        
        if X.shape[1] == 0:
            print("❌ No features after preprocessing!")
            return
        
        # Calculate importance with CI
        feature_importance, confidence_intervals = self.calculate_importance_with_ci(X, y, feature_names, n_bootstrap=50)
        
        # Group one-hot features
        grouped_importance = self.group_feature_importance(feature_importance)
        
        # Group confidence intervals
        grouped_ci = {}
        for label, _ in grouped_importance.items():
            # Find all features that map to this label
            matching_features = []
            for fname in feature_names:
                base_hp = fname.split('___')[0] if '___' in fname else fname
                hp_label, _ = self.hp_mapping.get(base_hp, (base_hp.split('.')[-1], 'Other'))
                if hp_label == label:
                    matching_features.append(fname)
            
            # Aggregate CI (take min/max across features)
            if matching_features:
                all_lows = [confidence_intervals[f][0] for f in matching_features]
                all_highs = [confidence_intervals[f][1] for f in matching_features]
                grouped_ci[label] = (min(all_lows), max(all_highs))
            else:
                grouped_ci[label] = (0.0, 0.0)
        
        # Create plot
        output_path = os.path.join(output_dir, 'cbramod_hp_sensitivity.png')
        
        self.create_sensitivity_plot(grouped_importance, grouped_ci, len(df), output_path)
        
        # Print summary with insights
        print("\n📋 Top 10 Most Sensitive Hyperparameters:")
        sorted_hps = sorted(grouped_importance.items(), key=lambda x: x[1][0], reverse=True)[:10]
        for i, (label, (importance, group)) in enumerate(sorted_hps):
            ci_low, ci_high = grouped_ci[label]
            significance = '***' if importance > 0.02 else '**' if importance > 0.01 else '*' if ci_low > 0 else ''
            print(f"  {i+1:2d}. {label:25} {importance:7.4f} [{ci_low:6.4f}, {ci_high:6.4f}] ({group}) {significance}")
        
        # Model insights
        print("\n🧠 Key Model Insights:")
        critical_params = [hp for hp in sorted_hps if hp[1][0] > 0.02]
        if critical_params:
            print(f"  • {len(critical_params)} hyperparameters show CRITICAL impact (importance > 0.02)")
        moderate_params = [hp for hp in sorted_hps if 0.01 < hp[1][0] <= 0.02]
        if moderate_params:
            print(f"  • {len(moderate_params)} hyperparameters show MODERATE impact (0.01 < importance ≤ 0.02)")
        
        # Category insights
        category_impact = {}
        for label, (importance, group) in sorted_hps:
            if group not in category_impact:
                category_impact[group] = []
            category_impact[group].append(importance)
        
        for category, impacts in category_impact.items():
            avg_impact = np.mean(impacts)
            print(f"  • {category} parameters: avg importance = {avg_impact:.4f} ({len(impacts)} params)")
        
        print(f"\n✅ Analysis complete! Plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Focused CBraMod Hyperparameter Sensitivity Analysis')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV with experimental results')
    parser.add_argument('--out', type=str, required=True, help='Output directory for plots')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"❌ CSV file not found: {args.csv}")
        return
    
    analyzer = FocusedHPAnalyzer()
    analyzer.run_analysis(args.csv, args.out)

if __name__ == '__main__':
    main()