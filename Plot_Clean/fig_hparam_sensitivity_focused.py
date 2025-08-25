#!/usr/bin/env python3
"""
fig_hparam_sensitivity_focused.py

Focused hyperparameter sensitivity analysis for CBraMod with proper filtering,
grouping, and publication-ready visualization.

Usage:
    python fig_hparam_sensitivity_focused.py --csv Plot_Clean/data/all_runs_flat.csv --out Plot_Clean/figures/
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

class FocusedHPAnalyzer:
    """Focused hyperparameter sensitivity analysis for CBraMod."""
    
    def __init__(self):
        # Semantic hyperparameter mapping
        self.hp_mapping = {
            # Optimization
            'contract.training.lr': ('Learning Rate', 'Optimization'),
            'contract.training.head_lr': ('Head LR', 'Optimization'), 
            'contract.training.backbone_lr': ('Backbone LR', 'Optimization'),
            'contract.training.weight_decay': ('Weight Decay', 'Optimization'),
            'contract.training.batch_size': ('Batch Size', 'Optimization'),
            'contract.training.epochs': ('Epochs', 'Optimization'),
            'cfg.epochs': ('Epochs', 'Optimization'),
            'cfg.training_epochs': ('Epochs', 'Optimization'),
            'contract.training.optimizer': ('Optimizer', 'Optimization'),
            'contract.training.scheduler': ('LR Schedule', 'Optimization'),
            'contract.training.use_amp': ('Mixed Precision', 'Optimization'),
            
            # Adaptation
            'contract.model.frozen': ('Frozen Backbone', 'Adaptation'),
            'contract.training.unfreeze_epoch': ('Unfreeze Epoch', 'Adaptation'),
            'contract.training.label_smoothing': ('Label Smoothing', 'Adaptation'),
            
            # ICL (In-Context Learning)
            'contract.icl.icl_mode': ('ICL Mode', 'ICL'),
            'contract.icl.k_support': ('K Support', 'ICL'),
            'contract.icl.icl_layers': ('ICL Layers', 'ICL'),
            'contract.icl.icl_hidden': ('ICL Hidden', 'ICL'),
            
            # Dataset/Augmentation  
            'contract.dataset.data_fraction': ('Data Fraction', 'Augmentation'),
            'contract.dataset.num_subjects_train': ('Training Subjects', 'Augmentation'),
        }
        
        # Columns to exclude (non-hyperparameters)
        self.exclude_patterns = {
            'cfg.subject_id', 'cfg.seed', 'cfg.data_splits_test', 'run_id', 'name', 'state',
            'duration_seconds', 'tags', 'label_scheme', 'contract.dataset.name', 
            'contract.dataset.dataset_names', 'contract.dataset.datasets', 'cfg.datasets',
            'contract.results.num_classes', 'contract.results.test_kappa', 
            'contract.results.test_f1', 'contract.results.test_accuracy',
            'contract.results.hours_of_data', 'sum.test_kappa', 'sum.test_accuracy',
            'sum.test_f1', 'sum.hours_of_data', 'sum._runtime', 'cfg.num_of_classes'
        }
        
        # Colors for semantic groups
        self.group_colors = {
            'Optimization': '#2E8B57',    # Sea Green
            'Adaptation': '#FF6347',      # Tomato  
            'ICL': '#4682B4',            # Steel Blue
            'Augmentation': '#9370DB',    # Medium Purple
            'Other': '#708090'            # Slate Gray
        }
    
    def load_and_clean_data(self, csv_path: str) -> pd.DataFrame:
        """Load and clean experimental data."""
        print(f"📊 Loading data: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"   Original: {len(df)} rows, {len(df.columns)} columns")
        
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
        
        # Calculate cross-validation score
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        print(f"   Model R² = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
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
                              model_r2: float, n_runs: int, output_path: str):
        """Create publication-ready horizontal bar plot."""
        print(f"🎨 Creating sensitivity plot: {output_path}")
        
        # Filter and sort by importance
        items = [(label, imp, group, grouped_ci.get(label, (0, 0))) 
                for label, (imp, group) in grouped_importance.items()]
        
        # Filter out very small effects (CI overlaps 0 and mean < 0.003)
        filtered_items = []
        for label, imp, group, (ci_low, ci_high) in items:
            if imp >= 0.003 or ci_low > 0:  # Keep if meaningful effect
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
        
        # Colors
        colors = [self.group_colors[group] for group in groups]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.5)))
        
        # Horizontal bars
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, importances, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Error bars
        xerr = [[imp - ci_low for imp, ci_low in zip(importances, ci_lows)],
                [ci_high - imp for imp, ci_high in zip(importances, ci_highs)]]
        ax.errorbar(importances, y_pos, xerr=xerr, fmt='none', color='black', capsize=3, alpha=0.7)
        
        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{i+1}. {label}" for i, label in enumerate(labels)])
        ax.set_xlabel('Mean Δκ when feature shuffled (permutation importance)', fontsize=12)
        
        # Title with model info
        title = f'CBraMod Hyperparameter Sensitivity Analysis\n'
        subtitle = f'Metric: Test Cohen\'s κ | N={n_runs} runs | Model R²={model_r2:.3f}'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha='center', fontsize=10, style='italic')
        
        # Grid and styling
        ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        # Legend (only if multiple groups)
        unique_groups = list(set(groups))
        if len(unique_groups) > 1:
            legend_elements = [plt.Rectangle((0,0),1,1, facecolor=self.group_colors[group], alpha=0.8, label=group) 
                             for group in unique_groups]
            ax.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        
        # Save both formats
        plt.savefig(output_path.replace('.png', '.svg'), format='svg', bbox_inches='tight', facecolor='white')
        plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"   Saved: {output_path} (PNG) and SVG")
        
        plt.close()
    
    def run_analysis(self, csv_path: str, output_dir: str):
        """Run complete focused hyperparameter sensitivity analysis."""
        print("🚀 CBraMod Focused Hyperparameter Sensitivity Analysis")
        print("=" * 60)
        
        # Create output directory
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
        
        # Calculate model R² for title
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
        
        self.create_sensitivity_plot(grouped_importance, grouped_ci, cv_r2, len(df), output_path)
        
        # Print summary
        print("\n📋 Top 10 Most Sensitive Hyperparameters:")
        sorted_hps = sorted(grouped_importance.items(), key=lambda x: x[1][0], reverse=True)[:10]
        for i, (label, (importance, group)) in enumerate(sorted_hps):
            ci_low, ci_high = grouped_ci[label]
            print(f"  {i+1:2d}. {label:25} {importance:7.4f} [{ci_low:6.4f}, {ci_high:6.4f}] ({group})")
        
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