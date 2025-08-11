#!/usr/bin/env python3
"""
Hyperparameter Impact Analysis for CBraMod
==========================================

Research Question: Which hyperparameters have the greatest impact on model performance?

This script analyzes the impact of different hyperparameters on CBraMod performance
using statistical methods and visualization techniques.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import wandb
import functools
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings 
warnings.filterwarnings('ignore')

# Set publication-quality plotting parameters
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 2,
    'grid.alpha': 0.3
})

class HyperparameterImpactAnalyzer:
    """Analyzes hyperparameter impact on CBraMod performance"""
    
    def __init__(self, entity="thibaut_hasle-epfl", project="CBraMod-earEEG-tuning", 
                 output_dir="Plot/figures"):
        self.entity = entity
        self.project = project
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Quality thresholds
        self.min_test_kappa = 0.4
        self.max_val_test_diff_pct = 5.0
        
        # Fetch and process data
        print("🔄 Fetching hyperparameter impact analysis data...")
        self.df = self._fetch_and_process_data()
        
    @functools.lru_cache(maxsize=1)
    def _fetch_wandb_runs(self):
        """Fetch WandB runs for hyperparameter analysis"""
        print("🔄 Fetching WandB runs...")
        api = wandb.Api()
        runs = api.runs(f"{self.entity}/{self.project}")
        
        filtered_runs = []
        for run in runs:
            config = run.config
            summary = run.summary
            
            # Filter for high-quality experiments with hyperparameter variation
            if (run.state == 'finished' and
                config.get('num_of_classes') == 5 and
                summary.get('test_kappa', 0) > self.min_test_kappa and
                bool(summary.keys())):
                
                filtered_runs.append(run)
        
        print(f"✅ Found {len(filtered_runs)} relevant runs for hyperparameter analysis")
        return filtered_runs
    
    def _fetch_and_process_data(self):
        """Process WandB data for hyperparameter analysis"""
        runs = self._fetch_wandb_runs()
        
        data = []
        for run in runs:
            config = run.config
            summary = run.summary
            
            # Performance metrics
            test_kappa = summary.get('test_kappa', 0)
            val_kappa = summary.get('val_kappa', 0)
            
            # Quality filter
            val_test_diff = 0
            is_high_quality = True
            if val_kappa > 0 and test_kappa > 0:
                val_test_diff = abs(val_kappa - test_kappa) / test_kappa * 100
                is_high_quality = val_test_diff <= self.max_val_test_diff_pct
            
            # Extract hyperparameters
            learning_rate = config.get('learning_rate', 1e-4)
            batch_size = config.get('batch_size', 64)
            optimizer = config.get('optimizer', 'AdamW')
            epochs = config.get('epochs', 100)
            
            # Training strategy parameters
            two_phase = config.get('two_phase_training', False)
            backbone_lr = config.get('backbone_lr', learning_rate)
            head_lr = config.get('head_lr', learning_rate)
            unfreeze_epoch = config.get('backbone_unfreeze_epoch', None)
            
            # Regularization parameters
            dropout = config.get('dropout', 0.1)
            weight_decay = config.get('weight_decay', 0.01)
            gradient_clip = config.get('gradient_clip_val', 1.0)
            
            # Architecture parameters
            model_dim = config.get('model_dim', 200)
            num_layers = config.get('num_layers', 12)
            num_heads = config.get('num_heads', 8)
            
            # Data parameters
            datasets_str = config.get('datasets', '')
            num_datasets = len([d.strip() for d in str(datasets_str).split(',') if d.strip()])
            
            run_data = {
                # Run identification
                'run_id': run.id,
                'run_name': run.name,
                
                # Performance
                'test_kappa': test_kappa,
                'val_kappa': val_kappa,
                'test_accuracy': summary.get('test_accuracy', 0),
                'val_test_diff_pct': val_test_diff,
                'is_high_quality': is_high_quality,
                
                # Core hyperparameters
                'learning_rate': learning_rate,
                'log_lr': np.log10(learning_rate),
                'batch_size': batch_size,
                'optimizer': optimizer,
                'epochs': epochs,
                
                # Training strategy
                'two_phase_training': two_phase,
                'backbone_lr': backbone_lr,
                'head_lr': head_lr,
                'lr_ratio': head_lr / backbone_lr if backbone_lr > 0 else 1.0,
                'unfreeze_epoch': unfreeze_epoch if unfreeze_epoch else epochs,
                'unfreeze_ratio': (unfreeze_epoch / epochs) if unfreeze_epoch else 1.0,
                
                # Regularization
                'dropout': dropout,
                'weight_decay': weight_decay,
                'gradient_clip': gradient_clip,
                
                # Architecture
                'model_dim': model_dim,
                'num_layers': num_layers,
                'num_heads': num_heads,
                'params_per_layer': model_dim * num_heads,
                
                # Data
                'num_datasets': num_datasets,
                
                # Training dynamics
                'convergence_speed': summary.get('best_epoch', epochs) / epochs,
                'final_train_loss': summary.get('train_loss', 0),
                'final_val_loss': summary.get('val_loss', 0),
            }
            data.append(run_data)
        
        df = pd.DataFrame(data)
        
        # Quality reporting
        print(f"📊 Processed {len(df)} hyperparameter experiments")
        high_quality = df[df['is_high_quality']]
        print(f"   - High quality runs: {len(high_quality)}/{len(df)}")
        if len(high_quality) > 0:
            print(f"   - Learning rate range: {high_quality['learning_rate'].min():.2e} - {high_quality['learning_rate'].max():.2e}")
            print(f"   - Batch size range: {high_quality['batch_size'].min():.0f} - {high_quality['batch_size'].max():.0f}")
        
        return df
    
    def create_hyperparameter_impact_analysis(self):
        """Create comprehensive 4-subplot hyperparameter impact analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Hyperparameter Impact Analysis for CBraMod\n' + 
                     'Research Question: Which Hyperparameters Most Impact Performance?', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # Subplot 1: Feature importance analysis
        self._plot_feature_importance(ax1)
        
        # Subplot 2: Learning rate impact
        self._plot_learning_rate_impact(ax2)
        
        # Subplot 3: Training strategy comparison
        self._plot_training_strategy_impact(ax3)
        
        # Subplot 4: Optimizer and regularization impact
        self._plot_optimizer_regularization_impact(ax4)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        output_path = os.path.join(self.output_dir, 'hyperparameter_impact_comprehensive_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved hyperparameter impact analysis: {output_path}")
        
        return fig
    
    def _plot_feature_importance(self, ax):
        """Subplot 1: Feature importance using Random Forest"""
        high_quality = self.df[self.df['is_high_quality']].copy()
        
        if len(high_quality) < 10:
            ax.text(0.5, 0.5, 'Insufficient data for feature importance', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Select features for importance analysis
        feature_cols = [
            'log_lr', 'batch_size', 'epochs', 'dropout', 'weight_decay',
            'gradient_clip', 'model_dim', 'num_layers', 'num_heads',
            'num_datasets', 'convergence_speed', 'lr_ratio'
        ]
        
        # Filter out missing features and encode categorical variables
        available_features = [col for col in feature_cols if col in high_quality.columns]
        
        if len(available_features) < 3:
            ax.text(0.5, 0.5, 'Insufficient features for analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Prepare data
        X = high_quality[available_features].dropna()
        y = high_quality.loc[X.index, 'test_kappa']
        
        if len(X) < 10:
            ax.text(0.5, 0.5, 'Insufficient clean data for RF analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Fit Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=True)
        
        # Plot horizontal bar chart
        bars = ax.barh(range(len(importance_df)), importance_df['importance'], 
                       alpha=0.7, color='steelblue')
        
        # Color the top 3 features differently
        for i, bar in enumerate(bars):
            if i >= len(bars) - 3:  # Top 3 features
                bar.set_color('darkred')
        
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_title('A) Hyperparameter Importance (Random Forest)', fontweight='bold')
        ax.set_xlabel('Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add R² score
        y_pred = rf.predict(X)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        ax.text(0.95, 0.05, f'R² = {r2:.3f}', transform=ax.transAxes, ha='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def _plot_learning_rate_impact(self, ax):
        """Subplot 2: Learning rate vs performance analysis"""
        high_quality = self.df[self.df['is_high_quality']].copy()
        
        if len(high_quality) < 5:
            ax.text(0.5, 0.5, 'Insufficient data for LR analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create learning rate bins
        high_quality['lr_bins'] = pd.qcut(high_quality['learning_rate'], 
                                         q=min(4, len(high_quality.groupby('learning_rate'))), 
                                         labels=['Very Low', 'Low', 'Medium', 'High'][:min(4, len(high_quality.groupby('learning_rate')))],
                                         duplicates='drop')
        
        # Get best performance per bin
        bin_stats = []
        for bin_label in high_quality['lr_bins'].cat.categories:
            bin_data = high_quality[high_quality['lr_bins'] == bin_label]
            if len(bin_data) > 0:
                bin_stats.append({
                    'bin': bin_label,
                    'lr_mean': bin_data['learning_rate'].mean(),
                    'best_kappa': bin_data['test_kappa'].max(),
                    'mean_kappa': bin_data['test_kappa'].mean(),
                    'count': len(bin_data)
                })
        
        if not bin_stats:
            ax.text(0.5, 0.5, 'No learning rate bins available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        bin_df = pd.DataFrame(bin_stats)
        
        # Plot learning rate impact
        x_pos = range(len(bin_df))
        bars = ax.bar(x_pos, bin_df['best_kappa'], alpha=0.7, color='green', 
                      label='Best Performance')
        ax.scatter(x_pos, bin_df['mean_kappa'], color='darkgreen', s=60, 
                  label='Mean Performance', zorder=5)
        
        # Add LR values as annotations
        for i, row in bin_df.iterrows():
            ax.text(i, row['best_kappa'] + 0.005, f"{row['lr_mean']:.1e}\nn={row['count']}", 
                   ha='center', fontsize=8, fontweight='bold')
        
        ax.set_title('B) Learning Rate Impact on Performance', fontweight='bold')
        ax.set_xlabel('Learning Rate Range')
        ax.set_ylabel('Test Kappa')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_df['bin'])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_training_strategy_impact(self, ax):
        """Subplot 3: Training strategy impact (two-phase vs single-phase)"""
        high_quality = self.df[self.df['is_high_quality']].copy()
        
        # Compare training strategies
        strategies = []
        for two_phase in [False, True]:
            strategy_data = high_quality[high_quality['two_phase_training'] == two_phase]
            if len(strategy_data) > 0:
                # Further break down by LR strategy if two-phase
                if two_phase and 'lr_ratio' in strategy_data.columns:
                    # High ratio = head LR >> backbone LR (recommended strategy)
                    high_ratio = strategy_data[strategy_data['lr_ratio'] > 5]
                    low_ratio = strategy_data[strategy_data['lr_ratio'] <= 5]
                    
                    for ratio_data, ratio_label in [(high_ratio, 'Two-Phase\n(High LR Ratio)'), 
                                                   (low_ratio, 'Two-Phase\n(Low LR Ratio)')]:
                        if len(ratio_data) > 0:
                            strategies.append({
                                'strategy': ratio_label,
                                'best_kappa': ratio_data['test_kappa'].max(),
                                'mean_kappa': ratio_data['test_kappa'].mean(),
                                'std_kappa': ratio_data['test_kappa'].std(),
                                'count': len(ratio_data),
                                'convergence': ratio_data['convergence_speed'].mean()
                            })
                else:
                    strategies.append({
                        'strategy': 'Two-Phase' if two_phase else 'Single-Phase',
                        'best_kappa': strategy_data['test_kappa'].max(),
                        'mean_kappa': strategy_data['test_kappa'].mean(),
                        'std_kappa': strategy_data['test_kappa'].std(),
                        'count': len(strategy_data),
                        'convergence': strategy_data['convergence_speed'].mean()
                    })
        
        if not strategies:
            ax.text(0.5, 0.5, 'No strategy data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        strategy_df = pd.DataFrame(strategies)
        
        # Plot strategy comparison
        x_pos = range(len(strategy_df))
        colors = ['orange', 'steelblue', 'darkblue'][:len(strategy_df)]
        
        bars = ax.bar(x_pos, strategy_df['best_kappa'], alpha=0.7, color=colors, 
                      label='Best Performance')
        ax.errorbar(x_pos, strategy_df['mean_kappa'], yerr=strategy_df['std_kappa'], 
                    fmt='o', color='darkred', label='Mean ± Std', capsize=5)
        
        # Add convergence speed as secondary info
        for i, row in strategy_df.iterrows():
            ax.text(i, row['best_kappa'] + 0.01, 
                   f"n={row['count']}\nConv:{row['convergence']:.2f}", 
                   ha='center', fontsize=8, fontweight='bold')
        
        ax.set_title('C) Training Strategy Impact', fontweight='bold')
        ax.set_xlabel('Training Strategy')
        ax.set_ylabel('Test Kappa')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(strategy_df['strategy'], rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_optimizer_regularization_impact(self, ax):
        """Subplot 4: Optimizer and regularization impact"""
        high_quality = self.df[self.df['is_high_quality']].copy()
        
        # Focus on optimizer comparison if available
        optimizer_stats = []
        optimizers = high_quality['optimizer'].unique()
        
        for optimizer in optimizers:
            opt_data = high_quality[high_quality['optimizer'] == optimizer]
            if len(opt_data) > 0:
                optimizer_stats.append({
                    'optimizer': optimizer,
                    'best_kappa': opt_data['test_kappa'].max(),
                    'mean_kappa': opt_data['test_kappa'].mean(),
                    'count': len(opt_data)
                })
        
        if len(optimizer_stats) > 1:
            # Plot optimizer comparison
            opt_df = pd.DataFrame(optimizer_stats)
            x_pos = range(len(opt_df))
            
            bars = ax.bar(x_pos, opt_df['best_kappa'], alpha=0.7, 
                         color=['blue', 'red', 'green'][:len(opt_df)], 
                         label='Best Performance')
            ax.scatter(x_pos, opt_df['mean_kappa'], color='darkred', s=60, 
                      label='Mean Performance', zorder=5)
            
            # Add sample counts
            for i, row in opt_df.iterrows():
                ax.text(i, row['best_kappa'] + 0.01, f"n={row['count']}", 
                       ha='center', fontsize=9, fontweight='bold')
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(opt_df['optimizer'])
            ax.set_title('D) Optimizer Impact on Performance', fontweight='bold')
            ax.set_xlabel('Optimizer')
            
        else:
            # Plot regularization impact instead
            if 'dropout' in high_quality.columns and high_quality['dropout'].nunique() > 1:
                # Dropout impact analysis
                high_quality['dropout_bins'] = pd.cut(high_quality['dropout'], 
                                                     bins=3, 
                                                     labels=['Low\n(<0.1)', 'Medium\n(0.1-0.2)', 'High\n(>0.2)'])
                
                dropout_stats = []
                for bin_label in high_quality['dropout_bins'].cat.categories:
                    bin_data = high_quality[high_quality['dropout_bins'] == bin_label]
                    if len(bin_data) > 0:
                        dropout_stats.append({
                            'bin': bin_label,
                            'best_kappa': bin_data['test_kappa'].max(),
                            'mean_kappa': bin_data['test_kappa'].mean(),
                            'count': len(bin_data)
                        })
                
                if dropout_stats:
                    dropout_df = pd.DataFrame(dropout_stats)
                    x_pos = range(len(dropout_df))
                    
                    bars = ax.bar(x_pos, dropout_df['best_kappa'], alpha=0.7, 
                                 color='purple', label='Best Performance')
                    ax.scatter(x_pos, dropout_df['mean_kappa'], color='darkred', s=60, 
                              label='Mean Performance', zorder=5)
                    
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(dropout_df['bin'])
                    ax.set_title('D) Dropout Regularization Impact', fontweight='bold')
                    ax.set_xlabel('Dropout Level')
            else:
                ax.text(0.5, 0.5, 'Insufficient regularization data', 
                       ha='center', va='center', transform=ax.transAxes)
                return
        
        ax.set_ylabel('Test Kappa')
        ax.legend()
        ax.grid(True, alpha=0.3)

def main():
    """Main analysis pipeline"""
    print("🚀 Starting Hyperparameter Impact Analysis...")
    
    # Initialize analyzer
    analyzer = HyperparameterImpactAnalyzer()
    
    # Create comprehensive analysis
    analyzer.create_hyperparameter_impact_analysis()
    
    print("✅ Hyperparameter impact analysis complete!")

if __name__ == "__main__":
    main()