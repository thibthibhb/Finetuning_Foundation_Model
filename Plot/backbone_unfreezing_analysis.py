#!/usr/bin/env python3
"""
Backbone Unfreezing Analysis for CBraMod
========================================

Research Question: How does the epoch at which the backbone is unfrozen impact 
fine-tuning performance, convergence speed, and overfitting in single-channel EEG scenarios?

This script analyzes the optimal timing for unfreezing the backbone transformer
in two-phase training strategies.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import wandb
import functools
from scipy import stats
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

class BackboneUnfreezingAnalyzer:
    """Analyzes backbone unfreezing strategies and their impact on performance"""
    
    def __init__(self, entity="thibaut_hasle-epfl", project="CBraMod-earEEG-tuning", 
                 output_dir="Plot/figures"):
        self.entity = entity
        self.project = project
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Quality thresholds for filtering
        self.min_test_kappa = 0.4
        self.max_val_test_diff_pct = 5.0
        
        # Fetch and process data
        print("🔄 Fetching backbone unfreezing analysis data...")
        self.df = self._fetch_and_process_data()
        
    @functools.lru_cache(maxsize=1)
    def _fetch_wandb_runs(self):
        """Fetch WandB runs with focus on two-phase training"""
        print("🔄 Fetching WandB runs...")
        api = wandb.Api()
        runs = api.runs(f"{self.entity}/{self.project}")
        
        filtered_runs = []
        for run in runs:
            config = run.config
            summary = run.summary
            
            # Filter for two-phase training experiments
            if (run.state == 'finished' and
                config.get('num_of_classes') == 5 and
                summary.get('test_kappa', 0) > self.min_test_kappa and
                bool(summary.keys())):
                
                filtered_runs.append(run)
        
        print(f"✅ Found {len(filtered_runs)} relevant runs for backbone analysis")
        return filtered_runs
    
    def _fetch_and_process_data(self):
        """Process WandB data for backbone unfreezing analysis"""
        runs = self._fetch_wandb_runs()
        
        data = []
        for run in runs:
            config = run.config
            summary = run.summary
            
            # Extract two-phase training information
            two_phase = config.get('two_phase_training', False)
            backbone_unfreeze_epoch = config.get('backbone_unfreeze_epoch', None)
            total_epochs = config.get('epochs', 100)
            
            # Performance metrics
            test_kappa = summary.get('test_kappa', 0)
            val_kappa = summary.get('val_kappa', 0)
            
            # Quality filter: val/test difference
            val_test_diff = 0
            is_high_quality = True
            if val_kappa > 0 and test_kappa > 0:
                val_test_diff = abs(val_kappa - test_kappa) / test_kappa * 100
                is_high_quality = val_test_diff <= self.max_val_test_diff_pct
            
            # Convergence metrics
            best_epoch = summary.get('best_epoch', total_epochs)
            convergence_speed = best_epoch / total_epochs if total_epochs > 0 else 1.0
            
            run_data = {
                # Run identification
                'run_id': run.id,
                'run_name': run.name,
                
                # Training strategy
                'two_phase_training': two_phase,
                'backbone_unfreeze_epoch': backbone_unfreeze_epoch,
                'total_epochs': total_epochs,
                'unfreeze_ratio': backbone_unfreeze_epoch / total_epochs if backbone_unfreeze_epoch else 0,
                
                # Performance
                'test_kappa': test_kappa,
                'val_kappa': val_kappa,
                'test_accuracy': summary.get('test_accuracy', 0),
                'val_test_diff_pct': val_test_diff,
                'is_high_quality': is_high_quality,
                
                # Convergence and overfitting
                'best_epoch': best_epoch,
                'convergence_speed': convergence_speed,
                'final_train_loss': summary.get('train_loss', 0),
                'final_val_loss': summary.get('val_loss', 0),
                'overfitting_indicator': summary.get('val_loss', 0) - summary.get('train_loss', 0),
                
                # Training configuration
                'learning_rate': config.get('learning_rate', 1e-4),
                'batch_size': config.get('batch_size', 64),
                'optimizer': config.get('optimizer', 'AdamW'),
            }
            data.append(run_data)
        
        df = pd.DataFrame(data)
        
        # Quality reporting
        print(f"📊 Processed {len(df)} backbone unfreezing experiments")
        high_quality = df[df['is_high_quality']]
        print(f"   - High quality runs (val/test diff ≤5%): {len(high_quality)}/{len(df)}")
        if len(high_quality) > 0:
            print(f"   - Test kappa range (high quality): {high_quality['test_kappa'].min():.3f} - {high_quality['test_kappa'].max():.3f}")
        
        return df
    
    def create_backbone_unfreezing_analysis(self):
        """Create comprehensive 4-subplot backbone unfreezing analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Backbone Unfreezing Strategy Analysis\n' + 
                     'Research Question: Impact of Unfreezing Timing on Performance', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # Subplot 1: Unfreezing timing vs performance
        self._plot_unfreezing_timing_performance(ax1)
        
        # Subplot 2: Convergence speed analysis
        self._plot_convergence_analysis(ax2)
        
        # Subplot 3: Overfitting analysis
        self._plot_overfitting_analysis(ax3)
        
        # Subplot 4: Training strategy comparison
        self._plot_training_strategy_comparison(ax4)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        output_path = os.path.join(self.output_dir, 'backbone_unfreezing_comprehensive_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved backbone unfreezing analysis: {output_path}")
        
        return fig
    
    def _plot_unfreezing_timing_performance(self, ax):
        """Subplot 1: Unfreezing timing vs peak performance"""
        # Focus on high-quality two-phase training runs
        df_clean = self.df[(self.df['is_high_quality']) & 
                           (self.df['two_phase_training']) &
                           (self.df['backbone_unfreeze_epoch'].notna())].copy()
        
        if len(df_clean) == 0:
            ax.text(0.5, 0.5, 'No high-quality two-phase runs found', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create unfreezing timing bins
        df_clean['unfreeze_bins'] = pd.cut(df_clean['unfreeze_ratio'], 
                                          bins=4, 
                                          labels=['Early\n(0-25%)', 'Mid-Early\n(25-50%)', 
                                                 'Mid-Late\n(50-75%)', 'Late\n(75-100%)'])
        
        # Get best performance per bin
        bin_stats = []
        for bin_label in df_clean['unfreeze_bins'].cat.categories:
            bin_data = df_clean[df_clean['unfreeze_bins'] == bin_label]
            if len(bin_data) > 0:
                bin_stats.append({
                    'bin': bin_label,
                    'best_kappa': bin_data['test_kappa'].max(),
                    'mean_kappa': bin_data['test_kappa'].mean(),
                    'count': len(bin_data),
                    'best_epoch_ratio': bin_data.loc[bin_data['test_kappa'].idxmax(), 'unfreeze_ratio']
                })
        
        if not bin_stats:
            ax.text(0.5, 0.5, 'No data in unfreezing bins', ha='center', va='center', transform=ax.transAxes)
            return
            
        bin_df = pd.DataFrame(bin_stats)
        
        # Plot performance vs unfreezing timing
        x_pos = range(len(bin_df))
        bars = ax.bar(x_pos, bin_df['best_kappa'], alpha=0.7, color='steelblue', 
                      label='Best Test Kappa')
        ax.scatter(x_pos, bin_df['mean_kappa'], color='darkred', s=60, 
                  label='Mean Test Kappa', zorder=5)
        
        # Add sample counts
        for i, row in bin_df.iterrows():
            ax.text(i, row['best_kappa'] + 0.01, f"n={row['count']}", 
                   ha='center', fontsize=9, fontweight='bold')
        
        ax.set_title('A) Backbone Unfreezing Timing vs Performance', fontweight='bold')
        ax.set_xlabel('Unfreezing Timing (% of Total Epochs)')
        ax.set_ylabel('Test Kappa')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_df['bin'])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_convergence_analysis(self, ax):
        """Subplot 2: Convergence speed vs unfreezing timing"""
        df_clean = self.df[(self.df['is_high_quality']) & 
                           (self.df['two_phase_training']) &
                           (self.df['convergence_speed'] > 0)].copy()
        
        if len(df_clean) == 0:
            ax.text(0.5, 0.5, 'No convergence data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Scatter plot: unfreezing ratio vs convergence speed, colored by performance
        scatter = ax.scatter(df_clean['unfreeze_ratio'], df_clean['convergence_speed'], 
                            c=df_clean['test_kappa'], cmap='viridis', alpha=0.7, s=60)
        
        # Add trend line
        if len(df_clean) >= 3:
            z = np.polyfit(df_clean['unfreeze_ratio'], df_clean['convergence_speed'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df_clean['unfreeze_ratio'].min(), 
                                 df_clean['unfreeze_ratio'].max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
        
        ax.set_title('B) Convergence Speed vs Unfreezing Timing', fontweight='bold')
        ax.set_xlabel('Backbone Unfreezing Ratio')
        ax.set_ylabel('Convergence Speed (Best Epoch / Total Epochs)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Color bar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Test Kappa', rotation=270, labelpad=15)
    
    def _plot_overfitting_analysis(self, ax):
        """Subplot 3: Overfitting analysis vs unfreezing timing"""
        df_clean = self.df[(self.df['is_high_quality']) & 
                           (self.df['two_phase_training']) &
                           (self.df['overfitting_indicator'].notna())].copy()
        
        if len(df_clean) == 0:
            ax.text(0.5, 0.5, 'No overfitting data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create bins for analysis
        df_clean['unfreeze_bins'] = pd.cut(df_clean['unfreeze_ratio'], 
                                          bins=3, 
                                          labels=['Early\n(<33%)', 'Middle\n(33-67%)', 'Late\n(>67%)'])
        
        # Box plot of overfitting indicator by unfreezing timing
        sns.boxplot(data=df_clean, x='unfreeze_bins', y='overfitting_indicator', ax=ax)
        
        # Add horizontal line at 0 (no overfitting)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No Overfitting')
        
        ax.set_title('C) Overfitting vs Unfreezing Timing', fontweight='bold')
        ax.set_xlabel('Backbone Unfreezing Timing')
        ax.set_ylabel('Overfitting Indicator (Val Loss - Train Loss)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_training_strategy_comparison(self, ax):
        """Subplot 4: Two-phase vs single-phase training comparison"""
        # Compare two-phase vs single-phase training
        high_quality = self.df[self.df['is_high_quality']]
        
        strategies = []
        for two_phase in [False, True]:
            strategy_data = high_quality[high_quality['two_phase_training'] == two_phase]
            if len(strategy_data) > 0:
                strategies.append({
                    'strategy': 'Two-Phase' if two_phase else 'Single-Phase',
                    'best_kappa': strategy_data['test_kappa'].max(),
                    'mean_kappa': strategy_data['test_kappa'].mean(),
                    'std_kappa': strategy_data['test_kappa'].std(),
                    'count': len(strategy_data),
                    'convergence_speed': strategy_data['convergence_speed'].mean()
                })
        
        if not strategies:
            ax.text(0.5, 0.5, 'No strategy data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        strategy_df = pd.DataFrame(strategies)
        
        # Plot comparison
        x_pos = range(len(strategy_df))
        bars = ax.bar(x_pos, strategy_df['best_kappa'], alpha=0.7, color=['orange', 'steelblue'], 
                      label='Best Performance')
        ax.errorbar(x_pos, strategy_df['mean_kappa'], yerr=strategy_df['std_kappa'], 
                    fmt='o', color='darkred', label='Mean ± Std', capsize=5)
        
        # Add improvement annotation
        if len(strategy_df) == 2:
            improvement = (strategy_df.iloc[1]['best_kappa'] - strategy_df.iloc[0]['best_kappa'])
            improvement_pct = improvement / strategy_df.iloc[0]['best_kappa'] * 100
            ax.annotate(f'+{improvement_pct:.1f}%', 
                       xy=(1, strategy_df.iloc[1]['best_kappa']), 
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=10, fontweight='bold', color='green')
        
        # Add sample counts
        for i, row in strategy_df.iterrows():
            ax.text(i, row['best_kappa'] + 0.01, f"n={row['count']}", 
                   ha='center', fontsize=9, fontweight='bold')
        
        ax.set_title('D) Training Strategy Comparison', fontweight='bold')
        ax.set_xlabel('Training Strategy')
        ax.set_ylabel('Test Kappa')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(strategy_df['strategy'])
        ax.legend()
        ax.grid(True, alpha=0.3)

def main():
    """Main analysis pipeline"""
    print("🚀 Starting Backbone Unfreezing Analysis...")
    
    # Initialize analyzer
    analyzer = BackboneUnfreezingAnalyzer()
    
    # Create comprehensive analysis
    analyzer.create_backbone_unfreezing_analysis()
    
    print("✅ Backbone unfreezing analysis complete!")

if __name__ == "__main__":
    main()