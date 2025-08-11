#!/usr/bin/env python3
"""
Scaling Laws Analysis for CBraMod
================================

Research Question: What are the scaling laws of foundation model fine-tuning performance 
with respect to (i) number of training samples, (ii) diversity of subjects, 
and (iii) quality of the labels?

This script analyzes empirical scaling relationships in EEG foundation model performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import wandb
import functools
from scipy import stats
from scipy.optimize import curve_fit
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

class ScalingLawsAnalyzer:
    """Analyzes scaling laws for CBraMod performance"""
    
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
        print("🔄 Fetching scaling laws analysis data...")
        self.df = self._fetch_and_process_data()
        
    @functools.lru_cache(maxsize=1)
    def _fetch_wandb_runs(self):
        """Fetch WandB runs for scaling analysis"""
        print("🔄 Fetching WandB runs...")
        api = wandb.Api()
        runs = api.runs(f"{self.entity}/{self.project}")
        
        filtered_runs = []
        for run in runs:
            config = run.config
            summary = run.summary
            
            # Filter for high-quality 5-class experiments
            if (run.state == 'finished' and
                config.get('num_of_classes') == 5 and
                summary.get('test_kappa', 0) > self.min_test_kappa and
                bool(summary.keys())):
                
                filtered_runs.append(run)
        
        print(f"✅ Found {len(filtered_runs)} relevant runs for scaling analysis")
        return filtered_runs
    
    def _estimate_subject_diversity(self, datasets_str):
        """Estimate subject diversity from dataset configuration"""
        datasets = str(datasets_str).split(',')
        diversity_score = 0
        
        # Score based on dataset diversity
        dataset_scores = {
            'ORP': 16,  # 16 subjects
            '2023_Open_N': 50,  # Many subjects
            '2019_Open_N': 30,  # Medium subjects
            '2017_Open_N': 20   # Fewer subjects
        }
        
        for dataset in datasets:
            dataset = dataset.strip()
            if dataset in dataset_scores:
                diversity_score += dataset_scores[dataset]
        
        return min(diversity_score, 100)  # Cap at 100 for normalization
    
    def _estimate_label_quality(self, config, summary):
        """Estimate label quality from training characteristics"""
        # Base quality score
        quality_score = 70
        
        # Adjust based on training characteristics
        if config.get('use_pretrained_weights', False):
            quality_score += 10  # Pretrained weights suggest better data
        
        if config.get('two_phase_training', False):
            quality_score += 5   # Two-phase training suggests careful setup
        
        # Adjust based on convergence behavior
        best_epoch = summary.get('best_epoch', 100)
        total_epochs = config.get('epochs', 100)
        if best_epoch < total_epochs * 0.8:  # Converged before end
            quality_score += 5
        
        # Adjust based on overfitting indicator
        train_loss = summary.get('train_loss', 0)
        val_loss = summary.get('val_loss', 0)
        if val_loss > 0 and train_loss > 0:
            overfitting = (val_loss - train_loss) / train_loss
            if overfitting < 0.1:  # Low overfitting suggests good labels
                quality_score += 10
            elif overfitting > 0.3:  # High overfitting suggests noisy labels
                quality_score -= 10
        
        return np.clip(quality_score, 0, 100)
    
    def _fetch_and_process_data(self):
        """Process WandB data for scaling laws analysis"""
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
            
            # Scaling factors
            total_samples = summary.get('total_train_samples', summary.get('train_samples', 0))
            datasets_str = config.get('datasets', '')
            num_datasets = len([d.strip() for d in str(datasets_str).split(',') if d.strip()])
            
            subject_diversity = self._estimate_subject_diversity(datasets_str)
            label_quality = self._estimate_label_quality(config, summary)
            
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
                
                # Scaling factors
                'total_samples': total_samples,
                'log_samples': np.log10(max(1, total_samples)),
                'subject_diversity': subject_diversity,
                'label_quality': label_quality,
                'num_datasets': num_datasets,
                
                # Additional factors
                'training_epochs': summary.get('best_epoch', config.get('epochs', 100)),
                'batch_size': config.get('batch_size', 64),
                'learning_rate': config.get('learning_rate', 1e-4),
                'two_phase_training': config.get('two_phase_training', False),
            }
            data.append(run_data)
        
        df = pd.DataFrame(data)
        
        # Quality reporting
        print(f"📊 Processed {len(df)} scaling law experiments")
        high_quality = df[df['is_high_quality']]
        print(f"   - High quality runs: {len(high_quality)}/{len(df)}")
        if len(high_quality) > 0:
            print(f"   - Sample range: {high_quality['total_samples'].min():.0f} - {high_quality['total_samples'].max():.0f}")
            print(f"   - Diversity range: {high_quality['subject_diversity'].min():.0f} - {high_quality['subject_diversity'].max():.0f}")
            print(f"   - Quality range: {high_quality['label_quality'].min():.0f} - {high_quality['label_quality'].max():.0f}")
        
        return df
    
    def create_scaling_laws_analysis(self):
        """Create comprehensive 4-subplot scaling laws analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Scaling Laws Analysis for CBraMod Foundation Model\n' + 
                     'Research Question: Performance Scaling with Data, Diversity & Quality', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # Subplot 1: Sample scaling law
        self._plot_sample_scaling_law(ax1)
        
        # Subplot 2: Subject diversity scaling
        self._plot_diversity_scaling(ax2)
        
        # Subplot 3: Label quality impact
        self._plot_quality_scaling(ax3)
        
        # Subplot 4: Multi-dimensional scaling
        self._plot_multidimensional_scaling(ax4)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        output_path = os.path.join(self.output_dir, 'scaling_laws_comprehensive_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved scaling laws analysis: {output_path}")
        
        return fig
    
    def _power_law(self, x, a, b):
        """Power law function: y = a * x^b"""
        return a * np.power(x, b)
    
    def _plot_sample_scaling_law(self, ax):
        """Subplot 1: Performance vs number of training samples (power law)"""
        high_quality = self.df[(self.df['is_high_quality']) & 
                               (self.df['total_samples'] > 0)].copy()
        
        if len(high_quality) < 3:
            ax.text(0.5, 0.5, 'Insufficient data for scaling analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Get best performance for each sample size bucket
        high_quality['sample_bins'] = pd.qcut(high_quality['total_samples'], 
                                             q=min(5, len(high_quality)), 
                                             duplicates='drop')
        
        bin_stats = []
        for bin_label in high_quality['sample_bins'].cat.categories:
            bin_data = high_quality[high_quality['sample_bins'] == bin_label]
            if len(bin_data) > 0:
                bin_stats.append({
                    'samples': bin_data['total_samples'].mean(),
                    'log_samples': np.log10(bin_data['total_samples'].mean()),
                    'best_kappa': bin_data['test_kappa'].max(),
                    'mean_kappa': bin_data['test_kappa'].mean(),
                    'count': len(bin_data)
                })
        
        if len(bin_stats) < 3:
            ax.text(0.5, 0.5, 'Insufficient bins for scaling', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        bin_df = pd.DataFrame(bin_stats)
        
        # Plot data points
        ax.scatter(bin_df['samples'], bin_df['best_kappa'], alpha=0.7, s=80, 
                  color='steelblue', label='Best Performance')
        ax.scatter(bin_df['samples'], bin_df['mean_kappa'], alpha=0.5, s=40, 
                  color='lightcoral', label='Mean Performance')
        
        # Fit power law to best performance
        try:
            popt, pcov = curve_fit(self._power_law, bin_df['samples'], bin_df['best_kappa'])
            x_fit = np.linspace(bin_df['samples'].min(), bin_df['samples'].max(), 100)
            y_fit = self._power_law(x_fit, *popt)
            ax.plot(x_fit, y_fit, 'r--', linewidth=2, 
                   label=f'Power Law: y = {popt[0]:.3f} × x^{popt[1]:.3f}')
            
            # Calculate R²
            y_pred = self._power_law(bin_df['samples'], *popt)
            r2 = 1 - np.sum((bin_df['best_kappa'] - y_pred)**2) / np.sum((bin_df['best_kappa'] - np.mean(bin_df['best_kappa']))**2)
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        except:
            pass
        
        ax.set_title('A) Sample Scaling Law', fontweight='bold')
        ax.set_xlabel('Number of Training Samples')
        ax.set_ylabel('Test Kappa')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    def _plot_diversity_scaling(self, ax):
        """Subplot 2: Performance vs subject diversity"""
        high_quality = self.df[self.df['is_high_quality']].copy()
        
        if len(high_quality) < 3:
            ax.text(0.5, 0.5, 'Insufficient data for diversity analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create diversity bins
        high_quality['diversity_bins'] = pd.cut(high_quality['subject_diversity'], 
                                               bins=4, 
                                               labels=['Low\n(0-25)', 'Medium\n(25-50)', 
                                                      'High\n(50-75)', 'Very High\n(75+)'])
        
        # Get statistics per bin
        bin_stats = []
        for bin_label in high_quality['diversity_bins'].cat.categories:
            bin_data = high_quality[high_quality['diversity_bins'] == bin_label]
            if len(bin_data) > 0:
                bin_stats.append({
                    'bin': bin_label,
                    'diversity': bin_data['subject_diversity'].mean(),
                    'best_kappa': bin_data['test_kappa'].max(),
                    'mean_kappa': bin_data['test_kappa'].mean(),
                    'count': len(bin_data)
                })
        
        if not bin_stats:
            ax.text(0.5, 0.5, 'No diversity bins available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        bin_df = pd.DataFrame(bin_stats)
        
        # Plot diversity effect
        x_pos = range(len(bin_df))
        bars = ax.bar(x_pos, bin_df['best_kappa'], alpha=0.7, color='green', 
                      label='Best Performance')
        ax.scatter(x_pos, bin_df['mean_kappa'], color='darkgreen', s=60, 
                  label='Mean Performance', zorder=5)
        
        # Add trend line
        if len(bin_df) >= 3:
            z = np.polyfit(range(len(bin_df)), bin_df['best_kappa'], 1)
            p = np.poly1d(z)
            ax.plot(x_pos, p(x_pos), 'r--', alpha=0.8, linewidth=2, label='Trend')
        
        # Add sample counts
        for i, row in bin_df.iterrows():
            ax.text(i, row['best_kappa'] + 0.01, f"n={row['count']}", 
                   ha='center', fontsize=9, fontweight='bold')
        
        ax.set_title('B) Subject Diversity Scaling', fontweight='bold')
        ax.set_xlabel('Subject Diversity Level')
        ax.set_ylabel('Test Kappa')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_df['bin'])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_quality_scaling(self, ax):
        """Subplot 3: Performance vs estimated label quality"""
        high_quality = self.df[self.df['is_high_quality']].copy()
        
        if len(high_quality) < 3:
            ax.text(0.5, 0.5, 'Insufficient data for quality analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Scatter plot with correlation
        ax.scatter(high_quality['label_quality'], high_quality['test_kappa'], 
                  alpha=0.7, s=60, color='purple')
        
        # Add correlation line
        if len(high_quality) >= 3:
            z = np.polyfit(high_quality['label_quality'], high_quality['test_kappa'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(high_quality['label_quality'].min(), 
                                 high_quality['label_quality'].max(), 100)
            ax.plot(x_range, p(x_range), 'r--', linewidth=2, label='Linear Fit')
            
            # Calculate correlation
            correlation, p_value = stats.pearsonr(high_quality['label_quality'], 
                                                 high_quality['test_kappa'])
            sig_text = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            ax.text(0.05, 0.95, f'r = {correlation:.3f} ({sig_text})', 
                   transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_title('C) Label Quality Impact', fontweight='bold')
        ax.set_xlabel('Estimated Label Quality Score')
        ax.set_ylabel('Test Kappa')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_multidimensional_scaling(self, ax):
        """Subplot 4: Multi-dimensional scaling visualization"""
        high_quality = self.df[self.df['is_high_quality']].copy()
        
        if len(high_quality) < 3:
            ax.text(0.5, 0.5, 'Insufficient data for multi-dim analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create composite scaling score
        # Normalize each factor to 0-1 range
        samples_norm = (high_quality['total_samples'] - high_quality['total_samples'].min()) / (high_quality['total_samples'].max() - high_quality['total_samples'].min())
        diversity_norm = high_quality['subject_diversity'] / 100  # Already 0-100
        quality_norm = high_quality['label_quality'] / 100       # Already 0-100
        
        # Weighted composite score
        composite_score = 0.4 * samples_norm + 0.3 * diversity_norm + 0.3 * quality_norm
        
        # Scatter plot with size proportional to composite score
        scatter = ax.scatter(composite_score, high_quality['test_kappa'], 
                           c=high_quality['subject_diversity'], 
                           s=high_quality['total_samples'] / high_quality['total_samples'].max() * 200 + 20,
                           alpha=0.7, cmap='viridis')
        
        # Add trend line
        if len(high_quality) >= 3:
            z = np.polyfit(composite_score, high_quality['test_kappa'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(composite_score.min(), composite_score.max(), 100)
            ax.plot(x_range, p(x_range), 'r--', linewidth=2, alpha=0.8)
            
            # Calculate R²
            y_pred = p(composite_score)
            r2 = 1 - np.sum((high_quality['test_kappa'] - y_pred)**2) / np.sum((high_quality['test_kappa'] - np.mean(high_quality['test_kappa']))**2)
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_title('D) Multi-Dimensional Scaling\n(Size ∝ Samples, Color = Diversity)', 
                    fontweight='bold')
        ax.set_xlabel('Composite Scaling Score\n(0.4×Samples + 0.3×Diversity + 0.3×Quality)')
        ax.set_ylabel('Test Kappa')
        ax.grid(True, alpha=0.3)
        
        # Color bar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Subject Diversity', rotation=270, labelpad=15)

def main():
    """Main analysis pipeline"""
    print("🚀 Starting Scaling Laws Analysis...")
    
    # Initialize analyzer
    analyzer = ScalingLawsAnalyzer()
    
    # Create comprehensive analysis
    analyzer.create_scaling_laws_analysis()
    
    print("✅ Scaling laws analysis complete!")

if __name__ == "__main__":
    main()