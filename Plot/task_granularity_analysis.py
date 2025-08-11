#!/usr/bin/env python3
"""
Task Granularity Analysis for CBraMod
====================================

Research Question: How does task granularity (e.g., 4-class vs 5-class sleep staging) 
affect model performance in single-channel ear-EEG?

This script compares 4-class vs 5-class sleep staging performance and analyzes
the trade-offs between task complexity and classification accuracy.
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

class TaskGranularityAnalyzer:
    """Analyzes 4-class vs 5-class sleep staging performance"""
    
    def __init__(self, entity="thibaut_hasle-epfl", project="CBraMod-earEEG-tuning", 
                 output_dir="Plot/figures"):
        self.entity = entity
        self.project = project
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Quality thresholds
        self.min_test_kappa_4class = 0.35  # Lower threshold for 4-class (harder task)
        self.min_test_kappa_5class = 0.4   # Higher threshold for 5-class
        self.max_val_test_diff_pct = 5.0
        
        # YASA baselines for comparison
        self.yasa_baselines = {
            4: {'kappa': 0.52, 'accuracy': 0.68},  # Estimated 4-class performance
            5: {'kappa': 0.446, 'accuracy': 0.60}  # Known 5-class performance
        }
        
        # Fetch and process data
        print("🔄 Fetching task granularity analysis data...")
        self.df = self._fetch_and_process_data()
        
    @functools.lru_cache(maxsize=1)
    def _fetch_wandb_runs(self):
        """Fetch WandB runs for both 4-class and 5-class experiments"""
        print("🔄 Fetching WandB runs...")
        api = wandb.Api()
        runs = api.runs(f"{self.entity}/{self.project}")
        
        filtered_runs = []
        for run in runs:
            config = run.config
            summary = run.summary
            
            num_classes = config.get('num_of_classes')
            test_kappa = summary.get('test_kappa', 0)
            
            # Filter for both 4-class and 5-class experiments
            min_kappa = (self.min_test_kappa_4class if num_classes == 4 
                        else self.min_test_kappa_5class)
            
            if (run.state == 'finished' and
                num_classes in [4, 5] and
                test_kappa > min_kappa and
                bool(summary.keys())):
                
                filtered_runs.append(run)
        
        print(f"✅ Found {len(filtered_runs)} relevant runs for task granularity analysis")
        return filtered_runs
    
    def _fetch_and_process_data(self):
        """Process WandB data for task granularity analysis"""
        runs = self._fetch_wandb_runs()
        
        data = []
        for run in runs:
            config = run.config
            summary = run.summary
            
            # Task configuration
            num_classes = config.get('num_of_classes')
            
            # Performance metrics
            test_kappa = summary.get('test_kappa', 0)
            val_kappa = summary.get('val_kappa', 0)
            test_accuracy = summary.get('test_accuracy', 0)
            
            # Quality filter
            val_test_diff = 0
            is_high_quality = True
            if val_kappa > 0 and test_kappa > 0:
                val_test_diff = abs(val_kappa - test_kappa) / test_kappa * 100
                is_high_quality = val_test_diff <= self.max_val_test_diff_pct
            
            # Per-class metrics (if available)
            per_class_f1 = {}
            class_names_4 = ['Wake', 'Light', 'Deep', 'REM']
            class_names_5 = ['Wake', 'N1', 'N2', 'N3', 'REM']
            
            class_names = class_names_4 if num_classes == 4 else class_names_5
            for i, class_name in enumerate(class_names):
                f1_key = f'test_f1_class_{i}'
                per_class_f1[class_name] = summary.get(f1_key, 0)
            
            # Training complexity indicators
            convergence_epochs = summary.get('best_epoch', config.get('epochs', 100))
            training_time = summary.get('training_time_hours', 0)
            
            run_data = {
                # Run identification
                'run_id': run.id,
                'run_name': run.name,
                
                # Task granularity
                'num_classes': num_classes,
                'task_type': f'{num_classes}-class',
                
                # Performance
                'test_kappa': test_kappa,
                'val_kappa': val_kappa,
                'test_accuracy': test_accuracy,
                'val_test_diff_pct': val_test_diff,
                'is_high_quality': is_high_quality,
                
                # Per-class performance
                **per_class_f1,
                
                # Training efficiency
                'convergence_epochs': convergence_epochs,
                'training_time_hours': training_time,
                'epochs_per_hour': convergence_epochs / training_time if training_time > 0 else 0,
                
                # YASA comparison
                'yasa_kappa_baseline': self.yasa_baselines[num_classes]['kappa'],
                'yasa_improvement_pct': (test_kappa - self.yasa_baselines[num_classes]['kappa']) / self.yasa_baselines[num_classes]['kappa'] * 100,
                
                # Configuration
                'learning_rate': config.get('learning_rate', 1e-4),
                'batch_size': config.get('batch_size', 64),
                'two_phase_training': config.get('two_phase_training', False),
            }
            data.append(run_data)
        
        df = pd.DataFrame(data)
        
        # Quality reporting
        print(f"📊 Processed {len(df)} task granularity experiments")
        for num_classes in [4, 5]:
            class_data = df[df['num_classes'] == num_classes]
            high_quality = class_data[class_data['is_high_quality']]
            print(f"   - {num_classes}-class: {len(high_quality)}/{len(class_data)} high quality")
            if len(high_quality) > 0:
                print(f"     Best kappa: {high_quality['test_kappa'].max():.3f}")
        
        return df
    
    def create_task_granularity_analysis(self):
        """Create comprehensive 4-subplot task granularity analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Task Granularity Analysis: 4-Class vs 5-Class Sleep Staging\n' + 
                     'Research Question: Impact of Classification Granularity on Performance', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # Subplot 1: Direct performance comparison
        self._plot_performance_comparison(ax1)
        
        # Subplot 2: Per-class performance analysis
        self._plot_per_class_performance(ax2)
        
        # Subplot 3: Training efficiency comparison
        self._plot_training_efficiency(ax3)
        
        # Subplot 4: YASA baseline comparison
        self._plot_yasa_baseline_comparison(ax4)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        output_path = os.path.join(self.output_dir, 'task_granularity_comprehensive_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved task granularity analysis: {output_path}")
        
        return fig
    
    def _plot_performance_comparison(self, ax):
        """Subplot 1: Direct 4-class vs 5-class performance comparison"""
        high_quality = self.df[self.df['is_high_quality']]
        
        if len(high_quality) == 0:
            ax.text(0.5, 0.5, 'No high-quality data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Group performance by task type
        task_stats = []
        for task_type in ['4-class', '5-class']:
            task_data = high_quality[high_quality['task_type'] == task_type]
            if len(task_data) > 0:
                task_stats.append({
                    'task_type': task_type,
                    'best_kappa': task_data['test_kappa'].max(),
                    'mean_kappa': task_data['test_kappa'].mean(),
                    'std_kappa': task_data['test_kappa'].std(),
                    'best_accuracy': task_data['test_accuracy'].max(),
                    'mean_accuracy': task_data['test_accuracy'].mean(),
                    'count': len(task_data)
                })
        
        if not task_stats:
            ax.text(0.5, 0.5, 'No task data available', ha='center', va='center', transform=ax.transAxes)
            return
            
        task_df = pd.DataFrame(task_stats)
        
        # Create side-by-side comparison
        x_pos = np.arange(len(task_df))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, task_df['best_kappa'], width, alpha=0.7, 
                       color='steelblue', label='Best Kappa')
        bars2 = ax.bar(x_pos + width/2, task_df['best_accuracy'], width, alpha=0.7, 
                       color='lightcoral', label='Best Accuracy')
        
        # Add mean performance points
        ax.scatter(x_pos - width/2, task_df['mean_kappa'], color='darkblue', s=60, 
                  label='Mean Kappa', zorder=5)
        ax.scatter(x_pos + width/2, task_df['mean_accuracy'], color='darkred', s=60, 
                  label='Mean Accuracy', zorder=5)
        
        # Add sample counts
        for i, row in task_df.iterrows():
            ax.text(i, max(row['best_kappa'], row['best_accuracy']) + 0.02, 
                   f"n={row['count']}", ha='center', fontsize=9, fontweight='bold')
        
        ax.set_title('A) 4-Class vs 5-Class Performance Comparison', fontweight='bold')
        ax.set_xlabel('Task Granularity')
        ax.set_ylabel('Performance Score')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(task_df['task_type'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.3, 1.0)
    
    def _plot_per_class_performance(self, ax):
        """Subplot 2: Per-class F1 score comparison"""
        high_quality = self.df[self.df['is_high_quality']]
        
        # Define class mappings
        class_mapping = {
            4: ['Wake', 'Light', 'Deep', 'REM'],
            5: ['Wake', 'N1', 'N2', 'N3', 'REM']
        }
        
        per_class_stats = []
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        
        for num_classes in [4, 5]:
            task_data = high_quality[high_quality['num_classes'] == num_classes]
            if len(task_data) == 0:
                continue
                
            class_names = class_mapping[num_classes]
            
            for i, class_name in enumerate(class_names):
                if class_name in task_data.columns:
                    class_f1_scores = task_data[class_name].dropna()
                    if len(class_f1_scores) > 0:
                        per_class_stats.append({
                            'task_type': f'{num_classes}-class',
                            'class_name': class_name,
                            'best_f1': class_f1_scores.max(),
                            'mean_f1': class_f1_scores.mean(),
                            'color': colors[i % len(colors)]
                        })
        
        if not per_class_stats:
            ax.text(0.5, 0.5, 'No per-class data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        per_class_df = pd.DataFrame(per_class_stats)
        
        # Create grouped bar chart
        task_types = per_class_df['task_type'].unique()
        class_names = per_class_df['class_name'].unique()
        
        x = np.arange(len(class_names))
        width = 0.35
        
        for i, task_type in enumerate(task_types):
            task_data = per_class_df[per_class_df['task_type'] == task_type]
            f1_scores = [task_data[task_data['class_name'] == cls]['best_f1'].iloc[0] 
                        if len(task_data[task_data['class_name'] == cls]) > 0 else 0 
                        for cls in class_names]
            
            bars = ax.bar(x + i * width, f1_scores, width, alpha=0.7, 
                         label=task_type)
        
        ax.set_title('B) Per-Class F1 Score Comparison', fontweight='bold')
        ax.set_xlabel('Sleep Stage')
        ax.set_ylabel('Best F1 Score')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(class_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_training_efficiency(self, ax):
        """Subplot 3: Training efficiency comparison"""
        high_quality = self.df[self.df['is_high_quality']]
        
        # Scatter plot: convergence epochs vs performance, colored by task type
        task_4_data = high_quality[high_quality['num_classes'] == 4]
        task_5_data = high_quality[high_quality['num_classes'] == 5]
        
        if len(task_4_data) > 0:
            ax.scatter(task_4_data['convergence_epochs'], task_4_data['test_kappa'], 
                      alpha=0.7, s=60, color='blue', label='4-class')
        
        if len(task_5_data) > 0:
            ax.scatter(task_5_data['convergence_epochs'], task_5_data['test_kappa'], 
                      alpha=0.7, s=60, color='red', label='5-class')
        
        # Add trend lines if enough data
        for data, color, label in [(task_4_data, 'blue', '4-class'), 
                                  (task_5_data, 'red', '5-class')]:
            if len(data) >= 3:
                z = np.polyfit(data['convergence_epochs'], data['test_kappa'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(data['convergence_epochs'].min(), 
                                     data['convergence_epochs'].max(), 100)
                ax.plot(x_trend, p(x_trend), color=color, linestyle='--', alpha=0.8)
        
        ax.set_title('C) Training Efficiency: Convergence vs Performance', fontweight='bold')
        ax.set_xlabel('Convergence Epochs')
        ax.set_ylabel('Test Kappa')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_yasa_baseline_comparison(self, ax):
        """Subplot 4: Improvement over YASA baseline"""
        high_quality = self.df[self.df['is_high_quality']]
        
        task_improvements = []
        for num_classes in [4, 5]:
            task_data = high_quality[high_quality['num_classes'] == num_classes]
            if len(task_data) > 0:
                best_improvement = task_data['yasa_improvement_pct'].max()
                mean_improvement = task_data['yasa_improvement_pct'].mean()
                yasa_baseline = self.yasa_baselines[num_classes]['kappa']
                
                task_improvements.append({
                    'task_type': f'{num_classes}-class',
                    'best_improvement_pct': best_improvement,
                    'mean_improvement_pct': mean_improvement,
                    'yasa_baseline': yasa_baseline,
                    'count': len(task_data)
                })
        
        if not task_improvements:
            ax.text(0.5, 0.5, 'No improvement data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        improvement_df = pd.DataFrame(task_improvements)
        
        # Plot improvements
        x_pos = range(len(improvement_df))
        bars = ax.bar(x_pos, improvement_df['best_improvement_pct'], alpha=0.7, 
                      color=['blue', 'red'], label='Best Improvement')
        ax.scatter(x_pos, improvement_df['mean_improvement_pct'], color='darkgreen', s=60, 
                  label='Mean Improvement', zorder=5)
        
        # Add baseline reference line
        ax.axhline(y=0, color='orange', linestyle='--', linewidth=2, 
                   label='YASA Baseline')
        
        # Add improvement annotations
        for i, row in improvement_df.iterrows():
            improvement = row['best_improvement_pct']
            color = 'green' if improvement > 0 else 'red'
            ax.annotate(f'{improvement:+.1f}%', 
                       xy=(i, improvement), xytext=(0, 5), textcoords='offset points',
                       ha='center', fontsize=10, fontweight='bold', color=color)
        
        ax.set_title('D) Improvement Over YASA Baseline', fontweight='bold')
        ax.set_xlabel('Task Granularity')
        ax.set_ylabel('Performance Improvement (%)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(improvement_df['task_type'])
        ax.legend()
        ax.grid(True, alpha=0.3)

def main():
    """Main analysis pipeline"""
    print("🚀 Starting Task Granularity Analysis...")
    
    # Initialize analyzer
    analyzer = TaskGranularityAnalyzer()
    
    # Create comprehensive analysis
    analyzer.create_task_granularity_analysis()
    
    print("✅ Task granularity analysis complete!")

if __name__ == "__main__":
    main()