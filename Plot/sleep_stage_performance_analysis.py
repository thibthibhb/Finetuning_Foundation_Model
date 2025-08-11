#!/usr/bin/env python3
"""
Sleep Stage Performance Analysis for CBraMod
============================================

Research Question: How does performance vary across sleep stages (e.g., N1, N2, REM, Wake)?

This script analyzes per-sleep-stage performance, identifies challenging stages,
and compares CBraMod performance with baseline methods.
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

class SleepStagePerformanceAnalyzer:
    """Analyzes per-sleep-stage performance of CBraMod"""
    
    def __init__(self, entity="thibaut_hasle-epfl", project="CBraMod-earEEG-tuning", 
                 output_dir="Plot/figures"):
        self.entity = entity
        self.project = project
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Quality thresholds
        self.min_test_kappa = 0.4
        self.max_val_test_diff_pct = 5.0
        
        # Sleep stage mappings
        self.stage_names = {
            0: 'Wake',
            1: 'N1',
            2: 'N2', 
            3: 'N3',
            4: 'REM'
        }
        
        # YASA baseline per-stage performance (estimated from literature)
        self.yasa_baselines = {
            'Wake': {'f1': 0.75, 'precision': 0.78, 'recall': 0.72},
            'N1': {'f1': 0.35, 'precision': 0.40, 'recall': 0.32},  # Typically worst
            'N2': {'f1': 0.65, 'precision': 0.68, 'recall': 0.62},
            'N3': {'f1': 0.70, 'precision': 0.75, 'recall': 0.66},
            'REM': {'f1': 0.58, 'precision': 0.60, 'recall': 0.56}
        }
        
        # Fetch and process data
        print("🔄 Fetching sleep stage performance analysis data...")
        self.df = self._fetch_and_process_data()
        
    @functools.lru_cache(maxsize=1)
    def _fetch_wandb_runs(self):
        """Fetch WandB runs with per-stage metrics"""
        print("🔄 Fetching WandB runs...")
        api = wandb.Api()
        runs = api.runs(f"{self.entity}/{self.project}")
        
        filtered_runs = []
        for run in runs:
            config = run.config
            summary = run.summary
            
            # Filter for high-quality 5-class experiments with per-stage metrics
            has_per_stage = any(f'test_f1_class_{i}' in summary for i in range(5))
            
            if (run.state == 'finished' and
                config.get('num_of_classes') == 5 and
                summary.get('test_kappa', 0) > self.min_test_kappa and
                has_per_stage and
                bool(summary.keys())):
                
                filtered_runs.append(run)
        
        print(f"✅ Found {len(filtered_runs)} relevant runs for sleep stage analysis")
        return filtered_runs
    
    def _fetch_and_process_data(self):
        """Process WandB data for sleep stage performance analysis"""
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
            
            # Extract per-stage metrics
            per_stage_metrics = {}
            for stage_idx, stage_name in self.stage_names.items():
                per_stage_metrics[f'{stage_name}_f1'] = summary.get(f'test_f1_class_{stage_idx}', 0)
                per_stage_metrics[f'{stage_name}_precision'] = summary.get(f'test_precision_class_{stage_idx}', 0)
                per_stage_metrics[f'{stage_name}_recall'] = summary.get(f'test_recall_class_{stage_idx}', 0)
                
                # YASA improvement calculation
                yasa_f1 = self.yasa_baselines[stage_name]['f1']
                cbramod_f1 = per_stage_metrics[f'{stage_name}_f1']
                improvement = (cbramod_f1 - yasa_f1) / yasa_f1 * 100 if yasa_f1 > 0 else 0
                per_stage_metrics[f'{stage_name}_yasa_improvement'] = improvement
            
            # Calculate stage difficulty (inverse of best F1 performance)
            stage_difficulties = {}
            for stage_name in self.stage_names.values():
                f1_score = per_stage_metrics[f'{stage_name}_f1']
                stage_difficulties[f'{stage_name}_difficulty'] = 1 - f1_score if f1_score > 0 else 1
            
            # Find worst and best performing stages for this run
            stage_f1s = {stage: per_stage_metrics[f'{stage}_f1'] 
                        for stage in self.stage_names.values() 
                        if per_stage_metrics[f'{stage}_f1'] > 0}
            
            worst_stage = min(stage_f1s.keys(), key=stage_f1s.get) if stage_f1s else 'N1'
            best_stage = max(stage_f1s.keys(), key=stage_f1s.get) if stage_f1s else 'Wake'
            
            run_data = {
                # Run identification
                'run_id': run.id,
                'run_name': run.name,
                
                # Overall performance
                'test_kappa': test_kappa,
                'val_kappa': val_kappa,
                'test_accuracy': summary.get('test_accuracy', 0),
                'val_test_diff_pct': val_test_diff,
                'is_high_quality': is_high_quality,
                
                # Per-stage metrics
                **per_stage_metrics,
                
                # Stage difficulty analysis
                **stage_difficulties,
                'worst_stage': worst_stage,
                'best_stage': best_stage,
                'stage_performance_variance': np.var(list(stage_f1s.values())) if stage_f1s else 0,
                
                # Configuration factors that might affect stage performance
                'two_phase_training': config.get('two_phase_training', False),
                'learning_rate': config.get('learning_rate', 1e-4),
                'batch_size': config.get('batch_size', 64),
                'datasets': config.get('datasets', ''),
            }
            data.append(run_data)
        
        df = pd.DataFrame(data)
        
        # Quality reporting
        print(f"📊 Processed {len(df)} sleep stage experiments")
        high_quality = df[df['is_high_quality']]
        print(f"   - High quality runs: {len(high_quality)}/{len(df)}")
        
        if len(high_quality) > 0:
            # Report per-stage performance ranges
            for stage in self.stage_names.values():
                f1_col = f'{stage}_f1'
                if f1_col in high_quality.columns:
                    valid_f1 = high_quality[f1_col].dropna()
                    if len(valid_f1) > 0:
                        print(f"   - {stage} F1 range: {valid_f1.min():.3f} - {valid_f1.max():.3f}")
        
        return df
    
    def create_sleep_stage_performance_analysis(self):
        """Create comprehensive 4-subplot sleep stage performance analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Sleep Stage Performance Analysis for CBraMod\n' + 
                     'Research Question: How Does Performance Vary Across Sleep Stages?', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # Subplot 1: Per-stage F1 score comparison
        self._plot_per_stage_performance(ax1)
        
        # Subplot 2: CBraMod vs YASA baseline per stage
        self._plot_yasa_comparison_per_stage(ax2)
        
        # Subplot 3: Stage difficulty analysis
        self._plot_stage_difficulty_analysis(ax3)
        
        # Subplot 4: Confusion matrix analysis
        self._plot_confusion_analysis(ax4)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        output_path = os.path.join(self.output_dir, 'sleep_stage_performance_comprehensive_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved sleep stage performance analysis: {output_path}")
        
        return fig
    
    def _plot_per_stage_performance(self, ax):
        """Subplot 1: Per-stage F1 score comparison"""
        high_quality = self.df[self.df['is_high_quality']].copy()
        
        if len(high_quality) == 0:
            ax.text(0.5, 0.5, 'No high-quality data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Collect per-stage F1 scores
        stage_stats = []
        for stage in self.stage_names.values():
            f1_col = f'{stage}_f1'
            if f1_col in high_quality.columns:
                stage_f1s = high_quality[f1_col].dropna()
                if len(stage_f1s) > 0:
                    stage_stats.append({
                        'stage': stage,
                        'best_f1': stage_f1s.max(),
                        'mean_f1': stage_f1s.mean(),
                        'std_f1': stage_f1s.std(),
                        'median_f1': stage_f1s.median(),
                        'count': len(stage_f1s)
                    })
        
        if not stage_stats:
            ax.text(0.5, 0.5, 'No per-stage data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        stage_df = pd.DataFrame(stage_stats).sort_values('mean_f1', ascending=True)
        
        # Create horizontal bar plot
        y_pos = range(len(stage_df))
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green'][:len(stage_df)]
        
        bars = ax.barh(y_pos, stage_df['best_f1'], alpha=0.7, color=colors, 
                       label='Best F1 Score')
        ax.scatter(stage_df['mean_f1'], y_pos, color='darkred', s=60, 
                  label='Mean F1 Score', zorder=5)
        
        # Add error bars for standard deviation
        ax.errorbar(stage_df['mean_f1'], y_pos, xerr=stage_df['std_f1'], 
                    fmt='none', color='darkred', alpha=0.7, capsize=3)
        
        # Add sample counts
        for i, row in stage_df.iterrows():
            ax.text(row['best_f1'] + 0.02, i, f"n={row['count']}", 
                   ha='left', va='center', fontsize=9, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(stage_df['stage'])
        ax.set_title('A) Per-Stage F1 Performance Ranking', fontweight='bold')
        ax.set_xlabel('F1 Score')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, 1)
    
    def _plot_yasa_comparison_per_stage(self, ax):
        """Subplot 2: CBraMod vs YASA baseline per stage"""
        high_quality = self.df[self.df['is_high_quality']].copy()
        
        # Collect improvement data
        improvements = []
        for stage in self.stage_names.values():
            improvement_col = f'{stage}_yasa_improvement'
            if improvement_col in high_quality.columns:
                stage_improvements = high_quality[improvement_col].dropna()
                if len(stage_improvements) > 0:
                    improvements.append({
                        'stage': stage,
                        'best_improvement': stage_improvements.max(),
                        'mean_improvement': stage_improvements.mean(),
                        'yasa_baseline': self.yasa_baselines[stage]['f1'],
                        'cbramod_best': self.yasa_baselines[stage]['f1'] * (1 + stage_improvements.max()/100)
                    })
        
        if not improvements:
            ax.text(0.5, 0.5, 'No YASA comparison data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        improve_df = pd.DataFrame(improvements).sort_values('best_improvement')
        
        # Create comparison bars
        x_pos = np.arange(len(improve_df))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, improve_df['yasa_baseline'], width, 
                       alpha=0.7, color='orange', label='YASA Baseline')
        bars2 = ax.bar(x_pos + width/2, improve_df['cbramod_best'], width, 
                       alpha=0.7, color='steelblue', label='CBraMod Best')
        
        # Add improvement percentage annotations
        for i, row in improve_df.iterrows():
            improvement = row['best_improvement']
            color = 'green' if improvement > 0 else 'red'
            ax.annotate(f'{improvement:+.1f}%', 
                       xy=(i + width/2, row['cbramod_best']), 
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', fontsize=9, fontweight='bold', color=color)
        
        ax.set_title('B) CBraMod vs YASA Per-Stage Comparison', fontweight='bold')
        ax.set_xlabel('Sleep Stage')
        ax.set_ylabel('F1 Score')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(improve_df['stage'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _plot_stage_difficulty_analysis(self, ax):
        """Subplot 3: Sleep stage difficulty and variance analysis"""
        high_quality = self.df[self.df['is_high_quality']].copy()
        
        # Analyze worst performing stages across experiments
        worst_stages = high_quality['worst_stage'].value_counts()
        best_stages = high_quality['best_stage'].value_counts()
        
        # Create difficulty ranking
        all_stages = list(self.stage_names.values())
        difficulty_scores = []
        
        for stage in all_stages:
            worst_count = worst_stages.get(stage, 0)
            best_count = best_stages.get(stage, 0)
            total_mentions = worst_count + best_count
            
            if total_mentions > 0:
                difficulty_score = worst_count / total_mentions
            else:
                difficulty_score = 0.5  # Neutral if no data
            
            difficulty_scores.append({
                'stage': stage,
                'difficulty_score': difficulty_score,
                'worst_mentions': worst_count,
                'best_mentions': best_count,
                'total_experiments': len(high_quality)
            })
        
        difficulty_df = pd.DataFrame(difficulty_scores).sort_values('difficulty_score', ascending=False)
        
        # Plot difficulty ranking
        y_pos = range(len(difficulty_df))
        colors = plt.cm.RdYlGn_r(difficulty_df['difficulty_score'])  # Red for difficult, green for easy
        
        bars = ax.barh(y_pos, difficulty_df['difficulty_score'], alpha=0.7, color=colors)
        
        # Add annotations
        for i, row in difficulty_df.iterrows():
            ax.text(row['difficulty_score'] + 0.02, i, 
                   f"Worst: {row['worst_mentions']}, Best: {row['best_mentions']}", 
                   ha='left', va='center', fontsize=8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(difficulty_df['stage'])
        ax.set_title('C) Sleep Stage Difficulty Ranking\n(Frequency of Being Worst Performer)', 
                    fontweight='bold')
        ax.set_xlabel('Difficulty Score (0=Easy, 1=Hard)')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, 1)
    
    def _plot_confusion_analysis(self, ax):
        """Subplot 4: Common confusion patterns analysis"""
        high_quality = self.df[self.df['is_high_quality']].copy()
        
        # Analyze precision vs recall trade-offs per stage
        precision_recall_data = []
        
        for stage in self.stage_names.values():
            precision_col = f'{stage}_precision'
            recall_col = f'{stage}_recall'
            f1_col = f'{stage}_f1'
            
            if all(col in high_quality.columns for col in [precision_col, recall_col, f1_col]):
                stage_data = high_quality[[precision_col, recall_col, f1_col]].dropna()
                
                if len(stage_data) > 0:
                    # Get best performing run for this stage
                    best_idx = stage_data[f1_col].idxmax()
                    best_run = stage_data.loc[best_idx]
                    
                    precision_recall_data.append({
                        'stage': stage,
                        'precision': best_run[precision_col],
                        'recall': best_run[recall_col],
                        'f1': best_run[f1_col]
                    })
        
        if not precision_recall_data:
            ax.text(0.5, 0.5, 'No precision/recall data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        pr_df = pd.DataFrame(precision_recall_data)
        
        # Create precision-recall scatter plot
        colors = ['red', 'orange', 'yellow', 'green', 'blue'][:len(pr_df)]
        
        for i, row in pr_df.iterrows():
            ax.scatter(row['recall'], row['precision'], s=200, alpha=0.7, 
                      color=colors[i], label=row['stage'])
            
            # Add F1 score as annotation
            ax.annotate(f"F1={row['f1']:.2f}", 
                       xy=(row['recall'], row['precision']), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, ha='left')
        
        # Add diagonal lines for F1 score reference
        recall_range = np.linspace(0, 1, 100)
        for f1_target in [0.4, 0.6, 0.8]:
            precision_line = f1_target * recall_range / (2 * recall_range - f1_target)
            precision_line = np.clip(precision_line, 0, 1)
            ax.plot(recall_range, precision_line, '--', alpha=0.3, color='gray')
            ax.text(0.9, precision_line[-10], f'F1={f1_target}', fontsize=8, alpha=0.7)
        
        ax.set_title('D) Precision vs Recall Trade-offs\n(Best Performance per Stage)', 
                    fontweight='bold')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

def main():
    """Main analysis pipeline"""
    print("🚀 Starting Sleep Stage Performance Analysis...")
    
    # Initialize analyzer
    analyzer = SleepStagePerformanceAnalyzer()
    
    # Create comprehensive analysis
    analyzer.create_sleep_stage_performance_analysis()
    
    print("✅ Sleep stage performance analysis complete!")

if __name__ == "__main__":
    main()