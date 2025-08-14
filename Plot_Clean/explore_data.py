#!/usr/bin/env python3
"""
Data Explorer for CBraMod Structured Runs
=========================================

This script provides utilities to explore and understand the structured
WandB run data before creating plots.

Usage:
    python explore_data.py --data-file Plot_Clean/data/all_runs.csv
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from config import *

logger = logging.getLogger(__name__)

class DataExplorer:
    """Explore and understand structured run data."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = None
        self.contract_data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load structured run data."""
        logger.info(f"Loading data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        
        # Parse JSON columns
        if 'contract_data' in self.df.columns:
            self.df['contract_parsed'] = self.df['contract_data'].apply(
                lambda x: json.loads(x) if pd.notna(x) else {}
            )
        
        if 'validation_status' in self.df.columns:
            self.df['validation_parsed'] = self.df['validation_status'].apply(
                lambda x: json.loads(x) if pd.notna(x) else {}
            )
        
        # Extract key metrics for easier access
        self._extract_key_metrics()
        
        logger.info(f"Loaded {len(self.df)} runs")
        return self.df
    
    def _extract_key_metrics(self):
        """Extract key metrics from contract data for easier analysis."""
        
        # Extract commonly used fields
        for idx, row in self.df.iterrows():
            contract = row.get('contract_parsed', {})
            
            # Results
            results = contract.get('results', {})
            self.df.loc[idx, 'kappa'] = results.get('test_kappa', np.nan)
            self.df.loc[idx, 'f1'] = results.get('test_f1', np.nan)
            self.df.loc[idx, 'accuracy'] = results.get('test_accuracy', np.nan)
            self.df.loc[idx, 'num_classes'] = results.get('num_classes', np.nan)
            
            # Training config
            training = contract.get('training', {})
            self.df.loc[idx, 'epochs'] = training.get('epochs', np.nan)
            self.df.loc[idx, 'batch_size'] = training.get('batch_size', np.nan)
            self.df.loc[idx, 'lr'] = training.get('lr', np.nan)
            self.df.loc[idx, 'optimizer'] = training.get('optimizer', 'unknown')
            
            # ICL config
            icl = contract.get('icl', {})
            self.df.loc[idx, 'icl_mode'] = icl.get('icl_mode', 'none')
            self.df.loc[idx, 'k_support'] = icl.get('k_support', 0)
            
            # Dataset
            dataset = contract.get('dataset', {})
            self.df.loc[idx, 'num_subjects'] = dataset.get('num_subjects_train', np.nan)
            self.df.loc[idx, 'data_fraction'] = dataset.get('data_fraction', np.nan)
            
            # Validation status
            validation = row.get('validation_parsed', {})
            self.df.loc[idx, 'validation_status'] = validation.get('status', 'UNKNOWN')
            self.df.loc[idx, 'quality_score'] = validation.get('score', 0)
    
    def generate_summary(self) -> Dict:
        """Generate a comprehensive data summary."""
        
        if self.df is None:
            self.load_data()
        
        summary = {
            'overview': self._get_overview(),
            'validation_status': self._get_validation_summary(),
            'performance_summary': self._get_performance_summary(),
            'configuration_summary': self._get_configuration_summary(),
            'icl_summary': self._get_icl_summary(),
            'data_completeness': self._get_completeness_summary(),
            'potential_issues': self._identify_issues()
        }
        
        return summary
    
    def _get_overview(self) -> Dict:
        """Get basic overview statistics."""
        return {
            'total_runs': len(self.df),
            'date_range': {
                'earliest': self.df['created_at'].min() if 'created_at' in self.df.columns else 'Unknown',
                'latest': self.df['created_at'].max() if 'created_at' in self.df.columns else 'Unknown'
            },
            'unique_names': len(self.df['name'].unique()) if 'name' in self.df.columns else 0,
            'run_states': self.df['state'].value_counts().to_dict() if 'state' in self.df.columns else {}
        }
    
    def _get_validation_summary(self) -> Dict:
        """Get validation status summary."""
        if 'validation_status' not in self.df.columns:
            return {'error': 'No validation status available'}
        
        status_counts = self.df['validation_status'].value_counts().to_dict()
        quality_stats = self.df['quality_score'].describe().to_dict() if 'quality_score' in self.df.columns else {}
        
        return {
            'status_distribution': status_counts,
            'quality_score_stats': quality_stats,
            'valid_runs_percentage': status_counts.get('VALID', 0) / len(self.df) * 100
        }
    
    def _get_performance_summary(self) -> Dict:
        """Get performance metrics summary."""
        metrics = ['kappa', 'f1', 'accuracy']
        summary = {}
        
        for metric in metrics:
            if metric in self.df.columns:
                data = self.df[metric].dropna()
                if len(data) > 0:
                    summary[metric] = {
                        'count': len(data),
                        'mean': float(data.mean()),
                        'std': float(data.std()),
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'median': float(data.median()),
                        'q25': float(data.quantile(0.25)),
                        'q75': float(data.quantile(0.75))
                    }
                    
                    # Performance categories
                    if metric == 'kappa':
                        thresholds = METRICS['test_kappa']['thresholds']
                        categories = {}
                        for category, threshold in thresholds.items():
                            categories[category] = int((data >= threshold).sum())
                        summary[metric]['categories'] = categories
        
        return summary
    
    def _get_configuration_summary(self) -> Dict:
        """Get training configuration summary."""
        
        config_fields = {
            'num_classes': 'Number of classes',
            'epochs': 'Training epochs',
            'batch_size': 'Batch size',
            'lr': 'Learning rate',
            'optimizer': 'Optimizer',
            'num_subjects': 'Number of training subjects',
            'data_fraction': 'Data fraction used'
        }
        
        summary = {}
        for field, description in config_fields.items():
            if field in self.df.columns:
                if self.df[field].dtype in ['object', 'string']:
                    # Categorical data
                    summary[field] = {
                        'description': description,
                        'unique_values': self.df[field].value_counts().to_dict()
                    }
                else:
                    # Numerical data
                    data = self.df[field].dropna()
                    if len(data) > 0:
                        summary[field] = {
                            'description': description,
                            'stats': {
                                'count': len(data),
                                'mean': float(data.mean()),
                                'min': float(data.min()),
                                'max': float(data.max()),
                                'unique_values': len(data.unique())
                            }
                        }
        
        return summary
    
    def _get_icl_summary(self) -> Dict:
        """Get ICL-specific summary."""
        if 'icl_mode' not in self.df.columns:
            return {'error': 'No ICL data available'}
        
        icl_summary = {
            'mode_distribution': self.df['icl_mode'].value_counts().to_dict(),
            'k_support_distribution': self.df['k_support'].value_counts().to_dict() if 'k_support' in self.df.columns else {},
        }
        
        # Performance by ICL mode
        if 'kappa' in self.df.columns:
            icl_performance = {}
            for mode in self.df['icl_mode'].unique():
                mode_data = self.df[self.df['icl_mode'] == mode]
                kappa_data = mode_data['kappa'].dropna()
                if len(kappa_data) > 0:
                    icl_performance[mode] = {
                        'count': len(kappa_data),
                        'mean_kappa': float(kappa_data.mean()),
                        'std_kappa': float(kappa_data.std()),
                        'best_kappa': float(kappa_data.max())
                    }
            icl_summary['performance_by_mode'] = icl_performance
        
        return icl_summary
    
    def _get_completeness_summary(self) -> Dict:
        """Assess data completeness."""
        
        key_fields = [
            'kappa', 'f1', 'accuracy', 'num_classes',
            'epochs', 'batch_size', 'lr', 'optimizer',
            'icl_mode', 'num_subjects'
        ]
        
        completeness = {}
        total_runs = len(self.df)
        
        for field in key_fields:
            if field in self.df.columns:
                non_null = self.df[field].notna().sum()
                completeness[field] = {
                    'available': int(non_null),
                    'missing': int(total_runs - non_null),
                    'percentage': float(non_null / total_runs * 100)
                }
        
        return completeness
    
    def _identify_issues(self) -> List[str]:
        """Identify potential data quality issues."""
        issues = []
        
        if self.df is None:
            return ['Data not loaded']
        
        # Check for critical missing data
        if 'kappa' in self.df.columns:
            missing_kappa = self.df['kappa'].isna().sum()
            if missing_kappa / len(self.df) > 0.2:
                issues.append(f"High proportion of missing kappa values: {missing_kappa}/{len(self.df)}")
        
        # Check for performance outliers
        if 'kappa' in self.df.columns:
            kappa_data = self.df['kappa'].dropna()
            if len(kappa_data) > 0:
                negative_kappa = (kappa_data < -0.1).sum()
                if negative_kappa / len(kappa_data) > 0.1:
                    issues.append(f"High proportion of very poor performance (κ < -0.1): {negative_kappa}/{len(kappa_data)}")
        
        # Check for configuration inconsistencies
        if 'num_classes' in self.df.columns:
            class_counts = self.df['num_classes'].value_counts()
            if len(class_counts) > 3:
                issues.append(f"Too many different class counts: {class_counts.to_dict()}")
        
        # Check for validation issues
        if 'validation_status' in self.df.columns:
            invalid_runs = (self.df['validation_status'] == 'INVALID').sum()
            if invalid_runs / len(self.df) > 0.3:
                issues.append(f"High proportion of invalid runs: {invalid_runs}/{len(self.df)}")
        
        # Check for temporal clustering (potential batch uploads)
        if 'created_at' in self.df.columns:
            try:
                timestamps = pd.to_datetime(self.df['created_at'])
                time_diffs = timestamps.diff().dt.total_seconds()
                rapid_uploads = (time_diffs < 60).sum()  # Less than 1 minute apart
                if rapid_uploads / len(self.df) > 0.5:
                    issues.append(f"Many runs created in rapid succession (potential batch upload): {rapid_uploads}/{len(self.df)}")
            except:
                issues.append("Could not parse creation timestamps")
        
        return issues
    
    def create_summary_plots(self, output_dir: Optional[str] = None):
        """Create summary visualization plots."""
        
        if self.df is None:
            self.load_data()
        
        if output_dir is None:
            output_dir = FIGURE_DIR / 'data_exploration'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting
        setup_plotting_style()
        
        # Plot 1: Performance distribution
        if 'kappa' in self.df.columns:
            self._plot_performance_distribution(output_dir)
        
        # Plot 2: Configuration overview
        self._plot_configuration_overview(output_dir)
        
        # Plot 3: ICL mode comparison
        if 'icl_mode' in self.df.columns:
            self._plot_icl_comparison(output_dir)
        
        # Plot 4: Data completeness
        self._plot_data_completeness(output_dir)
        
        logger.info(f"Summary plots saved to {output_dir}")
    
    def _plot_performance_distribution(self, output_dir: Path):
        """Plot performance metric distributions."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Metrics Distribution', fontsize=16, fontweight='bold')
        
        metrics = ['kappa', 'f1', 'accuracy']
        colors = [PLOT_CONFIG['colors']['primary'], PLOT_CONFIG['colors']['secondary'], PLOT_CONFIG['colors']['accent']]
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            if metric in self.df.columns:
                ax = axes[i//2, i%2]
                data = self.df[metric].dropna()
                
                # Histogram
                ax.hist(data, bins=30, alpha=0.7, color=color, edgecolor='black')
                ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.3f}')
                ax.axvline(data.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {data.median():.3f}')
                
                metric_info = get_metric_info(f'test_{metric}')
                ax.set_title(f'{metric_info["name"]} Distribution')
                ax.set_xlabel(metric_info['name'])
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Performance by number of classes
        if 'num_classes' in self.df.columns:
            ax = axes[1, 1]
            for num_classes in sorted(self.df['num_classes'].dropna().unique()):
                class_data = self.df[self.df['num_classes'] == num_classes]['kappa'].dropna()
                if len(class_data) > 0:
                    ax.hist(class_data, alpha=0.6, label=f'{int(num_classes)}-class', bins=20)
            
            ax.set_title("Cohen's κ by Number of Classes")
            ax.set_xlabel("Cohen's κ")
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_distribution.png', **PLOT_CONFIG['figure'])
        plt.close()
    
    def _plot_configuration_overview(self, output_dir: Path):
        """Plot configuration overview."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Configuration Overview', fontsize=16, fontweight='bold')
        
        # Number of classes
        if 'num_classes' in self.df.columns:
            class_counts = self.df['num_classes'].value_counts().sort_index()
            axes[0, 0].bar(class_counts.index.astype(str), class_counts.values, 
                          color=PLOT_CONFIG['colors']['primary'])
            axes[0, 0].set_title('Number of Classes')
            axes[0, 0].set_xlabel('Classes')
            axes[0, 0].set_ylabel('Number of Runs')
        
        # Epochs distribution
        if 'epochs' in self.df.columns:
            epoch_data = self.df['epochs'].dropna()
            axes[0, 1].hist(epoch_data, bins=20, color=PLOT_CONFIG['colors']['secondary'], alpha=0.7)
            axes[0, 1].set_title('Training Epochs')
            axes[0, 1].set_xlabel('Epochs')
            axes[0, 1].set_ylabel('Frequency')
        
        # Learning rate distribution  
        if 'lr' in self.df.columns:
            lr_data = self.df['lr'].dropna()
            axes[0, 2].hist(np.log10(lr_data), bins=20, color=PLOT_CONFIG['colors']['accent'], alpha=0.7)
            axes[0, 2].set_title('Learning Rate (log10)')
            axes[0, 2].set_xlabel('log10(Learning Rate)')
            axes[0, 2].set_ylabel('Frequency')
        
        # Optimizer distribution
        if 'optimizer' in self.df.columns:
            opt_counts = self.df['optimizer'].value_counts()
            axes[1, 0].bar(range(len(opt_counts)), opt_counts.values, 
                          color=PLOT_CONFIG['colors']['primary'])
            axes[1, 0].set_xticks(range(len(opt_counts)))
            axes[1, 0].set_xticklabels(opt_counts.index, rotation=45)
            axes[1, 0].set_title('Optimizer Usage')
            axes[1, 0].set_ylabel('Number of Runs')
        
        # Batch size distribution
        if 'batch_size' in self.df.columns:
            batch_counts = self.df['batch_size'].value_counts().sort_index()
            axes[1, 1].bar(batch_counts.index.astype(str), batch_counts.values,
                          color=PLOT_CONFIG['colors']['secondary'])
            axes[1, 1].set_title('Batch Size')
            axes[1, 1].set_xlabel('Batch Size')
            axes[1, 1].set_ylabel('Number of Runs')
        
        # Validation status
        if 'validation_status' in self.df.columns:
            val_counts = self.df['validation_status'].value_counts()
            colors = ['green' if status == 'VALID' else 'orange' if status == 'WARNING' else 'red' 
                     for status in val_counts.index]
            axes[1, 2].bar(val_counts.index, val_counts.values, color=colors)
            axes[1, 2].set_title('Validation Status')
            axes[1, 2].set_ylabel('Number of Runs')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'configuration_overview.png', **PLOT_CONFIG['figure'])
        plt.close()
    
    def _plot_icl_comparison(self, output_dir: Path):
        """Plot ICL mode comparison."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('In-Context Learning Analysis', fontsize=16, fontweight='bold')
        
        # ICL mode distribution
        icl_counts = self.df['icl_mode'].value_counts()
        colors = [ICL_CONFIG['modes'].get(mode, {}).get('color', PLOT_CONFIG['colors']['neutral']) 
                 for mode in icl_counts.index]
        axes[0, 0].bar(icl_counts.index, icl_counts.values, color=colors)
        axes[0, 0].set_title('ICL Mode Distribution')
        axes[0, 0].set_ylabel('Number of Runs')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Performance by ICL mode
        if 'kappa' in self.df.columns:
            icl_performance = []
            icl_modes = []
            for mode in self.df['icl_mode'].unique():
                mode_data = self.df[self.df['icl_mode'] == mode]['kappa'].dropna()
                if len(mode_data) > 0:
                    icl_performance.extend(mode_data)
                    icl_modes.extend([mode] * len(mode_data))
            
            if icl_performance:
                perf_df = pd.DataFrame({'mode': icl_modes, 'kappa': icl_performance})
                sns.boxplot(data=perf_df, x='mode', y='kappa', ax=axes[0, 1])
                axes[0, 1].set_title("Cohen's κ by ICL Mode")
                axes[0, 1].set_ylabel("Cohen's κ")
                axes[0, 1].tick_params(axis='x', rotation=45)
        
        # K-support distribution
        if 'k_support' in self.df.columns:
            k_counts = self.df['k_support'].value_counts().sort_index()
            axes[1, 0].bar(k_counts.index.astype(str), k_counts.values, 
                          color=PLOT_CONFIG['colors']['accent'])
            axes[1, 0].set_title('K-Support Distribution')
            axes[1, 0].set_xlabel('K Support')
            axes[1, 0].set_ylabel('Number of Runs')
        
        # Performance vs K-support
        if 'k_support' in self.df.columns and 'kappa' in self.df.columns:
            # Filter to ICL runs only
            icl_runs = self.df[self.df['icl_mode'] != 'none']
            if len(icl_runs) > 0:
                axes[1, 1].scatter(icl_runs['k_support'], icl_runs['kappa'], 
                                 alpha=0.6, color=PLOT_CONFIG['colors']['primary'])
                axes[1, 1].set_title("Performance vs K-Support")
                axes[1, 1].set_xlabel('K Support')
                axes[1, 1].set_ylabel("Cohen's κ")
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'icl_analysis.png', **PLOT_CONFIG['figure'])
        plt.close()
    
    def _plot_data_completeness(self, output_dir: Path):
        """Plot data completeness overview."""
        
        key_fields = [
            'kappa', 'f1', 'accuracy', 'num_classes',
            'epochs', 'batch_size', 'lr', 'optimizer',
            'icl_mode', 'num_subjects'
        ]
        
        completeness_data = []
        labels = []
        
        for field in key_fields:
            if field in self.df.columns:
                non_null_pct = self.df[field].notna().mean() * 100
                completeness_data.append(non_null_pct)
                labels.append(field)
        
        if completeness_data:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            colors = ['green' if pct > 90 else 'orange' if pct > 70 else 'red' 
                     for pct in completeness_data]
            
            bars = ax.barh(labels, completeness_data, color=colors)
            ax.set_xlabel('Completeness (%)')
            ax.set_title('Data Completeness by Field')
            ax.axvline(90, color='green', linestyle='--', alpha=0.7, label='90% threshold')
            ax.axvline(70, color='orange', linestyle='--', alpha=0.7, label='70% threshold')
            
            # Add percentage labels
            for bar, pct in zip(bars, completeness_data):
                ax.text(pct + 1, bar.get_y() + bar.get_height()/2, 
                       f'{pct:.1f}%', va='center')
            
            ax.legend()
            ax.set_xlim(0, 105)
            plt.tight_layout()
            plt.savefig(output_dir / 'data_completeness.png', **PLOT_CONFIG['figure'])
            plt.close()


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description='Explore CBraMod structured run data')
    parser.add_argument('--data-file', default='Plot_Clean/data/all_runs.csv', 
                       help='Path to structured runs CSV file')
    parser.add_argument('--output-dir', default='Plot_Clean/outputs/exploration',
                       help='Output directory for exploration results')
    parser.add_argument('--create-plots', action='store_true',
                       help='Create summary visualization plots')
    
    args = parser.parse_args()
    
    # Initialize explorer
    explorer = DataExplorer(args.data_file)
    
    try:
        # Load data
        df = explorer.load_data()
        print(f"✅ Loaded {len(df)} runs from {args.data_file}")
        
        # Generate summary
        summary = explorer.generate_summary()
        
        # Print summary to console
        print("\n" + "="*60)
        print("DATA EXPLORATION SUMMARY")
        print("="*60)
        
        # Overview
        print(f"\n📊 OVERVIEW")
        print(f"Total runs: {summary['overview']['total_runs']}")
        print(f"Date range: {summary['overview']['date_range']['earliest']} to {summary['overview']['date_range']['latest']}")
        print(f"Unique run names: {summary['overview']['unique_names']}")
        print(f"Run states: {summary['overview']['run_states']}")
        
        # Validation
        print(f"\n✅ VALIDATION STATUS")
        validation = summary['validation_status']
        if 'status_distribution' in validation:
            for status, count in validation['status_distribution'].items():
                print(f"  {status}: {count}")
        print(f"Valid runs: {validation.get('valid_runs_percentage', 0):.1f}%")
        
        # Performance
        print(f"\n🎯 PERFORMANCE SUMMARY")
        perf = summary['performance_summary']
        for metric, stats in perf.items():
            print(f"{metric.upper()}:")
            print(f"  Count: {stats['count']}, Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            if 'categories' in stats:
                print(f"  Categories: {stats['categories']}")
        
        # ICL Summary
        print(f"\n🧠 ICL ANALYSIS")
        icl = summary['icl_summary']
        if 'mode_distribution' in icl:
            print(f"Mode distribution: {icl['mode_distribution']}")
        if 'performance_by_mode' in icl:
            print("Performance by mode:")
            for mode, perf in icl['performance_by_mode'].items():
                print(f"  {mode}: {perf['mean_kappa']:.3f} ± {perf['std_kappa']:.3f} (n={perf['count']})")
        
        # Data completeness
        print(f"\n📋 DATA COMPLETENESS")
        completeness = summary['data_completeness']
        for field, stats in completeness.items():
            print(f"{field}: {stats['percentage']:.1f}% ({stats['available']}/{stats['available'] + stats['missing']})")
        
        # Issues
        print(f"\n⚠️  POTENTIAL ISSUES")
        issues = summary['potential_issues']
        if issues:
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("  No major issues detected")
        
        # Save detailed summary
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_dir / 'data_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n💾 Detailed summary saved to {output_dir / 'data_summary.json'}")
        
        # Create plots if requested
        if args.create_plots:
            print("\n📈 Creating summary plots...")
            explorer.create_summary_plots(output_dir)
            print(f"📈 Plots saved to {output_dir}")
        
        print("\n✅ Data exploration completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Exploration failed: {e}")
        raise


if __name__ == '__main__':
    main()