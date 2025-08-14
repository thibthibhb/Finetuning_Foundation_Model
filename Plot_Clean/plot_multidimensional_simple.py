#!/usr/bin/env python3
"""
Simple Multidimensional Leaderboard - just the essential function.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import ast

# Simple config
FIGURE_DIR = Path("Plot_Clean/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

def flatten_wandb_data(df):
    """Extract metrics from summary/config strings."""
    flattened_rows = []
    
    for idx, row in df.iterrows():
        flat_row = row.to_dict()
        
        # Parse summary string
        if 'summary' in row and isinstance(row['summary'], str):
            try:
                summary = ast.literal_eval(row['summary'])
                if isinstance(summary, dict):
                    for k, v in summary.items():
                        flat_row[k] = v
            except:
                pass
        
        # Parse config string  
        if 'config' in row and isinstance(row['config'], str):
            try:
                config = ast.literal_eval(row['config'])
                if isinstance(config, dict):
                    for k, v in config.items():
                        if k not in flat_row:  # Don't override summary
                            flat_row[k] = v
            except:
                pass
        
        flattened_rows.append(flat_row)
    
    return pd.DataFrame(flattened_rows)

def plot_multidimensional_leaderboard(df, cohort_name="All Runs"):
    """Create multi-dimensional performance visualization - standalone version."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Multi-Dimensional Performance Analysis\n{cohort_name}',
                 fontsize=16, fontweight='bold')
    
    # Colors
    primary_color = '#2E86AB'
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6C7B7F']
    
    # 1. κ vs F1-macro (if both exist)
    kappa_col = None
    f1_col = None
    
    for col in ['test_kappa', 'kappa', 'val_kappa']:
        if col in df.columns and df[col].notna().sum() > 0:
            kappa_col = col
            break
    
    for col in ['test_f1', 'f1', 'val_f1']:
        if col in df.columns and df[col].notna().sum() > 0:
            f1_col = col
            break
    
    if kappa_col and f1_col:
        valid_data = df.dropna(subset=[kappa_col, f1_col])
        ax1.scatter(valid_data[kappa_col], valid_data[f1_col], 
                   alpha=0.6, s=60, c=primary_color)
        ax1.set_xlabel("Cohen's κ")
        ax1.set_ylabel('Macro F1')
        ax1.set_title('κ vs Macro F1')
        ax1.grid(True, alpha=0.3)
        
        # Add diagonal line (perfect correlation)
        min_val = min(ax1.get_xlim()[0], ax1.get_ylim()[0])
        max_val = min(ax1.get_xlim()[1], ax1.get_ylim()[1])
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect correlation')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'κ vs F1 data not available', ha='center', va='center',
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title('κ vs Macro F1 (Not Available)')
    
    # 2. Performance vs Training Epochs
    epochs_col = None
    for col in ['epochs', 'epoch', 'num_epochs']:
        if col in df.columns and df[col].notna().sum() > 0:
            epochs_col = col
            break
    
    if kappa_col and epochs_col:
        valid_epochs = df.dropna(subset=[kappa_col, epochs_col])
        
        # Color by ICL mode if available
        if 'icl_mode' in df.columns:
            icl_modes = valid_epochs['icl_mode'].fillna('none').unique()
            for i, mode in enumerate(icl_modes):
                mode_data = valid_epochs[valid_epochs['icl_mode'].fillna('none') == mode]
                ax2.scatter(mode_data[epochs_col], mode_data[kappa_col],
                           alpha=0.7, s=60, c=colors[i % len(colors)], 
                           label=mode if mode != 'none' else 'Baseline')
            ax2.legend()
        else:
            ax2.scatter(valid_epochs[epochs_col], valid_epochs[kappa_col],
                       alpha=0.7, s=60, c=primary_color)
        
        ax2.set_xlabel('Training Epochs')
        ax2.set_ylabel("Cohen's κ")
        ax2.set_title('Performance vs Training Duration')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Epochs vs κ data not available', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Performance vs Training Duration (Not Available)')
    
    # 3. Optimizer Comparison (if available)
    if kappa_col and 'optimizer' in df.columns:
        optimizer_data = df.dropna(subset=[kappa_col, 'optimizer'])
        optimizers = optimizer_data['optimizer'].unique()
        
        if len(optimizers) > 1:
            positions = range(len(optimizers))
            violin_data = [optimizer_data[optimizer_data['optimizer'] == opt][kappa_col].values 
                          for opt in optimizers]
            
            parts = ax3.violinplot(violin_data, positions, widths=0.7, showmeans=True)
            ax3.set_xticks(positions)
            ax3.set_xticklabels(optimizers, rotation=45)
            ax3.set_ylabel("Cohen's κ")
            ax3.set_title('Performance by Optimizer')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, f'Only one optimizer: {optimizers[0]}', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Performance by Optimizer')
    else:
        ax3.text(0.5, 0.5, 'Optimizer data not available', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Performance by Optimizer (Not Available)')
    
    # 4. Learning Rate vs Performance
    lr_col = None
    for col in ['lr', 'learning_rate', 'head_lr']:
        if col in df.columns and df[col].notna().sum() > 0:
            lr_col = col
            break
    
    if kappa_col and lr_col:
        lr_data = df.dropna(subset=[kappa_col, lr_col])
        if len(lr_data) > 0:
            if epochs_col and epochs_col in lr_data.columns:
                scatter = ax4.scatter(lr_data[lr_col], lr_data[kappa_col], 
                                     c=lr_data[epochs_col], s=60, alpha=0.7, cmap='viridis')
                cbar = plt.colorbar(scatter, ax=ax4)
                cbar.set_label('Training Epochs')
            else:
                ax4.scatter(lr_data[lr_col], lr_data[kappa_col], 
                           s=60, alpha=0.7, c=primary_color)
            
            ax4.set_xlabel('Learning Rate')
            ax4.set_ylabel("Cohen's κ")
            ax4.set_title('Performance vs Learning Rate')
            ax4.set_xscale('log')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No valid LR vs κ data', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Performance vs Learning Rate')
    else:
        ax4.text(0.5, 0.5, 'Learning rate data not available', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Performance vs Learning Rate (Not Available)')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = FIGURE_DIR / 'multidimensional_leaderboard_simple.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {plot_path}")
    
    return fig, plot_path

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Simple multidimensional leaderboard')
    parser.add_argument('--data-file', type=str, required=True,
                       help='Path to CSV file')
    parser.add_argument('--cohort', type=str, default='All Runs',
                       help='Cohort name for title')
    
    args = parser.parse_args()
    
    print(f"🎯 Simple multidimensional leaderboard")
    print(f"📊 Data: {args.data_file}")
    
    try:
        # Load and flatten data
        df = pd.read_csv(args.data_file)
        print(f"📈 Loaded {len(df)} runs")
        
        df_flat = flatten_wandb_data(df)
        print(f"📊 After flattening: {len(df_flat)} runs")
        
        # Generate plot
        fig, plot_path = plot_multidimensional_leaderboard(df_flat, args.cohort)
        
        print(f"✅ Multidimensional leaderboard complete!")
        print(f"📁 Plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()