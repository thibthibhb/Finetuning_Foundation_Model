#!/usr/bin/env python3
"""
Dataset Composition Visualization for CBraMod
Creates comprehensive visualization of dataset composition showing subjects, nights, epochs, and class distributions.

Usage:
    python Plot_Clean/plot_dataset_composition.py --out Plot_Clean/figures
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib.patches import Rectangle
import seaborn as sns

# Import consistent figure styling
import sys; sys.path.append("../style"); from figure_style import setup_figure_style, get_color, save_figure

# Sleep stage colors (using consistent color scheme)
STAGE_COLORS = {
    'Wake': '#E74C3C',    # Red
    'N1': '#F39C12',      # Orange  
    'N2': '#3498DB',      # Blue
    'N3': '#323131',      # Dark gray instead of green
    'REM': '#9B59B6'      # Purple
}

# Dataset data (from analyze_class_distributions.py output)
DATASET_DATA = {
    'IDUN_EEG (ORP)': {
        'subjects': 13, 'nights': 39, 'total_epochs': 31006,
        'Wake': {'count': 5072, 'percent': 16.4},
        'N1': {'count': 1336, 'percent': 4.3},
        'N2': {'count': 13254, 'percent': 42.7},
        'N3': {'count': 6100, 'percent': 19.7},
        'REM': {'count': 5244, 'percent': 16.9}
    },
    'OpenNeuro 2023': {
        'subjects': 10, 'nights': 40, 'total_epochs': 33112,
        'Wake': {'count': 3006, 'percent': 9.1},
        'N1': {'count': 2738, 'percent': 8.3},
        'N2': {'count': 15586, 'percent': 47.1},
        'N3': {'count': 5334, 'percent': 16.1},
        'REM': {'count': 6448, 'percent': 19.5}
    },
    'OpenNeuro 2019': {
        'subjects': 20, 'nights': 156, 'total_epochs': 147570,
        'Wake': {'count': 22178, 'percent': 15.0},
        'N1': {'count': 10842, 'percent': 7.3},
        'N2': {'count': 63016, 'percent': 42.7},
        'N3': {'count': 24994, 'percent': 16.9},
        'REM': {'count': 26540, 'percent': 18.0}
    },
    'OpenNeuro 2017': {
        'subjects': 9, 'nights': 36, 'total_epochs': 29640,
        'Wake': {'count': 5080, 'percent': 17.1},
        'N1': {'count': 2100, 'percent': 7.1},
        'N2': {'count': 12732, 'percent': 43.0},
        'N3': {'count': 4144, 'percent': 14.0},
        'REM': {'count': 5584, 'percent': 18.8}
    }
}

# Overall totals
TOTAL_DATA = {
    'subjects': 52, 'nights': 271, 'total_epochs': 241328,
    'Wake': {'count': 35336, 'percent': 14.6},
    'N1': {'count': 17016, 'percent': 7.1},
    'N2': {'count': 104588, 'percent': 43.3},
    'N3': {'count': 40572, 'percent': 16.8},
    'REM': {'count': 43816, 'percent': 18.2}
}

def create_dataset_overview_plot(ax):
    """Create stacked bar chart showing dataset overview (subjects, nights, epochs)."""
    datasets = ['IDUN_EEG', 'OpenNeuro\n2023', 'OpenNeuro\n2019', 'OpenNeuro\n2017']  # Shorter names
    full_names = list(DATASET_DATA.keys())
    subjects = [DATASET_DATA[d]['subjects'] for d in full_names]
    nights = [DATASET_DATA[d]['nights'] for d in full_names]
    epochs_k = [DATASET_DATA[d]['total_epochs'] / 1000 for d in full_names]  # Convert to thousands
    
    x = np.arange(len(datasets))
    width = 0.25
    
    # Create grouped bars
    ax.bar(x - width, subjects, width, label='Subjects', color=get_color('cbramod'), alpha=0.8)
    ax.bar(x, nights, width, label='Nights', color=get_color('4_class'), alpha=0.8)
    ax.bar(x + width, epochs_k, width, label='Epochs (×1000)', color=get_color('5_class'), alpha=0.8)
    
    # Add value labels on bars with better positioning
    max_val = max(max(subjects), max(nights), max(epochs_k))
    for i, (s, n, e) in enumerate(zip(subjects, nights, epochs_k)):
        ax.text(i - width, s + max_val*0.01, str(s), ha='center', va='bottom', fontweight='bold', fontsize=6)
        ax.text(i, n + max_val*0.01, str(n), ha='center', va='bottom', fontweight='bold', fontsize=6)
        ax.text(i + width, e + max_val*0.01, f'{e:.0f}k', ha='center', va='bottom', fontweight='bold', fontsize=6)
    
    ax.set_xlabel('Dataset', fontweight='bold', fontsize=8)
    ax.set_ylabel('Count', fontweight='bold', fontsize=8)
    ax.set_title('Dataset Overview', fontweight='bold', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=7)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max_val * 1.15)
    ax.tick_params(axis='both', which='major', labelsize=7)

def create_class_distribution_plot(ax):
    """Create stacked bar chart showing class distributions across datasets plus overall average."""
    datasets = ['IDUN_EEG', '2023', '2019', '2017', 'Average']  # Added Average
    full_names = list(DATASET_DATA.keys())
    stages = ['Wake', 'N1', 'N2', 'N3', 'REM']
    
    # Prepare data (include overall totals for average)
    all_data = list(DATASET_DATA.values()) + [TOTAL_DATA]
    bottom = np.zeros(len(datasets))
    
    for stage in stages:
        percentages = [d[stage]['percent'] for d in all_data]
        ax.bar(datasets, percentages, bottom=bottom, label=stage, 
               color=STAGE_COLORS[stage], alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Add percentage labels for all segments (all in white)
        for i, (dataset, pct) in enumerate(zip(datasets, percentages)):
            if pct > 3:  # Only label segments that have enough space
                y_pos = bottom[i] + pct/2
                ax.text(i, y_pos, f'{pct:.0f}%', ha='center', va='center', 
                       fontweight='bold', fontsize=6, color='white')
        
        bottom += percentages
    
    ax.set_xlabel('Dataset', fontweight='bold', fontsize=8)
    ax.set_ylabel('Percentage (%)', fontweight='bold', fontsize=8)
    ax.set_title('Sleep Stage Distribution', fontweight='bold', fontsize=9)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=7)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='both', which='major', labelsize=7)



def add_summary_table(fig):
    """Add summary statistics table to the figure."""
    # Create summary data - only subjects/nights/epochs info
    summary_text = f"""Dataset Summary: 52 subjects, 271 nights, 241,328 epochs (30s each)"""
    
    fig.text(0.5, 0.05, summary_text, fontsize=8, ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))

def main():
    parser = argparse.ArgumentParser(description='Dataset Composition Visualization')
    parser.add_argument('--out', required=True, help='Output directory for figures')
    
    args = parser.parse_args()
    
    print("Creating dataset composition visualization...")
    
    # Set up consistent styling
    setup_figure_style()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(9, 4))
    
    # Create layout with two panels side by side:
    # Dataset overview (left) | Class distributions with average (right) 
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], 
                          left=0.1, right=0.82, top=0.85, bottom=0.22, 
                          wspace=0.4)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Dataset overview (left)
    ax2 = fig.add_subplot(gs[0, 1])  # Class distributions with average (right)
    
    # Create plots
    create_dataset_overview_plot(ax1)
    create_class_distribution_plot(ax2)
    
    # Add main title
    fig.suptitle('CBraMod Dataset Composition Analysis', 
                fontsize=12, fontweight='bold', y=0.94)
    
    # Add summary table
    add_summary_table(fig)
    
    # Save figure
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_path = output_dir / 'dataset_composition'
    saved_files = save_figure(fig, base_path)
    
    print(f"Dataset composition visualization saved to: {saved_files}")
    
    plt.show()

if __name__ == '__main__':
    main()