#!/usr/bin/env python3
"""
Compare Head vs Weights performances using Cohen's kappa scores.
Creates a figure showing performance comparison between different head and weight configurations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import sys; sys.path.append("../style"); from figure_style import setup_figure_style, get_color, save_figure

def parse_contract_data(contract_str):
    """Parse the contract_data JSON string to extract model parameters."""
    try:
        if pd.isna(contract_str) or contract_str == '':
            return None
        
        contract_data = json.loads(contract_str)
        
        # Extract key parameters
        model_info = contract_data.get('model', {})
        results = contract_data.get('results', {})
        training = contract_data.get('training', {})
        
        # Simplify foundation_dir to main weight types
        foundation_dir = model_info.get('foundation_dir', '')
        if 'BEST__loss10527.31.pth' in foundation_dir:
            weight_type = 'BEST__loss10527.31.pth'
        elif 'pretrained_weights.pth' in foundation_dir:
            weight_type = 'pretrained_weights.pth'
        elif 'BEST__loss8698.99.pth' in foundation_dir:
            weight_type = 'BEST__loss8698.99.pth'
        else:
            weight_type = 'other'
        
        return {
            'head_type': model_info.get('head_type', None),
            'weight_type': weight_type,
            'foundation_dir': foundation_dir,
            'test_kappa': results.get('test_kappa', None),
            'val_kappa': results.get('val_kappa', None),
            'lr': training.get('lr', None),
            'epochs': training.get('epochs', None)
        }
    except (json.JSONDecodeError, TypeError) as e:
        return None

def load_and_process_data(csv_path):
    """Load CSV data and extract head/weights information."""
    df = pd.read_csv(csv_path)
    
    # Parse contract data for each row
    parsed_data = []
    for _, row in df.iterrows():
        parsed = parse_contract_data(row['contract_data'])
        if parsed and parsed['test_kappa'] is not None:
            parsed_data.append(parsed)
    
    if not parsed_data:
        raise ValueError("No valid data found in CSV file")
    
    return pd.DataFrame(parsed_data)

def create_comparison_plot(data_df):
    """Create comparison plot of Head types vs Weight types performance."""
    setup_figure_style()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Performance by head type
    if 'head_type' in data_df.columns and not data_df['head_type'].isna().all():
        head_data = data_df.dropna(subset=['head_type', 'test_kappa'])
        if not head_data.empty:
            # Group by head type
            head_grouped = head_data.groupby('head_type')['test_kappa']
            
            head_types = []
            kappa_values = []
            kappa_errors = []
            
            for head_type, group in head_grouped:
                head_types.append(head_type.title())
                kappa_values.append(group.mean())
                kappa_errors.append(group.std() / np.sqrt(len(group)))  # SEM
            
            # Use specific colors for each head type
            colors = ["#0072B2", "#FC8D62", "#323131"][:len(head_types)]
            
            # Create boxplot data
            boxplot_data = [group.values for head_type, group in head_grouped]
            
            bp1 = ax1.boxplot(boxplot_data, positions=range(len(head_types)), 
                             patch_artist=True, widths=0.6)
            
            # Color the boxes
            for patch, color in zip(bp1['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add individual points
            for i, (head_type, group) in enumerate(head_grouped):
                x_pos = i + np.random.normal(0, 0.05, len(group))
                ax1.scatter(x_pos, group.values, color=colors[i], 
                          s=30, alpha=0.1)
            
            ax1.set_xticks(range(len(head_types)))
            ax1.set_xticklabels(head_types)
            ax1.set_ylabel('Test Cohen\'s Kappa')
            ax1.set_title('Performance by Head Type')
            ax1.grid(True, alpha=0.3)
            
            # Add sample sizes in the middle of boxplots
            for i, (head_type, group) in enumerate(head_grouped):
                median_val = group.median()
                ax1.text(i, median_val, f'n={len(group)}', 
                        ha='center', va='center', fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 2: Performance by weight type
    if 'weight_type' in data_df.columns:
        weight_data = data_df.dropna(subset=['weight_type', 'test_kappa'])
        # Filter out 'other' weights to focus on main two types
        weight_data = weight_data[weight_data['weight_type'].isin(['BEST__loss10527.31.pth', 'pretrained_weights.pth'])]
        
        if not weight_data.empty:
            # Group by weight type
            weight_grouped = weight_data.groupby('weight_type')['test_kappa']
            
            weight_types = []
            kappa_values = []
            kappa_errors = []
            
            for weight_type, group in weight_grouped:
                # Simplify labels
                if 'BEST__loss10527.31' in weight_type:
                    label = 'BEST Loss\n(10527.31)'
                elif 'pretrained_weights' in weight_type:
                    label = 'Standard\nPretrained'
                else:
                    label = weight_type
                    
                weight_types.append(label)
                kappa_values.append(group.mean())
                kappa_errors.append(group.std() / np.sqrt(len(group)))  # SEM
            
            colors = ["#0072B2", "#FC8D62"][:len(weight_types)]
            
            # Create boxplot data
            boxplot_data = [group.values for weight_type, group in weight_grouped]
            
            bp2 = ax2.boxplot(boxplot_data, positions=range(len(weight_types)), 
                             patch_artist=True, widths=0.6)
            
            # Color the boxes
            for patch, color in zip(bp2['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add individual points
            for i, (weight_type, group) in enumerate(weight_grouped):
                x_pos = i + np.random.normal(0, 0.05, len(group))
                ax2.scatter(x_pos, group.values, color=colors[i], 
                          s=30, alpha=0.1)
            
            ax2.set_xticks(range(len(weight_types)))
            ax2.set_xticklabels(weight_types)
            ax2.set_ylabel('Test Cohen\'s Kappa')
            ax2.set_title('Performance by Weight Type')
            ax2.grid(True, alpha=0.3)
            
            # Add sample sizes in the middle of boxplots
            for i, (weight_type, group) in enumerate(weight_grouped):
                median_val = group.median()
                ax2.text(i, median_val, f'n={len(group)}', 
                        ha='center', va='center', fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    # Load data
    csv_path = Path("../data/all_runs.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return
    
    print("Loading and processing data...")
    try:
        data_df = load_and_process_data(csv_path)
        print(f"Loaded {len(data_df)} valid runs")
        
        # Print summary statistics
        if 'head_type' in data_df.columns:
            head_counts = data_df['head_type'].value_counts()
            print(f"Head types available: {dict(head_counts)}")
        
        if 'weight_type' in data_df.columns:
            weight_counts = data_df['weight_type'].value_counts()
            print(f"Weight types available: {dict(weight_counts)}")
        
        # Create plot
        print("Creating comparison plot...")
        fig = create_comparison_plot(data_df)
        
        # Save figure
        output_path = Path("../figures/heads_vs_weights_comparison")
        save_figure(fig, output_path)
        
        plt.show()
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()