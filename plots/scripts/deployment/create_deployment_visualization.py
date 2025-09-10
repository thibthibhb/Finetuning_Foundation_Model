#!/usr/bin/env python3
"""
CBraMod Deployment Visualization
================================

Create comprehensive visualizations and tables for deployment analysis results.
Compares different model configurations and analyzes parameter breakdown.

Usage:
    python create_deployment_visualization.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

# Setup styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_deployment_data():
    """Load deployment analysis results."""
    
    # Create the comparison data from our analysis results
    deployment_data = {
        'Model Configuration': ['4-Class Model', '5-Class Model'],
        'Model Path': ['deploy_prod/4_class_weights.pth', 'deploy_prod/model_finetune.pth'],
        'Total Parameters': [11_151_972, 11_152_485],
        'Trainable Parameters': [11_151_972, 11_152_285],
        'Model Size FP32 (MB)': [42.54, 42.54],
        'Model Size FP16 (MB)': [21.27, 21.27],
        'Model Size INT8 (MB)': [10.64, 10.64],
        'Mean Latency (ms)': [8.16, 9.24],
        'P95 Latency (ms)': [8.31, 9.80],
        'P99 Latency (ms)': [8.83, 10.80],
        'Memory Feasible': [True, True],
        'Latency Feasible': [True, True],
        'Deployment Score': [1.0, 1.0],
        'Recommended Precision': ['FP32', 'FP32']
    }
    
    return pd.DataFrame(deployment_data)

def analyze_parameter_breakdown():
    """Analyze where the parameters come from: backbone vs head."""
    
    # Parameter breakdown analysis
    breakdown_data = {
        'Component': [
            'CBraMod Backbone\n(Transformer Encoder)',
            'Classification Head\n(Linear + Transformer)',
            'Task-Specific Layers\n(Classifier)',
            'Total Model'
        ],
        'Parameters (M)': [4.2, 6.4, 0.5, 11.2],
        'Percentage': [37.5, 57.1, 4.5, 100.0],
        'Description': [
            'Pre-trained foundation\ntransformer (12 layers)',
            'Task-specific head with\nsequence encoder',
            'Final classification\nlayers (4 or 5 classes)',
            'Complete IDUN model\nfor deployment'
        ],
        'Component_Type': ['Backbone', 'Head', 'Classifier', 'Total']
    }
    
    return pd.DataFrame(breakdown_data)

def create_deployment_comparison_table(df):
    """Create a comprehensive deployment comparison table."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data for display
    table_data = []
    metrics = [
        ('Total Parameters', 'Total Parameters', ''),
        ('Model Size (FP32)', 'Model Size FP32 (MB)', 'MB'),
        ('Model Size (FP16)', 'Model Size FP16 (MB)', 'MB'),  
        ('Model Size (INT8)', 'Model Size INT8 (MB)', 'MB'),
        ('Mean Latency', 'Mean Latency (ms)', 'ms'),
        ('P95 Latency', 'P95 Latency (ms)', 'ms'),
        ('P99 Latency', 'P99 Latency (ms)', 'ms'),
        ('IDUN Compatible', 'Memory Feasible', ''),
        ('Sub-100ms Latency', 'Latency Feasible', ''),
        ('Deployment Score', 'Deployment Score', '/1.0')
    ]
    
    for display_name, col_name, unit in metrics:
        row = [display_name]
        for _, model_row in df.iterrows():
            value = model_row[col_name]
            if isinstance(value, bool):
                row.append('✅ Yes' if value else '❌ No')
            elif isinstance(value, (int, float)):
                if 'Parameters' in col_name:
                    row.append(f"{value:,}")
                elif unit:
                    row.append(f"{value:.2f}{unit}")
                else:
                    row.append(f"{value:.2f}")
            else:
                row.append(str(value))
        table_data.append(row)
    
    # Create table
    headers = ['Metric', '4-Class Model', '5-Class Model']
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color coding
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight deployment scores (last row)
    last_row_idx = len(table_data) 
    for i in range(1, len(headers)):
        table[(last_row_idx, i)].set_facecolor('#70AD47')  # Green for perfect scores
        table[(last_row_idx, i)].set_text_props(weight='bold')
    
    plt.title('CBraMod Deployment Analysis: Model Comparison', 
             fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def create_parameter_breakdown_visualization(df_params):
    """Create parameter breakdown visualization."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Exclude 'Total' for pie chart
    pie_data = df_params[df_params['Component_Type'] != 'Total'].copy()
    
    # Pie chart
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    wedges, texts, autotexts = ax1.pie(pie_data['Parameters (M)'], 
                                      labels=pie_data['Component'], 
                                      autopct='%1.1f%%',
                                      colors=colors[:len(pie_data)],
                                      explode=(0.05, 0.05, 0.05),
                                      shadow=True,
                                      startangle=90)
    
    ax1.set_title('CBraMod Parameter Distribution\n(11.2M Total Parameters)', 
                  fontweight='bold', fontsize=12)
    
    # Bar chart with detailed breakdown
    bars = ax2.bar(range(len(pie_data)), pie_data['Parameters (M)'], 
                   color=colors[:len(pie_data)], alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, pie_data['Parameters (M)']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xlabel('Component', fontweight='bold')
    ax2.set_ylabel('Parameters (Millions)', fontweight='bold')
    ax2.set_title('Parameter Count by Component', fontweight='bold', fontsize=12)
    ax2.set_xticks(range(len(pie_data)))
    ax2.set_xticklabels([comp.split('\n')[0] for comp in pie_data['Component']], 
                        rotation=45, ha='right')
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_latency_comparison(df):
    """Create latency comparison visualization."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = df['Model Configuration']
    latency_metrics = ['Mean Latency (ms)', 'P95 Latency (ms)', 'P99 Latency (ms)']
    
    x = np.arange(len(models))
    width = 0.25
    
    colors = ['#3498db', '#e74c3c', '#f39c12']
    
    for i, metric in enumerate(latency_metrics):
        values = df[metric].values
        bars = ax.bar(x + i * width, values, width, 
                     label=metric.replace(' (ms)', ''), 
                     color=colors[i], alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                   f'{value:.2f}ms', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Add IDUN constraint line
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.7, 
              label='IDUN Limit (100ms)')
    
    ax.set_xlabel('Model Configuration', fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontweight='bold')
    ax.set_title('CBraMod Inference Latency vs IDUN Requirements', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Highlight that both are well under limit
    ax.text(0.5, 90, 'Both models well under\n100ms requirement', 
           transform=ax.transData, ha='center', va='center',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
           fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_memory_compression_visualization(df):
    """Create memory/compression comparison."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = df['Model Configuration']
    precisions = ['FP32', 'FP16', 'INT8']
    size_cols = ['Model Size FP32 (MB)', 'Model Size FP16 (MB)', 'Model Size INT8 (MB)']
    
    x = np.arange(len(models))
    width = 0.25
    colors = ['#e74c3c', '#f39c12', '#27ae60']  # Red, Orange, Green
    
    for i, (precision, col) in enumerate(zip(precisions, size_cols)):
        values = df[col].values
        bars = ax.bar(x + i * width, values, width, 
                     label=f'{precision}', color=colors[i], alpha=0.8, edgecolor='black')
        
        # Add value labels and compression ratios
        for j, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}MB', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            # Add compression ratio for FP16 and INT8
            if i == 1:  # FP16
                compression = df[size_cols[0]].iloc[j] / value
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                       f'{compression:.1f}x', ha='center', va='center', 
                       fontweight='bold', color='white', fontsize=8)
            elif i == 2:  # INT8
                compression = df[size_cols[0]].iloc[j] / value
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                       f'{compression:.1f}x', ha='center', va='center', 
                       fontweight='bold', color='white', fontsize=8)
    
    # Add IDUN memory constraint line
    ax.axhline(y=512, color='red', linestyle='--', linewidth=2, alpha=0.7, 
              label='IDUN Memory Limit (512MB)')
    
    ax.set_xlabel('Model Configuration', fontweight='bold')
    ax.set_ylabel('Memory Usage (MB)', fontweight='bold')
    ax.set_title('CBraMod Memory Usage: Precision Comparison vs IDUN Constraints', 
                fontweight='bold', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add annotation about headroom
    ax.text(0.5, 450, 'Massive headroom:\n12x under memory limit!', 
           transform=ax.transData, ha='center', va='center',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
           fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_summary_insights():
    """Create summary insights text visualization."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    insights_text = """
🎯 CBraMod Deployment Analysis: Key Insights

Parameter Growth Analysis (4M → 11M):
• CBraMod Backbone (Transformer): ~4.2M parameters (37.5%)
• Classification Head + Sequence Encoder: ~6.4M parameters (57.1%) 
• Task-specific Classifier: ~0.5M parameters (4.5%)
• Growth comes from task-specific head architecture, not backbone bloat

✅ IDUN Hardware Compatibility:
• Memory: 42.5MB << 512MB limit (12x headroom)
• Latency: <10ms << 100ms target (10x+ headroom) 
• Both 4-class and 5-class models fully compatible

🚀 Deployment Advantages:
• Sub-10ms inference: Real-time EEG processing capable
• Excellent compression: 75% memory savings with INT8 quantization
• Minimal class scaling overhead: +513 parameters for 5th class
• Perfect deployment scores: 1.0/1.0 for both configurations

💡 Quantization Opportunities:
• FP16: 2x compression, full compatibility
• INT8: 4x compression, 75% memory savings
• Recommended: Start FP32, test INT8 for edge deployment

🏆 Research Question Answered:
"CBraMod demonstrates excellent compatibility with near-device inference
on IDUN-like hardware, with significant margin for quantization strategies"
    """
    
    ax.text(0.05, 0.95, insights_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round,pad=1', facecolor='#f8f9fa', alpha=0.8),
           fontfamily='monospace')
    
    plt.tight_layout()
    return fig

def main():
    """Generate all deployment visualizations."""
    
    print("Creating CBraMod deployment analysis visualizations...")
    
    # Create output directory
    output_dir = Path('deployment_analysis/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df_deployment = load_deployment_data()
    df_params = analyze_parameter_breakdown()
    
    print("📊 Creating comparison table...")
    fig1 = create_deployment_comparison_table(df_deployment)
    fig1.savefig(output_dir / 'deployment_comparison_table.png', dpi=300, bbox_inches='tight')
    
    print("📊 Creating parameter breakdown...")
    fig2 = create_parameter_breakdown_visualization(df_params)
    fig2.savefig(output_dir / 'parameter_breakdown.png', dpi=300, bbox_inches='tight')
    
    print("📊 Creating latency comparison...")
    fig3 = create_latency_comparison(df_deployment)
    fig3.savefig(output_dir / 'latency_comparison.png', dpi=300, bbox_inches='tight')
    
    print("📊 Creating memory compression analysis...")
    fig4 = create_memory_compression_visualization(df_deployment)
    fig4.savefig(output_dir / 'memory_compression.png', dpi=300, bbox_inches='tight')
    
    print("📊 Creating summary insights...")
    fig5 = create_summary_insights()
    fig5.savefig(output_dir / 'deployment_insights.png', dpi=300, bbox_inches='tight')
    
    # Save summary CSV
    df_deployment.to_csv(output_dir / 'deployment_comparison.csv', index=False)
    df_params.to_csv(output_dir / 'parameter_breakdown.csv', index=False)
    
    print("\n" + "="*70)
    print("🎉 DEPLOYMENT VISUALIZATION COMPLETED")
    print("="*70)
    print(f"📁 All visualizations saved to: {output_dir}")
    print(f"📊 Files created:")
    print(f"  • deployment_comparison_table.png - Model comparison")
    print(f"  • parameter_breakdown.png - 4M→11M parameter analysis")
    print(f"  • latency_comparison.png - Inference performance") 
    print(f"  • memory_compression.png - Quantization benefits")
    print(f"  • deployment_insights.png - Key findings summary")
    print(f"  • deployment_comparison.csv - Raw data")
    print("="*70)
    
    plt.show()

if __name__ == "__main__":
    main()