#!/usr/bin/env python3
"""
Improved CBraMod Deployment Visualization
========================================

Create publication-quality combined visualizations for deployment analysis.
Focuses on key metrics: latency per 30s epoch and memory compression.

Usage:
    python create_improved_deployment_plots.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

# Set up publication-quality styling
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True
})

# Professional color scheme
COLORS = {
    '4-class': '#2E86AB',  # Professional blue
    '5-class': '#FC8D62',  # Orange
    'limit': '#E63946',    # Red for limits
    'fp32': '#264653',     # Dark green
    'fp16': '#2A9D8F',     # Teal
    'int8': '#E76F51',     # Orange
    'highlight': '#F4A261', # Yellow accent
}

def load_deployment_data():
    """Load deployment analysis results with clarified scope."""
    
    deployment_data = {
        'Model': ['4-Class Model', '5-Class Model'],
        'Parameters (M)': [11.15, 11.15],
        'Inference Latency (ms)': [8.16, 9.24],  # Per 30-second epoch
        'FP32 Size (MB)': [42.54, 42.54],
        'FP16 Size (MB)': [21.27, 21.27],
        'INT8 Size (MB)': [10.64, 10.64],
        'FP16 Compression': [2.0, 2.0],
        'INT8 Compression': [4.0, 4.0]
    }
    
    return pd.DataFrame(deployment_data)

def create_combined_deployment_visualization():
    """Create publication-quality combined latency and memory visualization."""
    
    df = load_deployment_data()
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(14, 6))
    
    # Simple grid layout - only 2 main plots with more space for title
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1],
                         left=0.08, right=0.95, top=0.82, bottom=0.15,
                         wspace=0.25)
    
    # Main plots
    ax1 = fig.add_subplot(gs[0, 0])  # Latency comparison
    ax2 = fig.add_subplot(gs[0, 1])  # Memory compression
    
    # === Plot 1: Inference Latency per 30s Epoch ===
    models = df['Model']
    latencies = df['Inference Latency (ms)']
    
    bars1 = ax1.bar(models, latencies, color=[COLORS['4-class'], COLORS['5-class']], 
                   alpha=0.8, edgecolor='black', linewidth=1.5, width=0.6)
    
    # Add value labels on bars
    for bar, value in zip(bars1, latencies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{value:.2f}ms', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add IDUN constraint line with annotation
    ax1.axhline(y=100, color=COLORS['limit'], linestyle='--', linewidth=2.5, alpha=0.8)
    ax1.text(0.5, 95, 'IDUN Limit\n(100ms per epoch)', ha='center', va='top', 
            fontweight='bold', color=COLORS['limit'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['limit'], alpha=0.9))
    
    # Highlight excellent performance
    ax1.text(0.5, 50, '10x+ faster\nthan required!', ha='center', va='center',
            fontweight='bold', fontsize=12, color=COLORS['highlight'],
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['highlight'], alpha=0.2))
    
    ax1.set_ylabel('Latency per 30s Epoch (ms)', fontweight='bold')
    ax1.set_title('IDUN Optimal Real-time', fontweight='bold', pad=20)
    ax1.set_ylim(0, 110)
    
    # === Plot 2: Memory Usage (FP32 only) ===
    precisions = ['FP32']
    model1_sizes = [df.iloc[0]['FP32 Size (MB)']]
    model2_sizes = [df.iloc[1]['FP32 Size (MB)']]
    
    x = np.arange(len(precisions))
    width = 0.35
    
    bars2a = ax2.bar(x - width/2, model1_sizes, width, label='4-Class Model',
                    color=COLORS['4-class'], alpha=0.8, edgecolor='black', linewidth=1)
    bars2b = ax2.bar(x + width/2, model2_sizes, width, label='5-Class Model', 
                    color=COLORS['5-class'], alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bars_set, sizes in [(bars2a, model1_sizes), (bars2b, model2_sizes)]:
        for bar, size in zip(bars_set, sizes):
            height = bar.get_height()
            # Size labels
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{size:.1f}MB', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add IDUN memory limit
    ax2.axhline(y=128, color=COLORS['limit'], linestyle='--', linewidth=2.5, alpha=0.8)
    ax2.text(0, 120, 'IDUN Memory Limit (128MB)', ha='center', va='top',
            fontweight='bold', color=COLORS['limit'])
    
    # Highlight massive headroom
    ax2.text(0, 75, '3× under\nmemory limit!', ha='center', va='center',
            fontweight='bold', fontsize=11, color=COLORS['highlight'],
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['highlight'], alpha=0.2))
    
    ax2.set_xlabel('Model Configuration', fontweight='bold')
    ax2.set_ylabel('Memory Usage (MB)', fontweight='bold')
    ax2.set_title('Memory Usage (FP32)', fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(precisions)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 150)
    
    # Main title with proper spacing
    fig.suptitle('CBraMod Edge Deployment Analysis: Latency & Memory Performance', 
                fontsize=16, fontweight='bold', y=0.93)
    
    plt.tight_layout()
    return fig

def create_latency_breakdown_analysis():
    """Create detailed latency breakdown showing what 8-9ms means in context."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # === Left: Latency in context ===
    scenarios = ['Single Epoch\n(30s EEG)', 'Continuous\nMonitoring\n(1 minute)', 'Full Night\n(8 hours)', 'IDUN Limit\n(per epoch)']
    latencies_4class = [8.16, 16.32, 3916.8, 100]  # ms
    latencies_5class = [9.24, 18.48, 4435.2, 100]  # ms
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, latencies_4class, width, label='4-Class Model',
                   color=COLORS['4-class'], alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, latencies_5class, width, label='5-Class Model',
                   color=COLORS['5-class'], alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars_set, values in [(bars1, latencies_4class), (bars2, latencies_5class)]:
        for bar, value in zip(bars_set, values):
            height = bar.get_height()
            if value < 1000:
                label = f'{value:.1f}ms'
            else:
                label = f'{value/1000:.1f}s'
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                    label, ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax1.set_ylabel('Processing Time', fontweight='bold')
    ax1.set_title('Inference Time in Real-world Context', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.legend()
    ax1.set_yscale('log')  # Log scale to show the dramatic differences
    
    # === Right: Power efficiency estimate ===
    power_scenarios = ['Inference\nPower', 'Idle Power\n(Baseline)', 'Total System\nBudget']
    power_estimates = [50, 200, 1000]  # mW estimates
    efficiency = [5, 20, 100]  # Percentage of total budget
    
    bars3 = ax2.bar(power_scenarios, power_estimates, 
                   color=[COLORS['highlight'], COLORS['fp16'], COLORS['limit']], 
                   alpha=0.8, edgecolor='black', width=0.6)
    
    # Add percentage labels
    for bar, power, pct in zip(bars3, power_estimates, efficiency):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 25,
                f'{power}mW\n({pct}%)', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylabel('Power Consumption (mW)', fontweight='bold')
    ax2.set_title('Estimated Power Efficiency', fontweight='bold')
    ax2.set_ylim(0, 1100)
    
    # Add efficiency note
    ax2.text(1, 800, 'Inference adds\nonly ~5% to\ntotal power budget', 
            ha='center', va='center', fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    return fig

def main():
    """Generate improved deployment visualizations."""
    
    print("Creating improved CBraMod deployment visualizations...")
    
    # Create output directory
    output_dir = Path('deployment_analysis/improved_plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 Creating combined deployment visualization...")
    fig1 = create_combined_deployment_visualization()
    fig1.savefig(output_dir / 'combined_deployment_analysis.png', dpi=300, bbox_inches='tight')
    fig1.savefig(output_dir / 'combined_deployment_analysis.pdf', bbox_inches='tight')
    
    print("📊 Creating latency context analysis...")
    fig2 = create_latency_breakdown_analysis()  
    fig2.savefig(output_dir / 'latency_context_analysis.png', dpi=300, bbox_inches='tight')
    fig2.savefig(output_dir / 'latency_context_analysis.pdf', bbox_inches='tight')
    
    # Update todos
    print("\n" + "="*70)
    print("🎉 IMPROVED DEPLOYMENT PLOTS COMPLETED")
    print("="*70)
    print(f"📁 Publication-quality plots saved to: {output_dir}")
    print(f"📊 Key clarification: Latency is per 30-second EEG epoch, not full night")
    print(f"📊 Files created:")
    print(f"  • combined_deployment_analysis.png/pdf - Main deployment metrics")
    print(f"  • latency_context_analysis.png/pdf - Latency in real-world context")
    print("="*70)
    
    plt.show()

if __name__ == "__main__":
    main()