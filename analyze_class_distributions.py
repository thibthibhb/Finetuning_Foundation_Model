#!/usr/bin/env python3
"""
Class Distribution Analysis for CBraMod Datasets

This script calculates the exact class distributions (Wake, N1, N2, N3, REM) 
across all datasets in the CBraMod project.

Results:
- IDUN dataset (ORP): 13 subjects, 39 nights
- OpenNeuro 2023: 10 subjects, 40 nights  
- OpenNeuro 2019: 20 subjects, 156 nights
- OpenNeuro 2017: 9 subjects, 36 nights
"""

import os
import numpy as np
from collections import Counter
from pathlib import Path
import pandas as pd

# Sleep stage class names
STAGE_NAMES = {
    0: 'Wake',
    1: 'N1', 
    2: 'N2',
    3: 'N3',
    4: 'REM'
}

def analyze_dataset_distribution(dataset_path, dataset_name):
    """Analyze class distribution for a single dataset."""
    label_dir = Path(dataset_path) / 'label_npy'
    
    if not label_dir.exists():
        print(f"❌ Label directory not found: {label_dir}")
        return None
        
    all_labels = []
    file_count = 0
    
    print(f"\n🔍 Analyzing {dataset_name}...")
    
    # Load all label files
    for label_file in sorted(label_dir.glob('*.npy')):
        try:
            labels = np.load(label_file)
            all_labels.extend(labels.flatten())
            file_count += 1
            
            if file_count <= 3:  # Show first few files for verification
                unique_labels, counts = np.unique(labels, return_counts=True)
                print(f"  📄 {label_file.name}: {len(labels)} epochs, classes: {dict(zip(unique_labels, counts))}")
                
        except Exception as e:
            print(f"  ❌ Error loading {label_file}: {e}")
            continue
    
    if not all_labels:
        print(f"  ❌ No valid labels found in {dataset_name}")
        return None
    
    # Calculate overall distribution
    class_counts = Counter(all_labels)
    total_epochs = len(all_labels)
    
    print(f"  ✅ Loaded {file_count} files, {total_epochs:,} total epochs")
    
    # Create detailed results
    results = {
        'dataset': dataset_name,
        'total_files': file_count,
        'total_epochs': total_epochs,
    }
    
    # Add class-specific counts and percentages
    for class_id in range(5):  # 0-4 for Wake, N1, N2, N3, REM
        stage_name = STAGE_NAMES[class_id]
        count = class_counts.get(class_id, 0)
        percentage = (count / total_epochs) * 100 if total_epochs > 0 else 0
        
        results[f'{stage_name}_count'] = count
        results[f'{stage_name}_percent'] = percentage
        
        print(f"    {stage_name}: {count:,} epochs ({percentage:.1f}%)")
    
    return results

def main():
    """Main analysis function."""
    print("🧠 CBraMod Dataset Class Distribution Analysis")
    print("=" * 50)
    
    # Define dataset paths
    base_path = Path("/root/cbramod/CBraMod/data/datasets/final_dataset")
    
    datasets = {
        'IDUN_EEG (ORP)': base_path / 'ORP',
        'OpenNeuro_2023': base_path / '2023_Open_N', 
        'OpenNeuro_2019': base_path / '2019_Open_N',
        'OpenNeuro_2017': base_path / '2017_Open_N'
    }
    
    all_results = []
    
    # Analyze each dataset
    for dataset_name, dataset_path in datasets.items():
        result = analyze_dataset_distribution(dataset_path, dataset_name)
        if result:
            all_results.append(result)
    
    # Create summary table
    if all_results:
        print("\n📊 SUMMARY TABLE")
        print("=" * 80)
        
        df = pd.DataFrame(all_results)
        
        # Display basic info
        print("\nDataset Overview:")
        basic_cols = ['dataset', 'total_files', 'total_epochs']
        print(df[basic_cols].to_string(index=False))
        
        # Display class distributions
        print("\nClass Distributions (counts):")
        class_cols = ['dataset'] + [f'{stage}_count' for stage in ['Wake', 'N1', 'N2', 'N3', 'REM']]
        print(df[class_cols].to_string(index=False))
        
        print("\nClass Distributions (percentages):")
        percent_cols = ['dataset'] + [f'{stage}_percent' for stage in ['Wake', 'N1', 'N2', 'N3', 'REM']]
        percent_df = df[percent_cols].copy()
        for col in percent_cols[1:]:
            percent_df[col] = percent_df[col].map('{:.1f}%'.format)
        print(percent_df.to_string(index=False))
        
        # Calculate totals
        print("\n🌍 GRAND TOTALS:")
        total_files = df['total_files'].sum()
        total_epochs = df['total_epochs'].sum()
        print(f"Total files across all datasets: {total_files}")
        print(f"Total epochs across all datasets: {total_epochs:,}")
        
        # Overall class distribution
        print("\nOverall class distribution across all datasets:")
        for stage in ['Wake', 'N1', 'N2', 'N3', 'REM']:
            total_count = df[f'{stage}_count'].sum()
            total_percent = (total_count / total_epochs) * 100
            print(f"  {stage}: {total_count:,} epochs ({total_percent:.1f}%)")
        
        # Save results
        output_file = "dataset_class_distributions.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        # Subject and night counts (from previous analysis)
        print("\n👥 SUBJECT & NIGHT COUNTS:")
        subject_info = {
            'IDUN_EEG (ORP)': {'subjects': 13, 'nights': 39},
            'OpenNeuro_2023': {'subjects': 10, 'nights': 40}, 
            'OpenNeuro_2019': {'subjects': 20, 'nights': 156},
            'OpenNeuro_2017': {'subjects': 9, 'nights': 36}
        }
        
        for dataset, info in subject_info.items():
            print(f"  {dataset}: {info['subjects']} subjects, {info['nights']} nights")
            
    else:
        print("❌ No valid datasets found for analysis")

if __name__ == "__main__":
    main()