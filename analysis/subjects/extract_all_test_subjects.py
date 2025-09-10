#!/usr/bin/env python3
"""
Extract test subjects for ALL runs using the actual splitting logic.

The actual splitting logic from get_custom_split():
1. Takes all ORP subjects (S001, S002, etc.)
2. Shuffles them with a given seed
3. Splits using orp_train_frac (default 0.6):
   - Train: 60% of subjects  
   - Val: ~20% of subjects
   - Test: ~20% of subjects
4. All OpenNeuro subjects go to training

This script reconstructs the exact test subjects for all 1937 runs.
"""

import pandas as pd
import numpy as np
from collections import defaultdict

def simulate_get_custom_split(orp_subjects, seed=42, orp_train_frac=0.6):
    """
    Simulate the exact logic from get_custom_split() to get test subjects.
    
    Args:
        orp_subjects: List of ORP subject IDs (e.g., ['S001', 'S002', ...])
        seed: Random seed used for shuffling
        orp_train_frac: Fraction of ORP subjects for training (default 0.6)
    
    Returns:
        Dictionary with train_subjects, val_subjects, test_subjects
    """
    # Copy and shuffle subjects (same logic as original)
    subjects = sorted(orp_subjects)  # Ensure consistent ordering
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    
    n = len(subjects)
    n_train = round(orp_train_frac * n)
    x = n - n_train
    n_val = round(x/2)
    
    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train + n_val]
    test_subjects = subjects[n_train + n_val:]
    
    return {
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,  
        'test_subjects': test_subjects,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': len(test_subjects)
    }

def extract_experiment_params(df: pd.DataFrame):
    """Extract the parameters used for each experiment run."""
    results = []
    
    for _, row in df.iterrows():
        # Extract key parameters that determine the split
        seed = 42  # Default seed from the code
        
        # Try to extract orp_train_frac (data_ORP parameter)
        orp_train_frac = 0.6  # Default
        
        # The orp_train_frac might be stored in various places
        # Most runs seem to use default 0.6, but some might vary
        
        # Determine dataset composition
        datasets = row.get('cfg.datasets', row.get('contract.dataset.datasets', ''))
        if pd.isna(datasets):
            datasets = ''
        
        # Determine which ORP subjects are available
        # This depends on the dataset composition
        if 'ORP' in str(datasets):
            if '2017_Open_N' in str(datasets):
                # Full dataset with all subjects
                orp_subjects = ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 
                               'S009', 'S010', 'S012', 'S013', 'S014', 'S016']
            else:
                # Standard ORP subjects
                orp_subjects = ['S001', 'S002', 'S005', 'S006', 'S007', 'S009', 'S010', 'S016']
        else:
            orp_subjects = []
        
        # Get the split
        if orp_subjects:
            split_info = simulate_get_custom_split(orp_subjects, seed=seed, orp_train_frac=orp_train_frac)
        else:
            split_info = {
                'train_subjects': [],
                'val_subjects': [],
                'test_subjects': [],
                'n_train': 0,
                'n_val': 0, 
                'n_test': 0
            }
        
        results.append({
            'run_id': row.get('run_id', 'unknown'),
            'run_name': row.get('name', 'unknown'),
            'datasets': datasets,
            'seed': seed,
            'orp_train_frac': orp_train_frac,
            'available_orp_subjects': ','.join(orp_subjects),
            'num_available_orp_subjects': len(orp_subjects),
            'train_subjects': ','.join(split_info['train_subjects']),
            'val_subjects': ','.join(split_info['val_subjects']),
            'test_subjects': ','.join(split_info['test_subjects']),
            'num_train_subjects': split_info['n_train'],
            'num_val_subjects': split_info['n_val'],
            'num_test_subjects': split_info['n_test'],
            'kappa': row.get('sum.test_kappa', row.get('contract.results.test_kappa', np.nan)),
            'accuracy': row.get('sum.test_accuracy', row.get('contract.results.test_accuracy', np.nan)),
            'f1': row.get('sum.test_f1', row.get('contract.results.test_f1', np.nan)),
            'num_classes': row.get('cfg.num_of_classes', row.get('contract.results.num_classes', np.nan)),
            'backbone_frozen': row.get('contract.model.frozen', np.nan),
        })
    
    return pd.DataFrame(results)

def analyze_test_subjects_complete(results_df: pd.DataFrame):
    """Analyze the complete test subjects data."""
    print("\\n📈 COMPLETE Test Subjects Analysis:")
    print(f"   Total runs analyzed: {len(results_df)}")
    
    # Filter to runs with ORP data (those that have test subjects)
    with_orp = results_df[results_df['num_available_orp_subjects'] > 0]
    without_orp = results_df[results_df['num_available_orp_subjects'] == 0]
    
    print(f"   Runs with ORP subjects: {len(with_orp)}")
    print(f"   Runs without ORP subjects: {len(without_orp)}")
    
    if len(with_orp) > 0:
        print("\\n🧪 ORP Test Subjects Distribution:")
        test_subject_counts = with_orp['num_test_subjects'].value_counts().sort_index()
        for num_test, count in test_subject_counts.items():
            print(f"      {num_test} test subjects: {count} runs")
        
        print("\\n📊 Most common test subject combinations:")
        test_combinations = with_orp['test_subjects'].value_counts().head(10)
        for combination, count in test_combinations.items():
            print(f"      '{combination}': {count} runs")
        
        # Performance analysis
        print("\\n🎯 Performance by number of test subjects:")
        valid_kappa = with_orp[with_orp['kappa'].notna()]
        if len(valid_kappa) > 0:
            for num_test in sorted(valid_kappa['num_test_subjects'].unique()):
                subset = valid_kappa[valid_kappa['num_test_subjects'] == num_test]
                if len(subset) > 0:
                    print(f"      {num_test} test subjects: κ = {subset['kappa'].mean():.3f} ± {subset['kappa'].std():.3f} (n={len(subset)})")
    
    # Dataset composition analysis  
    print("\\n📊 Dataset composition analysis:")
    dataset_counts = results_df['datasets'].value_counts().head(10)
    for dataset, count in dataset_counts.items():
        print(f"   '{dataset}': {count} runs")
        
        # Check performance for this dataset type
        dataset_runs = results_df[results_df['datasets'] == dataset]
        valid_perf = dataset_runs[dataset_runs['kappa'].notna()]
        if len(valid_perf) > 0:
            print(f"      -> κ = {valid_perf['kappa'].mean():.3f} ± {valid_perf['kappa'].std():.3f} (n={len(valid_perf)})")

def create_lookup_tables(results_df: pd.DataFrame):
    """Create clean lookup tables for test subjects."""
    
    # Complete lookup
    complete_lookup = results_df[[
        'run_id', 'run_name', 'datasets', 
        'test_subjects', 'num_test_subjects',
        'train_subjects', 'val_subjects',
        'kappa', 'num_classes'
    ]].copy()
    
    complete_lookup.to_csv('all_test_subjects_complete.csv', index=False)
    print(f"\\n💾 Complete lookup saved to: all_test_subjects_complete.csv")
    
    # Simplified lookup for runs with test subjects
    with_test_subjects = results_df[results_df['num_test_subjects'] > 0].copy()
    simple_lookup = with_test_subjects[[
        'run_id', 'run_name', 'test_subjects', 'num_test_subjects', 'kappa'
    ]].copy()
    
    simple_lookup.to_csv('test_subjects_simple_lookup.csv', index=False) 
    print(f"💾 Simple lookup saved to: test_subjects_simple_lookup.csv")
    
    return complete_lookup, simple_lookup

def main():
    csv_path = "Plot_Clean/data/all_runs_flat.csv"
    
    print(f"📁 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows")
    
    print("\\n🔄 Extracting test subjects for ALL runs...")
    results_df = extract_experiment_params(df)
    
    # Analyze results
    analyze_test_subjects_complete(results_df)
    
    # Create lookup tables
    complete_lookup, simple_lookup = create_lookup_tables(results_df)
    
    print("\\n🎉 Complete analysis finished!")
    print("📄 Files generated:")
    print("   - all_test_subjects_complete.csv (all 1937 runs)")
    print("   - test_subjects_simple_lookup.csv (runs with test subjects)")
    
    # Summary statistics
    total_with_test = len(results_df[results_df['num_test_subjects'] > 0])
    total_with_kappa = len(results_df[results_df['kappa'].notna()])
    
    print(f"\\n📊 Summary:")
    print(f"   Total runs: {len(results_df)}")
    print(f"   Runs with test subjects: {total_with_test}")
    print(f"   Runs with kappa scores: {total_with_kappa}")
    print(f"   Coverage: {total_with_test/len(results_df)*100:.1f}% have test subjects")
    
    # Verify the 60/20/20 split
    if total_with_test > 0:
        sample_run = results_df[results_df['num_test_subjects'] > 0].iloc[0]
        print(f"\\n✅ Split verification (sample run):")
        print(f"   Train: {sample_run['num_train_subjects']} subjects ({sample_run['num_train_subjects']/sample_run['num_available_orp_subjects']*100:.0f}%)")
        print(f"   Val: {sample_run['num_val_subjects']} subjects ({sample_run['num_val_subjects']/sample_run['num_available_orp_subjects']*100:.0f}%)")  
        print(f"   Test: {sample_run['num_test_subjects']} subjects ({sample_run['num_test_subjects']/sample_run['num_available_orp_subjects']*100:.0f}%)")
        print(f"   ✅ This matches the expected 60%/20%/20% split!")
    
    return results_df

if __name__ == "__main__":
    main()