#!/usr/bin/env python3
"""
Extract test subjects using LOSO (Leave-One-Subject-Out) logic.

Logic:
- cfg.subject_id = the subject used for TRAINING  
- Test subjects = ALL OTHER subjects in that dataset
- This is the standard LOSO cross-validation approach

This script reconstructs the test subjects for each run based on:
1. The dataset used (cfg.datasets)
2. The training subject (cfg.subject_id) 
3. The total subjects available per dataset
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# Known subject mappings for common datasets
DATASET_SUBJECTS = {
    'ORP': ['S001', 'S002','S003', 'S004', 'S005', 'S006', 'S007', 'S009', 'S010', 'S012', 'S013', 'S014','S016'],  # 8 subjects
    # Add more datasets as you discover their subject lists
}

def infer_dataset_subjects(df: pd.DataFrame) -> dict:
    """
    Infer the complete subject list for each dataset by analyzing all runs.
    """
    print("🔍 Inferring complete subject lists for each dataset...")
    
    dataset_subjects = defaultdict(set)
    
    # Group by dataset and collect all training subjects used
    for _, row in df.iterrows():
        if pd.notna(row.get('cfg.datasets')) and pd.notna(row.get('cfg.subject_id')):
            dataset = row['cfg.datasets'].strip()
            subject = row['cfg.subject_id']
            dataset_subjects[dataset].add(subject)
    
    # Convert to regular dict with sorted lists
    result = {}
    for dataset, subjects in dataset_subjects.items():
        result[dataset] = sorted(list(subjects))
        print(f"   {dataset}: {len(result[dataset])} subjects - {result[dataset]}")
    
    return result

def get_test_subjects_for_run(dataset: str, training_subject: str, all_subjects: dict) -> list:
    """
    Get the test subjects for a specific run using LOSO logic.
    
    Args:
        dataset: The dataset name
        training_subject: The subject used for training
        all_subjects: Dict mapping dataset -> list of all subjects
    
    Returns:
        List of test subjects (all subjects except the training one)
    """
    if dataset not in all_subjects:
        return []
    
    test_subjects = [s for s in all_subjects[dataset] if s != training_subject]
    return test_subjects

def extract_test_subjects_loso(csv_path: str):
    """
    Extract test subjects for each run using LOSO logic.
    """
    print(f"📁 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows")
    
    # Filter to rows that have both dataset and subject information
    valid_rows = df[df['cfg.datasets'].notna() & df['cfg.subject_id'].notna()]
    print(f"📊 Rows with dataset & subject info: {len(valid_rows)}")
    
    if len(valid_rows) == 0:
        print("❌ No rows with both cfg.datasets and cfg.subject_id")
        return None
    
    # Infer complete subject lists for each dataset
    dataset_subjects = infer_dataset_subjects(valid_rows)
    
    # Extract test subjects for each run
    results = []
    for _, row in valid_rows.iterrows():
        dataset = row['cfg.datasets'].strip()
        training_subject = row['cfg.subject_id']
        
        # Get test subjects using LOSO logic
        test_subjects = get_test_subjects_for_run(dataset, training_subject, dataset_subjects)
        
        # Create result entry
        result = {
            'run_id': row.get('run_id', 'unknown'),
            'run_name': row.get('name', 'unknown'),
            'dataset': dataset,
            'training_subject': training_subject,
            'test_subjects': test_subjects,
            'num_test_subjects': len(test_subjects),
            'test_subjects_str': ','.join(test_subjects),
            'kappa': row.get('sum.test_kappa', row.get('contract.results.test_kappa', np.nan)),
            'accuracy': row.get('sum.test_accuracy', row.get('contract.results.test_accuracy', np.nan)),
            'f1': row.get('sum.test_f1', row.get('contract.results.test_f1', np.nan)),
            'num_classes': row.get('cfg.num_of_classes', row.get('contract.results.num_classes', np.nan)),
            'backbone_frozen': row.get('contract.model.frozen', False),
            'epochs': row.get('contract.training.epochs', np.nan),
        }
        results.append(result)
    
    results_df = pd.DataFrame(results)
    print(f"✅ Extracted test subjects for {len(results_df)} runs")
    
    return results_df, dataset_subjects

def analyze_test_subjects(results_df: pd.DataFrame, dataset_subjects: dict):
    """Analyze the extracted test subjects."""
    print("\\n📈 Test subjects analysis:")
    
    # Distribution of test subjects per run
    test_counts = results_df['num_test_subjects'].value_counts().sort_index()
    print("   Test subjects per run:")
    for count, runs in test_counts.items():
        print(f"      {count} test subjects: {runs} runs")
    
    # Most common test subject combinations
    print("\\n🧪 Most common test subject combinations:")
    test_combinations = results_df['test_subjects_str'].value_counts().head(10)
    for combination, count in test_combinations.items():
        print(f"      {combination}: {count} runs")
    
    # Performance analysis by number of test subjects
    print("\\n🎯 Performance by number of test subjects:")
    valid_kappa = results_df[results_df['kappa'].notna()]
    if len(valid_kappa) > 0:
        for num_test in sorted(valid_kappa['num_test_subjects'].unique()):
            subset = valid_kappa[valid_kappa['num_test_subjects'] == num_test]
            if len(subset) > 0:
                print(f"      {num_test} test subjects: κ = {subset['kappa'].mean():.3f} ± {subset['kappa'].std():.3f} (n={len(subset)})")
    
    # Dataset-specific analysis
    print("\\n📊 Dataset-specific analysis:")
    for dataset in results_df['dataset'].unique():
        dataset_runs = results_df[results_df['dataset'] == dataset]
        dataset_kappa = dataset_runs[dataset_runs['kappa'].notna()]
        
        if len(dataset_kappa) > 0:
            print(f"   {dataset}:")
            print(f"      Runs: {len(dataset_runs)}")
            print(f"      Subjects in dataset: {len(dataset_subjects.get(dataset, []))}")
            print(f"      Mean κ: {dataset_kappa['kappa'].mean():.3f} ± {dataset_kappa['kappa'].std():.3f}")
            
            # Show subject-specific performance
            subject_performance = {}
            for training_subj in dataset_runs['training_subject'].unique():
                subj_runs = dataset_kappa[dataset_kappa['training_subject'] == training_subj]
                if len(subj_runs) > 0:
                    subject_performance[training_subj] = {
                        'runs': len(subj_runs),
                        'mean_kappa': subj_runs['kappa'].mean(),
                        'std_kappa': subj_runs['kappa'].std()
                    }
            
            print(f"      Performance by training subject:")
            for subj, perf in sorted(subject_performance.items()):
                print(f"         Train on {subj}: κ = {perf['mean_kappa']:.3f} ± {perf['std_kappa']:.3f} (n={perf['runs']})")

def create_test_subjects_lookup(results_df: pd.DataFrame):
    """Create a clean lookup table of test subjects for each run."""
    
    # Create simplified lookup
    lookup = results_df[['run_id', 'run_name', 'training_subject', 'test_subjects_str', 'num_test_subjects', 'kappa']].copy()
    lookup = lookup.sort_values(['run_name', 'training_subject'])
    
    # Save lookup table
    lookup.to_csv('test_subjects_lookup.csv', index=False)
    print(f"\\n💾 Test subjects lookup saved to: test_subjects_lookup.csv")
    
    return lookup

def main():
    csv_path = "Plot_Clean/data/all_runs_flat.csv"
    
    # Extract test subjects using LOSO logic
    results_df, dataset_subjects = extract_test_subjects_loso(csv_path)
    
    if results_df is None:
        return
    
    # Analyze the results
    analyze_test_subjects(results_df, dataset_subjects)
    
    # Create lookup table
    lookup = create_test_subjects_lookup(results_df)
    
    # Save full results
    results_df.to_csv('test_subjects_full_loso.csv', index=False)
    
    print("\\n🎉 Analysis complete!")
    print("📄 Files generated:")
    print("   - test_subjects_full_loso.csv (complete results)")
    print("   - test_subjects_lookup.csv (simplified lookup)")
    
    print("\\n🔍 Key findings:")
    print(f"   - Analyzed {len(results_df)} runs with LOSO test subject inference")
    print(f"   - Found {len(dataset_subjects)} unique datasets")
    for dataset, subjects in dataset_subjects.items():
        print(f"     * {dataset}: {len(subjects)} subjects")
    
    return results_df

if __name__ == "__main__":
    main()