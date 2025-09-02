#!/usr/bin/env python3
"""
Extract test subject IDs from all_runs_flat.csv

This script examines the cfg.data_splits_test column to extract which subjects
were used as test subjects in each experiment run.
"""

import pandas as pd
import numpy as np
import ast
import json
from collections import Counter
from pathlib import Path

def parse_test_subjects(test_splits_str):
    """
    Parse test subjects from the cfg.data_splits_test column.
    This column might contain various formats like:
    - JSON string
    - Python literal (list/dict)
    - Simple comma-separated values
    """
    if pd.isna(test_splits_str) or test_splits_str == '':
        return None
    
    try:
        # Try parsing as JSON first
        parsed = json.loads(test_splits_str)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict) and 'test' in parsed:
            return parsed['test']
        elif isinstance(parsed, dict):
            # Maybe the dict values contain the test subjects
            for key, value in parsed.items():
                if 'test' in str(key).lower() and isinstance(value, list):
                    return value
        return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    
    try:
        # Try parsing as Python literal
        parsed = ast.literal_eval(str(test_splits_str))
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict) and 'test' in parsed:
            return parsed['test']
        elif isinstance(parsed, dict):
            for key, value in parsed.items():
                if 'test' in str(key).lower() and isinstance(value, list):
                    return value
        return parsed
    except (ValueError, SyntaxError):
        pass
    
    # Try simple comma-separated parsing
    try:
        if ',' in str(test_splits_str):
            return [x.strip() for x in str(test_splits_str).split(',')]
        else:
            return [str(test_splits_str).strip()]
    except:
        return None

def extract_test_subjects_info(csv_path: str):
    """Extract and analyze test subject information from the CSV."""
    print(f"📁 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows")
    
    # Check if test splits column exists
    test_splits_col = 'cfg.data_splits_test'
    if test_splits_col not in df.columns:
        print(f"❌ Column '{test_splits_col}' not found in CSV")
        print("Available columns containing 'test':")
        test_cols = [col for col in df.columns if 'test' in col.lower()]
        for col in test_cols:
            print(f"   - {col}")
        return
    
    print(f"\\n📊 Analyzing {test_splits_col} column...")
    
    # Check how many rows have test split data
    non_empty = df[test_splits_col].notna() & (df[test_splits_col] != '')
    print(f"   Rows with test split data: {non_empty.sum()} / {len(df)}")
    
    if non_empty.sum() == 0:
        print("❌ No test split data found")
        return
    
    # Sample some entries to see the format
    print("\\n🔍 Sample test split entries:")
    sample_entries = df[non_empty][test_splits_col].head(5)
    for i, entry in enumerate(sample_entries):
        print(f"   {i+1}: {str(entry)[:100]}{'...' if len(str(entry)) > 100 else ''}")
    
    # Parse test subjects for all rows
    print("\\n🔄 Parsing test subjects...")
    test_subjects_data = []
    
    for idx, row in df[non_empty].iterrows():
        test_splits_str = row[test_splits_col]
        parsed_subjects = parse_test_subjects(test_splits_str)
        
        if parsed_subjects is not None:
            test_subjects_data.append({
                'run_id': row.get('run_id', idx),
                'name': row.get('name', 'unknown'),
                'test_subjects': parsed_subjects,
                'num_test_subjects': len(parsed_subjects) if isinstance(parsed_subjects, list) else 1,
                'kappa': row.get('sum.test_kappa', row.get('contract.results.test_kappa', np.nan)),
                'num_classes': row.get('cfg.num_of_classes', row.get('contract.results.num_classes', np.nan))
            })
    
    print(f"✅ Successfully parsed test subjects from {len(test_subjects_data)} runs")
    
    if not test_subjects_data:
        print("❌ No test subjects could be parsed")
        return
    
    # Analyze the parsed data
    print("\\n📈 Test subjects analysis:")
    
    # Count number of test subjects per run
    num_subjects_counts = Counter([d['num_test_subjects'] for d in test_subjects_data])
    print("   Test subjects per run distribution:")
    for num_subj, count in sorted(num_subjects_counts.items()):
        print(f"      {num_subj} subjects: {count} runs")
    
    # Find all unique test subjects
    all_test_subjects = set()
    for data in test_subjects_data:
        if isinstance(data['test_subjects'], list):
            all_test_subjects.update(data['test_subjects'])
        else:
            all_test_subjects.add(data['test_subjects'])
    
    print(f"\\n🧪 Unique test subjects found: {len(all_test_subjects)}")
    print("   Sample test subjects:")
    for i, subj in enumerate(sorted(list(all_test_subjects))[:10]):
        print(f"      {i+1}: {subj}")
    if len(all_test_subjects) > 10:
        print(f"      ... and {len(all_test_subjects) - 10} more")
    
    # Count how often each subject was used for testing
    subject_usage = Counter()
    for data in test_subjects_data:
        if isinstance(data['test_subjects'], list):
            for subj in data['test_subjects']:
                subject_usage[subj] += 1
        else:
            subject_usage[data['test_subjects']] += 1
    
    print("\\n📊 Most frequently used test subjects:")
    for subj, count in subject_usage.most_common(10):
        print(f"      {subj}: used in {count} runs")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(test_subjects_data)
    
    # Save detailed results
    output_file = "test_subjects_analysis.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"\\n💾 Detailed results saved to: {output_file}")
    
    # Performance analysis by test subjects
    print("\\n🎯 Performance analysis:")
    if summary_df['kappa'].notna().sum() > 0:
        valid_kappa = summary_df[summary_df['kappa'].notna()]
        print(f"   Runs with valid kappa: {len(valid_kappa)}")
        print(f"   Mean kappa: {valid_kappa['kappa'].mean():.3f} ± {valid_kappa['kappa'].std():.3f}")
        print(f"   Kappa range: [{valid_kappa['kappa'].min():.3f}, {valid_kappa['kappa'].max():.3f}]")
        
        # Performance by number of test subjects
        print("\\n   Performance by number of test subjects:")
        for num_subj in sorted(valid_kappa['num_test_subjects'].unique()):
            subset = valid_kappa[valid_kappa['num_test_subjects'] == num_subj]
            if len(subset) > 0:
                print(f"      {num_subj} test subjects: κ = {subset['kappa'].mean():.3f} ± {subset['kappa'].std():.3f} (n={len(subset)})")
    
    return summary_df

def create_test_subjects_summary(csv_path: str):
    """Create a focused summary of test subjects for each run."""
    print("🎯 Creating focused test subjects summary...")
    
    df = pd.read_csv(csv_path)
    test_splits_col = 'cfg.data_splits_test'
    
    if test_splits_col not in df.columns:
        print(f"❌ No {test_splits_col} column found")
        return
    
    # Create clean summary
    results = []
    for _, row in df.iterrows():
        if pd.notna(row[test_splits_col]) and row[test_splits_col] != '':
            parsed = parse_test_subjects(row[test_splits_col])
            if parsed is not None:
                results.append({
                    'run_name': row.get('name', 'unknown'),
                    'run_id': row.get('run_id', 'unknown'),
                    'test_subjects': str(parsed),
                    'num_test_subjects': len(parsed) if isinstance(parsed, list) else 1,
                    'kappa': row.get('sum.test_kappa', row.get('contract.results.test_kappa', np.nan)),
                    'num_classes': row.get('cfg.num_of_classes', row.get('contract.results.num_classes', np.nan)),
                    'backbone_frozen': row.get('contract.model.frozen', False)
                })
    
    summary_df = pd.DataFrame(results)
    summary_df.to_csv('test_subjects_summary.csv', index=False)
    print(f"✅ Clean summary saved to: test_subjects_summary.csv ({len(summary_df)} runs)")
    
    return summary_df

def main():
    csv_path = "Plot_Clean/data/all_runs_flat.csv"
    
    # Full analysis
    full_results = extract_test_subjects_info(csv_path)
    
    # Clean summary
    summary_results = create_test_subjects_summary(csv_path)
    
    print("\\n🎉 Analysis complete!")
    print("📄 Files generated:")
    print("   - test_subjects_analysis.csv (detailed)")
    print("   - test_subjects_summary.csv (clean summary)")

if __name__ == "__main__":
    main()