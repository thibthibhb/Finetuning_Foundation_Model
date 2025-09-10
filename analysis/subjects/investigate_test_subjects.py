#!/usr/bin/env python3
"""
Investigate test subject information in the CSV data.
Look for any columns that might contain actual test subject IDs.
"""

import pandas as pd
import numpy as np

def investigate_columns(csv_path: str):
    """Investigate all columns that might contain test subject information."""
    print(f"📁 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Look for columns that might contain subject information
    subject_related_cols = [col for col in df.columns if any(keyword in col.lower() 
                           for keyword in ['subject', 'split', 'test', 'train', 'val', 'id'])]
    
    print(f"\\n🔍 Columns potentially related to subjects/splits ({len(subject_related_cols)}):")
    for col in subject_related_cols:
        non_null = df[col].notna().sum()
        unique_vals = df[col].nunique()
        print(f"   {col}: {non_null} non-null, {unique_vals} unique values")
        
        # Show sample values
        sample_vals = df[col].dropna().head(3).tolist()
        print(f"      Sample: {sample_vals}")
    
    # Special focus on cfg.subject_id since that's likely the training subject
    print("\\n🧪 Detailed analysis of cfg.subject_id:")
    if 'cfg.subject_id' in df.columns:
        subject_ids = df['cfg.subject_id'].dropna()
        print(f"   Non-null subject IDs: {len(subject_ids)}")
        print(f"   Unique subject IDs: {subject_ids.nunique()}")
        print(f"   Subject ID range: {subject_ids.min()} to {subject_ids.max()}")
        
        # Most common subject IDs
        subject_counts = subject_ids.value_counts().head(10)
        print(f"   Most common subject IDs:")
        for subj_id, count in subject_counts.items():
            print(f"      Subject {subj_id}: {count} runs")
    
    # Check if there are any patterns in the data that might indicate test subjects
    print("\\n🔍 Looking for patterns that might indicate test subject splits...")
    
    # Check for columns with JSON-like or list-like data
    for col in df.columns:
        sample_values = df[col].dropna().head(5)
        for val in sample_values:
            val_str = str(val)
            if (('[' in val_str and ']' in val_str) or 
                ('{' in val_str and '}' in val_str) or
                (',' in val_str and len(val_str) > 10)):
                print(f"   {col} might contain structured data:")
                print(f"      Sample: {val_str[:100]}...")
                break
    
    # Check run names for patterns
    print("\\n🏷️ Sample run names (might contain subject info):")
    if 'name' in df.columns:
        sample_names = df['name'].dropna().head(10).tolist()
        for name in sample_names:
            print(f"   {name}")
    
    return df

def examine_training_vs_test_logic(df: pd.DataFrame):
    """
    Try to understand the training vs test logic from the data.
    Since cfg.subject_id likely represents the training subject,
    the test subjects would be all OTHER subjects in the dataset.
    """
    print("\\n🤔 Understanding training vs test subject logic...")
    
    if 'cfg.subject_id' not in df.columns:
        print("❌ No cfg.subject_id column found")
        return
    
    # Get info about datasets and subjects
    datasets_info = {}
    if 'cfg.datasets' in df.columns:
        print("\\n📊 Dataset information:")
        for _, row in df.iterrows():
            if pd.notna(row['cfg.subject_id']) and pd.notna(row['cfg.datasets']):
                dataset = row['cfg.datasets'] 
                subject_id = row['cfg.subject_id']
                
                if dataset not in datasets_info:
                    datasets_info[dataset] = {'subjects': set(), 'runs': 0}
                datasets_info[dataset]['subjects'].add(subject_id)
                datasets_info[dataset]['runs'] += 1
        
        for dataset, info in datasets_info.items():
            print(f"   {dataset}: {info['runs']} runs, {len(info['subjects'])} unique training subjects")
            print(f"      Training subjects: {sorted(list(info['subjects']))[:10]}{'...' if len(info['subjects']) > 10 else ''}")
    
    # Try to infer test subjects based on the logic:
    # If we're doing Leave-One-Subject-Out (LOSO), then:
    # - Training subject = the one specified in cfg.subject_id  
    # - Test subjects = all others in that dataset
    
    print("\\n💡 Inferring test subjects using LOSO logic...")
    print("   Logic: If cfg.subject_id = training subject, then test subjects = all other subjects in dataset")
    
    # For this we'd need to know the full subject list for each dataset
    # Let's check if there are any hints in the data
    
    if 'contract.dataset.num_subjects_train' in df.columns:
        train_subjects = df['contract.dataset.num_subjects_train'].dropna()
        if len(train_subjects) > 0:
            print(f"   Training subjects per run (from contract): {train_subjects.value_counts().head()}")
    
    # Check total subjects in datasets
    if 'sum.hours_of_data' in df.columns and 'cfg.subject_id' in df.columns:
        print("\\n📈 Sample analysis for LOSO inference:")
        sample_data = df[df['cfg.subject_id'].notna()].head(5)
        for _, row in sample_data.iterrows():
            subj_id = row['cfg.subject_id']
            dataset = row.get('cfg.datasets', 'unknown')
            hours = row.get('sum.hours_of_data', 'unknown')
            print(f"   Run {row.get('name', 'unknown')}: Training on subject {subj_id} ({dataset}), {hours} hours")

def main():
    csv_path = "Plot_Clean/data/all_runs_flat.csv"
    
    df = investigate_columns(csv_path)
    examine_training_vs_test_logic(df)
    
    print("\\n🎯 CONCLUSION:")
    print("The CSV contains cfg.subject_id which represents the TRAINING subject.")
    print("The test subjects are likely all OTHER subjects in the dataset (LOSO approach).")
    print("The actual test subject IDs are not explicitly stored in this CSV.")
    print("\\nTo get test subject IDs, you would need:")
    print("1. The full subject list for each dataset")
    print("2. Remove the training subject (cfg.subject_id) to get test subjects")

if __name__ == "__main__":
    main()