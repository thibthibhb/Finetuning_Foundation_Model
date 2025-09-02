#!/usr/bin/env python3
"""
Investigate the missing test subject information for the majority of runs.

Out of 1937 rows, only 262 have cfg.subject_id. Let's find where the test 
subject information is stored for the remaining ~1675 rows.
"""

import pandas as pd
import numpy as np
import json
import ast

def comprehensive_column_analysis(df: pd.DataFrame):
    """Analyze all columns for potential test subject information."""
    print("🔍 Comprehensive analysis of ALL columns for test subject info...")
    
    # Look for any column that might contain structured data (JSON, lists, etc.)
    potential_columns = []
    
    for col in df.columns:
        # Skip obviously unrelated columns
        if any(skip in col.lower() for skip in ['duration', 'lr', 'epoch', 'batch', 'weight']):
            continue
            
        # Get sample of non-null values
        sample_values = df[col].dropna().head(20)
        
        for val in sample_values:
            val_str = str(val)
            # Look for structured data indicators
            if (len(val_str) > 10 and 
                (('[' in val_str and ']' in val_str) or 
                 ('{' in val_str and '}' in val_str) or
                 (',' in val_str and ('S0' in val_str or 'subj' in val_str.lower())) or
                 ('test' in val_str.lower() and 'train' in val_str.lower()))):
                potential_columns.append(col)
                print(f"   🎯 {col}: {val_str[:100]}...")
                break
    
    return potential_columns

def analyze_rows_without_subject_id(df: pd.DataFrame):
    """Focus on the rows that don't have cfg.subject_id."""
    print("\\n📊 Analyzing rows WITHOUT cfg.subject_id...")
    
    no_subject_id = df[df['cfg.subject_id'].isna()]
    print(f"   Rows without cfg.subject_id: {len(no_subject_id)} / {len(df)}")
    
    # Check what these rows have instead
    print("\\n🔍 What do these rows contain?")
    
    # Check run names
    print("   Sample run names:")
    sample_names = no_subject_id['name'].dropna().head(10).tolist()
    for name in sample_names:
        print(f"      {name}")
    
    # Check for any columns that might contain subject info
    subject_related = [col for col in no_subject_id.columns if 'subject' in col.lower()]
    print(f"\\n   Subject-related columns: {subject_related}")
    
    for col in subject_related:
        non_null = no_subject_id[col].notna().sum()
        if non_null > 0:
            print(f"      {col}: {non_null} non-null values")
            samples = no_subject_id[col].dropna().head(5).tolist()
            print(f"         Samples: {samples}")
    
    # Check if these rows have different experiment types
    if 'contract.icl.icl_mode' in no_subject_id.columns:
        icl_modes = no_subject_id['contract.icl.icl_mode'].value_counts(dropna=False)
        print(f"\\n   ICL modes in non-subject-id rows:")
        print(icl_modes)
    
    return no_subject_id

def deep_search_for_splits(df: pd.DataFrame):
    """Deep search in all text-like columns for train/test split information."""
    print("\\n🕵️ Deep search for train/test split information...")
    
    # Look through all object/string columns
    text_columns = df.select_dtypes(include=['object']).columns
    
    split_indicators = ['train', 'test', 'val', 'split', 'fold', 'subject']
    
    for col in text_columns:
        if col in ['run_id', 'name']:  # Skip obviously irrelevant
            continue
            
        # Sample values from this column
        sample_vals = df[col].dropna().head(50)
        
        for val in sample_vals:
            val_str = str(val).lower()
            
            # Check if this value might contain split information
            if (len(val_str) > 15 and 
                sum(indicator in val_str for indicator in split_indicators) >= 2):
                
                print(f"   🎯 Found potential split info in {col}:")
                print(f"      {str(val)[:200]}...")
                
                # Try to parse if it looks like structured data
                try:
                    if val_str.startswith('[') or val_str.startswith('{'):
                        parsed = json.loads(str(val))
                        print(f"      Parsed JSON: {type(parsed)} with keys/length: {len(parsed) if hasattr(parsed, '__len__') else 'N/A'}")
                        if isinstance(parsed, dict):
                            print(f"         Keys: {list(parsed.keys())[:5]}")
                except:
                    try:
                        parsed = ast.literal_eval(str(val))
                        print(f"      Parsed literal: {type(parsed)}")
                    except:
                        pass
                
                break  # Just show first example per column

def check_data_source_patterns(df: pd.DataFrame):
    """Check if different types of experiments store data differently."""
    print("\\n🔬 Checking data source patterns...")
    
    # Group by different characteristics
    print("   By experiment type (name patterns):")
    name_patterns = {}
    for _, row in df.head(100).iterrows():  # Sample first 100
        name = str(row.get('name', ''))
        # Extract pattern (before underscore or hyphen)
        pattern = name.split('_')[0].split('-')[0]
        if pattern not in name_patterns:
            name_patterns[pattern] = 0
        name_patterns[pattern] += 1
    
    for pattern, count in sorted(name_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"      {pattern}: {count} runs")
        
        # Check if this pattern has subject info
        pattern_runs = df[df['name'].str.startswith(pattern, na=False)]
        subject_info = pattern_runs['cfg.subject_id'].notna().sum()
        print(f"         -> {subject_info}/{len(pattern_runs)} have cfg.subject_id")

def main():
    csv_path = "Plot_Clean/data/all_runs_flat.csv"
    
    print(f"📁 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
    
    print(f"\\n📊 Quick stats:")
    print(f"   Rows with cfg.subject_id: {df['cfg.subject_id'].notna().sum()}")
    print(f"   Rows WITHOUT cfg.subject_id: {df['cfg.subject_id'].isna().sum()}")
    print(f"   Percentage missing: {df['cfg.subject_id'].isna().sum() / len(df) * 100:.1f}%")
    
    # Comprehensive column analysis
    potential_cols = comprehensive_column_analysis(df)
    
    # Analyze rows without subject_id
    no_subject_rows = analyze_rows_without_subject_id(df)
    
    # Deep search for split info
    deep_search_for_splits(df)
    
    # Check patterns
    check_data_source_patterns(df)
    
    print("\\n💡 Next steps to find test subjects:")
    print("1. Check if WandB runs contain more detailed split information")
    print("2. Look for experiment logs or config files")
    print("3. Check if different experiment types store splits differently")
    print("4. Examine the actual training scripts for how they handle data splits")

if __name__ == "__main__":
    main()