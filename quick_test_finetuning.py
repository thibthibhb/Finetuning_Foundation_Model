#!/usr/bin/env python3
"""
Quick test of the optimized finetuning pipeline without pretrained weights
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from run_optimized_finetuning import main as run_pipeline

def main():
    """Run a quick test of the finetuning pipeline"""
    
    # Override sys.argv to simulate command line arguments
    test_args = [
        'quick_test_finetuning.py',
        '--config-env', 'finetuning_optimized', 
        '--epochs', '2',  # Very short for testing
        '--batch-size', '32',  # Smaller batch for testing
        '--check-only'  # Just check, don't actually train
    ]
    
    original_argv = sys.argv
    try:
        sys.argv = test_args
        result = run_pipeline()
        return result
    finally:
        sys.argv = original_argv

if __name__ == '__main__':
    success = main()
    print(f"\n{'✅ Test passed!' if success else '❌ Test failed!'}")
    sys.exit(0 if success else 1)