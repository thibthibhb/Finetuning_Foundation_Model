#!/usr/bin/env python3
"""
Simple runner for Figure 7

Just calls the plotting function - no training, no complexity.
Focuses on getting ONE good comparison figure working.
"""

import sys
import os

# Make sure we can import from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig7_ICL_simple import main as plot_main

def main():
    print("🚀 Generating simple ICL vs Fine-tuning comparison...")
    print("📊 Focus: 5-class Cohen's kappa comparison")
    print("=" * 50)
    
    # Call the plotting function
    result = plot_main()
    
    if result == 0:
        print("=" * 50)
        print("✅ SUCCESS: Figure generated!")
        print("📁 Check ../figures/fig7_method_comparison.pdf")
    else:
        print("=" * 50)
        print("❌ FAILED: Check error messages above")
    
    return result

if __name__ == '__main__':
    sys.exit(main())