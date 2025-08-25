# CBraMod Subjects vs Minutes: Comprehensive Testing Results
## Iso-Hours Analysis (Matched Comparisons)
Comparing different allocation strategies within equal total hour bins:
- **0-100h**: Best = 1-5 subj (Δκ=-0.010)
- **100-200h**: Best = 6-10 subj (Δκ=0.085)
- **200-300h**: Best = 11-20 subj (Δκ=-0.031)
- **300-400h**: Best = 11-20 subj (Δκ=0.028)
- **400-500h**: Best = 11-20 subj (Δκ=0.031)
- **1200-1300h**: Best = 20+ subj (Δκ=0.070)
- **1700-1800h**: Best = 20+ subj (Δκ=0.092)
- **1800-1900h**: Best = 20+ subj (Δκ=0.115)

## Mixed-Effects Regression Analysis

Standardized coefficients:
- **num_subjects**: -0.0195
- **minutes_per_subject**: -0.0263
- **total_hours**: 0.0548

## Partial Dependence Analysis
Random Forest feature importance:
- **minutes_per_subject**: 0.5857
- **total_hours**: 0.3646
- **num_subjects**: 0.0496

**Diminishing returns detected** at ~1024 minutes per subject

## Sensitivity Analysis
Robustness check across different data selection methods:
- **filtered**: subjects (r=0.305)
- **full**: subjects (r=0.138)
- **random**: total_hours (r=-0.042)

## Overall Conclusion
**Mixed evidence**: No clear consensus on most important factor. Top factors: {'minutes_per_subject': 1, 'subjects': 2, 'total_hours': 1}

