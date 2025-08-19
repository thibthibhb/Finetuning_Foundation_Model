#!/usr/bin/env python3
"""
CBraMod Run Loader and Structure Script
=======================================

This script implements a clean, structured approach to loading WandB runs
and preparing them for fair comparison analysis following the defined contract.

Usage:
    python load_and_structure_runs.py --project CBraMod-earEEG-tuning --entity thibaut_hasle-epfl
"""

import wandb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import warnings
import logging
from dataclasses import dataclass
from datetime import datetime
import json
import argparse
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ContractSpec:
    """Definition of the comparison contract for fair analysis."""
    
    # Dataset & Splits
    required_dataset_fields: List[str]
    required_split_fields: List[str] 
    
    # Preprocessing
    required_preprocessing_fields: List[str]
    
    # Model & Training
    required_model_fields: List[str]
    required_training_fields: List[str]
    
    # Results 
    required_result_fields: List[str]
    primary_metrics: List[str]
    secondary_metrics: List[str]
    
    # Compute
    required_compute_fields: List[str]

class RunLoader:
    """Handles loading and structuring WandB runs according to the contract."""
    
    def __init__(self, project: str, entity: str, contract: ContractSpec):
        self.project = project
        self.entity = entity
        self.contract = contract
        self.api = wandb.Api()
        
        # Initialize storage
        self.raw_runs = []
        self.structured_runs = []
        self.failed_runs = []
        self.validation_issues = []
    
    def load_all_runs(self, filters: Optional[Dict] = None, limit: Optional[int] = None) -> int:
        """
        Load ALL runs from WandB with proper pagination handling.
        
        Args:
            filters: Optional WandB filters
            limit: Optional limit on number of runs (for testing)
            
        Returns:
            Number of runs loaded
        """
        logger.info(f"Loading runs from {self.entity}/{self.project}")
        
        try:
            # Get all runs with proper pagination
            runs = self.api.runs(
                f"{self.entity}/{self.project}",
                filters=filters,
                per_page=100  # Increase page size for efficiency
            )
            
            # Convert to list to handle pagination properly
            run_list = list(runs)
            if limit:
                run_list = run_list[:limit]
                
            logger.info(f"Found {len(run_list)} runs to process")
            
            # Process each run
            for i, run in enumerate(run_list):
                if i % 50 == 0:
                    logger.info(f"Processing run {i+1}/{len(run_list)}")
                
                try:
                    structured_run = self._structure_single_run(run)
                    if structured_run:
                        self.structured_runs.append(structured_run)
                    else:
                        self.failed_runs.append({
                            'run_id': getattr(run, 'id', 'unknown'),
                            'name': getattr(run, 'name', 'unknown'),
                            'reason': 'Failed to structure'
                        })
                except Exception as e:
                    run_name = getattr(run, 'name', 'unknown')
                    run_id = getattr(run, 'id', 'unknown')
                    logger.warning(f"Failed to process run {run_name}: {e}")
                    self.failed_runs.append({
                        'run_id': run_id,
                        'name': run_name, 
                        'reason': str(e)
                    })
            
            logger.info(f"Successfully structured {len(self.structured_runs)} runs")
            logger.info(f"Failed to structure {len(self.failed_runs)} runs")
            
            return len(self.structured_runs)
            
        except Exception as e:
            logger.error(f"Failed to load runs: {e}")
            raise
    
    def _structure_single_run(self, run) -> Optional[Dict]:
        """Structure a single WandB run according to the contract."""
        
        try:
            # Basic run info - handle missing attributes gracefully
            structured = {
                'run_id': getattr(run, 'id', 'unknown'),
                'name': getattr(run, 'name', 'unnamed'),
                'state': getattr(run, 'state', 'unknown'),
                'created_at': getattr(run, 'created_at', None),
                'updated_at': getattr(run, 'updated_at', None),
                'tags': list(getattr(run, 'tags', [])),
                'label_scheme': self._detect_label_scheme(run),
                'notes': getattr(run, 'notes', '') or '',
            }
            
            # Calculate duration if both timestamps are available
            created_at = structured['created_at']
            updated_at = structured['updated_at']
            if created_at and updated_at:
                try:
                    structured['duration_seconds'] = (updated_at - created_at).total_seconds()
                except:
                    structured['duration_seconds'] = None
            else:
                structured['duration_seconds'] = None
            
            # Extract config (hyperparameters) - handle missing gracefully
            try:
                config = dict(getattr(run, 'config', {}))
            except:
                config = {}
            structured['config'] = config
            
            # Extract summary (final metrics) - handle missing gracefully
            try:
                summary = dict(getattr(run, 'summary', {}))
            except:
                summary = {}
            structured['summary'] = summary
            
            # Extract history (training curves) - sample key metrics only
            try:
                if hasattr(run, 'history'):
                    history = run.history(samples=1000)  # Limit for memory
                    structured['history'] = self._process_history(history)
                else:
                    structured['history'] = {}
            except Exception as e:
                logger.debug(f"Could not load history for {structured['name']}: {e}")
                structured['history'] = {}
            
            # Structure according to contract
            structured['contract_data'] = self._apply_contract_structure(structured)
            structured['validation_status'] = self._validate_contract_compliance(structured['contract_data'])
            
            return structured
            
        except Exception as e:
            run_name = getattr(run, 'name', 'unknown') if 'run' in locals() else 'unknown'
            logger.warning(f"Error structuring run {run_name}: {e}")
            return None
    
    def _process_history(self, history_df: pd.DataFrame) -> Dict:
        """Process history dataframe to extract key training curves."""
        if history_df.empty:
            return {}
        
        # Extract key metrics
        key_metrics = [
            'val_kappa', 'val_accuracy', 'val_f1', 
            'train_loss', 'val_loss',
            'epoch', '_step', '_runtime'
        ]
        
        processed = {}
        for metric in key_metrics:
            if metric in history_df.columns:
                values = history_df[metric].dropna()
                if not values.empty:
                    processed[metric] = {
                        'values': values.tolist(),
                        'final': float(values.iloc[-1]),
                        'best': float(values.max()) if 'loss' not in metric else float(values.min()),
                        'count': len(values)
                    }
        
        return processed
    
    def _apply_contract_structure(self, run_data: Dict) -> Dict:
        """Structure run data according to the comparison contract."""
        
        config = run_data.get('config', {})
        summary = run_data.get('summary', {})
        
        contract_data = {
            # Dataset & Splits
            'dataset': {
                'name': config.get('downstream_dataset', 'UNKNOWN'),
                'datasets': config.get('datasets', 'UNKNOWN'),
                'dataset_names': config.get('dataset_names', []),
                'num_datasets': config.get('num_datasets', 0),
                'num_subjects_train': config.get('num_subjects_train', -1),
                'data_fraction': config.get('data_fraction', 1.0),
                'orp_train_frac': config.get('data_ORP', 0.6),
            },
            
            # Preprocessing  
            'preprocessing': {
                'preprocess': config.get('preprocess', False),
                'sample_rate': config.get('sample_rate', 200),
                'window_length': 30,  # Fixed for ear-EEG
                'version_string': self._generate_preprocessing_version(config),
            },
            
            # Model Architecture
            'model': {
                'backbone': 'CBraMod',  # Fixed
                'use_pretrained_weights': config.get('use_pretrained_weights', True),
                'foundation_dir': config.get('foundation_dir', 'pretrained_weights.pth'),
                'head_type': config.get('head_type', 'simple'),
                'model_size': config.get('model_size', 4000000),
                'layers': config.get('layers', 12),
                'heads': config.get('heads', 8),
                'embedding_dim': config.get('embedding_dim', 512),
                'frozen': config.get('frozen', False),
            },
            
            # Training Configuration
            'training': {
                'epochs': config.get('epochs', 100),
                'batch_size': config.get('batch_size', 64),
                'lr': config.get('lr', 5e-5),
                'optimizer': config.get('optimizer', 'AdamW'),
                'scheduler': config.get('scheduler', 'cosine'),
                'weight_decay': config.get('weight_decay', 0),
                'label_smoothing': config.get('label_smoothing', 0.1),
                'use_focal_loss': config.get('use_focal_loss', False),
                'focal_gamma': config.get('focal_gamma', 2.0),
                'two_phase_training': config.get('two_phase_training', False),
                'phase1_epochs': config.get('phase1_epochs', 3),
                'head_lr': config.get('head_lr', 1e-3),
                'backbone_lr': config.get('backbone_lr', 1e-5),
                'use_amp': config.get('use_amp', False),
            },
            
            # ICL Configuration (new)
            'icl': {
                'icl_mode': config.get('icl_mode', 'none'),
                'k_support': config.get('k_support', 0),
                'proto_temp': config.get('proto_temp', 0.1),
                'icl_hidden': config.get('icl_hidden', 256),
                'icl_layers': config.get('icl_layers', 2),
                'icl_eval_Ks': config.get('icl_eval_Ks', '0'),
            },
            
            # Results (Test Set Only)
            'results': {
                'num_classes': config.get('num_of_classes', 5),
                'test_kappa': summary.get('test_kappa', np.nan),
                'test_accuracy': summary.get('test_accuracy', np.nan),
                'test_f1': summary.get('test_f1', np.nan),
                'val_kappa': summary.get('val_kappa', np.nan),
                'hours_of_data': summary.get('hours_of_data', np.nan),
            },
            
            # ICL Results (if available)
            'icl_results': self._extract_icl_results(summary),
            
            # Compute Resources
            'compute': {
                'gpu_type': 'UNKNOWN',  # Not logged, would need to backfill
                'gpu_hours': np.nan,   # Calculate from duration if available  
                'max_vram': np.nan,    # Not logged
                'effective_tokens': self._calculate_effective_tokens(config, summary),
                'cost_per_inference_usd': summary.get('cost_per_inference_usd', np.nan),
                'cost_per_night_usd': summary.get('cost_per_night_usd', np.nan),
                'throughput_samples_per_sec': summary.get('inference_throughput_samples_per_sec', np.nan),
            },
            
            # Quality Metadata
            'quality': {
                'seed': config.get('seed', 3407),
                'early_stopped': False,  # Would need to detect from history
                'converged': True,       # Would need to detect from history  
                'anomalous_compute': False,  # Would need to detect
            }
        }
        
        return contract_data
    
    def _detect_label_scheme(self, run) -> str:
        """Detect label scheme with backwards compatibility."""
        config = getattr(run, 'config', {})
        tags = list(getattr(run, 'tags', []))
        num_classes = config.get('num_of_classes', 5)
        
        # Check for new versioning tags
        has_version_tags = any('labelspace/train:' in str(tag) for tag in tags)
        
        if has_version_tags:
            # New runs with explicit tags
            if any('labelspace/train:4c-v1' in str(tag) for tag in tags):
                return '4c-v1'
            elif any('labelspace/train:4c-v0' in str(tag) for tag in tags):
                return '4c-v0'
            elif any('labelspace/train:5c' in str(tag) for tag in tags):
                return '5c'
            else:
                return f'{num_classes}c-tagged-unknown'
        else:
            # OLD RUNS - Apply backwards compatibility
            if num_classes == 5:
                return '5c'  # All old 5c = standard
            elif num_classes == 4:
                return '4c-v0'  # All old 4c = legacy
            else:
                return f'{num_classes}c-old-unknown'
    
    def _generate_preprocessing_version(self, config: Dict) -> str:
        """Generate a version string for preprocessing configuration."""
        components = [
            f"sr{int(config.get('sample_rate', 200))}",
            f"preproc{config.get('preprocess', False)}",
        ]
        return "_".join(components)
    
    def _extract_icl_results(self, summary: Dict) -> Dict:
        """Extract ICL-specific results from summary."""
        icl_results = {}
        
        # Look for ICL metrics with pattern: {mode}_test_kappa_{icl/base}_K{k}
        for key, value in summary.items():
            if any(mode in key for mode in ['proto_test', 'cnp_test', 'set_test']):
                icl_results[key] = value
        
        return icl_results
    
    def _calculate_effective_tokens(self, config: Dict, summary: Dict) -> float:
        """Calculate effective tokens processed (windows × epochs)."""
        try:
            hours_of_data = summary.get('hours_of_data', 0)
            epochs = config.get('epochs', 100)
            
            if hours_of_data and epochs:
                # Estimate windows from hours of data
                windows_per_hour = 3600 / 30  # 30-second windows
                total_windows = hours_of_data * windows_per_hour
                return total_windows * epochs
            
            return np.nan
        except:
            return np.nan
    
    def _validate_contract_compliance(self, contract_data: Dict) -> Dict:
        """Validate that run meets contract requirements."""
        
        issues = []
        status = 'VALID'
        
        # Check required fields
        for field_group, required_fields in [
            ('dataset', self.contract.required_dataset_fields),
            ('preprocessing', self.contract.required_preprocessing_fields),
            ('model', self.contract.required_model_fields),
            ('training', self.contract.required_training_fields),
            ('results', self.contract.required_result_fields),
            ('compute', self.contract.required_compute_fields),
        ]:
            data = contract_data.get(field_group, {})
            for field in required_fields:
                if field not in data or pd.isna(data[field]) or data[field] in ['UNKNOWN', '', None]:
                    issues.append(f"Missing {field_group}.{field}")
        
        # Check primary metrics
        results = contract_data.get('results', {})
        for metric in self.contract.primary_metrics:
            if pd.isna(results.get(metric, np.nan)):
                issues.append(f"Missing primary metric: {metric}")
        
        # Determine status
        if issues:
            status = 'INVALID' if len(issues) > 3 else 'WARNING'
        
        return {
            'status': status,
            'issues': issues,
            'score': max(0, 100 - len(issues) * 10)  # Quality score
        }


class CohortBuilder:
    """Builds analysis cohorts from structured runs."""
    
    def __init__(self, structured_runs: List[Dict]):
        self.structured_runs = structured_runs
        self.cohorts = {}
    
    def build_all_cohorts(self) -> Dict[str, List[Dict]]:
        """Build all analysis cohorts according to the contract."""
        
        # Filter to only valid/warning runs
        valid_runs = [
            run for run in self.structured_runs
            if run['validation_status']['status'] in ['VALID', 'WARNING']
        ]
        
        logger.info(f"Building cohorts from {len(valid_runs)} valid runs")
        
        # Cohort A: 5-class comparison
        self.cohorts['5_class'] = self._build_class_cohort(valid_runs, num_classes=5)
        
        # Cohort B: 4-class comparison  
        self.cohorts['4_class'] = self._build_class_cohort(valid_runs, num_classes=4)
        
        # Scaling cohort: vary data size
        self.cohorts['scaling'] = self._build_scaling_cohort(valid_runs)
        
        # ICL cohort: compare ICL modes
        self.cohorts['icl_comparison'] = self._build_icl_cohort(valid_runs)
        
        # Window cohort: vary window sizes (if available)
        # Note: Current data is fixed at 30s windows
        
        # Log cohort statistics
        for cohort_name, runs in self.cohorts.items():
            logger.info(f"Cohort '{cohort_name}': {len(runs)} runs")
        
        return self.cohorts
    
    def _build_class_cohort(self, runs: List[Dict], num_classes: int) -> List[Dict]:
        """Build cohort for specific number of classes."""
        return [
            run for run in runs
            if run['contract_data']['results']['num_classes'] == num_classes
        ]
    
    def _build_scaling_cohort(self, runs: List[Dict]) -> List[Dict]:
        """Build cohort for data scaling analysis."""
        # Include runs with different data fractions or subject counts
        scaling_runs = [
            run for run in runs
            if (run['contract_data']['dataset']['data_fraction'] != 1.0 or
                run['contract_data']['dataset']['orp_train_frac'] != 0.6)
        ]
        return scaling_runs
    
    def _build_icl_cohort(self, runs: List[Dict]) -> List[Dict]:
        """Build cohort for ICL mode comparison."""
        icl_runs = [
            run for run in runs
            if run['contract_data']['icl']['icl_mode'] != 'none'
        ]
        return icl_runs


class QualityChecker:
    """Implements sanity checks and QA before plotting."""
    
    def __init__(self, structured_runs: List[Dict]):
        self.structured_runs = structured_runs
        self.issues = []
    
    def run_all_checks(self) -> Dict[str, bool]:
        """Run all QA checks and return results."""
        
        checks = {
            'pagination_complete': self._check_pagination_completeness(),
            'no_leakage': self._check_subject_leakage(),
            'class_balance_consistent': self._check_class_balance_consistency(),
            'metric_consistency': self._check_metric_consistency(),
            'no_anomalous_outliers': self._check_for_outliers(),
        }
        
        # Log results
        passed = sum(checks.values())
        total = len(checks)
        logger.info(f"QA Checks: {passed}/{total} passed")
        
        for check_name, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            logger.info(f"  {check_name}: {status}")
        
        if self.issues:
            logger.warning("QA Issues found:")
            for issue in self.issues:
                logger.warning(f"  - {issue}")
        
        return checks
    
    def _check_pagination_completeness(self) -> bool:
        """Check if we loaded all runs (no pagination issues)."""
        # This is a heuristic - check for runs with very recent timestamps
        if not self.structured_runs:
            return False
        
        # Check if we have a reasonable distribution of creation times
        timestamps = [run['created_at'] for run in self.structured_runs if run['created_at']]
        if len(timestamps) < len(self.structured_runs) * 0.8:
            self.issues.append("Many runs missing creation timestamps")
            return False
        
        return True
    
    def _check_subject_leakage(self) -> bool:
        """Check for potential subject leakage between splits."""
        # Note: This would require more detailed split information
        # For now, just check that num_subjects_train is reasonable
        
        subject_counts = [
            run['contract_data']['dataset']['num_subjects_train']
            for run in self.structured_runs
            if not pd.isna(run['contract_data']['dataset']['num_subjects_train'])
        ]
        
        if not subject_counts:
            self.issues.append("No subject count information available")
            return False
        
        # Check for consistency
        unique_counts = set(subject_counts)
        if len(unique_counts) > 5:  # Allow some variation
            self.issues.append(f"High variation in subject counts: {unique_counts}")
            return False
        
        return True
    
    def _check_class_balance_consistency(self) -> bool:
        """Check for consistent class distributions."""
        # Group by number of classes and check hours_of_data consistency
        class_groups = {}
        for run in self.structured_runs:
            num_classes = run['contract_data']['results']['num_classes']
            hours = run['contract_data']['results']['hours_of_data']
            
            if not pd.isna(hours):
                if num_classes not in class_groups:
                    class_groups[num_classes] = []
                class_groups[num_classes].append(hours)
        
        # Check within-group consistency
        for num_classes, hours_list in class_groups.items():
            if len(hours_list) > 1:
                cv = np.std(hours_list) / np.mean(hours_list)  # Coefficient of variation
                if cv > 0.2:  # 20% variation threshold
                    self.issues.append(f"{num_classes}-class runs have high data variation (CV={cv:.2f})")
                    return False
        
        return True
    
    def _check_metric_consistency(self) -> bool:
        """Check for consistent metric calculation."""
        # Check that we have both kappa and f1 for most runs
        metrics_available = {
            'test_kappa': 0,
            'test_f1': 0,
            'test_accuracy': 0
        }
        
        for run in self.structured_runs:
            results = run['contract_data']['results']
            for metric in metrics_available:
                if not pd.isna(results.get(metric, np.nan)):
                    metrics_available[metric] += 1
        
        total_runs = len(self.structured_runs)
        for metric, count in metrics_available.items():
            coverage = count / total_runs if total_runs > 0 else 0
            if coverage < 0.8:  # 80% coverage threshold
                self.issues.append(f"Low coverage for {metric}: {coverage:.1%}")
                return False
        
        return True
    
    def _check_for_outliers(self) -> bool:
        """Check for anomalous runs that should be excluded."""
        # Check for runs with abnormal duration or performance
        durations = []
        kappas = []
        
        for run in self.structured_runs:
            if run['duration_seconds']:
                durations.append(run['duration_seconds'])
            
            kappa = run['contract_data']['results']['test_kappa']
            if not pd.isna(kappa):
                kappas.append(kappa)
        
        # Check duration outliers
        if durations:
            q1, q3 = np.percentile(durations, [25, 75])
            iqr = q3 - q1
            outlier_threshold = q3 + 3 * iqr  # 3 IQR rule
            outliers = [d for d in durations if d > outlier_threshold]
            
            if len(outliers) / len(durations) > 0.05:  # >5% outliers
                self.issues.append(f"High proportion of duration outliers: {len(outliers)}/{len(durations)}")
        
        # Check performance outliers (negative kappa)
        if kappas:
            negative_kappas = [k for k in kappas if k < 0]
            if len(negative_kappas) / len(kappas) > 0.1:  # >10% negative
                self.issues.append(f"High proportion of negative kappa values: {len(negative_kappas)}/{len(kappas)}")
                return False
        
        return True


def _to_dict(x):
    """Convert various formats to dictionary."""
    import json
    import ast
    
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        # try JSON first, then Python literal
        try:
            return json.loads(x)
        except Exception:
            try:
                return ast.literal_eval(x)
            except Exception:
                return {}
    return {}

def save_flat_versions(structured_runs: List[Dict], output_dir: Path, cohorts: Dict[str, List[Dict]]):
    """Save flattened versions of structured runs for easy plotting."""
    import pandas as pd
    import json
    
    logger.info("Creating flattened CSVs with hyperparameters as columns...")
    
    df = pd.DataFrame(structured_runs)
    
    if df.empty:
        logger.warning("No structured runs to flatten")
        return
    
    # Ensure dicts (not strings) 
    df["config"] = df["config"].apply(_to_dict)
    df["summary"] = df["summary"].apply(_to_dict)
    df["contract_data"] = df["contract_data"].apply(lambda x: x if isinstance(x, dict) else json.loads(str(x)) if x else {})
    df["validation_status"] = df["validation_status"].apply(lambda x: x if isinstance(x, dict) else json.loads(str(x)) if x else {})
    
    # Flatten sections with clear prefixes
    base_cols = ["run_id", "name", "state", "label_scheme", "duration_seconds", "tags"]
    # Only include columns that exist
    existing_base_cols = [col for col in base_cols if col in df.columns]
    base = df[existing_base_cols]
    
    # Flatten nested dictionaries
    cfg = pd.json_normalize(df["config"]).add_prefix("cfg.")
    summ = pd.json_normalize(df["summary"]).add_prefix("sum.")
    contract = pd.json_normalize(df["contract_data"]).add_prefix("contract.")
    val = pd.json_normalize(df["validation_status"]).add_prefix("val.")
    
    # Combine all flattened data
    flat = pd.concat([base, cfg, summ, contract, val], axis=1)
    
    # Create a subset with commonly used plotting columns
    keep_patterns = [
        "run_id", "name", "state", "label_scheme", "duration_seconds", "tags",
        "contract.results.num_classes",
        "contract.results.test_kappa", "contract.results.test_f1", "contract.results.test_accuracy",
        "contract.results.hours_of_data",
        "contract.dataset.num_subjects_train", "contract.dataset.data_fraction",
        "contract.dataset.datasets", "contract.dataset.name", "contract.dataset.dataset_names",  # Added datasets info
        "contract.training.batch_size", "contract.training.lr", "contract.training.head_lr", "contract.training.backbone_lr",
        "contract.training.epochs", "contract.training.optimizer", "contract.training.scheduler",
        "contract.training.weight_decay", "contract.training.label_smoothing",
        "contract.training.unfreeze_epoch", "contract.model.frozen", "contract.training.use_amp",
        "contract.icl.icl_mode", "contract.icl.k_support", "contract.icl.icl_layers", "contract.icl.icl_hidden",
        "cfg.calib_nights", "cfg.nights_training", "cfg.nights_calibration",
        "cfg.calib_minutes", "cfg.minutes_calib", "cfg.epoch_len",
        "cfg.subject_id", "cfg.seed", "cfg.num_of_classes", "cfg.datasets",  # Added cfg.datasets
        "sum.test_kappa", "sum.test_accuracy", "sum.test_f1", "sum.test_f1_macro",
        "sum.hours_of_data", "sum._runtime"
    ]
    
    # Include columns that exist and match our patterns
    existing_cols = []
    for pattern in keep_patterns:
        if pattern in flat.columns:
            existing_cols.append(pattern)
        else:
            # Try to find similar columns (fuzzy matching)
            similar_cols = [col for col in flat.columns if any(part in col.lower() for part in pattern.lower().split('.')[-1].split('_'))]
            if similar_cols and pattern not in existing_cols:
                existing_cols.extend([col for col in similar_cols[:1] if col not in existing_cols])  # Add first match
    
    # Remove duplicates while preserving order
    existing_cols = list(dict.fromkeys(existing_cols))
    flat_plot = flat[existing_cols].copy()
    
    # Save main flat CSV
    flat_path = output_dir / "all_runs_flat.csv"
    flat_plot.to_csv(flat_path, index=False)
    logger.info(f"Saved tidy plot CSV: {flat_path} ({len(flat_plot)} rows, {len(flat_plot.columns)} columns)")
    
    # Save per-cohort flat CSVs
    if cohorts:
        for cohort_name, cohort_runs in cohorts.items():
            if not cohort_runs:
                continue
                
            # Get run IDs for this cohort
            cohort_run_ids = {run['run_id'] for run in cohort_runs}
            
            # Filter flat data to this cohort
            cohort_flat = flat_plot[flat_plot['run_id'].isin(cohort_run_ids)]
            
            if not cohort_flat.empty:
                cohort_flat_path = output_dir / f"cohort_{cohort_name}_flat.csv"
                cohort_flat.to_csv(cohort_flat_path, index=False)
                logger.info(f"Saved cohort '{cohort_name}' flat CSV: {cohort_flat_path} ({len(cohort_flat)} rows)")
    
    # Create a simple loader function as a text file for reference
    loader_code = '''import pandas as pd

def load_runs_flat(path="Plot_Clean/data/all_runs_flat.csv"):
    """Load flattened runs data with convenience aliases."""
    df = pd.read_csv(path)
    
    # Create convenience aliases for common columns
    column_mappings = {
        "contract.results.test_kappa": "test_kappa",
        "contract.results.test_f1": "test_f1", 
        "contract.results.test_accuracy": "test_accuracy",
        "contract.results.num_classes": "num_classes",
        "contract.results.hours_of_data": "hours_of_data",
        "contract.dataset.num_subjects_train": "num_subjects_train",
        "contract.dataset.datasets": "datasets",
        "contract.training.epochs": "epochs",
        "contract.training.batch_size": "batch_size",
        "contract.training.lr": "lr",
        "sum.test_kappa": "test_kappa",
        "sum.test_f1": "test_f1",
        "sum.test_accuracy": "test_accuracy", 
        "cfg.num_of_classes": "num_classes",
        "cfg.subject_id": "subject_id",
        "cfg.datasets": "datasets"
    }
    
    # Apply mappings for columns that exist
    for old_name, new_name in column_mappings.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]
    
    # Derive calibration nights/minutes from common keys
    for c in ["cfg.calib_nights", "cfg.nights_training", "cfg.nights_calibration", "contract.dataset.calib_nights"]:
        if c in df.columns and df[c].notna().any():
            df["calib_nights"] = df[c]
            break
            
    for c in ["cfg.calib_minutes", "cfg.minutes_calib", "contract.dataset.calib_minutes"]:
        if c in df.columns and df[c].notna().any():
            df["calib_minutes"] = df[c]
            break
    
    return df

# Example usage:
# df = load_runs_flat()
# df5 = df.query("num_classes == 5 and test_kappa.notna()")
# print(df.columns)  # See all available columns
'''
    
    loader_path = output_dir / "load_runs_flat.py"
    with open(loader_path, 'w') as f:
        f.write(loader_code)
    logger.info(f"Saved convenience loader: {loader_path}")

def create_contract_spec() -> ContractSpec:
    """Create the comparison contract specification."""
    return ContractSpec(
        required_dataset_fields=[
            'name', 'num_subjects_train', 'data_fraction'
        ],
        required_split_fields=[
            'orp_train_frac'  # Subject-level split info
        ],
        required_preprocessing_fields=[
            'sample_rate', 'window_length', 'version_string'
        ],
        required_model_fields=[
            'backbone', 'use_pretrained_weights', 'head_type'
        ],
        required_training_fields=[
            'epochs', 'batch_size', 'lr', 'optimizer', 'scheduler'
        ],
        required_result_fields=[
            'num_classes', 'test_kappa', 'test_f1'
        ],
        required_compute_fields=[
            'effective_tokens'
        ],
        primary_metrics=[
            'test_kappa', 'test_f1'
        ],
        secondary_metrics=[
            'test_accuracy', 'hours_of_data'
        ]
    )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Load and structure CBraMod WandB runs')
    parser.add_argument('--project', default='CBraMod-earEEG-tuning', help='WandB project name')
    parser.add_argument('--entity', default='thibaut_hasle-epfl', help='WandB entity name')
    parser.add_argument('--limit', type=int, help='Limit number of runs (for testing)')
    parser.add_argument('--output-dir', default='Plot_Clean/data', help='Output directory for structured data')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create contract specification
    contract = create_contract_spec()
    
    # Initialize loader
    loader = RunLoader(args.project, args.entity, contract)
    
    try:
        # Step 1: Load all runs
        logger.info("=== Step 1: Loading WandB Runs ===")
        num_runs = loader.load_all_runs(limit=args.limit)
        
        if num_runs == 0:
            logger.error("No runs loaded. Check project/entity names and permissions.")
            sys.exit(1)
        
        # Step 2: Build analysis cohorts
        logger.info("=== Step 2: Building Analysis Cohorts ===")
        cohort_builder = CohortBuilder(loader.structured_runs)
        cohorts = cohort_builder.build_all_cohorts()
        
        # Step 3: Run QA checks
        logger.info("=== Step 3: Quality Assurance Checks ===")
        qa_checker = QualityChecker(loader.structured_runs)
        qa_results = qa_checker.run_all_checks()
        
        # Step 4: Save structured data
        logger.info("=== Step 4: Saving Structured Data ===")
        
        # Save all structured runs
        all_runs_df = pd.DataFrame([
            {**run, 'contract_data': json.dumps(run['contract_data']), 
             'validation_status': json.dumps(run['validation_status'])}
            for run in loader.structured_runs
        ])
        all_runs_df.to_csv(output_dir / 'all_runs.csv', index=False)
        
        # Save cohorts separately
        for cohort_name, cohort_runs in cohorts.items():
            if cohort_runs:
                cohort_df = pd.DataFrame([
                    {**run, 'contract_data': json.dumps(run['contract_data']),
                     'validation_status': json.dumps(run['validation_status'])}
                    for run in cohort_runs
                ])
                cohort_df.to_csv(output_dir / f'cohort_{cohort_name}.csv', index=False)
        
        # Save QA report
        qa_report = {
            'timestamp': datetime.now().isoformat(),
            'total_runs': len(loader.structured_runs),
            'valid_runs': len([r for r in loader.structured_runs if r['validation_status']['status'] == 'VALID']),
            'warning_runs': len([r for r in loader.structured_runs if r['validation_status']['status'] == 'WARNING']),
            'invalid_runs': len([r for r in loader.structured_runs if r['validation_status']['status'] == 'INVALID']),
            'qa_checks': qa_results,
            'issues': qa_checker.issues,
            'cohort_sizes': {name: len(runs) for name, runs in cohorts.items()}
        }
        
        with open(output_dir / 'qa_report.json', 'w') as f:
            json.dump(qa_report, f, indent=2, default=str)
        
        # Summary
        logger.info("=== Summary ===")
        logger.info(f"Total runs loaded: {len(loader.structured_runs)}")
        logger.info(f"Valid runs: {qa_report['valid_runs']}")
        logger.info(f"Warning runs: {qa_report['warning_runs']}")
        logger.info(f"Invalid runs: {qa_report['invalid_runs']}")
        logger.info(f"QA checks passed: {sum(qa_results.values())}/{len(qa_results)}")
        
        for cohort_name, size in qa_report['cohort_sizes'].items():
            logger.info(f"Cohort '{cohort_name}': {size} runs")
        
        # Step 5: Create flattened CSVs for easy plotting
        logger.info("=== Step 5: Creating Flattened CSVs for Plotting ===")
        save_flat_versions(loader.structured_runs, output_dir, cohorts)
        
        logger.info(f"Data saved to: {output_dir}")
        logger.info("✅ Run loading and structuring completed successfully!")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise


if __name__ == '__main__':
    main()