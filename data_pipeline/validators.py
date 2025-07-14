import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    test_name: str
    status: bool
    severity: ValidationSeverity
    message: str
    metadata: Dict[str, Any] = None

class EEGDataValidator:
    """Enhanced EEG-specific data validation with comprehensive quality checks"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'sample_rate': 200,
            'expected_channels': [1, 22],  # Support both single and multi-channel
            'max_amplitude': 1000,  # µV
            'min_amplitude': -1000,  # µV
            'max_variance_threshold': 10000,
            'min_variance_threshold': 0.01,
            'artifact_threshold': 500,  # µV
            'zero_variance_tolerance': 0.05,  # 5% of epochs can have zero variance
            'nan_tolerance': 0.0,  # 0% tolerance for NaN values
            'expected_epoch_length': 6000,  # 30 seconds * 200 Hz
            'frequency_range': [0.5, 100],  # Hz
        }

    def validate_single_file(self, data: np.ndarray, labels: np.ndarray, 
                           file_path: str = "unknown") -> List[ValidationResult]:
        """Comprehensive validation of a single EEG file"""
        results = []
        
        # Basic integrity checks
        results.extend(self._check_basic_integrity(data, labels, file_path))
        
        # EEG-specific quality checks
        results.extend(self._check_eeg_quality(data, file_path))
        
        # Label validation
        results.extend(self._check_label_integrity(labels, file_path))
        
        # Signal characteristics
        results.extend(self._check_signal_characteristics(data, file_path))
        
        return results

    def _check_basic_integrity(self, data: np.ndarray, labels: np.ndarray, 
                             file_path: str) -> List[ValidationResult]:
        """Enhanced version of the original check_integrity function"""
        results = []
        
        # Check for NaN/Inf values
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        total_points = data.size
        
        if nan_count > 0:
            nan_ratio = nan_count / total_points
            severity = ValidationSeverity.CRITICAL if nan_ratio > self.config['nan_tolerance'] else ValidationSeverity.WARNING
            results.append(ValidationResult(
                test_name="nan_check",
                status=nan_ratio <= self.config['nan_tolerance'],
                severity=severity,
                message=f"Found {nan_count} NaN values ({nan_ratio:.4f} ratio) in {file_path}",
                metadata={'nan_count': int(nan_count), 'nan_ratio': float(nan_ratio)}
            ))
        
        if inf_count > 0:
            results.append(ValidationResult(
                test_name="infinity_check",
                status=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Found {inf_count} infinite values in {file_path}",
                metadata={'inf_count': int(inf_count)}
            ))
        
        # Check amplitude range
        max_val = np.abs(data).max()
        amplitude_ok = max_val <= self.config['max_amplitude']
        results.append(ValidationResult(
            test_name="amplitude_range",
            status=amplitude_ok,
            severity=ValidationSeverity.ERROR if not amplitude_ok else ValidationSeverity.INFO,
            message=f"Max amplitude: {max_val:.2f} µV (limit: {self.config['max_amplitude']} µV)",
            metadata={'max_amplitude': float(max_val), 'limit': self.config['max_amplitude']}
        ))
        
        # Check data-label alignment
        if len(data.shape) > 1:
            data_epochs = data.shape[0]
        else:
            data_epochs = len(data) // self.config['expected_epoch_length']
            
        label_epochs = len(labels)
        alignment_ok = data_epochs == label_epochs
        
        results.append(ValidationResult(
            test_name="data_label_alignment",
            status=alignment_ok,
            severity=ValidationSeverity.CRITICAL if not alignment_ok else ValidationSeverity.INFO,
            message=f"Data epochs: {data_epochs}, Label epochs: {label_epochs}",
            metadata={'data_epochs': data_epochs, 'label_epochs': label_epochs}
        ))
        
        return results

    def _check_eeg_quality(self, data: np.ndarray, file_path: str) -> List[ValidationResult]:
        """EEG-specific signal quality checks"""
        results = []
        
        # Reshape data for analysis if needed
        if len(data.shape) == 1:
            # Assume single channel, reshape to epochs
            epoch_length = self.config['expected_epoch_length']
            n_epochs = len(data) // epoch_length
            if n_epochs > 0:
                data_epochs = data[:n_epochs * epoch_length].reshape(n_epochs, -1)
            else:
                data_epochs = data.reshape(1, -1)
        else:
            data_epochs = data.reshape(data.shape[0], -1)
        
        # Check variance per epoch
        variances = np.var(data_epochs, axis=1)
        zero_var_count = np.sum(variances < self.config['min_variance_threshold'])
        zero_var_ratio = zero_var_count / len(variances)
        
        variance_ok = zero_var_ratio <= self.config['zero_variance_tolerance']
        results.append(ValidationResult(
            test_name="signal_variance",
            status=variance_ok,
            severity=ValidationSeverity.WARNING if not variance_ok else ValidationSeverity.INFO,
            message=f"Zero variance epochs: {zero_var_count}/{len(variances)} ({zero_var_ratio:.3f})",
            metadata={
                'zero_var_count': int(zero_var_count),
                'total_epochs': len(variances),
                'zero_var_ratio': float(zero_var_ratio),
                'mean_variance': float(np.mean(variances))
            }
        ))
        
        # Check for artifacts (high amplitude spikes)
        artifact_epochs = []
        for i, epoch in enumerate(data_epochs):
            if np.any(np.abs(epoch) > self.config['artifact_threshold']):
                artifact_epochs.append(i)
        
        artifact_ratio = len(artifact_epochs) / len(data_epochs)
        results.append(ValidationResult(
            test_name="artifact_detection",
            status=artifact_ratio < 0.1,  # Less than 10% artifacts
            severity=ValidationSeverity.WARNING if artifact_ratio >= 0.1 else ValidationSeverity.INFO,
            message=f"Artifact epochs: {len(artifact_epochs)}/{len(data_epochs)} ({artifact_ratio:.3f})",
            metadata={
                'artifact_epochs': artifact_epochs,
                'artifact_ratio': float(artifact_ratio),
                'artifact_threshold': self.config['artifact_threshold']
            }
        ))
        
        return results

    def _check_label_integrity(self, labels: np.ndarray, file_path: str) -> List[ValidationResult]:
        """Validate sleep stage labels"""
        results = []
        
        # Check label range (0-4 for 5-class, 0-3 for 4-class)
        unique_labels = np.unique(labels)
        expected_labels = set(range(5))  # 0-4
        unexpected_labels = set(unique_labels) - expected_labels
        
        if unexpected_labels:
            results.append(ValidationResult(
                test_name="label_range",
                status=False,
                severity=ValidationSeverity.ERROR,
                message=f"Unexpected label values: {unexpected_labels}",
                metadata={'unexpected_labels': list(unexpected_labels), 'unique_labels': unique_labels.tolist()}
            ))
        else:
            results.append(ValidationResult(
                test_name="label_range",
                status=True,
                severity=ValidationSeverity.INFO,
                message=f"Valid label range: {unique_labels}",
                metadata={'unique_labels': unique_labels.tolist()}
            ))
        
        # Check label distribution
        label_counts = {int(label): int(count) for label, count in zip(*np.unique(labels, return_counts=True))}
        min_class_ratio = min(label_counts.values()) / len(labels)
        
        results.append(ValidationResult(
            test_name="label_distribution",
            status=min_class_ratio > 0.01,  # At least 1% per class
            severity=ValidationSeverity.WARNING if min_class_ratio <= 0.01 else ValidationSeverity.INFO,
            message=f"Label distribution: {label_counts}, min class ratio: {min_class_ratio:.3f}",
            metadata={'label_distribution': label_counts, 'min_class_ratio': float(min_class_ratio)}
        ))
        
        return results

    def _check_signal_characteristics(self, data: np.ndarray, file_path: str) -> List[ValidationResult]:
        """Check EEG signal characteristics"""
        results = []
        
        # Basic statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        # Check for DC offset (mean should be close to 0)
        dc_offset_ok = abs(mean_val) < 50  # Less than 50 µV DC offset
        results.append(ValidationResult(
            test_name="dc_offset",
            status=dc_offset_ok,
            severity=ValidationSeverity.WARNING if not dc_offset_ok else ValidationSeverity.INFO,
            message=f"DC offset: {mean_val:.2f} µV",
            metadata={'dc_offset': float(mean_val), 'std': float(std_val)}
        ))
        
        return results

class DataQualityChecker:
    """Aggregate data quality analysis across datasets"""
    
    def __init__(self, validator: EEGDataValidator = None):
        self.validator = validator or EEGDataValidator()
        self.quality_report = {
            'total_files': 0,
            'passed_files': 0,
            'failed_files': 0,
            'warnings': 0,
            'errors': 0,
            'validation_results': []
        }
    
    def validate_dataset(self, dataset) -> Dict[str, Any]:
        """Run validation on entire dataset"""
        self.quality_report = {
            'total_files': 0,
            'passed_files': 0,
            'failed_files': 0,
            'warnings': 0,
            'errors': 0,
            'validation_results': []
        }
        
        # Get metadata from dataset
        metadata = dataset.get_metadata() if hasattr(dataset, 'get_metadata') else []
        
        if hasattr(dataset, 'seqs_labels_path_pair'):
            # Validate original files
            for seq_path, label_path in dataset.seqs_labels_path_pair:
                try:
                    seq_data = np.load(seq_path)
                    label_data = np.load(label_path)
                    
                    file_results = self.validator.validate_single_file(
                        seq_data, label_data, seq_path
                    )
                    
                    self._update_quality_report(file_results, seq_path)
                    
                except Exception as e:
                    error_result = ValidationResult(
                        test_name="file_loading",
                        status=False,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Failed to load {seq_path}: {str(e)}",
                        metadata={'error': str(e), 'file_path': seq_path}
                    )
                    self._update_quality_report([error_result], seq_path)
        
        return self.quality_report
    
    def _update_quality_report(self, results: List[ValidationResult], file_path: str):
        """Update quality report with validation results"""
        self.quality_report['total_files'] += 1
        
        has_errors = any(r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                        and not r.status for r in results)
        has_warnings = any(r.severity == ValidationSeverity.WARNING and not r.status for r in results)
        
        if has_errors:
            self.quality_report['failed_files'] += 1
            self.quality_report['errors'] += sum(1 for r in results 
                                               if r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                                               and not r.status)
        else:
            self.quality_report['passed_files'] += 1
        
        if has_warnings:
            self.quality_report['warnings'] += sum(1 for r in results 
                                                  if r.severity == ValidationSeverity.WARNING 
                                                  and not r.status)
        
        # Store detailed results
        file_summary = {
            'file_path': file_path,
            'status': 'passed' if not has_errors else 'failed',
            'results': [
                {
                    'test_name': r.test_name,
                    'status': r.status,
                    'severity': r.severity.value,
                    'message': r.message,
                    'metadata': r.metadata
                } for r in results
            ]
        }
        self.quality_report['validation_results'].append(file_summary)
    
    def save_report(self, output_path: str):
        """Save detailed quality report"""
        with open(output_path, 'w') as f:
            json.dump(self.quality_report, f, indent=2, default=str)
    
    def print_summary(self):
        """Print quality summary"""
        report = self.quality_report
        print(f"\n📊 Data Quality Report")
        print(f"{'='*50}")
        print(f"Total files: {report['total_files']}")
        print(f"✅ Passed: {report['passed_files']} ({report['passed_files']/max(report['total_files'],1)*100:.1f}%)")
        print(f"❌ Failed: {report['failed_files']} ({report['failed_files']/max(report['total_files'],1)*100:.1f}%)")
        print(f"⚠️  Warnings: {report['warnings']}")
        print(f"🚨 Errors: {report['errors']}")
        print(f"{'='*50}")