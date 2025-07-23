import os
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Import the original dataset
from .idun_datasets import MemoryEfficientKFoldDataset, LoadDataset

# Import data pipeline components
from data_pipeline.validators import EEGDataValidator, DataQualityChecker
from data_pipeline.versioning import DataVersionManager
from data_pipeline.monitoring import DataQualityMonitor
from data_pipeline.lineage import DataLineageTracker

# Import class configuration system
from class_configurations import ClassConfigurationManager, create_enhanced_dataset_with_classes

class EnhancedEEGDataset(MemoryEfficientKFoldDataset):
    """Enhanced EEG dataset with comprehensive data pipeline integration"""
    
    def __init__(self, seqs_labels_path_pair, enable_data_pipeline: bool = True,
                 dataset_name: str = "eeg_dataset", dataset_version: str = "1.0",
                 classification_scheme: str = "4_class"):
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.enable_data_pipeline = enable_data_pipeline
        self.classification_scheme = classification_scheme
        self.logger = logging.getLogger(__name__)
        
        # Initialize class configuration BEFORE parent init
        self.class_manager = ClassConfigurationManager()
        self.class_config = self.class_manager.get_configuration(classification_scheme)
        self.logger.info(f"🏷️ Using {self.class_config.name} classification with {self.class_config.num_of_classes} classes")
        
        # Initialize data pipeline components
        if self.enable_data_pipeline:
            self.validator = EEGDataValidator()
            self.quality_checker = DataQualityChecker(self.validator)
            self.version_manager = DataVersionManager()
            self.quality_monitor = DataQualityMonitor()
            self.lineage_tracker = DataLineageTracker()
            
            # Register dataset version
            self.dataset_id = self._register_dataset_version(seqs_labels_path_pair)
            
            # Start lineage tracking
            self.transformation_id = self.lineage_tracker.start_transformation(
                name="dataset_loading",
                description=f"Loading {dataset_name} dataset",
                parameters={
                    'dataset_name': dataset_name,
                    'version': dataset_version,
                    'file_count': len(seqs_labels_path_pair)
                }
            )
        
        # Run data validation before loading
        if self.enable_data_pipeline:
            self._run_pre_load_validation(seqs_labels_path_pair)
        
        # Custom data loading with proper class handling
        self._load_data_with_custom_classes(seqs_labels_path_pair)
        
        # Post-load activities
        if self.enable_data_pipeline:
            self._run_post_load_activities()
    
    def _load_data_with_custom_classes(self, seqs_labels_path_pair):
        """Custom data loading that respects classification scheme"""
        self.samples = []
        self.metadata = []
        
        for seq_path, label_path in seqs_labels_path_pair:
            base = os.path.basename(seq_path)
            sid = base.split("_")[0]

            try:
                seq = np.load(seq_path)
                label = np.load(label_path)
            except Exception as e:
                print(f"[❌] Failed loading {seq_path}: {e}")
                continue

            if seq.shape[0] != label.shape[0]:
                print(f"[ERROR] Mismatch: {seq.shape[0]} vs {label.shape[0]} in {seq_path}")
                continue

            # Use parent's integrity check method
            if not self.check_integrity(seq, seq_path):
                print(f"[❌] Integrity check failed: {seq_path}")
                continue

            label = label.astype(int)
            
            # Apply class mapping based on classification scheme
            if self.classification_scheme == "4_class":
                # Apply 4-class remapping: 0,1→0, 2→1, 3→2, 4→3
                label = np.array([self._apply_4_class_mapping(l) for l in label], dtype=int)
                self.logger.debug(f"Applied 4-class mapping for {base}")
            elif self.classification_scheme == "5_class":
                # Keep original labels: 0→0, 1→1, 2→2, 3→3, 4→4
                label = np.array([self._apply_5_class_mapping(l) for l in label], dtype=int)
                self.logger.debug(f"Applied 5-class mapping (identity) for {base}")
            else:
                # Use configurable mapping
                label = np.array([self.remap_label(l) for l in label], dtype=int)

            for i in range(len(label)):
                if label[i] is None or label[i] < 0:
                    continue
                self.samples.append((seq[i], int(label[i])))
                self.metadata.append({
                    'subject': sid,
                    'file': base,
                    'index': i,
                    'path': seq_path
                })

        if len(self.samples) == 0:
            raise ValueError("All samples filtered out. Empty dataset.")

        # Log the class distribution
        all_labels = [sample[1] for sample in self.samples]
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        
        print(f"[✅] Loaded total {len(self.samples)} valid samples from {len(seqs_labels_path_pair)} files.")
        print(f"[📊] Class distribution: {dict(zip(unique_labels, counts))}")
        print(f"[🏷️] Using {self.classification_scheme} with {len(unique_labels)} unique labels: {unique_labels}")
    
    def _apply_4_class_mapping(self, l):
        """Apply 4-class mapping: merge wake and movement"""
        if l in [0, 1]:
            return 0  # Wake (includes movement)
        elif l == 2:
            return 1  # Light
        elif l == 3:
            return 2  # Deep
        elif l == 4:
            return 3  # REM
        else:
            raise ValueError(f"Unknown label value: {l}")
    
    def _apply_5_class_mapping(self, l):
        """Apply 5-class mapping: keep all classes separate"""
        if l in [0, 1, 2, 3, 4]:
            return l  # Identity mapping
        else:
            raise ValueError(f"Unknown label value: {l}")
    
    def remap_label(self, l):
        """Override the remap_label method to use configurable mapping"""
        try:
            if l in self.class_config.label_mapping:
                return self.class_config.label_mapping[l]
            else:
                raise ValueError(f"Unknown label value: {l} for scheme {self.classification_scheme}")
        except Exception as e:
            self.logger.error(f"Error remapping label {l}: {e}")
            raise
    
    def check_integrity(self, data, filename):
        """Use parent class's integrity check method"""
        if np.isnan(data).any() or np.isinf(data).any():
            return False
        if np.abs(data).max() > 1e6:
            return False
        variances = np.var(data.reshape(data.shape[0], -1), axis=1)
        if np.sum(variances == 0) > 0:
            return False
        if np.any(np.abs(data) >= 32768):
            return False
        return True
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample by index"""
        from utils.util import to_tensor
        import torch
        epoch, label = self.samples[idx]
        return to_tensor(epoch), torch.tensor(label, dtype=torch.long)

    def get_metadata(self):
        """Return metadata for the dataset"""
        return self.metadata
    
    def _register_dataset_version(self, seqs_labels_path_pair) -> str:
        """Register dataset version for tracking"""
        # Calculate metadata
        metadata = {
            'file_count': len(seqs_labels_path_pair),
            'sample_files': [str(p[0]) for p in seqs_labels_path_pair[:3]],  # First 3 as examples
            'created_at': datetime.now().isoformat(),
            'format': 'numpy_arrays',
            'label_mapping': {
                '0': 'Wake (0,1->0)',
                '1': 'Light (2->1)', 
                '2': 'Deep (3->2)',
                '3': 'REM (4->3)'
            }
        }
        
        # Use first file path as representative path
        representative_path = str(Path(seqs_labels_path_pair[0][0]).parent.parent)

        # Try to register with version manager, but handle path issues gracefully
        dataset_id = None
        try:
            dataset_id = self.version_manager.register_dataset(
                dataset_name=self.dataset_name,
                description=f"EEG sleep staging dataset v{self.dataset_version}",
                tags=['eeg', 'sleep_staging', 'preprocessed']
            )
        except FileNotFoundError as e:
            self.logger.warning(f"Could not register dataset with version manager: {e}")
            self.logger.info("Continuing with lineage tracking only...")
            dataset_id = f"{self.dataset_name}_{self.dataset_version}_manual"
        
        # Register in lineage tracker
        lineage_dataset_id = self.lineage_tracker.register_dataset_version(
            name=self.dataset_name,
            version=self.dataset_version,
            path=representative_path,
            metadata=metadata
        )
        
        return lineage_dataset_id
    
    def _run_pre_load_validation(self, seqs_labels_path_pair):
        """Run validation before loading data"""
        self.logger.info(f"🔍 Running pre-load validation on {len(seqs_labels_path_pair)} files...")
        
        # Sample a few files for quick validation
        sample_size = min(5, len(seqs_labels_path_pair))
        sample_files = seqs_labels_path_pair[:sample_size]
        
        validation_results = []
        for seq_path, label_path in sample_files:
            try:
                seq_data = np.load(seq_path)
                label_data = np.load(label_path)
                
                file_results = self.validator.validate_single_file(
                    seq_data, label_data, seq_path
                )
                validation_results.extend(file_results)
                
            except Exception as e:
                self.logger.error(f"Validation error for {seq_path}: {e}")
        
        # Check for critical issues
        critical_issues = [r for r in validation_results 
                          if not r.status and r.severity.value in ['critical', 'error']]
        
        if critical_issues:
            error_msg = f"Critical data quality issues found: {len(critical_issues)} errors"
            self.logger.error(error_msg)
            for issue in critical_issues[:3]:  # Show first 3
                self.logger.error(f"  - {issue.message}")
            
            # You might want to raise an exception here in production
            # raise ValueError(error_msg)
        else:
            self.logger.info("✅ Pre-load validation passed")
    
    def _run_post_load_activities(self):
        """Run activities after successful data loading"""
        try:
            # Record quality metrics
            self._record_quality_metrics()
            
            # Run full quality monitoring
            self._run_quality_monitoring()
            
            # Complete lineage tracking
            self.lineage_tracker.end_transformation(
                self.transformation_id, 
                status="completed"
            )
            
            self.logger.info(f"✅ Successfully loaded {len(self.samples)} samples with data pipeline")
            
        except Exception as e:
            self.logger.error(f"Error in post-load activities: {e}")
            if hasattr(self, 'transformation_id'):
                self.lineage_tracker.end_transformation(
                    self.transformation_id,
                    status="failed"
                )
    
    def _record_quality_metrics(self):
        """Record quality metrics for the loaded dataset"""
        if not hasattr(self, 'samples') or not self.samples:
            return
        
        # Calculate dataset-level metrics
        total_samples = len(self.samples)
        
        # Label distribution
        labels = [sample[1] for sample in self.samples]
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_distribution = dict(zip(unique_labels.tolist(), counts.tolist()))
        
        # Record metrics in lineage tracker
        self.lineage_tracker.record_quality_metric(
            self.dataset_id, "total_samples", float(total_samples)
        )
        
        self.lineage_tracker.record_quality_metric(
            self.dataset_id, "unique_classes", float(len(unique_labels))
        )
        
        # Record class balance (minimum class percentage)
        min_class_pct = min(counts) / total_samples * 100
        self.lineage_tracker.record_quality_metric(
            self.dataset_id, "min_class_percentage", float(min_class_pct)
        )
        
        # Record subject count if available
        if hasattr(self, 'metadata'):
            unique_subjects = len(set(meta['subject'] for meta in self.metadata))
            self.lineage_tracker.record_quality_metric(
                self.dataset_id, "unique_subjects", float(unique_subjects)
            )
    
    def _run_quality_monitoring(self):
        """Run comprehensive quality monitoring"""
        try:
            # Create a temporary dataset-like object for monitoring
            temp_dataset = type('TempDataset', (), {
                'seqs_labels_path_pair': getattr(self, 'seqs_labels_path_pair', []),
                'get_metadata': lambda: getattr(self, 'metadata', [])
            })()
            
            # Run quality monitoring
            monitor_result = self.quality_monitor.run_quality_check(
                temp_dataset, self.dataset_name
            )
            
            # Log any alerts
            if monitor_result['alerts']:
                self.logger.warning(f"⚠️ Quality alerts for {self.dataset_name}:")
                for alert in monitor_result['alerts']:
                    self.logger.warning(f"  - {alert['message']}")
        
        except Exception as e:
            self.logger.error(f"Error in quality monitoring: {e}")
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive data quality report"""
        if not self.enable_data_pipeline:
            return {"error": "Data pipeline not enabled"}
        
        try:
            # Create temporary dataset for validation
            temp_dataset = type('TempDataset', (), {
                'seqs_labels_path_pair': getattr(self, 'seqs_labels_path_pair', []),
                'get_metadata': lambda: getattr(self, 'metadata', [])
            })()
            
            quality_report = self.quality_checker.validate_dataset(temp_dataset)
            
            # Add dataset-specific information
            quality_report['dataset_info'] = {
                'name': self.dataset_name,
                'version': self.dataset_version,
                'dataset_id': getattr(self, 'dataset_id', None),
                'total_samples': len(getattr(self, 'samples', [])),
                'unique_subjects': len(set(meta['subject'] for meta in getattr(self, 'metadata', []))),
                'generated_at': datetime.now().isoformat()
            }
            
            return quality_report
            
        except Exception as e:
            return {"error": f"Failed to generate quality report: {str(e)}"}
    
    def get_lineage_info(self) -> Dict[str, Any]:
        """Get lineage information for this dataset"""
        if not self.enable_data_pipeline:
            return {"error": "Data pipeline not enabled"}
        
        try:
            lineage = self.lineage_tracker.get_dataset_lineage(self.dataset_id)
            quality_history = self.lineage_tracker.get_dataset_quality_history(self.dataset_id)
            
            return {
                'dataset_id': self.dataset_id,
                'lineage': lineage,
                'quality_history': quality_history,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to get lineage info: {str(e)}"}
    
    def save_quality_report(self, output_path: str):
        """Save quality report to file"""
        quality_report = self.get_data_quality_report()
        
        with open(output_path, 'w') as f:
            import json
            json.dump(quality_report, f, indent=2, default=str)
        
        self.logger.info(f"💾 Quality report saved to {output_path}")

class EnhancedLoadDataset(LoadDataset):
    """Enhanced data loader with pipeline integration"""
    
    def __init__(self, params, enable_data_pipeline: bool = True):
        self.enable_data_pipeline = enable_data_pipeline
        
        # Initialize data pipeline components
        if self.enable_data_pipeline:
            self.version_manager = DataVersionManager()
        
        # Call parent initialization
        super().__init__(params)
        
        # Register datasets with version manager
        if self.enable_data_pipeline:
            self._register_source_datasets()
    
    def _register_source_datasets(self):
        """Register source datasets for version tracking"""
        for dataset_name in self.dataset_names:
            try:
                dataset_path = os.path.join(self.params.datasets_dir, dataset_name)
                
                if os.path.exists(dataset_path):
                    self.version_manager.register_dataset(
                        dataset_name=dataset_name,
                        description=f"Source dataset: {dataset_name}",
                        tags=['source', 'raw_data']
                    )
                else:
                    logging.getLogger(__name__).debug(f"Dataset path not found for registration: {dataset_path}")
            except Exception as e:
                logging.getLogger(__name__).debug(f"Failed to register {dataset_name}: {e}")
    
    def create_enhanced_dataset(self, dataset_name: str = None, 
                              dataset_version: str = "1.0",
                              classification_scheme: str = "4_class") -> EnhancedEEGDataset:
        """Create enhanced dataset with pipeline integration"""
        if dataset_name is None:
            dataset_name = f"combined_{'+'.join(self.dataset_names)}"
        
        return EnhancedEEGDataset(
            seqs_labels_path_pair=self.seqs_labels_path_pair,
            enable_data_pipeline=self.enable_data_pipeline,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            classification_scheme=classification_scheme
        )