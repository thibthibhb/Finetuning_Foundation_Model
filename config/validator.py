"""
Configuration validation using schema validation.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from .exceptions import ValidationError, SchemaError


logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Configuration validator that checks configuration against predefined schemas.
    """
    
    def __init__(self):
        """Initialize the configuration validator."""
        self.schema = self._get_base_schema()
    
    def validate(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValidationError: If validation fails
        """
        errors = []
        
        try:
            # Validate required sections
            errors.extend(self._validate_required_sections(config))
            
            # Validate model configuration
            if 'model' in config:
                errors.extend(self._validate_model_config(config['model']))
            
            # Validate training configuration
            if 'training' in config:
                errors.extend(self._validate_training_config(config['training']))
            
            # Validate data configuration
            if 'data' in config:
                errors.extend(self._validate_data_config(config['data']))
            
            # Validate paths configuration
            if 'paths' in config:
                errors.extend(self._validate_paths_config(config['paths']))
            
            # Validate device configuration
            if 'device' in config:
                errors.extend(self._validate_device_config(config['device']))
            
            if errors:
                error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
                raise ValidationError(error_msg)
            
            logger.info("Configuration validation passed")
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Validation error: {str(e)}") from e
    
    def _validate_required_sections(self, config: Dict[str, Any]) -> List[str]:
        """Validate that required configuration sections are present."""
        errors = []
        required_sections = ['model', 'training', 'data', 'paths', 'device']
        
        for section in required_sections:
            if section not in config:
                errors.append(f"Required section missing: {section}")
        
        return errors
    
    def _validate_model_config(self, model_config: Dict[str, Any]) -> List[str]:
        """Validate model configuration."""
        errors = []
        
        # Validate backbone configuration
        if 'backbone' in model_config:
            backbone = model_config['backbone']
            required_fields = ['in_dim', 'out_dim', 'd_model', 'n_layer', 'nhead']
            
            for field in required_fields:
                if field not in backbone:
                    errors.append(f"model.backbone.{field} is required")
                elif not isinstance(backbone[field], int) or backbone[field] <= 0:
                    errors.append(f"model.backbone.{field} must be a positive integer")
            
            # Validate dropout
            if 'dropout' in backbone:
                dropout = backbone['dropout']
                if not isinstance(dropout, (int, float)) or not (0.0 <= dropout <= 1.0):
                    errors.append("model.backbone.dropout must be between 0.0 and 1.0")
        
        # Validate pretrained weights configuration
        if 'pretrained_weights' in model_config:
            pw_config = model_config['pretrained_weights']
            if 'enabled' in pw_config and pw_config['enabled']:
                if 'path' not in pw_config or not pw_config['path']:
                    errors.append("model.pretrained_weights.path is required when enabled")
        
        return errors
    
    def _validate_training_config(self, training_config: Dict[str, Any]) -> List[str]:
        """Validate training configuration."""
        errors = []
        
        # Validate basic training parameters
        numeric_fields = {
            'epochs': (int, lambda x: x > 0),
            'batch_size': (int, lambda x: x > 0),
            'learning_rate': ((int, float), lambda x: x > 0),
            'weight_decay': ((int, float), lambda x: x >= 0),
            'clip_value': ((int, float), lambda x: x > 0),
            'label_smoothing': ((int, float), lambda x: 0 <= x <= 1)
        }
        
        for field, (expected_type, validator) in numeric_fields.items():
            if field in training_config:
                value = training_config[field]
                if not isinstance(value, expected_type):
                    errors.append(f"training.{field} must be of type {expected_type}")
                elif not validator(value):
                    errors.append(f"training.{field} has invalid value: {value}")
        
        # Validate optimizer
        if 'optimizer' in training_config:
            valid_optimizers = ['Adam', 'AdamW', 'SGD', 'Lion']
            if training_config['optimizer'] not in valid_optimizers:
                errors.append(f"training.optimizer must be one of: {valid_optimizers}")
        
        # Validate scheduler configuration
        if 'scheduler' in training_config:
            scheduler_config = training_config['scheduler']
            if 'type' in scheduler_config:
                valid_schedulers = ['cosine', 'cosine_warmup', 'step', 'plateau', 'exponential', 'none']
                if scheduler_config['type'] not in valid_schedulers:
                    errors.append(f"training.scheduler.type must be one of: {valid_schedulers}")
        
        return errors
    
    def _validate_data_config(self, data_config: Dict[str, Any]) -> List[str]:
        """Validate data configuration."""
        errors = []
        
        # Validate numeric fields
        numeric_fields = {
            'sample_rate': (int, lambda x: x > 0),
            'num_workers': (int, lambda x: x >= 0)
        }
        
        for field, (expected_type, validator) in numeric_fields.items():
            if field in data_config:
                value = data_config[field]
                if not isinstance(value, expected_type):
                    errors.append(f"data.{field} must be of type {expected_type}")
                elif not validator(value):
                    errors.append(f"data.{field} has invalid value: {value}")
        
        # Validate data fraction
        if 'data_fraction' in data_config:
            data_fraction = data_config['data_fraction']
            if not isinstance(data_fraction, (int, float)) or not (0 < data_fraction <= 1):
                errors.append("data.data_fraction must be between 0 and 1")
        
        # Validate splits
        if 'splits' in data_config:
            splits = data_config['splits']
            split_sum = 0
            for split_name in ['train', 'val', 'test']:
                if split_name in splits:
                    split_value = splits[split_name]
                    if not isinstance(split_value, (int, float)) or not (0 < split_value < 1):
                        errors.append(f"data.splits.{split_name} must be between 0 and 1")
                    split_sum += split_value
            
            if abs(split_sum - 1.0) > 0.01:  # Allow small floating point errors
                errors.append(f"data.splits must sum to 1.0, got {split_sum}")
        
        return errors
    
    def _validate_paths_config(self, paths_config: Dict[str, Any]) -> List[str]:
        """Validate paths configuration."""
        errors = []
        
        required_paths = ['datasets_dir', 'model_dir', 'log_dir']
        
        for path_name in required_paths:
            if path_name not in paths_config:
                errors.append(f"paths.{path_name} is required")
            elif not isinstance(paths_config[path_name], str):
                errors.append(f"paths.{path_name} must be a string")
        
        return errors
    
    def _validate_device_config(self, device_config: Dict[str, Any]) -> List[str]:
        """Validate device configuration."""
        errors = []
        
        # Validate CUDA configuration
        if 'cuda' in device_config:
            cuda_config = device_config['cuda']
            
            if 'enabled' in cuda_config and not isinstance(cuda_config['enabled'], bool):
                errors.append("device.cuda.enabled must be a boolean")
            
            if 'device_id' in cuda_config:
                device_id = cuda_config['device_id']
                if not isinstance(device_id, int) or device_id < 0:
                    errors.append("device.cuda.device_id must be a non-negative integer")
        
        # Validate distributed configuration
        if 'distributed' in device_config:
            dist_config = device_config['distributed']
            
            if 'enabled' in dist_config and not isinstance(dist_config['enabled'], bool):
                errors.append("device.distributed.enabled must be a boolean")
            
            if 'backend' in dist_config:
                valid_backends = ['nccl', 'gloo', 'mpi']
                if dist_config['backend'] not in valid_backends:
                    errors.append(f"device.distributed.backend must be one of: {valid_backends}")
        
        return errors
    
    def _get_base_schema(self) -> Dict[str, Any]:
        """Get the base schema definition."""
        # This is a simplified schema definition
        # In a more robust implementation, you might use libraries like cerberus or jsonschema
        return {
            "model": {
                "required": True,
                "type": "dict"
            },
            "training": {
                "required": True,
                "type": "dict"
            },
            "data": {
                "required": True,
                "type": "dict"
            },
            "paths": {
                "required": True,
                "type": "dict"
            },
            "device": {
                "required": True,
                "type": "dict"
            }
        }