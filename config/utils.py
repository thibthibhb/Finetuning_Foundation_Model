"""
Utility functions for configuration management.
"""

import argparse
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

from .config_loader import load_config, get_config_loader
from .env_loader import get_env
from .secrets import get_secrets_manager
from .exceptions import ConfigError


logger = logging.getLogger(__name__)


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Setup logging based on configuration.
    
    Args:
        config: Configuration dictionary
    """
    log_config = config.get('experiment', {}).get('logging', {})
    
    level = log_config.get('level', 'INFO')
    format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        force=True  # Override any existing configuration
    )
    
    # Create log directory if saving logs
    if log_config.get('save_logs', False):
        log_dir = Path(config.get('paths', {}).get('log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Add file handler
        file_handler = logging.FileHandler(log_dir / 'cbramod.log')
        file_handler.setFormatter(logging.Formatter(format_str))
        logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Logging configured with level: {level}")


def create_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary directories based on configuration.
    
    Args:
        config: Configuration dictionary
    """
    paths_config = config.get('paths', {})
    
    directories = [
        'model_dir',
        'log_dir', 
        'results_dir',
        'temp_dir',
        'cache_dir'
    ]
    
    for dir_key in directories:
        if dir_key in paths_config:
            dir_path = Path(paths_config[dir_key])
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")


def convert_config_to_args(config: Dict[str, Any]) -> argparse.Namespace:
    """
    Convert configuration dictionary to argparse.Namespace for compatibility
    with existing argument-based code.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        argparse.Namespace with configuration values
    """
    args = argparse.Namespace()
    
    # Flatten the configuration dictionary
    flattened = flatten_config(config)
    
    # Set attributes on the namespace
    for key, value in flattened.items():
        setattr(args, key, value)
    
    return args


def flatten_config(config: Dict[str, Any], prefix: str = '', separator: str = '_') -> Dict[str, Any]:
    """
    Flatten a nested configuration dictionary.
    
    Args:
        config: Configuration dictionary to flatten
        prefix: Prefix for keys
        separator: Separator for nested keys
        
    Returns:
        Flattened dictionary
        
    Example:
        >>> config = {'model': {'backbone': {'d_model': 200}}}
        >>> flatten_config(config)
        {'model_backbone_d_model': 200}
    """
    flattened = {}
    
    for key, value in config.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            flattened.update(flatten_config(value, new_key, separator))
        else:
            flattened[new_key] = value
    
    return flattened


def unflatten_config(flattened: Dict[str, Any], separator: str = '_') -> Dict[str, Any]:
    """
    Unflatten a configuration dictionary.
    
    Args:
        flattened: Flattened configuration dictionary
        separator: Separator used for nested keys
        
    Returns:
        Nested configuration dictionary
    """
    config = {}
    
    for key, value in flattened.items():
        keys = key.split(separator)
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs take precedence over earlier ones.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    from .config_loader import ConfigLoader
    
    loader = ConfigLoader()
    result = {}
    
    for config in configs:
        result = loader._merge_configs(result, config)
    
    return result


def validate_paths(config: Dict[str, Any]) -> None:
    """
    Validate that required paths exist and are accessible.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ConfigError: If required paths don't exist or are not accessible
    """
    paths_config = config.get('paths', {})
    
    # Check if datasets directory exists
    if 'datasets_dir' in paths_config:
        datasets_dir = Path(paths_config['datasets_dir'])
        if not datasets_dir.exists():
            logger.warning(f"Datasets directory does not exist: {datasets_dir}")
    
    # Check pretrained weights path
    model_config = config.get('model', {})
    if model_config.get('pretrained_weights', {}).get('enabled', False):
        weights_path = Path(model_config['pretrained_weights']['path'])
        if not weights_path.exists():
            logger.warning(f"Pretrained weights file does not exist: {weights_path}")


def setup_environment_from_config(config: Dict[str, Any]) -> None:
    """
    Setup environment variables based on configuration.
    
    Args:
        config: Configuration dictionary
    """
    import os
    
    # Set CUDA device
    device_config = config.get('device', {})
    if device_config.get('cuda', {}).get('enabled', False):
        device_id = device_config['cuda'].get('device_id', 0)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    
    # Set reproducibility settings
    repro_config = config.get('reproducibility', {})
    if repro_config.get('deterministic', False):
        os.environ['PYTHONHASHSEED'] = str(repro_config.get('seed', 3407))


def get_device_from_config(config: Dict[str, Any]) -> str:
    """
    Get device string from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Device string (e.g., 'cuda:0', 'cpu')
    """
    device_config = config.get('device', {})
    
    if device_config.get('cuda', {}).get('enabled', False):
        device_id = device_config['cuda'].get('device_id', 0)
        return f"cuda:{device_id}"
    else:
        return "cpu"


def init_config_system(environment: Optional[str] = None, 
                      config_dir: Optional[str] = None,
                      create_dirs: bool = True,
                      setup_logs: bool = True) -> Dict[str, Any]:
    """
    Initialize the complete configuration system.
    
    Args:
        environment: Environment name (development, production, testing)
        config_dir: Configuration directory path
        create_dirs: Whether to create necessary directories
        setup_logs: Whether to setup logging
        
    Returns:
        Loaded configuration dictionary
    """
    try:
        # Determine environment
        if environment is None:
            environment = get_env('CBRAMOD_ENV', 'development')
        
        # Load configuration
        config = load_config(environment=environment)
        
        # Setup logging
        if setup_logs:
            setup_logging(config)
        
        # Create directories
        if create_dirs:
            create_directories(config)
        
        # Setup environment variables
        setup_environment_from_config(config)
        
        # Validate paths
        validate_paths(config)
        
        logger.info(f"Configuration system initialized for environment: {environment}")
        return config
        
    except Exception as e:
        # Use basic logging if config logging failed
        logging.basicConfig(level=logging.ERROR)
        logger.error(f"Failed to initialize configuration system: {e}")
        raise


def get_wandb_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get Weights & Biases configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        WandB configuration dictionary
    """
    secrets_manager = get_secrets_manager()
    wandb_config = secrets_manager.get_wandb_config()
    
    # Merge with config file settings
    exp_config = config.get('experiment', {}).get('tracking', {})
    
    return {
        'project': wandb_config.get('project') or exp_config.get('project_name'),
        'entity': wandb_config.get('entity'),
        'name': exp_config.get('run_name'),
        'config': config,
        'mode': 'online' if exp_config.get('enabled', True) else 'disabled'
    }


class ConfigManager:
    """
    High-level configuration manager that provides easy access to configuration values.
    """
    
    def __init__(self, environment: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            environment: Environment name
        """
        self.config = init_config_system(environment)
        self.secrets = get_secrets_manager()
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by key path."""
        loader = get_config_loader()
        return loader.get_value(key_path, default)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config.get('training', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.config.get('data', {})
    
    def get_device(self) -> str:
        """Get device string."""
        return get_device_from_config(self.config)
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.config.get('debug', {}).get('enabled', False)
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret value."""
        return self.secrets.get_secret(key, default)