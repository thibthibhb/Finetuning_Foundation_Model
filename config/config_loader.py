"""
Configuration loader with schema validation and environment variable support.
"""

import os
import re
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from copy import deepcopy

from .exceptions import ConfigError, ValidationError, EnvironmentError
from .validator import ConfigValidator


logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Configuration loader that handles environment-specific configs,
    schema validation, and environment variable substitution.
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_dir: Directory containing configuration files.
                       Defaults to 'config' in the current directory.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent
        
        self.config_dir = Path(config_dir)
        self.validator = ConfigValidator()
        self._loaded_config: Optional[Dict[str, Any]] = None
        
        if not self.config_dir.exists():
            raise ConfigError(f"Configuration directory does not exist: {self.config_dir}")
    
    def load_config(self, 
                   environment: str = "development",
                   validate: bool = True,
                   substitute_env_vars: bool = True) -> Dict[str, Any]:
        """
        Load configuration for the specified environment.
        
        Args:
            environment: Environment name (development, production, testing, etc.)
            validate: Whether to validate the configuration against schema
            substitute_env_vars: Whether to substitute environment variables
            
        Returns:
            Loaded and processed configuration dictionary
            
        Raises:
            ConfigError: If configuration cannot be loaded
            ValidationError: If configuration validation fails
        """
        try:
            # Load base configuration
            base_config = self._load_yaml_file(self.config_dir / "base.yaml")
            
            # Load environment-specific configuration
            env_config_path = self.config_dir / "environments" / f"{environment}.yaml"
            if env_config_path.exists():
                env_config = self._load_yaml_file(env_config_path)
                # Merge configurations (environment overrides base)
                config = self._merge_configs(base_config, env_config)
            else:
                logger.warning(f"Environment config not found: {env_config_path}")
                config = base_config
            
            # Add environment metadata
            config['_meta'] = {
                'environment': environment,
                'config_dir': str(self.config_dir),
                'loaded_at': self._get_timestamp()
            }
            
            # Substitute environment variables
            if substitute_env_vars:
                config = self._substitute_env_vars(config)
            
            # Validate configuration
            if validate:
                self.validator.validate(config)
            
            # Cache the loaded configuration
            self._loaded_config = config
            
            logger.info(f"Successfully loaded configuration for environment: {environment}")
            return config
            
        except Exception as e:
            raise ConfigError(f"Failed to load configuration: {str(e)}") from e
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the currently loaded configuration.
        
        Returns:
            Currently loaded configuration
            
        Raises:
            ConfigError: If no configuration has been loaded
        """
        if self._loaded_config is None:
            raise ConfigError("No configuration loaded. Call load_config() first.")
        
        return self._loaded_config.copy()
    
    def get_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get a specific configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., 'model.backbone.d_model')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config_loader.get_value('training.learning_rate')
            5e-4
            >>> config_loader.get_value('model.backbone.n_layer')
            12
        """
        config = self.get_config()
        
        keys = key_path.split('.')
        current = config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise ConfigError(f"Configuration key not found: {key_path}")
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load and parse a YAML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
                if content is None:
                    return {}
                return content
        except FileNotFoundError:
            raise ConfigError(f"Configuration file not found: {file_path}")
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {file_path}: {str(e)}")
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two configuration dictionaries.
        Override values take precedence over base values.
        """
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Substitute environment variables in configuration values.
        
        Supports patterns like:
        - ${VAR_NAME}
        - ${VAR_NAME:default_value}
        """
        def substitute_recursive(obj):
            if isinstance(obj, dict):
                return {key: substitute_recursive(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [substitute_recursive(item) for item in obj]
            elif isinstance(obj, str):
                return self._substitute_string(obj)
            else:
                return obj
        
        return substitute_recursive(config)
    
    def _substitute_string(self, value: str) -> str:
        """Substitute environment variables in a string value."""
        # Pattern: ${VAR_NAME} or ${VAR_NAME:default}
        pattern = r'\$\{([^}]+)\}'
        
        def replacer(match):
            var_expr = match.group(1)
            
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
            else:
                var_name, default_value = var_expr, None
            
            env_value = os.getenv(var_name.strip())
            
            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            else:
                raise EnvironmentError(f"Environment variable not found: {var_name}")
        
        return re.sub(pattern, replacer, value)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO format string."""
        from datetime import datetime
        return datetime.now().isoformat()


# Singleton instance for global access
_config_loader: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """Get the global configuration loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def load_config(environment: str = None, **kwargs) -> Dict[str, Any]:
    """
    Load configuration for the specified environment.
    
    Args:
        environment: Environment name. If None, tries to get from CBRAMOD_ENV
        **kwargs: Additional arguments passed to ConfigLoader.load_config()
        
    Returns:
        Loaded configuration dictionary
    """
    if environment is None:
        environment = os.getenv('CBRAMOD_ENV', 'development')
    
    loader = get_config_loader()
    return loader.load_config(environment=environment, **kwargs)


def get_config() -> Dict[str, Any]:
    """Get the currently loaded configuration."""
    loader = get_config_loader()
    return loader.get_config()