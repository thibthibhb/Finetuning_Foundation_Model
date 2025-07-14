"""
Environment variable loader with .env file support.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union

from .exceptions import EnvironmentError


logger = logging.getLogger(__name__)


class EnvironmentLoader:
    """
    Environment variable loader that supports .env files and provides
    type conversion and validation.
    """
    
    def __init__(self, env_file: Optional[Union[str, Path]] = None):
        """
        Initialize the environment loader.
        
        Args:
            env_file: Path to .env file. If None, looks for .env in current directory.
        """
        self.env_file = env_file
        self.loaded_vars: Dict[str, str] = {}
        
        # Try to load .env file
        self._load_env_file()
    
    def _load_env_file(self) -> None:
        """Load variables from .env file if it exists."""
        if self.env_file is None:
            # Look for .env in current directory and parent directories
            current_dir = Path.cwd()
            while current_dir != current_dir.parent:
                env_path = current_dir / '.env'
                if env_path.exists():
                    self.env_file = env_path
                    break
                current_dir = current_dir.parent
        
        if self.env_file and Path(self.env_file).exists():
            try:
                self._parse_env_file(Path(self.env_file))
                logger.info(f"Loaded environment variables from {self.env_file}")
            except Exception as e:
                logger.warning(f"Failed to load .env file {self.env_file}: {e}")
        else:
            logger.debug("No .env file found or specified")
    
    def _parse_env_file(self, env_path: Path) -> None:
        """Parse .env file and load variables."""
        with open(env_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse key=value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # Set environment variable if not already set
                    if key not in os.environ:
                        os.environ[key] = value
                        self.loaded_vars[key] = value
                else:
                    logger.warning(f"Invalid line in .env file at line {line_num}: {line}")
    
    def get_env(self, 
                key: str, 
                default: Optional[str] = None, 
                required: bool = False) -> Optional[str]:
        """
        Get environment variable value.
        
        Args:
            key: Environment variable name
            default: Default value if variable is not set
            required: Whether the variable is required
            
        Returns:
            Environment variable value or default
            
        Raises:
            EnvironmentError: If required variable is not set
        """
        value = os.getenv(key, default)
        
        if required and value is None:
            raise EnvironmentError(f"Required environment variable not set: {key}")
        
        return value
    
    def get_env_bool(self, 
                     key: str, 
                     default: bool = False, 
                     required: bool = False) -> bool:
        """
        Get environment variable as boolean.
        
        Args:
            key: Environment variable name
            default: Default value if variable is not set
            required: Whether the variable is required
            
        Returns:
            Boolean value
        """
        value = self.get_env(key, str(default) if not required else None, required)
        
        if value is None:
            return default
        
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
    
    def get_env_int(self, 
                    key: str, 
                    default: Optional[int] = None, 
                    required: bool = False) -> Optional[int]:
        """
        Get environment variable as integer.
        
        Args:
            key: Environment variable name
            default: Default value if variable is not set
            required: Whether the variable is required
            
        Returns:
            Integer value or default
            
        Raises:
            EnvironmentError: If value cannot be converted to int
        """
        value = self.get_env(key, str(default) if default is not None else None, required)
        
        if value is None:
            return default
        
        try:
            return int(value)
        except ValueError as e:
            raise EnvironmentError(f"Cannot convert {key}='{value}' to integer") from e
    
    def get_env_float(self, 
                      key: str, 
                      default: Optional[float] = None, 
                      required: bool = False) -> Optional[float]:
        """
        Get environment variable as float.
        
        Args:
            key: Environment variable name
            default: Default value if variable is not set
            required: Whether the variable is required
            
        Returns:
            Float value or default
            
        Raises:
            EnvironmentError: If value cannot be converted to float
        """
        value = self.get_env(key, str(default) if default is not None else None, required)
        
        if value is None:
            return default
        
        try:
            return float(value)
        except ValueError as e:
            raise EnvironmentError(f"Cannot convert {key}='{value}' to float") from e
    
    def get_env_list(self, 
                     key: str, 
                     separator: str = ',', 
                     default: Optional[list] = None, 
                     required: bool = False) -> Optional[list]:
        """
        Get environment variable as list.
        
        Args:
            key: Environment variable name
            separator: Separator for splitting the value
            default: Default value if variable is not set
            required: Whether the variable is required
            
        Returns:
            List of values or default
        """
        value = self.get_env(key, None, required)
        
        if value is None:
            return default or []
        
        return [item.strip() for item in value.split(separator) if item.strip()]
    
    def get_all_env_vars(self) -> Dict[str, str]:
        """Get all environment variables as a dictionary."""
        return dict(os.environ)
    
    def get_loaded_vars(self) -> Dict[str, str]:
        """Get variables that were loaded from .env file."""
        return self.loaded_vars.copy()


# Global environment loader instance
_env_loader: Optional[EnvironmentLoader] = None


def get_env_loader() -> EnvironmentLoader:
    """Get the global environment loader instance."""
    global _env_loader
    if _env_loader is None:
        _env_loader = EnvironmentLoader()
    return _env_loader


def get_env(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Get environment variable value."""
    return get_env_loader().get_env(key, default, required)


def get_env_bool(key: str, default: bool = False, required: bool = False) -> bool:
    """Get environment variable as boolean."""
    return get_env_loader().get_env_bool(key, default, required)


def get_env_int(key: str, default: Optional[int] = None, required: bool = False) -> Optional[int]:
    """Get environment variable as integer."""
    return get_env_loader().get_env_int(key, default, required)


def get_env_float(key: str, default: Optional[float] = None, required: bool = False) -> Optional[float]:
    """Get environment variable as float."""
    return get_env_loader().get_env_float(key, default, required)