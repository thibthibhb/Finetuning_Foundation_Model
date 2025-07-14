"""
Custom exceptions for configuration management.
"""


class ConfigError(Exception):
    """Base exception for configuration-related errors."""
    pass


class ValidationError(ConfigError):
    """Raised when configuration validation fails."""
    pass


class EnvironmentError(ConfigError):
    """Raised when environment-related configuration issues occur."""
    pass


class SecretError(ConfigError):
    """Raised when secret management operations fail."""
    pass


class SchemaError(ConfigError):
    """Raised when configuration schema is invalid."""
    pass