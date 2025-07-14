"""
Secrets management for CBraMod configuration.

Supports multiple backends:
- Environment variables (default)
- AWS Secrets Manager
- HashiCorp Vault
- Local encrypted files
"""

import os
import json
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path
from abc import ABC, abstractmethod

from .exceptions import SecretError
from .env_loader import get_env


logger = logging.getLogger(__name__)


class SecretBackend(ABC):
    """Abstract base class for secret storage backends."""
    
    @abstractmethod
    def get_secret(self, key: str) -> Optional[str]:
        """Get a secret by key."""
        pass
    
    @abstractmethod
    def set_secret(self, key: str, value: str) -> None:
        """Set a secret value."""
        pass
    
    @abstractmethod
    def delete_secret(self, key: str) -> None:
        """Delete a secret."""
        pass
    
    @abstractmethod
    def list_secrets(self) -> list:
        """List all secret keys."""
        pass


class EnvironmentSecretBackend(SecretBackend):
    """Secret backend using environment variables."""
    
    def __init__(self, prefix: str = "CBRAMOD_SECRET_"):
        """
        Initialize environment secret backend.
        
        Args:
            prefix: Prefix for secret environment variables
        """
        self.prefix = prefix
    
    def get_secret(self, key: str) -> Optional[str]:
        """Get secret from environment variable."""
        env_key = f"{self.prefix}{key.upper()}"
        return os.getenv(env_key)
    
    def set_secret(self, key: str, value: str) -> None:
        """Set secret as environment variable."""
        env_key = f"{self.prefix}{key.upper()}"
        os.environ[env_key] = value
    
    def delete_secret(self, key: str) -> None:
        """Delete secret from environment."""
        env_key = f"{self.prefix}{key.upper()}"
        if env_key in os.environ:
            del os.environ[env_key]
    
    def list_secrets(self) -> list:
        """List all secret keys."""
        secrets = []
        for key in os.environ:
            if key.startswith(self.prefix):
                secret_key = key[len(self.prefix):].lower()
                secrets.append(secret_key)
        return secrets


class AWSSecretsManagerBackend(SecretBackend):
    """Secret backend using AWS Secrets Manager."""
    
    def __init__(self, region_name: Optional[str] = None):
        """
        Initialize AWS Secrets Manager backend.
        
        Args:
            region_name: AWS region name
        """
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            self.boto3 = boto3
            self.ClientError = ClientError
            
            if region_name is None:
                region_name = get_env('AWS_REGION', 'us-east-1')
            
            self.client = boto3.client('secretsmanager', region_name=region_name)
            
        except ImportError:
            raise SecretError("boto3 is required for AWS Secrets Manager backend")
    
    def get_secret(self, key: str) -> Optional[str]:
        """Get secret from AWS Secrets Manager."""
        try:
            response = self.client.get_secret_value(SecretId=key)
            return response['SecretString']
        except self.ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                return None
            raise SecretError(f"Failed to get secret {key}: {e}")
    
    def set_secret(self, key: str, value: str) -> None:
        """Set secret in AWS Secrets Manager."""
        try:
            # Try to update existing secret
            self.client.update_secret(SecretId=key, SecretString=value)
        except self.ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                # Create new secret
                self.client.create_secret(Name=key, SecretString=value)
            else:
                raise SecretError(f"Failed to set secret {key}: {e}")
    
    def delete_secret(self, key: str) -> None:
        """Delete secret from AWS Secrets Manager."""
        try:
            self.client.delete_secret(SecretId=key, ForceDeleteWithoutRecovery=True)
        except self.ClientError as e:
            if e.response['Error']['Code'] != 'ResourceNotFoundException':
                raise SecretError(f"Failed to delete secret {key}: {e}")
    
    def list_secrets(self) -> list:
        """List all secrets in AWS Secrets Manager."""
        try:
            response = self.client.list_secrets()
            return [secret['Name'] for secret in response.get('SecretList', [])]
        except self.ClientError as e:
            raise SecretError(f"Failed to list secrets: {e}")


class FileSecretBackend(SecretBackend):
    """Secret backend using local encrypted files."""
    
    def __init__(self, secrets_dir: Union[str, Path], encryption_key: Optional[str] = None):
        """
        Initialize file secret backend.
        
        Args:
            secrets_dir: Directory to store secret files
            encryption_key: Key for encrypting secrets
        """
        self.secrets_dir = Path(secrets_dir)
        self.secrets_dir.mkdir(exist_ok=True, mode=0o700)  # Secure permissions
        
        if encryption_key is None:
            encryption_key = get_env('ENCRYPTION_KEY')
        
        if encryption_key:
            try:
                from cryptography.fernet import Fernet
                self.cipher = Fernet(encryption_key.encode())
                self.encrypted = True
            except ImportError:
                logger.warning("cryptography package not available, storing secrets unencrypted")
                self.encrypted = False
        else:
            logger.warning("No encryption key provided, storing secrets unencrypted")
            self.encrypted = False
    
    def _get_secret_path(self, key: str) -> Path:
        """Get path for secret file."""
        return self.secrets_dir / f"{key}.secret"
    
    def get_secret(self, key: str) -> Optional[str]:
        """Get secret from file."""
        secret_path = self._get_secret_path(key)
        
        if not secret_path.exists():
            return None
        
        try:
            with open(secret_path, 'rb') as f:
                data = f.read()
            
            if self.encrypted:
                data = self.cipher.decrypt(data)
            
            return data.decode('utf-8')
        except Exception as e:
            raise SecretError(f"Failed to read secret {key}: {e}")
    
    def set_secret(self, key: str, value: str) -> None:
        """Set secret in file."""
        secret_path = self._get_secret_path(key)
        
        try:
            data = value.encode('utf-8')
            
            if self.encrypted:
                data = self.cipher.encrypt(data)
            
            with open(secret_path, 'wb') as f:
                f.write(data)
            
            # Set secure permissions
            secret_path.chmod(0o600)
            
        except Exception as e:
            raise SecretError(f"Failed to write secret {key}: {e}")
    
    def delete_secret(self, key: str) -> None:
        """Delete secret file."""
        secret_path = self._get_secret_path(key)
        
        if secret_path.exists():
            secret_path.unlink()
    
    def list_secrets(self) -> list:
        """List all secret files."""
        secrets = []
        for secret_file in self.secrets_dir.glob("*.secret"):
            secret_key = secret_file.stem
            secrets.append(secret_key)
        return secrets


class SecretsManager:
    """
    Main secrets manager that orchestrates different backends.
    """
    
    def __init__(self, backend: Optional[SecretBackend] = None):
        """
        Initialize secrets manager.
        
        Args:
            backend: Secret storage backend. If None, uses environment backend.
        """
        if backend is None:
            backend = EnvironmentSecretBackend()
        
        self.backend = backend
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret value.
        
        Args:
            key: Secret key
            default: Default value if secret is not found
            
        Returns:
            Secret value or default
        """
        try:
            value = self.backend.get_secret(key)
            return value if value is not None else default
        except Exception as e:
            logger.error(f"Failed to get secret {key}: {e}")
            return default
    
    def set_secret(self, key: str, value: str) -> None:
        """
        Set a secret value.
        
        Args:
            key: Secret key
            value: Secret value
        """
        self.backend.set_secret(key, value)
        logger.info(f"Secret {key} has been set")
    
    def delete_secret(self, key: str) -> None:
        """
        Delete a secret.
        
        Args:
            key: Secret key to delete
        """
        self.backend.delete_secret(key)
        logger.info(f"Secret {key} has been deleted")
    
    def list_secrets(self) -> list:
        """List all available secrets."""
        return self.backend.list_secrets()
    
    def get_database_config(self) -> Dict[str, str]:
        """Get database configuration from secrets."""
        return {
            'host': self.get_secret('database_host', 'localhost'),
            'port': self.get_secret('database_port', '5432'),
            'name': self.get_secret('database_name', 'cbramod'),
            'user': self.get_secret('database_user'),
            'password': self.get_secret('database_password'),
            'url': self.get_secret('database_url')
        }
    
    def get_aws_config(self) -> Dict[str, str]:
        """Get AWS configuration from secrets."""
        return {
            'access_key_id': self.get_secret('aws_access_key_id'),
            'secret_access_key': self.get_secret('aws_secret_access_key'),
            'region': self.get_secret('aws_region', 'us-east-1'),
            's3_bucket': self.get_secret('aws_s3_bucket')
        }
    
    def get_wandb_config(self) -> Dict[str, str]:
        """Get Weights & Biases configuration from secrets."""
        return {
            'api_key': self.get_secret('wandb_api_key'),
            'project': self.get_secret('wandb_project', 'cbramod-experiments'),
            'entity': self.get_secret('wandb_entity')
        }


def create_secrets_manager(backend_type: str = "environment") -> SecretsManager:
    """
    Create a secrets manager with the specified backend.
    
    Args:
        backend_type: Type of backend ('environment', 'aws', 'file')
        
    Returns:
        Configured secrets manager
    """
    if backend_type == "environment":
        backend = EnvironmentSecretBackend()
    elif backend_type == "aws":
        backend = AWSSecretsManagerBackend()
    elif backend_type == "file":
        secrets_dir = get_env('SECRETS_DIR', 'secrets')
        backend = FileSecretBackend(secrets_dir)
    else:
        raise SecretError(f"Unknown backend type: {backend_type}")
    
    return SecretsManager(backend)


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get the global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        backend_type = get_env('SECRETS_BACKEND', 'environment')
        _secrets_manager = create_secrets_manager(backend_type)
    return _secrets_manager


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get a secret value using the global secrets manager."""
    return get_secrets_manager().get_secret(key, default)