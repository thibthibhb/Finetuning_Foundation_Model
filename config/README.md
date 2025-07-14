# CBraMod Configuration Management System

This directory contains a comprehensive configuration management system for CBraMod that provides:

- ✅ **Environment-specific configurations** (development, production, testing)
- ✅ **Schema validation** for configuration files
- ✅ **Environment variable substitution** with .env file support
- ✅ **Secrets management** with multiple backends
- ✅ **Type safety** and validation
- ✅ **Backward compatibility** with existing code

## 🚀 Quick Start

### 1. Basic Usage

```python
from config import ConfigManager

# Initialize configuration for development environment
config = ConfigManager(environment='development')

# Get configuration values
learning_rate = config.get('training.learning_rate')
batch_size = config.get('training.batch_size')
model_config = config.get_model_config()

# Get device configuration
device = config.get_device()  # Returns 'cuda:0' or 'cpu'
```

### 2. Environment Setup

Copy the example environment file and customize it:

```bash
cp .env.example .env
# Edit .env with your specific values
```

Set your environment:

```bash
export CBRAMOD_ENV=development  # or production, testing
```

### 3. Run Training with Configuration

```bash
# Use the new configuration-aware script
python finetune_main_with_config.py --config-env development

# Override specific values
python finetune_main_with_config.py --epochs 50 --batch-size 128
```

## 📁 Directory Structure

```
config/
├── __init__.py              # Main exports
├── README.md               # This file
├── base.yaml               # Base configuration
├── environments/           # Environment-specific configs
│   ├── development.yaml    # Development settings
│   ├── production.yaml     # Production settings
│   └── testing.yaml        # Testing/CI settings
├── schemas/                # Configuration schemas
├── config_loader.py        # Configuration loading logic
├── validator.py            # Configuration validation
├── env_loader.py           # Environment variable handling
├── secrets.py              # Secrets management
├── utils.py                # Utility functions
└── exceptions.py           # Custom exceptions
```

## ⚙️ Configuration Files

### Base Configuration (`base.yaml`)

Contains default settings that apply to all environments:

```yaml
# Model architecture
model:
  backbone:
    d_model: 200
    n_layer: 12
    nhead: 8
    dropout: 0.1

# Training parameters
training:
  epochs: 100
  batch_size: 64
  learning_rate: 5e-4
  optimizer: "AdamW"

# Data configuration
data:
  sample_rate: 200
  num_workers: 8
```

### Environment-Specific Configurations

Environment configs inherit from base and override specific values:

- `development.yaml` - Fast iteration settings
- `production.yaml` - Optimized for production deployment
- `testing.yaml` - Minimal settings for CI/CD

## 🔧 Configuration Loading

### Automatic Loading

The system automatically loads configuration based on the `CBRAMOD_ENV` environment variable:

```python
from config.utils import init_config_system

# Loads based on CBRAMOD_ENV (defaults to 'development')
config = init_config_system()
```

### Manual Loading

```python
from config import load_config

# Load specific environment
config = load_config(environment='production')

# Access values
learning_rate = config['training']['learning_rate']
```

### Using ConfigManager (Recommended)

```python
from config.utils import ConfigManager

config_manager = ConfigManager('production')

# Type-safe access
lr = config_manager.get('training.learning_rate', default=1e-3)
model_config = config_manager.get_model_config()
device = config_manager.get_device()
```

## 🌍 Environment Variables

### .env File Support

Create a `.env` file in the project root:

```bash
# Environment
CBRAMOD_ENV=development

# Paths
DATA_ROOT=/data/cbramod
MODEL_ROOT=/models/cbramod

# Secrets
WANDB_API_KEY=your_key_here
AWS_ACCESS_KEY_ID=your_key_here
```

### Variable Substitution

Use environment variables in configuration files:

```yaml
paths:
  datasets_dir: "${DATA_ROOT}/datasets"
  model_dir: "${MODEL_ROOT}/saved_models"
  log_dir: "${LOG_ROOT}/logs"

# With defaults
database:
  host: "${DB_HOST:localhost}"
  port: "${DB_PORT:5432}"
```

### Environment Loading

```python
from config.env_loader import get_env, get_env_bool, get_env_int

# Basic usage
api_key = get_env('WANDB_API_KEY', required=True)
debug_mode = get_env_bool('DEBUG', default=False)
batch_size = get_env_int('BATCH_SIZE', default=64)
```

## 🔐 Secrets Management

### Environment Backend (Default)

Stores secrets as environment variables with a prefix:

```python
from config.secrets import get_secret

# Set secrets
export CBRAMOD_SECRET_WANDB_API_KEY="your_key"
export CBRAMOD_SECRET_AWS_ACCESS_KEY="your_key"

# Get secrets
api_key = get_secret('wandb_api_key')
```

### AWS Secrets Manager Backend

```python
from config.secrets import create_secrets_manager

# Use AWS Secrets Manager
secrets = create_secrets_manager('aws')
api_key = secrets.get_secret('wandb_api_key')
```

### File Backend (Encrypted)

```python
# Set encryption key
export ENCRYPTION_KEY="your_base64_key"

# Use file backend
secrets = create_secrets_manager('file')
secrets.set_secret('api_key', 'secret_value')
```

## ✅ Validation

### Automatic Validation

Configuration is automatically validated when loaded:

```python
from config import load_config, ValidationError

try:
    config = load_config('production', validate=True)
except ValidationError as e:
    print(f"Configuration error: {e}")
```

### Manual Validation

```python
from config.validator import ConfigValidator

validator = ConfigValidator()
validator.validate(config)
```

### Schema Validation

The validator checks:
- Required sections are present
- Numeric values are in valid ranges
- String values match expected patterns
- Path configurations are valid

## 🔄 Migration from Existing Code

### Step 1: Analyze Current Code

```bash
python migrate_to_config.py --project-dir . --generate-config
```

### Step 2: Update Code

**Before:**
```python
epochs = 100
batch_size = 64
learning_rate = 5e-4
```

**After:**
```python
from config.utils import ConfigManager

config = ConfigManager()
epochs = config.get('training.epochs')
batch_size = config.get('training.batch_size')
learning_rate = config.get('training.learning_rate')
```

### Step 3: Backward Compatibility

Use the conversion utility for existing argparse-based code:

```python
from config.utils import convert_config_to_args

config_manager = ConfigManager()
args = convert_config_to_args(config_manager.config)

# Now 'args' can be used with existing code
trainer = Trainer(args, data_loader, model)
```

## 🎯 Best Practices

### 1. Configuration Organization

- Keep environment-specific values in environment configs
- Use base.yaml for common defaults
- Group related settings in sections

### 2. Environment Variables

- Use environment variables for deployment-specific values
- Provide sensible defaults in configuration files
- Use secrets management for sensitive data

### 3. Validation

- Always validate configuration in production
- Use type hints and validation schemas
- Fail fast on configuration errors

### 4. Secrets

- Never commit secrets to version control
- Use appropriate secrets backend for your environment
- Rotate secrets regularly

### 5. Testing

- Use the testing environment for CI/CD
- Override values for specific test scenarios
- Validate configuration changes in tests

## 🚀 Integration Examples

### Training Script Integration

```python
from config.utils import ConfigManager, setup_logging

def main():
    # Initialize configuration
    config_manager = ConfigManager()
    
    # Setup logging based on config
    setup_logging(config_manager.config)
    
    # Get training parameters
    training_config = config_manager.get_training_config()
    
    # Create model with config
    model = create_model(config_manager.get_model_config())
    
    # Setup device
    device = torch.device(config_manager.get_device())
    model.to(device)
```

### Deployment Integration

```python
from config.utils import ConfigManager
from config.secrets import get_secrets_manager

def deploy():
    config = ConfigManager('production')
    secrets = get_secrets_manager()
    
    # Get deployment configuration
    aws_config = secrets.get_aws_config()
    model_path = config.get('paths.model_dir')
    
    # Deploy model
    deploy_to_sagemaker(aws_config, model_path)
```

## 🔍 Troubleshooting

### Common Issues

1. **Configuration Not Found**
   ```
   ConfigError: Configuration file not found
   ```
   Solution: Ensure config files exist and CBRAMOD_ENV is set correctly.

2. **Environment Variable Not Found**
   ```
   EnvironmentError: Environment variable not found: DATA_ROOT
   ```
   Solution: Set the required environment variable or provide a default.

3. **Validation Error**
   ```
   ValidationError: training.learning_rate must be positive
   ```
   Solution: Check configuration values match the expected schema.

### Debug Mode

Enable debug logging to troubleshoot:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

config = ConfigManager()
```

### Configuration Inspection

```python
# Print current configuration
config_manager = ConfigManager()
print(json.dumps(config_manager.config, indent=2))

# Check loaded environment variables
from config.env_loader import get_env_loader
print(get_env_loader().get_loaded_vars())
```

## 📚 API Reference

### ConfigManager

Main interface for configuration access:

- `get(key_path, default)` - Get configuration value by dot notation
- `get_model_config()` - Get model configuration section
- `get_training_config()` - Get training configuration section
- `get_data_config()` - Get data configuration section
- `get_device()` - Get device string (cuda:0, cpu)
- `get_secret(key, default)` - Get secret value

### Functions

- `load_config(environment)` - Load configuration for environment
- `init_config_system()` - Initialize complete configuration system
- `get_secret(key, default)` - Get secret value
- `get_env(key, default, required)` - Get environment variable

For complete API documentation, see the docstrings in each module.



###   ✅ Configuration Management System Complete

  🏗️ What Was Built:

  1. Configuration Structure:
    - config/base.yaml - Default settings for all environments
    - config/environments/ - Environment-specific overrides (development, production, testing)
    - Schema validation for type safety and correctness
  2. Core Components:
    - ConfigLoader - Loads and merges configurations with validation
    - ConfigValidator - Validates configuration against schemas
    - EnvironmentLoader - Handles .env files and environment variables
    - SecretsManager - Multi-backend secrets management (env vars, AWS, files)
    - ConfigManager - High-level interface for easy access
  3. Utilities:
    - Environment variable substitution with ${VAR_NAME:default} syntax
    - Backward compatibility helpers for existing argparse-based code
    - Migration analysis tool to identify hardcoded values
    - Comprehensive documentation and examples