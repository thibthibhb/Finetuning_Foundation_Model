#!/usr/bin/env python3
"""
DVC (Data Version Control) setup script for CBraMod project.

This script initializes DVC for the project and sets up data versioning for the datasets.
"""

import os
import subprocess
import sys
import json
from pathlib import Path
import logging

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def run_command(command, cwd=None, check=True):
    """Run shell command and return result"""
    logger = logging.getLogger(__name__)
    logger.info(f"Running: {' '.join(command) if isinstance(command, list) else command}")
    
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check
        )
        
        if result.stdout:
            logger.info(f"Output: {result.stdout.strip()}")
        
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        if check:
            raise
        return e

def check_dvc_installed():
    """Check if DVC is installed"""
    try:
        result = subprocess.run(["dvc", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_dvc():
    """Install DVC"""
    logger = logging.getLogger(__name__)
    logger.info("Installing DVC...")
    
    try:
        # Try installing with pip
        subprocess.run([sys.executable, "-m", "pip", "install", "dvc"], check=True)
        logger.info("✅ DVC installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install DVC: {e}")
        return False

def init_dvc(project_root):
    """Initialize DVC in the project"""
    logger = logging.getLogger(__name__)
    
    # Check if DVC is already initialized
    dvc_dir = project_root / ".dvc"
    if dvc_dir.exists():
        logger.info("DVC already initialized")
        return True
    
    try:
        # Initialize DVC
        run_command(["dvc", "init"], cwd=project_root)
        logger.info("✅ DVC initialized successfully")
        
        # Create .dvcignore file
        dvcignore_path = project_root / ".dvcignore"
        with open(dvcignore_path, 'w') as f:
            f.write("""# DVC ignore file
# Temporary files
*.tmp
*.temp
__pycache__/
*.pyc
.pytest_cache/

# Logs
*.log
logs/

# Editor files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db
""")
        
        logger.info("Created .dvcignore file")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize DVC: {e}")
        return False

def setup_git_hooks(project_root):
    """Setup git hooks for DVC"""
    logger = logging.getLogger(__name__)
    
    try:
        # Install DVC git hooks
        run_command(["dvc", "install"], cwd=project_root)
        logger.info("✅ DVC git hooks installed")
        return True
    except Exception as e:
        logger.error(f"Failed to install DVC git hooks: {e}")
        return False

def add_datasets_to_dvc(project_root):
    """Add datasets to DVC tracking"""
    logger = logging.getLogger(__name__)
    
    datasets_dir = project_root / "Datasets"
    if not datasets_dir.exists():
        logger.warning(f"Datasets directory not found: {datasets_dir}")
        return False
    
    # Find dataset subdirectories
    dataset_dirs = [d for d in datasets_dir.iterdir() if d.is_dir()]
    
    if not dataset_dirs:
        logger.warning("No dataset directories found")
        return False
    
    logger.info(f"Found {len(dataset_dirs)} dataset directories")
    
    # Add each dataset directory to DVC
    for dataset_dir in dataset_dirs:
        try:
            logger.info(f"Adding {dataset_dir.name} to DVC...")
            
            # Check if already tracked
            dvc_file = dataset_dir.with_suffix('.dvc')
            if dvc_file.exists():
                logger.info(f"  {dataset_dir.name} already tracked by DVC")
                continue
            
            # Add to DVC
            run_command(["dvc", "add", str(dataset_dir)], cwd=project_root)
            logger.info(f"  ✅ Added {dataset_dir.name}")
            
        except Exception as e:
            logger.error(f"  ❌ Failed to add {dataset_dir.name}: {e}")
    
    return True

def create_dvc_config(project_root):
    """Create DVC configuration"""
    logger = logging.getLogger(__name__)
    
    dvc_config = {
        'core': {
            'analytics': False,
            'autostage': True
        },
        'cache': {
            'type': 'reflink,copy'
        }
    }
    
    config_dir = project_root / ".dvc"
    config_file = config_dir / "config"
    
    try:
        # Read existing config if it exists
        if config_file.exists():
            result = run_command(["dvc", "config", "--local", "--list"], 
                               cwd=project_root, check=False)
            if result.returncode == 0:
                logger.info("DVC config already exists")
                return True
        
        # Set configuration options
        run_command(["dvc", "config", "core.analytics", "false"], cwd=project_root)
        run_command(["dvc", "config", "core.autostage", "true"], cwd=project_root)
        
        logger.info("✅ DVC configuration created")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create DVC config: {e}")
        return False

def create_data_pipeline_stage(project_root):
    """Create DVC pipeline stage for data processing"""
    logger = logging.getLogger(__name__)
    
    dvc_yaml_content = """stages:
  data_validation:
    cmd: python examples/data_pipeline_example.py
    deps:
    - examples/data_pipeline_example.py
    - cbramod/datasets/
    - data_pipeline/
    outs:
    - data_quality_report.json
    - data_quality_monitor.db
    - data_lineage.db
    metrics:
    - data_quality_report.json:
        cache: false
    
  data_preprocessing:
    cmd: python -c "print('Data preprocessing placeholder')"
    deps:
    - Datasets/
    params:
    - config/base.yaml:
        - data.sample_rate
        - data.num_workers
    outs:
    - preprocessed_data/:
        cache: true
"""
    
    try:
        dvc_yaml_path = project_root / "dvc.yaml"
        with open(dvc_yaml_path, 'w') as f:
            f.write(dvc_yaml_content)
        
        logger.info("✅ Created DVC pipeline configuration (dvc.yaml)")
        
        # Create params file if it doesn't exist
        params_file = project_root / "params.yaml"
        if not params_file.exists():
            params_content = """data:
  sample_rate: 200
  num_workers: 8
  validation_split: 0.2

training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.0005
"""
            with open(params_file, 'w') as f:
                f.write(params_content)
            
            logger.info("✅ Created params.yaml file")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create DVC pipeline: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("""
🎉 DVC Setup Complete!

📋 Next Steps:
1. Review the generated DVC files:
   - .dvc/config          # DVC configuration
   - dvc.yaml            # DVC pipeline definition
   - params.yaml         # Parameters for pipeline
   - Datasets/*.dvc      # Dataset tracking files

2. Commit DVC files to git:
   git add .dvc/ dvc.yaml params.yaml Datasets/*.dvc .dvcignore
   git commit -m "Initialize DVC for data versioning"

3. Setup remote storage (optional):
   dvc remote add -d myremote s3://my-bucket/dvcstore
   dvc remote modify myremote profile myprofile

4. Run data pipeline:
   dvc repro

5. Track data changes:
   dvc status                 # Check for changes
   dvc add Datasets/new_data  # Add new data
   dvc push                   # Push to remote storage

📚 DVC Documentation: https://dvc.org/doc
""")

def main():
    """Main setup function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🚀 CBraMod DVC Setup")
    print("=" * 30)
    
    project_root = Path.cwd()
    logger.info(f"Project root: {project_root}")
    
    try:
        # 1. Check if DVC is installed
        print("\n1. Checking DVC installation...")
        if not check_dvc_installed():
            print("DVC not found. Installing...")
            if not install_dvc():
                print("❌ Failed to install DVC")
                return False
        else:
            print("✅ DVC is already installed")
        
        # 2. Initialize DVC
        print("\n2. Initializing DVC...")
        if not init_dvc(project_root):
            print("❌ Failed to initialize DVC")
            return False
        
        # 3. Setup git hooks
        print("\n3. Setting up git hooks...")
        setup_git_hooks(project_root)
        
        # 4. Create DVC configuration
        print("\n4. Creating DVC configuration...")
        create_dvc_config(project_root)
        
        # 5. Add datasets to DVC (if they exist)
        print("\n5. Adding datasets to DVC...")
        add_datasets_to_dvc(project_root)
        
        # 6. Create DVC pipeline
        print("\n6. Creating DVC pipeline...")
        create_data_pipeline_stage(project_root)
        
        # 7. Print next steps
        print_next_steps()
        
        return True
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\n❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)