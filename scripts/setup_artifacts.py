#!/usr/bin/env python3
"""
Setup script for CBraMod artifacts directory structure.
Creates the standardized artifacts directory layout and moves existing files.
"""

import os
import shutil
import sys
from pathlib import Path
import logging

# Add config to path
sys.path.insert(0, str(Path(__file__).parent.parent / "config"))
from paths import PathConfig

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_artifacts_structure(path_config: PathConfig):
    """Create the artifacts directory structure."""
    logger = logging.getLogger(__name__)
    
    logger.info("Creating artifacts directory structure...")
    path_config.ensure_directories()
    
    # Create .gitkeep files to ensure directories are tracked
    gitkeep_dirs = [
        path_config.pretrained_models,
        path_config.finetuned_models,
        path_config.production_models,
        path_config.optuna_studies,
        path_config.reproducibility,
        path_config.wandb_artifacts,
        path_config.logs_root,
        path_config.cache_root,
        path_config.results_root
    ]
    
    for directory in gitkeep_dirs:
        gitkeep_file = directory / ".gitkeep"
        if not gitkeep_file.exists():
            gitkeep_file.touch()
            logger.info(f"Created .gitkeep in {directory.name}")
    
    logger.info("✅ Artifacts directory structure created")

def move_existing_files(path_config: PathConfig):
    """Move existing files to the new artifacts structure."""
    logger = logging.getLogger(__name__)
    project_root = path_config.project_root
    
    # Define file movements
    movements = [
        # Model files
        ("saved_models", path_config.finetuned_models),
        ("cbramod/saved_models", path_config.finetuned_models),
        ("cbramod/weights", path_config.pretrained_models),
        ("data/saved_models", path_config.finetuned_models),
        
        # Experiment tracking
        ("optuna_studies", path_config.optuna_studies),
        ("reproducibility", path_config.reproducibility),
        ("wandb", path_config.wandb_artifacts),
        ("cbramod/wandb", path_config.wandb_artifacts),
        
        # Logs
        ("logs", path_config.logs_root),
        
        # Cache and results
        ("cache", path_config.cache_root),
        ("results", path_config.results_root),
        
        # Database files
        ("data_lineage.db", path_config.cache_root / "data_lineage.db"),
        ("data_quality_monitor.db", path_config.cache_root / "data_quality_monitor.db"),
        
        # JSON results
        ("*.json", path_config.results_root),  # Pattern for result files
    ]
    
    logger.info("Moving existing files to artifacts structure...")
    
    for source, destination in movements:
        source_path = project_root / source
        
        # Handle pattern matching for JSON files
        if source.endswith("*.json"):
            json_files = list(project_root.glob("*_results_*.json"))
            json_files.extend(list(project_root.glob("finetuning_report_*.json")))
            
            for json_file in json_files:
                if json_file.exists():
                    dest_file = destination / json_file.name
                    try:
                        shutil.move(str(json_file), str(dest_file))
                        logger.info(f"Moved {json_file.name} to {destination.name}/")
                    except Exception as e:
                        logger.warning(f"Failed to move {json_file.name}: {e}")
            continue
        
        # Handle single file
        if source_path.is_file():
            try:
                shutil.move(str(source_path), str(destination))
                logger.info(f"Moved {source} to {destination}")
            except Exception as e:
                logger.warning(f"Failed to move {source}: {e}")
        
        # Handle directory
        elif source_path.is_dir():
            try:
                if destination.exists():
                    # If destination exists, move contents
                    for item in source_path.iterdir():
                        dest_item = destination / item.name
                        if item.is_file():
                            shutil.move(str(item), str(dest_item))
                        elif item.is_dir():
                            shutil.move(str(item), str(dest_item))
                    
                    # Remove empty source directory
                    if not any(source_path.iterdir()):
                        source_path.rmdir()
                        logger.info(f"Moved contents of {source} to {destination.name}/")
                else:
                    # Move entire directory
                    shutil.move(str(source_path), str(destination))
                    logger.info(f"Moved {source} to {destination}")
            except Exception as e:
                logger.warning(f"Failed to move {source}: {e}")
    
    logger.info("✅ File migration completed")

def update_gitignore(path_config: PathConfig):
    """Update .gitignore to use the new artifacts structure."""
    logger = logging.getLogger(__name__)
    project_root = path_config.project_root
    gitignore_path = project_root / ".gitignore"
    
    if not gitignore_path.exists():
        logger.warning(".gitignore not found, creating new one")
        gitignore_content = ""
    else:
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read()
    
    # New gitignore section for artifacts
    artifacts_section = '''
# CBraMod Artifacts Directory
artifacts/
!artifacts/.gitkeep
!artifacts/*/.gitkeep

# Keep important artifact subdirectories tracked (empty)
!artifacts/models/
!artifacts/models/pretrained/
!artifacts/models/finetuned/
!artifacts/models/production/
!artifacts/experiments/
!artifacts/logs/
!artifacts/cache/
!artifacts/results/
'''
    
    # Remove old artifact-related entries
    lines_to_remove = [
        'saved_models/', 'checkpoints/', 'weights/', 'logs/', 'wandb/', 'mlruns/',
        'experiments/', 'results/', 'outputs/', 'runs/', 'optuna_studies/',
        'reproducibility/', 'cache/', '*.db', '*.log'
    ]
    
    updated_lines = []
    for line in gitignore_content.split('\n'):
        if not any(remove_pattern in line for remove_pattern in lines_to_remove):
            updated_lines.append(line)
    
    # Add artifacts section
    updated_content = '\n'.join(updated_lines) + artifacts_section
    
    # Write updated gitignore
    with open(gitignore_path, 'w') as f:
        f.write(updated_content)
    
    logger.info("✅ Updated .gitignore for artifacts structure")

def create_env_file(path_config: PathConfig):
    """Create environment file with artifact paths."""
    logger = logging.getLogger(__name__)
    project_root = path_config.project_root
    env_file = project_root / ".env.artifacts"
    
    env_vars = path_config.get_environment_variables()
    
    with open(env_file, 'w') as f:
        f.write("# CBraMod Artifacts Environment Variables\n")
        f.write("# Source this file to set artifact paths\n\n")
        
        for name, value in env_vars.items():
            f.write(f"export {name}={value}\n")
    
    logger.info(f"✅ Created {env_file} with environment variables")

def print_setup_summary(path_config: PathConfig):
    """Print setup summary."""
    print("\n" + "="*60)
    print("🎉 CBraMod Artifacts Setup Complete!")
    print("="*60)
    
    path_config.print_path_summary()
    
    print("\n📋 Next Steps:")
    print("1. Review the artifacts directory structure")
    print("2. Source environment variables:")
    print("   source .env.artifacts")
    print("3. Commit changes to git:")
    print("   git add artifacts/ .gitignore .env.artifacts")
    print("   git commit -m 'Setup artifacts directory structure'")
    print("4. Update any remaining hardcoded paths in your code")
    print("5. Test your training pipeline with the new structure")

def main():
    """Main setup function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🚀 CBraMod Artifacts Setup")
    print("=" * 30)
    
    try:
        # Initialize path configuration
        path_config = PathConfig()
        
        # Create artifacts structure
        create_artifacts_structure(path_config)
        
        # Move existing files
        move_existing_files(path_config)
        
        # Update .gitignore
        update_gitignore(path_config)
        
        # Create environment file
        create_env_file(path_config)
        
        # Print summary
        print_setup_summary(path_config)
        
        return True
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)