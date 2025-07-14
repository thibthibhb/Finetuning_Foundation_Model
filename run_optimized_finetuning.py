#!/usr/bin/env python3
"""
Complete optimized finetuning pipeline for CBraMod.

This script demonstrates the full pipeline:
1. Data pipeline robustness with quality checks
2. Enhanced finetuning with pretrained weights
3. Comprehensive experiment tracking
4. Model validation and selection

Usage:
    python run_optimized_finetuning.py --config-env finetuning_optimized
    python run_optimized_finetuning.py --help
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from enhanced_finetune_main import EnhancedFinetunePipeline, create_enhanced_params
from config.utils import ConfigManager
from data_pipeline.versioning import DataVersionManager
from data_pipeline.monitoring import DataQualityMonitor

def setup_logging():
    """Setup comprehensive logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"optimized_finetuning_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    return log_file

def print_banner():
    """Print startup banner"""
    print("""
🚀 CBraMod Optimized Finetuning Pipeline
==========================================

This pipeline includes:
✅ Robust data pipeline with quality validation
✅ Enhanced pretrained weight loading
✅ Optimized training strategies
✅ Comprehensive experiment tracking
✅ Automated model validation
✅ Data versioning and lineage tracking

""")

def check_prerequisites():
    """Check that necessary files and directories exist"""
    logger = logging.getLogger(__name__)
    
    required_dirs = [
        "Datasets",
        "config",
        "data_pipeline",
        "cbramod",
        "experiments"
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        logger.error(f"Missing required directories: {missing_dirs}")
        return False
    
    # Check for pretrained weights
    config_manager = ConfigManager('finetuning_optimized')
    pretrained_path = config_manager.get('model.pretrained_weights.path')
    
    if pretrained_path and not Path(pretrained_path).exists():
        logger.warning(f"Pretrained weights not found at: {pretrained_path}")
        logger.warning("Training will proceed with random initialization")
    
    return True

def initialize_data_pipeline():
    """Initialize data pipeline components"""
    logger = logging.getLogger(__name__)
    logger.info("🔧 Initializing data pipeline components...")
    
    # Initialize version manager
    version_manager = DataVersionManager()
    
    # Initialize quality monitor
    quality_monitor = DataQualityMonitor()
    
    # Register datasets if they exist
    datasets_dir = Path("Datasets")
    if datasets_dir.exists():
        for dataset_dir in datasets_dir.iterdir():
            if dataset_dir.is_dir():
                try:
                    version_manager.register_dataset(
                        dataset_name=dataset_dir.name,
                        description=f"EEG dataset: {dataset_dir.name}",
                        tags=['eeg', 'sleep_staging']
                    )
                except Exception as e:
                    logger.debug(f"Dataset {dataset_dir.name} already registered or error: {e}")
    
    logger.info("✅ Data pipeline components initialized")
    return version_manager, quality_monitor

def run_complete_pipeline(args):
    """Run the complete optimized finetuning pipeline"""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize configuration
        config_manager = ConfigManager(environment=args.config_env)
        logger.info(f"📋 Loaded configuration: {args.config_env}")
        
        # Initialize data pipeline
        version_manager, quality_monitor = initialize_data_pipeline()
        
        # Create enhanced parameters
        params = create_enhanced_params(config_manager, args)
        
        # Display configuration
        print("\n📊 Configuration Summary:")
        print(f"  Environment: {args.config_env}")
        print(f"  Epochs: {params.epochs}")
        print(f"  Batch size: {params.batch_size}")
        print(f"  Learning rate: {params.lr}")
        print(f"  Pretrained weights: {params.use_pretrained_weights}")
        print(f"  Adaptive LR: {getattr(params, 'adaptive_lr', False)}")
        print(f"  Weighted sampler: {getattr(params, 'use_weighted_sampler', False)}")
        print(f"  Early stopping: {getattr(params, 'early_stopping_patience', 'disabled')}")
        print(f"  Scheduler: {getattr(params, 'scheduler', 'none')}")
        
        # Create snapshot before training
        snapshot_id = version_manager.create_snapshot(
            name="pre_training_snapshot",
            description=f"Dataset state before training with {args.config_env} config"
        )
        logger.info(f"📸 Created pre-training snapshot: {snapshot_id}")
        
        # Create and run enhanced pipeline
        pipeline = EnhancedFinetunePipeline(config_manager)
        results = pipeline.run_enhanced_training(params)
        
        # Create post-training snapshot
        post_snapshot_id = version_manager.create_snapshot(
            name="post_training_snapshot",
            description=f"Dataset state after training - Kappa: {results['kappa']:.4f}"
        )
        logger.info(f"📸 Created post-training snapshot: {post_snapshot_id}")
        
        # Display results
        print("\n🎉 Training Completed Successfully!")
        print("=" * 50)
        print(f"📊 Results:")
        print(f"  Kappa Score: {results['kappa']:.4f}")
        print(f"  Accuracy:    {results['accuracy']:.4f}")
        print(f"  F1-Score:    {results['f1_score']:.4f}")
        print("=" * 50)
        
        # Check if results meet thresholds
        kappa_threshold = config_manager.get('validation.kappa_threshold', 0.55)
        acc_threshold = config_manager.get('validation.accuracy_threshold', 0.65)
        
        if results['kappa'] >= kappa_threshold and results['accuracy'] >= acc_threshold:
            print("✅ Model meets quality thresholds!")
        else:
            print("⚠️  Model below quality thresholds - consider retraining with different hyperparameters")
        
        # Generate final report
        generate_final_report(results, config_manager, version_manager, quality_monitor)
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

def generate_final_report(results, config_manager, version_manager, quality_monitor):
    """Generate comprehensive final report"""
    logger = logging.getLogger(__name__)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"finetuning_report_{timestamp}.json"
    
    # Gather comprehensive information
    report = {
        "timestamp": timestamp,
        "configuration": config_manager.config,
        "results": results,
        "data_snapshots": version_manager.list_snapshots()[-2:],  # Last 2 snapshots
        "quality_trends": {},
        "recommendations": []
    }
    
    # Add quality trends if available
    try:
        for dataset_name in ["ORP", "OpenNeuro_2017"]:
            trends = quality_monitor.get_quality_trends(dataset_name, days=7)
            if 'message' not in trends:
                report["quality_trends"][dataset_name] = trends
    except Exception as e:
        logger.debug(f"Could not get quality trends: {e}")
    
    # Generate recommendations
    if results['kappa'] < 0.60:
        report["recommendations"].append("Consider increasing training epochs or adjusting learning rate")
    if results['accuracy'] < 0.70:
        report["recommendations"].append("Try data augmentation or different preprocessing")
    if results['f1_score'] < 0.65:
        report["recommendations"].append("Address class imbalance with weighted loss or sampling")
    
    # Save report
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"📋 Final report saved to: {report_path}")
    print(f"\n📋 Comprehensive report saved to: {report_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Complete optimized finetuning pipeline for CBraMod',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with optimized configuration
  python run_optimized_finetuning.py --config-env finetuning_optimized
  
  # Run with custom parameters
  python run_optimized_finetuning.py --config-env finetuning_optimized --epochs 150 --lr 0.001
  
  # Run with all enhancements enabled
  python run_optimized_finetuning.py --config-env finetuning_optimized --adaptive-lr --weighted-sampler
        """
    )
    
    # Configuration
    parser.add_argument(
        '--config-env', 
        type=str, 
        default='finetuning_optimized',
        help='Configuration environment (default: finetuning_optimized)'
    )
    
    # Training parameters
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device ID')
    
    # Enhanced features
    parser.add_argument(
        '--adaptive-lr', 
        action='store_true',
        help='Use adaptive learning rates for pretrained vs new layers'
    )
    parser.add_argument(
        '--weighted-sampler', 
        action='store_true',
        help='Use weighted random sampler for class balancing'
    )
    parser.add_argument(
        '--early-stopping-patience', 
        type=int, 
        help='Early stopping patience (epochs)'
    )
    parser.add_argument(
        '--scheduler', 
        type=str,
        choices=['cosine', 'cosine_warmup', 'plateau', 'exponential'],
        help='Learning rate scheduler type'
    )
    
    # Other options
    parser.add_argument(
        '--check-only', 
        action='store_true',
        help='Only check prerequisites and configuration, do not train'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print banner
    print_banner()
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting optimized finetuning pipeline (log: {log_file})")
    
    try:
        # Check prerequisites
        if not check_prerequisites():
            print("❌ Prerequisites check failed. Please ensure all required files are present.")
            return False
        
        print("✅ Prerequisites check passed")
        
        if args.check_only:
            print("🔍 Check-only mode - exiting without training")
            return True
        
        # Run complete pipeline
        results = run_complete_pipeline(args)
        
        print(f"\n🚀 Pipeline completed successfully!")
        print(f"📊 Final Kappa Score: {results['kappa']:.4f}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n❌ Pipeline failed: {e}")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)