#!/usr/bin/env python3
"""
Example script demonstrating the enhanced data pipeline capabilities for CBraMod.

This script shows how to:
1. Load data with comprehensive validation
2. Monitor data quality
3. Track data lineage
4. Version datasets
5. Generate quality reports
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.utils import ConfigManager
from cbramod.datasets.enhanced_dataset import EnhancedLoadDataset, EnhancedEEGDataset
from data_pipeline.validators import EEGDataValidator, DataQualityChecker
from data_pipeline.monitoring import DataQualityMonitor
from data_pipeline.versioning import DataVersionManager

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data_pipeline_example.log')
        ]
    )

def main():
    """Main example function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🚀 CBraMod Enhanced Data Pipeline Example")
    print("=" * 50)
    
    try:
        # 1. Initialize configuration
        print("\n📋 1. Initializing Configuration...")
        config_manager = ConfigManager('development')
        
        # Mock parameters for demonstration
        class MockParams:
            def __init__(self, config):
                self.datasets_dir = config.get('paths.datasets_dir', 'Datasets')
                self.dataset_names = ['OpenNeuro_2017', 'ORP']  # Example datasets
                self.num_datasets = len(self.dataset_names)
        
        params = MockParams(config_manager.config)
        print(f"✅ Using datasets: {params.dataset_names}")
        
        # 2. Enhanced data loading with pipeline
        print("\n📊 2. Loading Data with Enhanced Pipeline...")
        enhanced_loader = EnhancedLoadDataset(params, enable_data_pipeline=True)
        
        # Create enhanced dataset
        dataset = enhanced_loader.create_enhanced_dataset(
            dataset_name="cbramod_training_data",
            dataset_version="1.1"
        )
        
        print(f"✅ Loaded {len(dataset.samples)} samples")
        print(f"✅ Dataset ID: {getattr(dataset, 'dataset_id', 'N/A')}")
        
        # 3. Generate quality report
        print("\n🔍 3. Generating Quality Report...")
        quality_report = dataset.get_data_quality_report()
        
        print(f"📊 Quality Summary:")
        if 'dataset_info' in quality_report:
            info = quality_report['dataset_info']
            print(f"   Total samples: {info.get('total_samples', 'N/A')}")
            print(f"   Unique subjects: {info.get('unique_subjects', 'N/A')}")
        
        if 'total_files' in quality_report:
            print(f"   Files processed: {quality_report['total_files']}")
            print(f"   Files passed: {quality_report['passed_files']}")
            print(f"   Files failed: {quality_report['failed_files']}")
            print(f"   Warnings: {quality_report['warnings']}")
            print(f"   Errors: {quality_report['errors']}")
        
        # Save quality report
        report_path = "data_quality_report.json"
        dataset.save_quality_report(report_path)
        print(f"💾 Quality report saved to: {report_path}")
        
        # 4. Data lineage information
        print("\n🔗 4. Data Lineage Information...")
        lineage_info = dataset.get_lineage_info()
        
        if 'error' not in lineage_info:
            print(f"📋 Dataset ID: {lineage_info.get('dataset_id', 'N/A')}")
            
            # Quality history
            quality_history = lineage_info.get('quality_history', {})
            if quality_history:
                print(f"📈 Quality metrics tracked: {list(quality_history.keys())}")
            else:
                print("📈 No quality history available yet")
        else:
            print(f"❌ Lineage error: {lineage_info['error']}")
        
        # 5. Data versioning demonstration
        print("\n🔄 5. Data Versioning...")
        version_manager = DataVersionManager()
        
        # Check for dataset changes
        if hasattr(dataset, 'dataset_name'):
            change_status = version_manager.check_dataset_changes(dataset.dataset_name)
            print(f"📊 Dataset change status: {change_status.get('status', 'unknown')}")
            print(f"📝 Message: {change_status.get('message', 'N/A')}")
        
        # Create snapshot
        snapshot_id = version_manager.create_snapshot(
            name="training_ready_snapshot",
            description="Dataset ready for training run",
            datasets=[dataset.dataset_name] if hasattr(dataset, 'dataset_name') else []
        )
        print(f"📸 Created snapshot: {snapshot_id}")
        
        # 6. Data quality monitoring trends
        print("\n📈 6. Quality Monitoring...")
        quality_monitor = DataQualityMonitor()
        
        # Get quality trends (if any historical data exists)
        if hasattr(dataset, 'dataset_name'):
            trends = quality_monitor.get_quality_trends(dataset.dataset_name, days=7)
            
            if 'message' in trends:
                print(f"📊 {trends['message']}")
            else:
                print(f"📊 Quality trends (last 7 days):")
                print(f"   Total runs: {trends.get('total_runs', 0)}")
                if trends.get('total_runs', 0) > 0:
                    trend_data = trends.get('trends', {})
                    if 'error_rate' in trend_data:
                        error_trend = trend_data['error_rate']
                        print(f"   Error rate: {error_trend.get('current', 0):.3f} ({error_trend.get('trend', 'unknown')})")
                    if 'pass_rate' in trend_data:
                        pass_trend = trend_data['pass_rate']  
                        print(f"   Pass rate: {pass_trend.get('current', 0):.3f} ({pass_trend.get('trend', 'unknown')})")
        
        # 7. Recent alerts
        print("\n🚨 7. Recent Alerts...")
        recent_alerts = quality_monitor.get_recent_alerts(hours=24)
        
        if recent_alerts:
            print(f"⚠️  Found {len(recent_alerts)} recent alerts:")
            for alert in recent_alerts[:3]:  # Show first 3
                print(f"   - {alert.get('type', 'unknown')}: {alert.get('message', 'N/A')}")
        else:
            print("✅ No recent alerts")
        
        print("\n🎉 Data Pipeline Example Completed Successfully!")
        print("=" * 50)
        
        # Summary of generated files
        print("\n📄 Generated Files:")
        print(f"   - Quality report: {report_path}")
        print(f"   - Quality monitoring DB: data_quality_monitor.db")
        print(f"   - Data lineage DB: data_lineage.db")
        print(f"   - Version tracking: .data_versions.json")
        print(f"   - Log file: data_pipeline_example.log")
        
    except Exception as e:
        logger.error(f"Error in data pipeline example: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()