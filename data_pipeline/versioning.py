import os
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

class DataVersionManager:
    """Simple data versioning system using DVC and git for tracking dataset changes"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.data_dir = self.project_root / "Datasets"
        self.version_file = self.project_root / ".data_versions.json"
        self.logger = logging.getLogger(__name__)
        
        # Initialize version tracking
        self._init_versioning()
    
    def _init_versioning(self):
        """Initialize data versioning system"""
        if not self.version_file.exists():
            initial_versions = {
                "format_version": "1.0",
                "created_at": datetime.now().isoformat(),
                "datasets": {},
                "snapshots": []
            }
            with open(self.version_file, 'w') as f:
                json.dump(initial_versions, f, indent=2)
    
    def _load_versions(self) -> Dict[str, Any]:
        """Load version tracking data"""
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                return json.load(f)
        return {"datasets": {}, "snapshots": []}
    
    def _save_versions(self, versions: Dict[str, Any]):
        """Save version tracking data"""
        with open(self.version_file, 'w') as f:
            json.dump(versions, f, indent=2, default=str)
    
    def _calculate_dataset_hash(self, dataset_path: Path) -> str:
        """Calculate hash of dataset directory contents"""
        hasher = hashlib.md5()
        
        if not dataset_path.exists():
            return ""
        
        # Get all files and their modification times
        files_info = []
        for root, dirs, files in os.walk(dataset_path):
            for file in sorted(files):
                file_path = Path(root) / file
                if file_path.suffix in ['.npy', '.csv', '.edf', '.h5']:  # Only data files
                    stat = file_path.stat()
                    files_info.append(f"{file_path.relative_to(dataset_path)}:{stat.st_size}:{stat.st_mtime}")
        
        # Hash the file information
        hasher.update('\n'.join(files_info).encode())
        return hasher.hexdigest()
    
    def register_dataset(self, dataset_name: str, description: str = "", 
                        tags: List[str] = None) -> str:
        """Register a dataset for version tracking"""
        dataset_path = self.data_dir / dataset_name
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        
        versions = self._load_versions()
        current_hash = self._calculate_dataset_hash(dataset_path)
        
        version_info = {
            "name": dataset_name,
            "path": str(dataset_path.relative_to(self.project_root)),
            "description": description,
            "tags": tags or [],
            "hash": current_hash,
            "registered_at": datetime.now().isoformat(),
            "size_mb": self._get_directory_size(dataset_path),
            "file_count": self._count_data_files(dataset_path)
        }
        
        versions["datasets"][dataset_name] = version_info
        self._save_versions(versions)
        
        self.logger.info(f"Registered dataset '{dataset_name}' with hash {current_hash[:8]}")
        return current_hash
    
    def check_dataset_changes(self, dataset_name: str) -> Dict[str, Any]:
        """Check if dataset has changed since last registration"""
        versions = self._load_versions()
        
        if dataset_name not in versions["datasets"]:
            return {"status": "untracked", "message": "Dataset not registered"}
        
        dataset_info = versions["datasets"][dataset_name]
        dataset_path = self.project_root / dataset_info["path"]
        
        if not dataset_path.exists():
            return {"status": "missing", "message": "Dataset directory not found"}
        
        current_hash = self._calculate_dataset_hash(dataset_path)
        registered_hash = dataset_info["hash"]
        
        if current_hash == registered_hash:
            return {"status": "unchanged", "message": "Dataset unchanged"}
        else:
            return {
                "status": "changed",
                "message": "Dataset has been modified",
                "old_hash": registered_hash,
                "new_hash": current_hash,
                "changed_at": datetime.now().isoformat()
            }
    
    def create_snapshot(self, name: str, datasets: List[str] = None, 
                       description: str = "") -> str:
        """Create a snapshot of current dataset states"""
        versions = self._load_versions()
        
        if datasets is None:
            datasets = list(versions["datasets"].keys())
        
        snapshot_data = {
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "datasets": {}
        }
        
        for dataset_name in datasets:
            if dataset_name in versions["datasets"]:
                dataset_info = versions["datasets"][dataset_name]
                dataset_path = self.project_root / dataset_info["path"]
                current_hash = self._calculate_dataset_hash(dataset_path)
                
                snapshot_data["datasets"][dataset_name] = {
                    "hash": current_hash,
                    "path": dataset_info["path"],
                    "size_mb": self._get_directory_size(dataset_path),
                    "file_count": self._count_data_files(dataset_path)
                }
        
        snapshot_id = hashlib.md5(json.dumps(snapshot_data, sort_keys=True).encode()).hexdigest()[:8]
        snapshot_data["id"] = snapshot_id
        
        versions["snapshots"].append(snapshot_data)
        self._save_versions(versions)
        
        self.logger.info(f"Created snapshot '{name}' with ID {snapshot_id}")
        return snapshot_id
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all dataset snapshots"""
        versions = self._load_versions()
        return versions.get("snapshots", [])
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered dataset"""
        versions = self._load_versions()
        return versions["datasets"].get(dataset_name)
    
    def _get_directory_size(self, directory: Path) -> float:
        """Get directory size in MB"""
        total_size = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in ['.npy', '.csv', '.edf', '.h5']:
                    total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _count_data_files(self, directory: Path) -> int:
        """Count data files in directory"""
        count = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                if Path(file).suffix in ['.npy', '.csv', '.edf', '.h5']:
                    count += 1
        return count
    
    def init_dvc(self) -> bool:
        """Initialize DVC for the project"""
        try:
            # Check if DVC is already initialized
            if (self.project_root / ".dvc").exists():
                self.logger.info("DVC already initialized")
                return True
            
            # Initialize DVC
            result = subprocess.run(
                ["dvc", "init"], 
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info("DVC initialized successfully")
                return True
            else:
                self.logger.error(f"DVC initialization failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            self.logger.warning("DVC not installed. Install with: pip install dvc")
            return False
    
    def add_to_dvc(self, dataset_name: str) -> bool:
        """Add dataset to DVC tracking"""
        if not self.init_dvc():
            return False
        
        versions = self._load_versions()
        if dataset_name not in versions["datasets"]:
            self.logger.error(f"Dataset '{dataset_name}' not registered")
            return False
        
        dataset_info = versions["datasets"][dataset_name]
        dataset_path = self.project_root / dataset_info["path"]
        
        try:
            # Add to DVC
            result = subprocess.run(
                ["dvc", "add", str(dataset_path)],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info(f"Added '{dataset_name}' to DVC tracking")
                
                # Update dataset info
                dataset_info["dvc_tracked"] = True
                dataset_info["dvc_file"] = f"{dataset_path}.dvc"
                self._save_versions(versions)
                
                return True
            else:
                self.logger.error(f"DVC add failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding to DVC: {str(e)}")
            return False
    
    def validate_data_integrity(self, dataset_name: str) -> Dict[str, Any]:
        """Validate dataset integrity using stored hash"""
        check_result = self.check_dataset_changes(dataset_name)
        
        return {
            "dataset": dataset_name,
            "integrity_status": check_result["status"],
            "message": check_result["message"],
            "validated_at": datetime.now().isoformat(),
            "details": check_result
        }