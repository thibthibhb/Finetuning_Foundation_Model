"""
Memory Management Utility for CBraMod Training Pipeline
Handles cleanup between training trials, checkpoint management, and memory leak prevention.
"""

import os
import gc
import glob
import torch
import psutil
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryManager:
    """Comprehensive memory management for training sessions."""
    
    def __init__(self, 
                 checkpoint_dir: str = "saved_models/finetuned",
                 max_checkpoints: int = 5,
                 memory_threshold_mb: float = 1000.0):
        """
        Initialize memory manager.
        
        Args:
            checkpoint_dir: Directory containing model checkpoints
            max_checkpoints: Maximum number of checkpoints to keep per trial
            memory_threshold_mb: Log warning if memory usage exceeds this (MB)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.memory_threshold_mb = memory_threshold_mb
        self.initial_memory = None
        
    def cleanup_between_trials(self, trial_id: Optional[str] = None):
        """
        Comprehensive cleanup between training trials.
        
        Args:
            trial_id: Optional trial identifier for logging
        """
        logger.info(f"Starting cleanup between trials{f' (Trial: {trial_id})' if trial_id else ''}")
        
        # Clear PyTorch cache
        self._clear_torch_cache()
        
        # Force garbage collection
        self._force_garbage_collection()
        
        # Log memory status
        self._log_memory_status()
        
        logger.info("Trial cleanup completed")
    
    def _clear_torch_cache(self):
        """Clear PyTorch GPU and CPU caches."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Cleared CUDA cache")
        
        # Clear CPU cache if available
        if hasattr(torch, 'clear_autocast_cache'):
            torch.clear_autocast_cache()
            logger.debug("Cleared autocast cache")
    
    def _force_garbage_collection(self):
        """Force Python garbage collection multiple times."""
        for i in range(3):
            collected = gc.collect()
            if collected > 0:
                logger.debug(f"GC cycle {i+1}: collected {collected} objects")
        
        # Force collection of all generations
        for gen in range(3):
            gc.collect(gen)
    
    def manage_checkpoints(self, model_path: str) -> bool:
        """
        Manage checkpoint files, keeping only the best ones.
        
        Args:
            model_path: Path to the newly saved model
            
        Returns:
            True if checkpoint was saved, False if skipped due to limits
        """
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all checkpoint files in the directory
        checkpoint_pattern = str(self.checkpoint_dir / "epoch*_acc_*.pth")
        existing_checkpoints = glob.glob(checkpoint_pattern)
        
        # Sort by modification time (newest first)
        existing_checkpoints.sort(key=os.path.getmtime, reverse=True)
        
        # If we exceed max_checkpoints, remove oldest ones
        if len(existing_checkpoints) >= self.max_checkpoints:
            checkpoints_to_remove = existing_checkpoints[self.max_checkpoints-1:]
            for old_checkpoint in checkpoints_to_remove:
                try:
                    os.remove(old_checkpoint)
                    logger.info(f"Removed old checkpoint: {os.path.basename(old_checkpoint)}")
                except OSError as e:
                    logger.warning(f"Failed to remove {old_checkpoint}: {e}")
        
        return True
    
    
    def start_memory_monitoring(self):
        """Start monitoring memory usage."""
        self.initial_memory = self._get_memory_usage()
        logger.info(f"Initial memory usage: {self.initial_memory:.1f} MB")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _log_memory_status(self):
        """Log current memory status and check for potential leaks."""
        current_memory = self._get_memory_usage()
        
        if self.initial_memory is not None:
            memory_diff = current_memory - self.initial_memory
            logger.info(f"Memory usage: {current_memory:.1f} MB (Δ{memory_diff:+.1f} MB)")
            
            if memory_diff > self.memory_threshold_mb:
                logger.warning(f"Potential memory leak detected! Memory increased by {memory_diff:.1f} MB")
        else:
            logger.info(f"Current memory usage: {current_memory:.1f} MB")
        
        # Log GPU memory if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_cached = torch.cuda.memory_reserved() / 1024 / 1024
            logger.info(f"GPU memory: {gpu_memory:.1f} MB allocated, {gpu_cached:.1f} MB cached")
    
    def cleanup_training_artifacts(self, directories: List[str] = None):
        """
        Clean up training artifacts like temporary files, logs, etc.
        
        Args:
            directories: List of directories to clean. If None, uses defaults.
        """
        if directories is None:
            directories = ["temp", "logs/temp", "runs", "__pycache__"]
        
        for directory in directories:
            dir_path = Path(directory)
            if dir_path.exists():
                try:
                    import shutil
                    shutil.rmtree(dir_path)
                    logger.info(f"Cleaned up directory: {directory}")
                except OSError as e:
                    logger.warning(f"Failed to clean {directory}: {e}")
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get summary of checkpoint files."""
        if not self.checkpoint_dir.exists():
            return {"total_files": 0, "total_size_mb": 0, "oldest_file": None, "newest_file": None}
        
        checkpoint_files = list(self.checkpoint_dir.rglob("*.pth"))
        
        if not checkpoint_files:
            return {"total_files": 0, "total_size_mb": 0, "oldest_file": None, "newest_file": None}
        
        total_size = sum(f.stat().st_size for f in checkpoint_files)
        sorted_files = sorted(checkpoint_files, key=lambda f: f.stat().st_mtime)
        
        return {
            "total_files": len(checkpoint_files),
            "total_size_mb": total_size / 1024 / 1024,
            "oldest_file": sorted_files[0].name if sorted_files else None,
            "newest_file": sorted_files[-1].name if sorted_files else None
        }
    
    def force_cleanup_all_checkpoints(self):
        """Emergency cleanup - remove ALL checkpoint files."""
        if not self.checkpoint_dir.exists():
            return
        
        checkpoint_files = list(self.checkpoint_dir.rglob("*.pth"))
        removed_count = 0
        
        for checkpoint_file in checkpoint_files:
            try:
                checkpoint_file.unlink()
                removed_count += 1
            except OSError as e:
                logger.warning(f"Failed to remove {checkpoint_file}: {e}")
        
        logger.warning(f"Emergency cleanup: removed {removed_count} checkpoint files")

class TrainingMemoryTracker:
    """Context manager for tracking memory during training."""
    
    def __init__(self, memory_manager: MemoryManager, trial_name: str = "training"):
        self.memory_manager = memory_manager
        self.trial_name = trial_name
    
    def __enter__(self):
        self.memory_manager.start_memory_monitoring()
        logger.info(f"Started memory tracking for {self.trial_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.memory_manager.cleanup_between_trials(self.trial_name)
        logger.info(f"Completed memory tracking for {self.trial_name}")