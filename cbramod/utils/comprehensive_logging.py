"""
Comprehensive Logging System for CBraMod

This module provides standardized logging with structured output, 
performance monitoring, and multiple log destinations.
"""

import logging
import json
import time
import psutil
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
from contextlib import contextmanager
import functools
import traceback


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def __init__(self, include_context=True, include_performance=True):
        self.include_context = include_context
        self.include_performance = include_performance
        super().__init__()
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add context information if available
        if self.include_context and hasattr(record, 'context'):
            log_entry['context'] = record.context
        
        # Add performance metrics if available
        if self.include_performance:
            if hasattr(record, 'performance'):
                log_entry['performance'] = record.performance
            
            # Add system metrics
            log_entry['system'] = {
                "memory_percent": psutil.virtual_memory().percent,
                "cpu_percent": psutil.cpu_percent()
            }
            
            # Add GPU metrics if available
            if torch.cuda.is_available():
                log_entry['system']['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
                log_entry['system']['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry, default=str)


class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_time = None
        self.start_memory = None
        self.start_gpu_memory = None
    
    def start(self, operation_name: str):
        """Start monitoring an operation"""
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used / 1024**3  # GB
        
        if torch.cuda.is_available():
            self.start_gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        
        self.logger.info(f"Started operation: {operation_name}", extra={
            'context': {'operation': operation_name, 'phase': 'start'},
            'performance': {
                'memory_gb': self.start_memory,
                'gpu_memory_gb': self.start_gpu_memory
            }
        })
    
    def end(self, operation_name: str, additional_metrics: Optional[Dict] = None):
        """End monitoring and log results"""
        if self.start_time is None:
            self.logger.warning(f"Performance monitor not started for {operation_name}")
            return
        
        end_time = time.time()
        duration = end_time - self.start_time
        end_memory = psutil.virtual_memory().used / 1024**3  # GB
        memory_delta = end_memory - self.start_memory
        
        metrics = {
            'duration_seconds': duration,
            'memory_start_gb': self.start_memory,
            'memory_end_gb': end_memory,
            'memory_delta_gb': memory_delta
        }
        
        if torch.cuda.is_available() and self.start_gpu_memory is not None:
            end_gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            metrics.update({
                'gpu_memory_start_gb': self.start_gpu_memory,
                'gpu_memory_end_gb': end_gpu_memory,
                'gpu_memory_delta_gb': end_gpu_memory - self.start_gpu_memory
            })
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        self.logger.info(f"Completed operation: {operation_name}", extra={
            'context': {'operation': operation_name, 'phase': 'end'},
            'performance': metrics
        })
        
        # Reset for next operation
        self.start_time = None
        self.start_memory = None
        self.start_gpu_memory = None


class CBraModLogger:
    """Comprehensive logging system for CBraMod"""
    
    def __init__(self, config_manager, name: str = "cbramod"):
        self.config_manager = config_manager
        self.name = name
        self.logger = None
        self.performance_monitor = None
        self.context_stack = []
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup comprehensive logging configuration"""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Get logging configuration
        log_config = self.config_manager.get('logging', {})
        
        # Setup handlers based on configuration
        self._setup_console_handler(log_config.get('handlers', {}).get('console', {}))
        self._setup_file_handler(log_config.get('handlers', {}).get('file', {}))
        self._setup_experiment_handler(log_config.get('handlers', {}).get('experiment', {}))
        self._setup_error_handler(log_config.get('handlers', {}).get('error', {}))
        
        # Setup performance monitor
        self.performance_monitor = PerformanceMonitor(self.logger)
    
    def _setup_console_handler(self, config: Dict):
        """Setup console logging handler"""
        if not config.get('enabled', True):
            return
        
        handler = logging.StreamHandler()
        handler.setLevel(getattr(logging, config.get('level', 'INFO')))
        
        if self.config_manager.get('logging.structured.enabled', False):
            formatter = StructuredFormatter(
                include_context=self.config_manager.get('logging.structured.include_context', True),
                include_performance=self.config_manager.get('logging.structured.include_performance', True)
            )
        else:
            format_str = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            formatter = logging.Formatter(format_str)
        
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def _setup_file_handler(self, config: Dict):
        """Setup file logging handler"""
        if not config.get('enabled', True):
            return
        
        log_path = Path(config.get('path', 'logs/cbramod_detailed.log'))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_path)
        handler.setLevel(getattr(logging, config.get('level', 'DEBUG')))
        
        if self.config_manager.get('logging.structured.enabled', False):
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def _setup_experiment_handler(self, config: Dict):
        """Setup experiment-specific logging handler"""
        if not config.get('enabled', True):
            return
        
        log_path = Path(config.get('path', 'logs/experiments.log'))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        class ExperimentFilter(logging.Filter):
            def __init__(self, pattern):
                super().__init__()
                self.pattern = pattern.lower()
            
            def filter(self, record):
                message = record.getMessage().lower()
                return any(keyword in message for keyword in self.pattern.split('|'))
        
        handler = logging.FileHandler(log_path)
        handler.setLevel(getattr(logging, config.get('level', 'INFO')))
        handler.addFilter(ExperimentFilter(config.get('filter_pattern', 'experiment|trial|hyperopt')))
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def _setup_error_handler(self, config: Dict):
        """Setup error-only logging handler"""
        if not config.get('enabled', True):
            return
        
        log_path = Path(config.get('path', 'logs/errors.log'))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_path)
        handler.setLevel(logging.ERROR)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(exc_info)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    @contextmanager
    def context(self, **kwargs):
        """Add context to all log messages within this block"""
        self.context_stack.append(kwargs)
        try:
            yield
        finally:
            self.context_stack.pop()
    
    def _get_current_context(self):
        """Get current logging context"""
        context = {}
        for ctx in self.context_stack:
            context.update(ctx)
        return context
    
    def _log_with_context(self, level, message, *args, **kwargs):
        """Log message with current context"""
        extra = kwargs.get('extra', {})
        current_context = self._get_current_context()
        
        if current_context:
            extra['context'] = {**extra.get('context', {}), **current_context}
        
        kwargs['extra'] = extra
        self.logger.log(level, message, *args, **kwargs)
    
    def debug(self, message, *args, **kwargs):
        self._log_with_context(logging.DEBUG, message, *args, **kwargs)
    
    def info(self, message, *args, **kwargs):
        self._log_with_context(logging.INFO, message, *args, **kwargs)
    
    def warning(self, message, *args, **kwargs):
        self._log_with_context(logging.WARNING, message, *args, **kwargs)
    
    def error(self, message, *args, **kwargs):
        self._log_with_context(logging.ERROR, message, *args, **kwargs)
    
    def critical(self, message, *args, **kwargs):
        self._log_with_context(logging.CRITICAL, message, *args, **kwargs)
    
    def exception(self, message, *args, **kwargs):
        """Log exception with full traceback"""
        kwargs['exc_info'] = True
        self.error(message, *args, **kwargs)


def log_performance(operation_name: str = None):
    """Decorator to automatically log performance of functions"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get logger from first argument (usually self)
            logger = None
            if args and hasattr(args[0], 'logger') and hasattr(args[0].logger, 'performance_monitor'):
                logger = args[0].logger
            
            if logger and logger.performance_monitor:
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                logger.performance_monitor.start(op_name)
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Add function-specific metrics if result is a dict
                    additional_metrics = {}
                    if isinstance(result, dict) and 'kappa' in result:
                        additional_metrics = {
                            'kappa': result.get('kappa'),
                            'accuracy': result.get('accuracy'),
                            'f1_score': result.get('f1_score')
                        }
                    
                    logger.performance_monitor.end(op_name, additional_metrics)
                    return result
                    
                except Exception as e:
                    logger.performance_monitor.end(op_name, {'error': str(e)})
                    raise
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class StandardizedErrorHandler:
    """Standardized error handling patterns"""
    
    def __init__(self, logger: CBraModLogger):
        self.logger = logger
    
    @contextmanager
    def handle_errors(self, operation: str, reraise: bool = True, default_return=None):
        """Context manager for standardized error handling"""
        try:
            with self.logger.context(operation=operation):
                yield
        except KeyError as e:
            self.logger.error(f"Configuration error in {operation}: Missing key {e}")
            if reraise:
                raise ConfigurationError(f"Missing configuration key: {e}") from e
            return default_return
        except ValueError as e:
            self.logger.error(f"Value error in {operation}: {e}")
            if reraise:
                raise ValidationError(f"Invalid value in {operation}: {e}") from e
            return default_return
        except FileNotFoundError as e:
            self.logger.error(f"File not found in {operation}: {e}")
            if reraise:
                raise ResourceError(f"Required file not found: {e}") from e
            return default_return
        except RuntimeError as e:
            if "CUDA" in str(e) or "memory" in str(e).lower():
                self.logger.error(f"CUDA/Memory error in {operation}: {e}")
                if reraise:
                    raise ResourceError(f"GPU/Memory error: {e}") from e
            else:
                self.logger.error(f"Runtime error in {operation}: {e}")
                if reraise:
                    raise
            return default_return
        except Exception as e:
            self.logger.exception(f"Unexpected error in {operation}")
            if reraise:
                raise
            return default_return


# Custom exception classes for better error categorization
class CBraModError(Exception):
    """Base exception for CBraMod"""
    pass

class ConfigurationError(CBraModError):
    """Configuration-related errors"""
    pass

class ValidationError(CBraModError):
    """Data validation errors"""
    pass

class ResourceError(CBraModError):
    """Resource-related errors (files, GPU, memory)"""
    pass

class TrainingError(CBraModError):
    """Training-related errors"""
    pass