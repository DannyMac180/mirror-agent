"""Google Cloud Platform logging configuration module."""

import os
import time
from typing import Dict, Any, Optional, List
from google.cloud import logging

class MirrorAgentLogger:
    """Wrapper for GCP logging with Mirror Agent specific functionality."""
    
    def __init__(self, log_name: str = "mirror-agent"):
        """Initialize the logger.
        
        Args:
            log_name: Name of the log to write to
        """
        self.client = logging.Client()
        self.logger = self.client.logger(log_name)
        self.start_time = None
        
    def log_info(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log an info message with context.
        
        Args:
            message: Info message
            context: Additional context
        """
        info_data = {
            'event': 'info',
            'message': message,
            'timestamp': time.time()
        }
        if context:
            info_data.update(context)
            
        self.logger.log_struct(info_data, severity='INFO')
        
    def log_start_indexing(self, vault_path: str, total_files: int):
        """Log the start of an indexing operation.
        
        Args:
            vault_path: Path to the Obsidian vault
            total_files: Total number of files to process
        """
        self.start_time = time.time()
        self.logger.log_struct({
            'event': 'indexing_started',
            'vault_path': vault_path,
            'total_files': total_files,
            'timestamp': self.start_time
        }, severity='INFO')
        
    def log_end_indexing(self, processed_files: int, success: bool):
        """Log the end of an indexing operation.
        
        Args:
            processed_files: Number of files successfully processed
            success: Whether the operation completed successfully
        """
        end_time = time.time()
        duration = end_time - (self.start_time or end_time)
        
        self.logger.log_struct({
            'event': 'indexing_completed',
            'success': success,
            'processed_files': processed_files,
            'duration_seconds': duration,
            'timestamp': end_time
        }, severity='INFO' if success else 'ERROR')
        
    def log_batch_failure(self, batch_files: List[str], error: Exception):
        """Log a batch processing failure.
        
        Args:
            batch_files: List of files in the failed batch
            error: The exception that caused the failure
        """
        self.logger.log_struct({
            'event': 'batch_failure',
            'files': batch_files,
            'error': str(error),
            'error_type': error.__class__.__name__,
            'timestamp': time.time()
        }, severity='ERROR')
        
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        metrics['event'] = 'performance_metrics'
        metrics['timestamp'] = time.time()
        self.logger.log_struct(metrics, severity='INFO')
        
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log an error with context.
        
        Args:
            error: The exception to log
            context: Additional context about the error
        """
        error_data = {
            'event': 'error',
            'error': str(error),
            'error_type': error.__class__.__name__,
            'timestamp': time.time()
        }
        if context:
            error_data.update(context)
            
        self.logger.log_struct(error_data, severity='ERROR')
        
    def log_warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log a warning with context.
        
        Args:
            message: Warning message
            context: Additional context
        """
        warning_data = {
            'event': 'warning',
            'message': message,
            'timestamp': time.time()
        }
        if context:
            warning_data.update(context)
            
        self.logger.log_struct(warning_data, severity='WARNING')

def get_logger(log_name: str = "mirror-agent") -> Optional[MirrorAgentLogger]:
    """Get or create a Mirror Agent logger instance.
    
    Args:
        log_name: Name of the log to write to
        
    Returns:
        Optional[MirrorAgentLogger]: Logger instance or None if setup fails
    """
    try:
        return MirrorAgentLogger(log_name)
    except Exception as e:
        print(f"Failed to setup GCP logging: {e}")
        return None

if __name__ == "__main__":
    # Test the logging setup
    logger = get_logger()
    if logger:
        # Test various logging functions
        logger.log_info("Starting test run", {
            "test_type": "manual",
            "environment": "development"
        })
        logger.log_start_indexing("/path/to/vault", 100)
        logger.log_performance_metrics({
            "batch_processing_time": 1.5,
            "memory_usage_mb": 256
        })
        logger.log_warning("Some files might be skipped", {
            "reason": "permissions"
        })
        logger.log_batch_failure(
            ["doc1.md", "doc2.md"],
            ValueError("Invalid content")
        )
        logger.log_end_indexing(95, True)
        print("Test logs sent successfully")
    else:
        print("Failed to initialize logger") 