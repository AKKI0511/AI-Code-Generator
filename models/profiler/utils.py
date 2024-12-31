import logging
import time
import functools
import traceback
from typing import Any, Dict, Callable, Optional, TypeVar, cast
from pathlib import Path
import json
from datetime import datetime
import psutil
import sys
import os
from contextlib import contextmanager, asynccontextmanager

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

# Type variable for generic function type
F = TypeVar('F', bound=Callable[..., Any])

def measure_execution_time(func: F) -> F:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to be measured
        
    Returns:
        Wrapped function with timing measurement
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log execution time
            logger.info(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
            
            # Add execution time to result if it's a dict
            if isinstance(result, dict):
                result['execution_time'] = execution_time
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Error in function '{func.__name__}' after {execution_time:.4f} seconds: {str(e)}"
            )
            raise
    
    return cast(F, wrapper)

def log_metrics(metrics: Dict[str, Any], 
                log_file: Optional[Path] = None) -> None:
    """
    Log profiling metrics for debugging.
    
    Args:
        metrics: Dictionary containing metrics to log
        log_file: Optional path to log file
    """
    try:
        # Format metrics for logging
        formatted_metrics = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        # Log to file if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, 'a') as f:
                json.dump(formatted_metrics, f)
                f.write('\n')
        
        # Log to console
        logger.info(f"Profiling metrics: {formatted_metrics}")
        
    except Exception as e:
        logger.error(f"Error logging metrics: {e}")

def get_system_info() -> Dict[str, Any]:
    """
    Get current system information and resources.
    
    Returns:
        Dict containing system information
    """
    try:
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu': {
                'count': cpu_count,
                'percent': psutil.cpu_percent(interval=1),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            },
            'python': {
                'version': sys.version,
                'platform': sys.platform,
                'executable': sys.executable
            }
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {}

def safe_delete(path: Path) -> bool:
    """
    Safely delete a file or directory.
    
    Args:
        path: Path to file or directory to delete
        
    Returns:
        bool: Whether deletion was successful
    """
    try:
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            import shutil
            shutil.rmtree(path)
        return True
    except Exception as e:
        logger.error(f"Error deleting {path}: {e}")
        return False
    
@asynccontextmanager
async def temporary_environment(temp_dir: str = 'temp'):
    """Async context manager for creating a temporary execution environment."""
    temp_path = Path(temp_dir)
    original_cwd = Path.cwd()
    try:
        temp_path.mkdir(parents=True, exist_ok=True)
        os.chdir(temp_path)
        yield
    finally:
        os.chdir(original_cwd)

@contextmanager
class TemporaryEnvironment:
    """Context manager for creating a temporary execution environment"""
    def __init__(self, temp_dir: str = 'temp'):
        self.temp_dir = Path(temp_dir)
        self.original_cwd = Path.cwd()

    def __enter__(self):
        """Set up temporary environment"""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(self.temp_dir)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original environment"""
        os.chdir(self.original_cwd)

    async def __aenter__(self):
        """Async setup for temporary environment."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async restore for temporary environment."""
        self.__exit__(exc_type, exc_val, exc_tb)

def format_size(size_bytes: int) -> str:
    """
    Format byte size to human readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.23 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"

def create_error_report(error: Exception,
                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a detailed error report.
    
    Args:
        error: Exception that occurred
        context: Optional context information
        
    Returns:
        Dict containing error details
    """
    return {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'traceback': traceback.format_exc(),
        'timestamp': datetime.now().isoformat(),
        'system_info': get_system_info(),
        'context': context or {}
    }

def example_function(n: int) -> int:
    """
    Example function to demonstrate execution time measurement.
    
    Args:
        n: Integer input
        
    Returns:
        Square of the input integer
    """
    time.sleep(n)  # Simulate work
    return n * n

def main() -> None:
    """
    Main function to demonstrate the usage of utilities.
    """
    # Example with execution time measurement
    measured_example_function = measure_execution_time(example_function)
    result = measured_example_function(2)
    
    # Example with temporary environment
    with TemporaryEnvironment(PYTHONPATH="/custom/path"):
        result = example_function(2)
        
    # Log the metrics
    metrics = {
        'result': result,
        'system_info': get_system_info()
    }
    log_metrics(metrics, Path('profiling_logs.jsonl'))

if __name__ == "__main__":
    main()
