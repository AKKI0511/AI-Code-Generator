from collections import defaultdict
import logging
import platform
import time
from pathlib import Path
import psutil
import json
from typing import Dict, Any, Optional
from functools import wraps
from memory_profiler import profile as memory_profile
from line_profiler import LineProfiler

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

class PerformanceMetricsCollector:
    """
    Collects system performance metrics using psutil, memory_profiler, and line_profiler.
    
    Features:
    - Automated CPU and memory profiling using psutil
    - Function-level memory profiling using memory_profiler
    - Line-by-line execution profiling using line_profiler
    - System resource monitoring
    - Metric persistence
    """
    
    def __init__(self):
        """Initialize the metrics collector with psutil process tracking."""
        self.process = psutil.Process()
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._line_profiler = LineProfiler()
        self.metrics_data: Dict[str, Any] = {}
        self.total_execution_time: float = 0.0
    
    def profile_memory(self, func):
        """
        Decorator that combines memory_profiler with custom metric collection.
        
        Args:
            func: The function to profile
            
        Returns:
            Decorated function with memory profiling
        """
        @wraps(func)
        @memory_profile
        def wrapper(*args, **kwargs):
            
            # Collect initial metrics
            initial_metrics = self._collect_system_metrics()
            start_time = time.time()
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Collect final metrics
            end_time = time.time()
            final_metrics = self._collect_system_metrics()
            
            # Calculate differential metrics
            self.metrics_data[func.__name__].update({
                "duration": end_time - start_time,
                "memory_delta": final_metrics["memory"]["used"] - initial_metrics["memory"]["used"],
                "cpu_percent": final_metrics["cpu"]["percent"],
            })
            
            # Update total execution time
            self.total_execution_time += self.metrics_data[func.__name__]["duration"]
            
            return result
        return wrapper
    
    def profile_execution(self, func):
        """
        Decorator that combines line_profiler with custom metric collection.

        Args:
            func: The function to profile

        Returns:
            Decorated function with line-by-line profiling
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Add the undecorated function to the line profiler
            undecorated_func = func.__wrapped__ if hasattr(func, "__wrapped__") else func
            self._line_profiler.add_function(undecorated_func)

            # Start line profiling
            self._line_profiler.enable()
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable line profiling after execution
                self._line_profiler.disable()

            # Collect line-by-line stats
            stats = self._collect_line_profile_stats(self._line_profiler, undecorated_func)
            if func.__name__ not in self.metrics_data:
                self.metrics_data[func.__name__] = {}
            self.metrics_data[func.__name__]["line_profile"] = stats
            return result

        return wrapper
    
    def _collect_line_profile_stats(self, profiler: LineProfiler, func):
        """
        Collect and format line-by-line profiling stats.

        Args:
            profiler: The LineProfiler instance
            func: The undecorated function being profiled

        Returns:
            Dictionary with line-by-line profiling data
        """
        stats = {}
        try:
            profile_stats = profiler.get_stats()
            for key, timing in profile_stats.timings.items():
                filename, start_line, function_name = key

                # Match the function name and file path
                if function_name == func.__name__ and filename == func.__code__.co_filename:
                    for (line, hits, time) in timing:
                        stats[line-start_line-1] = {
                            "hits": hits,
                            "time_microseconds": time,
                            # "filename": filename.split('\\')[-1],
                            "function": function_name,
                        }
            return stats

        except Exception as e:
            logger.error(f"Error collecting line profile stats: {e}")
            return {"error": str(e)}
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """
        Collect current system metrics using psutil.
        
        Returns:
            Dictionary containing current system metrics
        """
        try:
            # Get memory information
            virtual_memory = psutil.virtual_memory()
            swap_memory = psutil.swap_memory()
            
            # Get CPU information
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            
            # Get process-specific information
            process_memory = self.process.memory_info()
            process_cpu = self.process.cpu_percent()
            
            # Calculate the number of CPU cores utilized
            cpu_cores_utilized = len([x for x in psutil.cpu_percent(percpu=True, interval=0.1) if x > 0])
            
            # Get disk I/O information
            disk_io = psutil.disk_io_counters()
            
            return {
                "memory": {
                    "used": virtual_memory.used,
                    "percent": virtual_memory.percent,
                    "swap_percent": swap_memory.percent
                },
                "cpu": {
                    "percent": cpu_percent,
                    "process_percent": process_cpu,
                    "cpu_cores_utilized": cpu_cores_utilized
                },
            }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the collected performance metrics.
        
        Returns:
            Dictionary containing all collected metrics
        """
        try:
            current_metrics = self._collect_system_metrics()
            
            metrics = {
                "system": current_metrics,
                "profiled_functions": self.metrics_data,
                "total_execution_time": self.total_execution_time,
                "system_info": {
                    "python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "cpu_cores": psutil.cpu_count(),
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {"error": str(e)}
    
    def save_metrics(self, file_path: Path) -> None:
        """
        Save metrics to a JSON file.
        
        Args:
            file_path: Path to save the metrics
        """
        try:
            metrics = self.get_metrics()
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "w") as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Saved metrics to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            raise

# Example usage
if __name__ == "__main__":
    collector = PerformanceMetricsCollector()
    
    # Example function with both memory and line profiling
    @collector.profile_memory
    @collector.profile_execution
    def intensive_operation():
        """Example function that performs memory-intensive operations."""
        data = []
        for i in range(1_000_000):
            data.append(i ** 2)
        return sum(data)
    
    try:
        # Execute the profiled function
        result = intensive_operation()
        
        # Get and display metrics
        metrics = collector.get_metrics()
        
        print("\nPerformance Metrics:")
        print(f"Execution time: {metrics['total_execution_time']}s")
        print(f"Memory Usage (%): {metrics['system']['memory']['percent']} %")
        print(f"CPU Usage: {metrics['system']['cpu']['process_percent']}%")
        print(f"Function Duration: {metrics['profiled_functions']['intensive_operation']['duration']:.2f} seconds")
        
        # Save metrics to file
        collector.save_metrics(Path("performance_metrics.json"))
        
    except Exception as e:
        logger.error(f"Error in example usage: {e}")