import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable
import os
import time
import json
from functools import wraps

from .config import Config, get_config
from .code_execution_wrapper import CodeExecutionWrapper
from .dependency_manager import DependencyManager
from .report_generator import ProfilerReportGenerator
from .utils import create_error_report, measure_execution_time
from .performance_metrics import PerformanceMetricsCollector

class ProfileHandler:
    """
    Main entry point for the Real-Time Code Profiler system.
    Coordinates profiling process and manages component interactions with enhanced
    metrics collection and LLM-based optimization suggestions.
    """
    
    def __init__(self, config_or_path: Optional[Union[Path, Config]] = None):
        """
        Initialize the ProfileHandler with configuration.

        Args:
            config_or_path: Configuration object or path to config file.
                          If None, default configuration will be used.
        """
        # Initialize configuration
        self.config = (config_or_path if isinstance(config_or_path, Config) 
                      else get_config(config_or_path))
        
        # Setup logging
        self.config.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        try:
            # Initialize components
            self._initialize_components()
                
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            raise

    @classmethod
    def from_config(cls, config: Config) -> 'ProfileHandler':
        """
        Create a ProfileHandler instance from a Config object.

        Args:
            config (Config): Configuration object

        Returns:
            ProfileHandler: Initialized handler
        """
        return cls(config)

    @classmethod
    def from_path(cls, config_path: Path) -> 'ProfileHandler':
        """
        Create a ProfileHandler instance from a config file path.

        Args:
            config_path (Path): Path to configuration file

        Returns:
            ProfileHandler: Initialized handler
        """
        return cls(config_path)

    def _initialize_components(self) -> None:
        """Initialize all required components with proper error handling."""
        # self.file_handler = FileHandler(config=self.config)

        # Initialize dependency manager
        self.dependency_manager = DependencyManager(config=self.config)
        
        # Initialize code executor
        self.code_executor = CodeExecutionWrapper(config=self.config)
        
        # Initialize metrics collector with new implementation
        self.metrics_collector = PerformanceMetricsCollector()
        
        # Initialize report generator if API key is available
        self._initialize_report_generator()
        
        # Create temp directory if needed
        if self.config.execution.sandbox_enabled:
            os.makedirs(self.config.execution.temp_directory, exist_ok=True)

    def _initialize_report_generator(self) -> None:
        """Initialize the report generator with API key handling."""
        api_key = os.getenv('GROQ_API_KEY') or self.config.env_vars.get('GROQ_API_KEY')
        if api_key:
            self.config.env_vars['GROQ_API_KEY'] = api_key
            self.report_generator = ProfilerReportGenerator(config=self.config)
            self.logger.info(f"Report generator initialized with model: "
                           f"{self.config.report.llm_config['model']}")
        else:
            self.report_generator = None
            self.logger.warning("No Groq API key found - Report generation disabled")

    def _profile_function(self, func):
        """
        Decorator to apply both memory and execution profiling to a function.
        
        Args:
            func: Function to profile
        
        Returns:
            Decorated function with profiling enabled
        """
        @wraps(func)
        @self.metrics_collector.profile_memory
        @self.metrics_collector.profile_execution
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper

    @measure_execution_time
    async def run_code_with_report(
        self, 
        code: str, 
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute provided code with comprehensive profiling and optimization suggestions.

        Args:
            code: Python code to execute and analyze
            progress_callback: Optional callback function(percentage: float, message: str)
                             for progress updates

        Returns:
            Dictionary containing execution results, metrics, and optimization suggestions
        """
        try:
            # Validate and prepare dependencies
            if progress_callback:
                progress_callback(0.1, "Validating dependencies...")
            await self._handle_dependencies(code)
            
            # Execute code with configured limits
            if progress_callback:
                progress_callback(0.3, "Executing and profiling code...")
            execution_result = await self._execute_profiled_code(code)
            
            if not execution_result['success']:
                raise RuntimeError(execution_result.get('error', 'Unknown execution error'))
            
            # Collect and process metrics
            if progress_callback:
                progress_callback(0.6, "Collecting performance metrics...")
            metrics = self._collect_analysis_metrics(execution_result, code)

            # Generate report if available
            if self.report_generator and self.config.report.save_reports:
                if progress_callback:
                    progress_callback(0.8, "Generating optimization report...")
                await self._generate_and_save_report(execution_result, metrics)
            
            execution_result['metrics'] = metrics
            
            if progress_callback:
                progress_callback(1.0, "Analysis complete!")
            return execution_result
            
        except Exception as e:
            if progress_callback:
                progress_callback(1.0, f"Error: {str(e)}")
            self._handle_execution_error(e, code)
        finally:
            # Cleanup temp directory if configured
            if (self.config.execution.sandbox_enabled and 
                self.config.execution.cleanup_temp and 
                os.path.exists(self.config.execution.temp_directory)):
                try:
                    import shutil
                    shutil.rmtree(self.config.execution.temp_directory)
                except Exception as e:
                    self.logger.error(f"Failed to cleanup temp directory: {e}")

    async def _handle_dependencies(self, code: str) -> Dict[str, Any]:
        """
        Validate and install required dependencies.

        Args:
            code: Python code to analyze for dependencies

        Returns:
            Dictionary containing dependency validation results
        """
        deps_result = self.dependency_manager.validate_environment(code)
        
        if not deps_result['valid']:
            if not self.config.dependency.allow_install:
                raise RuntimeError(f"Missing dependencies: {deps_result['missing_packages']}")
                
            async with asyncio.timeout(self.config.dependency.max_install_time):
                install_results = await self.dependency_manager.install_dependencies(
                    deps_result['missing_packages']
                )
                if not all(install_results.values()):
                    raise RuntimeError("Failed to install all required dependencies")
                
        return self.dependency_manager.validate_environment(code)

    async def _execute_profiled_code(self, code: str) -> Dict[str, Any]:
        """
        Execute code with profiling enabled.

        Args:
            code: Python code to execute

        Returns:
            Dictionary containing execution results
        """
        # Wrap the execution with profiling decorators
        @self._profile_function
        async def execute_code():
            return await self.code_executor.execute_with_report(code)
        
        return await execute_code()
    
    def _collect_analysis_metrics(
        self, 
        execution_result: Dict[str, Any],
        code: str
    ) -> Dict[str, Any]:
        """
        Collect and organize all analysis metrics.

        Args:
            execution_result: Results from code execution
            code: Code snippet being analyzed

        Returns:
            Dictionary containing organized metrics and analysis
        """
        metrics = self.metrics_collector.get_metrics()
        
        return {
            'code': code,
            'execution': {
                'time': metrics['total_execution_time'],
                'success': execution_result['success'],
                'namespace': execution_result.get('namespace', {}),
                'execution_context': execution_result.get('execution_context', {})
            },
            'memory': metrics['system']['memory'],
            'cpu': metrics['system']['cpu'],
            'system_info': metrics['system_info'],
            'profiled_functions': metrics['profiled_functions']
        }

    async def _generate_and_save_report(
        self,
        execution_result: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> None:
        """
        Generate and save optimization report.

        Args:
            execution_result: Results from code execution
            metrics: Collected performance metrics
        """
        try:
            report = await self.report_generator.generate_report(
                metrics=metrics,
                formats=self.config.report.report_formats
            )
            execution_result['llm'] = report
            
            if self.config.report.save_reports:
                report_path = Path(self.config.report.report_output_dir)
                report_path.mkdir(exist_ok=True)
                
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                report_file = report_path / f"profile_report_{timestamp}.json"
                
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2)
                self.logger.info(f"Global results saved at {report_file}")
                    
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            execution_result['report_error'] = str(e)

    def _handle_execution_error(self, error: Exception, code: str) -> Dict[str, Any]:
        """
        Handle and format execution errors.

        Args:
            error: Exception that occurred
            code: Original code being executed

        Returns:
            Dictionary containing error information and metrics
        """
        return {
            'success': False,
            'error': str(error),
            'error_report': create_error_report(error, {
                'code': code,
                'metrics': self.metrics_collector.get_metrics()
            })
        }

    def __del__(self):
        """Cleanup resources on deletion."""
        if self.config.execution.cleanup_temp:
            try:
                if hasattr(self, 'code_executor'):
                    self.code_executor.cleanup()
            except Exception as e:
                self.logger.error(f"Cleanup failed: {e}")

# Example usage
if __name__ == "__main__":
    import asyncio
    from pathlib import Path
    from .config import ExecutionConfig, DependencyConfig, ProfilerConfig
    
    async def main():
        """Demonstrate various ways to use the ProfileHandler"""
        
        # Example 1: Basic usage with default configuration
        print("\n=== Example 1: Basic Usage ===")
        profiler = ProfileHandler()
        
        code = """
def factorial(n):
    return 1 if n <= 1 else n * factorial(n-1)

result = factorial(10)
print(f"Factorial of 10 is {result}")
"""
        result = await profiler.run_code_with_report(code)
        if result['success']:
            print(f"Peak Memory: {result['metrics']['memory']['peak_mb']:.2f} MB")
            print(f"CPU Usage: {result['metrics']['cpu']['average_percent']:.1f}%")

        # Example 2: Using custom configuration
        print("\n=== Example 2: Custom Configuration ===")
        custom_config = Config(
            execution=ExecutionConfig(
                max_execution_time=30.0,
                max_memory_mb=1024,
                sandbox_enabled=True,
                allowed_modules={'numpy', 'pandas', 'matplotlib'}
            ),
            profiler=ProfilerConfig(
                enable_cpu_profiling=True,
                enable_memory_profiling=True,
                function_call_tracking=True,
                memory_growth_tracking=True
            ),
            dependency=DependencyConfig(
                allow_install=True,
                trusted_sources=['pypi'],
                verify_checksums=True
            )
        )
        
        profiler = ProfileHandler.from_config(custom_config)
        
        data_analysis_code = """
import numpy as np
import pandas as pd

# Create sample data
data = np.random.randn(1000, 3)
df = pd.DataFrame(data, columns=['A', 'B', 'C'])

# Perform calculations
result = {
    'mean': df.mean().to_dict(),
    'std': df.std().to_dict(),
    'correlation': df.corr().to_dict()
}
print("Analysis complete!")
"""
        result = await profiler.run_code_with_report(data_analysis_code)
        
        # Example 3: Loading configuration from file
        print("\n=== Example 3: Configuration from File ===")
        config_path = Path("config/profiler_config.yaml")
        if config_path.exists():
            profiler = ProfileHandler.from_path(config_path)
            
            # Example with async code
            async_code = """
import asyncio

async def process_data(x):
    await asyncio.sleep(0.1)
    return x * x

async def main():
    tasks = [process_data(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    print(f"Processed results: {results}")

await main()
"""
            result = await profiler.run_code_with_report(async_code)

        # Example 4: Error handling and reporting
        print("\n=== Example 4: Error Handling ===")
        error_code = """
def buggy_function():
    # This will raise a ZeroDivisionError
    return 1/0

result = buggy_function()
"""
        result = await profiler.run_code_with_report(error_code)
        if not result['success']:
            print(f"Error caught: {result['error']}")
            if 'error_report' in result:
                print("Error Report:")
                print(f"  Type: {result['error_report']['error_type']}")
                print(f"  Message: {result['error_report']['error_message']}")

    # Run the examples
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"\nError in example execution: {e}")
        raise

"""
Real-Time Code Profiler Usage Guide:

1. Basic Usage:   ```python
   from models.profiler import ProfileHandler
   
   profiler = ProfileHandler()
   result = await profiler.run_code_with_report("your_code_here")   ```

2. Custom Configuration:   ```python
   from models.profiler import ProfileHandler, Config, ExecutionConfig
   
   config = Config(
       execution=ExecutionConfig(
           max_execution_time=30.0,
           sandbox_enabled=True
       )
   )
   profiler = ProfileHandler.from_config(config)   ```

3. Configuration from File:   ```python
   from pathlib import Path
   
   profiler = ProfileHandler.from_path(Path("config.yaml"))   ```

4. Accessing Results:   ```python
   result = await profiler.run_code_with_report(code)
   if result['success']:
       memory_usage = result['metrics']['memory']['peak_mb']
       cpu_usage = result['metrics']['cpu']['average_percent']
       function_profiles = result['metrics']['function_profiles']   ```

Key Features:
- Real-time performance monitoring
- Memory usage tracking
- CPU profiling
- Function call analysis
- Dependency management
- Sandboxed execution
- Detailed error reporting
- Async code support

For more information, visit: https://github.com/AKKI0511/AI-Code-Generator
"""
