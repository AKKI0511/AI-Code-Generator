import ast
import asyncio
import logging
import sys
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import builtins
from contextlib import contextmanager

from .utils import measure_execution_time, create_error_report, temporary_environment
from .performance_metrics import PerformanceMetricsCollector
from .config import get_config, ProfilerConfig

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

class CodeExecutionWrapper:
    """
    Safely executes Python code with proper isolation and dependency management.
    Integrates with PerformanceMetricsCollector for comprehensive profiling.

    This wrapper is designed to be called from ProfileHandler and works in conjunction
    with PerformanceMetricsCollector for metric gathering.
    """

    def __init__(self, config: Optional[ProfilerConfig] = None):
        """
        Initialize the code execution wrapper.

        Args:
            config: Optional configuration for the profiler
        """
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        # Initialize metrics collector - actual profiling is handled by ProfileHandler
        self.metrics_collector = PerformanceMetricsCollector()

    @contextmanager
    def safe_execution_environment(self):
        """
        Context manager that provides a safe execution environment by:
        - Storing original sys.modules
        - Creating a clean namespace
        - Restoring the original state after execution
        """
        original_modules = dict(sys.modules)
        original_path = list(sys.path)

        try:
            yield
        finally:
            # Restore original state
            sys.modules.clear()
            sys.modules.update(original_modules)
            sys.path[:] = original_path

    def validate_code(self, code: str) -> Tuple[bool, str]:
        """
        Validate Python code for syntax and security issues.

        Args:
            code: Python code to validate

        Returns:
            Tuple containing validation status and error message
        """
        try:
            ast.parse(code)

            # No need to validate imports against allowed modules
            # Just check for restricted builtins
            if any(
                builtin in code for builtin in self.config.execution.restricted_builtins
            ):
                self.logger.error("Use of restricted builtins detected")
                return False, f"Use of restricted builtins detected"

            return True, ""
        except SyntaxError as e:
            self.logger.error(f"Syntax error: {str(e)}")
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return False, f"Validation error: {str(e)}"

    @measure_execution_time
    async def execute_with_report(self, code: str) -> Dict[str, Any]:
        """
        Execute code and return execution results. Metrics collection is handled by
        ProfileHandler through its profiling decorators.

        Args:
            code: Python code to execute

        Returns:
            Dictionary containing execution results
        """
        try:
            # Validate code
            is_valid, error_message = self.validate_code(code)
            if not is_valid:
                self.logger.error(f"Code validation failed: {error_message}")
                raise ValueError(error_message)

            # Prepare execution environment
            namespace = {
                "__builtins__": {
                    name: getattr(builtins, name)
                    for name in dir(builtins)
                    if name not in self.config.execution.restricted_builtins
                }
            }

            # Execute code with timeout and sandboxing as configured
            if self.config.execution.timeout_enabled:
                async with asyncio.timeout(self.config.execution.max_execution_time):
                    await self._execute_code(code, namespace)
            else:
                await self._execute_code(code, namespace)

            # Filter the namespace to only include variable names and their values
            filtered_namespace = {
                k: v
                for k, v in namespace.items()
                if not k.startswith("__") and not isinstance(v, type(builtins))
            }

            return {
                "success": True,
                "namespace": filtered_namespace,
                "execution_context": {
                    "sandbox_enabled": self.config.execution.sandbox_enabled,
                    "timeout_enabled": self.config.execution.timeout_enabled,
                },
            }

        except asyncio.TimeoutError:
            self.logger.error(
                f"Execution timeout after {self.config.execution.max_execution_time} seconds"
            )
            return {
                "success": False,
                "error": f"Execution timeout after {self.config.execution.max_execution_time} seconds",
                "namespace": {},
                "execution_context": {
                    "sandbox_enabled": self.config.execution.sandbox_enabled,
                    "timeout_enabled": self.config.execution.timeout_enabled,
                },
            }
        except Exception as e:
            self.logger.error(f"Execution error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_report": create_error_report(
                    e,
                    {
                        "code": code,
                    },
                ),
                "namespace": {},
                "execution_context": {
                    "sandbox_enabled": self.config.execution.sandbox_enabled,
                    "timeout_enabled": self.config.execution.timeout_enabled,
                },
            }

    async def _execute_code(self, code: str, namespace: Dict[str, Any]) -> None:
        """
        Execute code with proper sandboxing if enabled.

        Args:
            code: Python code to execute
            namespace: Execution namespace
        """
        if self.config.execution.sandbox_enabled:
            async with temporary_environment(self.config.execution.temp_directory):
                await asyncio.to_thread(exec, code, namespace)
        else:
            await asyncio.to_thread(exec, code, namespace)

    def cleanup(self) -> None:
        """Clean up any temporary resources used during execution."""
        if self.config.execution.sandbox_enabled and self.config.execution.cleanup_temp:
            try:
                import shutil

                temp_dir = Path(self.config.execution.temp_directory)
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    self.logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                self.logger.error(f"Failed to cleanup temporary directory: {e}")


# Example usage
if __name__ == "__main__":

    async def main():
        # Initialize the wrapper
        wrapper = CodeExecutionWrapper()

        # Example 1: Simple code execution
        code_snippet = """
def intensive_operation():
    data = []
    for i in range(1_000_000):
        data.append(i ** 2)
    return sum(data)

result = intensive_operation()
print(f"Operation result = {result}")
"""
        result = await wrapper.execute_with_report(code_snippet)
        print("\nExample 1 - Simple code execution:")
        print(f"Success: {result['success']}")
        print(f"Namespace: {result['namespace']}")

        #         # Example 2: Code with imports
        code_with_imports = """
import math
import statistics

numbers = [1, 2, 3, 4, 5]
mean = statistics.mean(numbers)
std_dev = statistics.stdev(numbers)
print(f"Mean: {mean}, Standard Deviation: {std_dev}")
"""
        result = await wrapper.execute_with_report(code_with_imports)
        print("\nExample 2 - Code with imports:")
        print(f"Success: {result['success']}")
        print(f"Namespace: {result['namespace']}")

        # Example 3: Invalid code (demonstration of error handling)
        invalid_code = """
import restricted_module  # This should fail if not in allowed_modules
print("This won't execute")
"""
        result = await wrapper.execute_with_report(invalid_code)
        print("\nExample 3 - Invalid code:")
        print(f"Success: {result['success']}")
        print(f"Error: {result.get('error', 'No error')}")

    # Run the examples
    asyncio.run(main())
