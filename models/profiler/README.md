# Profiler Module

The Profiler module is a comprehensive system designed for profiling and optimizing Python code. It provides real-time performance monitoring, dependency management, and detailed reporting, leveraging AI for optimization suggestions.

## Features

- **Code Execution and Profiling**: Safely execute Python code with isolation and gather performance metrics.
- **Dependency Management**: Detect, validate, and manage Python package dependencies.
- **Real-Time Profiling**: Monitor execution time, memory usage, and CPU utilization.
- **AI-Powered Optimization**: Generate actionable optimization suggestions using AI.
- **Report Generation**: Create detailed reports in multiple formats (JSON, Markdown, PDF).
- **Sandboxed Execution**: Execute code in a controlled environment to ensure safety.
- **Error Handling**: Comprehensive error reporting and handling.
- **Profiler**: Integrate with PerformanceMetricsCollector for comprehensive profiling and metric gathering.

## Components

### 1. Code Execution Wrapper
- **File**: `models/profiler/code_execution_wrapper.py`
- **Purpose**: Executes Python code with isolation and integrates with `PerformanceMetricsCollector` for profiling.
- **Key Methods**:
  - `validate_code`: Validates code for syntax and security.
  - `execute_with_report`: Executes code and returns execution results with metrics.

### 2. Dependency Manager
- **File**: `models/profiler/dependency_manager.py`
- **Purpose**: Manages Python package dependencies, including detection and installation.
- **Key Methods**:
  - `check_dependencies`: Checks for missing or outdated packages.
  - `install_dependencies`: Installs required packages.

### 3. Real-Time Profiler
- **File**: `models/profiler/fallback/real_time_profiler.py`
- **Purpose**: Profiles code execution in real-time and provides optimization recommendations.
- **Key Methods**:
  - `profile_code_files`: Profiles multiple Python files.
  - `get_optimization_recommendations`: Generates optimization suggestions.

### 4. Performance Metrics Collector
- **File**: `models/profiler/performance_metrics.py`
- **Purpose**: Collects and manages system performance metrics.
- **Key Methods**:
  - `profile_execution`: Decorator for line-by-line execution profiling.
  - `get_metrics`: Retrieves collected metrics.

### 5. Report Generator
- **File**: `models/profiler/report_generator.py`
- **Purpose**: Generates comprehensive profiling reports using LLM-enhanced analysis.
- **Key Methods**:
  - `generate_report`: Creates reports in JSON, Markdown, and PDF formats.

### 6. File Handler
- **File**: `models/profiler/file_handler.py`
- **Purpose**: Manages file operations, including saving and cleaning up temporary files.
- **Key Methods**:
  - `save_files_locally`: Saves files to a local directory.
  - `cleanup_files`: Cleans up temporary files and directories.

## Configuration

- **File**: `models/profiler/config.py`
- **Purpose**: Defines configuration settings for execution, profiling, dependencies, and reporting.
- **Key Classes**:
  - `ExecutionConfig`: Settings for code execution.
  - `DependencyConfig`: Settings for dependency management.
  - `ReportConfig`: Settings for report generation.

## Usage

### Basic Example

```python
from models.profiler.profiler_handler import ProfileHandler

async def main():
    profiler = ProfileHandler()
    code = """
    def factorial(n):
        return 1 if n <= 1 else n * factorial(n-1)
    result = factorial(10)
    print(f"Factorial of 10 is {result}")
    """
    result = await profiler.run_code_with_report(code)
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Running Tests

- **File**: `models/profiler/tests/test_profiler.py`
- **Purpose**: Contains test cases to validate the functionality of the profiler module.
- **Command**: Run tests using `python models/profiler/tests/test_profiler.py`.

## Logging

- Configured across modules to provide detailed execution logs.
- Log format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

## Contributing

Contributions are welcome! Please ensure that your code adheres to the existing style and passes all tests.

## License

This project is licensed under the MIT License.

For more information, visit the [GitHub repository](https://github.com/AKKI0511/AI-Code-Generator).
