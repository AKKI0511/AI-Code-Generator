import asyncio
import os
from pathlib import Path
import sys
import time
import json

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from models.profiler.profiler_handler import ProfileHandler
from models.profiler.config import ProfilerConfig, Config, ExecutionConfig, DependencyConfig, ReportConfig

async def test_profiler():
    try:
        # Create config with test settings
        config = Config(
            execution=ExecutionConfig(
                max_execution_time=300.0,
                max_memory_mb=1024,
                sandbox_enabled=True,
                cleanup_temp=True
            ),
            profiler=ProfilerConfig(
                enable_cpu_profiling=True,
                enable_memory_profiling=True,
                function_call_tracking=True,
                loop_detection=True,
                memory_growth_tracking=True
            ),
            dependency=DependencyConfig(
                allow_install=True,
                trusted_sources=['pypi'],
                verify_checksums=True
            ),
            report=ReportConfig(
                report_formats=['md'],
            )
        )
        
        # Try to get API key from environment
        api_key = os.getenv('GROQ_API_KEY')
        if api_key:
            config.env_vars['GROQ_API_KEY'] = api_key
            print("\n✅ Found Groq API key - Report generation will be enabled")
        else:
            print("\n⚠️ No Groq API key found - Report generation will be disabled")
        
        # Initialize profiler
        print("\n Initializing profiler...")
        profiler = ProfileHandler(config)
        
        # Comprehensive Test: Data Processing and Analysis
        print("\n📊 Comprehensive Test: Data Processing and Analysis")
        test_code = """
def inefficient_sum_of_squares(n):
    total = 0
    for i in range(n):
        total += i * i  # Inefficiently calculating the sum of squares
    return total

def analyze_large_data(count):
    # Simulate processing a large dataset
    print("Starting analysis...")
    result = inefficient_sum_of_squares(count)
    print(f"Total sum of squares up to {count}: {result}")

# Main execution
analyze_large_data(10_000_000)  # Use a large number to create a performance issue
"""
        print("Running data processing and analysis...")
        result = await profiler.run_code_with_report(test_code)
        print_test_results(result, "Data Processing and Analysis")

    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        raise

def print_test_results(result: dict, test_name: str):
    """Print formatted test results as pretty JSON, filtering out non-serializable parts."""
    print(f"\nResults for {test_name}:")
    print("=" * 50)

    # Create a filtered version of the result that only includes serializable items
    serializable_result = {}
    
    for key, value in result.items():
        try:
            # Attempt to serialize the value to check if it's serializable
            json.dumps(value)
            serializable_result[key] = value
        except (TypeError, ValueError):
            # If not serializable, skip this key
            print(f"Skip non-serializable key: {key}")

    # Print the filtered result as pretty JSON
    print(json.dumps(serializable_result, indent=4))  # Pretty print the filtered result

if __name__ == "__main__":
    print("\n🧪 Starting Profiler Test Suite")
    print("=" * 50)
    
    start_time = time.time()
    asyncio.run(test_profiler())
    duration = time.time() - start_time
    
    print("\n✨ Test Suite Completed")
    print(f"Total duration: {duration:.2f} seconds")
