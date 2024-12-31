import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import logging
import os
from pathlib import Path
import time
import cProfile
import pstats
import functools
import tracemalloc
import psutil
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Callable, List, Tuple
from io import StringIO
import traceback

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import tqdm
import sys

# directory reach
directory = Path(__file__).absolute()

# setting path
sys.path.append(str(directory.parent.parent))
from code_assistant import CodeAssistant

# Configure logging with more detailed format
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

class RealTimeProfiler:
    """
    A comprehensive code profiling system with AI-powered optimization recommendations.
    
    Supports detailed performance analysis, including:
    - Execution time tracking
    - Memory usage monitoring
    - CPU utilization analysis
    - Bottleneck identification
    - AI-generated optimization suggestions
    """
    
    def __init__(self, 
                 max_execution_time: float = 10.0, 
                 max_memory_mb: int = 100,
                 code_assistant: CodeAssistant = None):
        """
        Initialize the RealTimeProfiler with configurable constraints.
        
        Args:
            max_execution_time (float): Maximum allowed execution time in seconds.
            max_memory_mb (int): Maximum allowed memory usage in megabytes.
            code_assistant (CodeAssistant): CodeAssistant instance for LLM calls.
        """
        self.max_execution_time = max_execution_time
        self.max_memory_mb = max_memory_mb
        self.code_assistant = code_assistant
        
        # Initialize LLM for optimization suggestions
        self.llm = ChatGroq(
            api_key=self.code_assistant.groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=2000,
        )
        
        # Optimization suggestion prompt
        self.optimization_prompt = ChatPromptTemplate.from_template("""
        Analyze the following Python code performance metrics:
        Execution Time: {execution_time} seconds
        Peak Memory Usage: {peak_memory} MB
        CPU Usage: {cpu_usage}%
        
        Provide concise, actionable Python optimization suggestions considering:
        1. Algorithmic improvements
        2. Memory efficiency
        3. Computational complexity reduction
        
        Suggestions should be specific and implementable.
        """)
        self.logger = logging.getLogger(__name__)
    
    def profile_code(self, 
                     code: str, 
                     globals_dict: Optional[Dict[str, Any]] = None,
                     locals_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Profile the execution of a given Python code snippet.
        
        Args:
            code (str): Python code to profile
            globals_dict (dict, optional): Global variables context
            locals_dict (dict, optional): Local variables context
        
        Returns:
            Dict containing detailed performance metrics
        """
        # Safety checks
        if not code or len(code.strip()) == 0:
            raise ValueError("Empty code snippet provided")
        
        # Initialize tracking
        tracemalloc.start()
        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=None)
        
        # Profiling setup
        profiler = cProfile.Profile()
        
        try:
            # Execute code with profiling
            profiler.enable()
            exec(code, globals_dict or {}, locals_dict or {})
            profiler.disable()
            
            # Collect metrics
            end_time = time.time()
            execution_time = end_time - start_time
            current, peak = tracemalloc.get_traced_memory()
            peak_memory_mb = peak / (1024 * 1024)
            end_cpu = psutil.cpu_percent(interval=None)
            
            # Performance validation
            if execution_time > self.max_execution_time:
                raise TimeoutError(f"Execution exceeded {self.max_execution_time} seconds")
            
            if peak_memory_mb > self.max_memory_mb:
                raise MemoryError(f"Peak memory usage {peak_memory_mb:.2f} MB exceeded limit")
            
            # Capture profiler statistics
            stats_stream = StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream).sort_stats('cumulative')
            stats.print_stats(10)  # Top 10 time-consuming functions
            
            return {
                "execution_time": round(execution_time, 4),
                "peak_memory_mb": round(peak_memory_mb, 2),
                "cpu_usage": round((end_cpu + start_cpu) / 2, 2),
                "profile_stats": stats_stream.getvalue()
            }
        
        except Exception as e:
            raise RuntimeError(f"Code execution failed: {str(e)}")
        
        finally:
            tracemalloc.stop()
    
    def profile_code_files(
        self,
        directory_path: str,
        file_pattern: str = "*.py",
        max_workers: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Profile a list of Python code files."""
        try:
            directory = Path(directory_path)
            if not directory.exists():
                raise RuntimeError(f"Directory {directory_path} does not exist")
            
            files = list(directory.rglob(file_pattern))
            if not files:
                self.logger.warning(f"No files matching pattern {file_pattern} found in {directory_path}")
                return []
            
            # Start timing
            start_time = time.time()
            
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self._profile_file, file) for file in files]
                for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Profiling files"):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Thread execution error: {e}")
            
            # Calculate total time
            total_time = time.time() - start_time
            self.logger.info(f"Total profiling time: {total_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Error profiling code files: {e}")
        
    def _profile_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Profile a single Python code file by executing it and capturing performance metrics.
        
        Args:
            file_path (Path): Path to the Python file to profile
        
        Returns:
            Dict containing performance metrics or error information
        """
        try:
            # Read the file content
            with file_path.open('r', encoding='utf-8') as f:
                code = f.read()
            
            # Create a new module namespace
            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            
            # Add the file's directory to sys.path temporarily
            file_dir = str(file_path.parent.absolute())
            sys.path.insert(0, file_dir)
            
            try:
                # Create globals dict with module's namespace
                globals_dict = module.__dict__.copy()
                
                # Add builtins and __file__ to globals
                globals_dict['__file__'] = str(file_path)
                globals_dict['__name__'] = module_name
                globals_dict['__package__'] = None

                # Extract and import all necessary modules
                import_statements = self._extract_imports(code)

                self.logger.info(f"Import statements: \n{import_statements}\n")

                for statement in import_statements:
                    try:
                        exec(statement, globals_dict)
                    except Exception as e:
                        self.logger.error(f"Could not import module: {e}")
                
                # Profile the code execution with the proper context
                metrics = self.profile_code(code, globals_dict=globals_dict)
                
                # Include file path in the metrics
                metrics['file_path'] = str(file_path)
                metrics['status'] = 'Executed successfully'
                
                return metrics
                
            finally:
                # Remove the temporarily added path
                sys.path.pop(0)
            
        except Exception as e:
            self.logger.error(f"Error profiling file {file_path}: {e}")
            return {
                'file_path': str(file_path),
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        
    def _extract_imports(self, code: str) -> List[str]:
        """
        Extract all import statements from the given code.

        Args:
            code (str): Python code to extract imports from
        
        Returns:
            List[str]: List of import statements
        """
        try:
            import_statements = []
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Handle regular imports (import x, import x.y)
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_statements.append(f"import {alias.name}")
                
                # Handle from imports (from x import y)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    names = [alias.name for alias in node.names]
                    if node.level > 0:  # Handle relative imports
                        module = '.' * node.level + module
                    import_statements.append(
                        f"from {module} import {', '.join(names)}"
                    )

            # Remove duplicates while preserving order
            seen = set()
            unique_imports = []
            for stmt in import_statements:
                if stmt not in seen:
                    seen.add(stmt)
                    unique_imports.append(stmt)
                    
            return unique_imports
            
        except SyntaxError as e:
            self.logger.error(f"Syntax error while parsing imports: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error extracting imports: {e}")
            return []


    def get_optimization_recommendations(self, metrics: list[Dict[str, Any]]) -> list[str]:
        """
        Generate AI-powered optimization suggestions based on performance metrics.
        
        Args:
            metrics (dict): Performance metrics from profile_code
        
        Returns:
            list[str]: AI-generated optimization recommendations
        """
        try:
            recommendations = []

            for metric in metrics:
                if 'error' in metric:
                    recommendations.append(f"Error profiling {metric['file_path']}: {metric['error']}")
                    continue

                chain = self.optimization_prompt | self.llm | StrOutputParser()

                recommendations.append(chain.invoke({
                    "execution_time": metric["execution_time"],
                    "peak_memory": metric["peak_memory_mb"],
                    "cpu_usage": metric["cpu_usage"]
                }))
                
            return recommendations
        
        except Exception as e:
            self.logger.error(f"Error generating optimization recommendations: {e}")
            return []
    
    def visualize_bottlenecks(self, metrics: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        Create a bottleneck visualization using NetworkX.
        
        Args:
            metrics (list[dict]): Performance metrics
        
        Returns:
            nx.DiGraph: Graph representing performance bottlenecks
        """
        G = nx.DiGraph()
        
        # Add performance metric nodes
        G.add_node("Execution Time", weight=metrics["execution_time"])
        G.add_node("Memory Usage", weight=metrics["peak_memory_mb"])
        G.add_node("CPU Usage", weight=metrics["cpu_usage"])
        
        # Add edges based on relative performance impact
        if metrics["execution_time"] > 5:
            G.add_edge("Execution Time", "Performance Critical")
        
        if metrics["peak_memory_mb"] > 50:
            G.add_edge("Memory Usage", "Memory Intensive")
        
        if metrics["cpu_usage"] > 70:
            G.add_edge("CPU Usage", "High Computational Load")
        
        return G
    
    def plot_bottleneck_graph(self, graph: nx.DiGraph) -> None:
        """
        Render and save a bottleneck visualization graph.
        
        Args:
            graph (nx.DiGraph): Bottleneck graph to visualize
        """
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_color='lightblue', 
                node_size=2000, font_size=10, font_weight='bold')
        
        plt.title("Performance Bottleneck Analysis")
        plt.tight_layout()
        plt.savefig('bottleneck_graph.png')
        plt.close()

def main():
    """
    Example usage and demonstration of the RealTimeProfiler.
    """
    code_assistant = CodeAssistant()
    profiler = RealTimeProfiler(code_assistant=code_assistant)
    
    try:
        # Profile the code
        file_metrics = profiler.profile_code_files("./test_profiler")

        # Get AI optimization recommendations
        recommendations = profiler.get_optimization_recommendations(file_metrics)
        print("Optimization Recommendations:")
        for recommendation in recommendations:
            print("-" * 20)
            print(recommendation)
            print("\n")
        
        # # Visualize bottlenecks
        bottleneck_graph = profiler.visualize_bottlenecks(file_metrics)
        profiler.plot_bottleneck_graph(bottleneck_graph)
        
    except Exception as e:
        print(f"Profiling error: {e}")

if __name__ == "__main__":
    main()