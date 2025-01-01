import streamlit as st
import pandas as pd
import json
from typing import Dict, Any, List
import asyncio
from datetime import datetime


class ProfilerPage:
    """
    Profiler Page for the Streamlit application.
    Provides code profiling, optimization suggestions, and performance visualization.
    """

    def __init__(self):
        """Initialize ProfilerPage with configuration and handler."""

        self.profiler = st.session_state.profiler
        self.config = st.session_state.config
        self._configure_page()

    def _configure_page(self):
        """Configure the page layout and styling."""
        st.markdown(
            """
            <style>
            .metric-card {
                background-color: rgba(28, 131, 225, 0.1);
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem 0;
                border: 1px solid rgba(28, 131, 225, 0.2);
            }
            .recommendation-card {
                background-color: rgba(0, 200, 0, 0.1);
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem 0;
                border: 1px solid rgba(0, 200, 0, 0.2);
            }
            .warning-card {
                background-color: rgba(255, 165, 0, 0.1);
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem 0;
                border: 1px solid rgba(255, 165, 0, 0.2);
            }
            .stButton button {
                width: 100%;
                margin-top: 1rem;
            }
            .code-editor {
                border: 1px solid #ccc;
                border-radius: 4px;
                margin: 1rem 0;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def render(self):
        """Render the main profiler interface."""
        st.title("Code Performance Profiler")
        st.markdown("Analyze and optimize your Python code for better performance.")

        # Sidebar controls in first column
        with st.sidebar:
            self._render_sidebar_controls()

        # Main content area in second column
        if st.session_state.profiler_code:
            st.markdown("### Current Code")
            st.code(st.session_state.profiler_code, language="python")
        
        # Display Results if Available
        if st.session_state.profiler_results:
            self._render_results(st.session_state.profiler_results)

    def _handle_code_input(self, code: str):
        """Callback to handle code input changes."""
        st.session_state.profiler_code = code

    def _handle_file_upload(self, uploaded_file):
        """Callback to handle file uploads."""
        if uploaded_file is not None:
            st.session_state.profiler_code = uploaded_file.getvalue().decode()

    def _handle_example_load(self):
        """Callback to handle loading example code."""
        st.session_state.profiler_code = self._get_example_code()

    def _render_sidebar_controls(self):
        """Render all control elements in the sidebar."""
        st.markdown("---")
        st.markdown("### Code Input")
        
        # Input method selection using radio buttons
        st.radio(
            "Choose input method:",
            ["Enter Code", "Upload File", "Try Example"],
            key="input_method",
            on_change=self._handle_input_method_change
        )

        # Handle different input methods
        if st.session_state.input_method == "Enter Code":
            st.text_area(
                "Python code to profile:",
                value=st.session_state.profiler_code,
                height=200,
                key="code_input",
                help="Enter the Python code you want to analyze",
                on_change=lambda: self._handle_code_input(st.session_state.code_input)
            )
                
        elif st.session_state.input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload Python file",
                type=["py"],
                key="file_uploader",
                on_change=lambda: self._handle_file_upload(st.session_state.file_uploader)
            )
                
        else:  # Try Example
            if st.button("Load Example", on_click=self._handle_example_load):
                pass

        # Profiling Options
        st.markdown("### Profiling Options")
        
        # Use callbacks for checkbox changes
        def update_cpu_profiling():
            self.config.profiler.enable_cpu_profiling = st.session_state.cpu_profiling
            
        def update_memory_profiling():
            self.config.profiler.enable_memory_profiling = st.session_state.memory_profiling
                        
        def update_timeout():
            self.config.execution.max_execution_time = st.session_state.timeout

        st.checkbox(
            "Enable CPU Profiling",
            value=self.config.profiler.enable_cpu_profiling,
            key="cpu_profiling",
            on_change=update_cpu_profiling
        )
        
        st.checkbox(
            "Enable Memory Profiling",
            value=self.config.profiler.enable_memory_profiling,
            key="memory_profiling",
            on_change=update_memory_profiling
        )
                
        st.number_input(
            "Execution Timeout (seconds)",
            value=self.config.execution.max_execution_time,
            min_value=1.0,
            max_value=300.0,
            key="timeout",
            on_change=update_timeout
        )

        # Start Profiling Button
        if st.session_state.profiler_code and st.button("Start Profiling"):
            with st.spinner("Profiling code..."):
                asyncio.run(self._run_profiler(st.session_state.profiler_code))

    def _handle_input_method_change(self):
        """Handle input method changes and clear irrelevant state."""
        # Clear code if switching to example mode
        if st.session_state.input_method == "Try Example":
            st.session_state.profiler_code = ""
        # Clear file uploader if switching away from upload
        elif st.session_state.input_method != "Upload File":
            if "file_uploader" in st.session_state:
                del st.session_state.file_uploader

    async def _run_profiler(self, code: str):
        """Run the profiler on the provided code."""
        try:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run profiling with progress updates
            results = await self.profiler.run_code_with_report(
                code,
                progress_callback=lambda p, msg: self._update_progress(progress_bar, status_text, p, msg)
            )
            
            # Store results
            st.session_state.profiler_results = results
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Show success message
            st.success("Profiling completed successfully!")

        except Exception as e:
            st.error(f"Error during profiling: {str(e)}")
            if hasattr(e, '__traceback__'):
                st.exception(e)

    def _update_progress(self, progress_bar, status_text, percentage: float, message: str):
        """Update the progress bar and status message."""
        progress_bar.progress(percentage)
        status_text.text(message)

    def _render_results(self, results: Dict[str, Any]):
        """Render the profiling results based on the actual output format."""
        st.markdown("### Results")

        if not results.get('success', False):
            st.error(f"Profiling failed: {results.get('error', 'Unknown error')}")
            if 'error_report' in results:
                with st.expander("Error Details"):
                    st.markdown(f"**Error Type:** {results['error_report']['error_type']}")
                    st.markdown(f"**Message:** {results['error_report']['error_message']}")
            return

        # Metrics Overview
        st.markdown("### Performance Metrics")
        
        # Create three columns for key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            exec_time = results.get('metrics', {}).get('execution', {}).get('time', 0)
            st.metric("Execution Time", f"{exec_time:.3f}s")
            
        with col2:
            memory_used = results.get('metrics', {}).get('memory', {}).get('used', 0)
            memory_mb = memory_used / (1024 * 1024)
            st.metric("Peak Memory", f"{memory_mb:.1f} MB")
            
        with col3:
            cpu_percent = results.get('metrics', {}).get('cpu', {}).get('percent', 0)
            st.metric("CPU Usage", f"{cpu_percent:.1f}%")

        # System Information
        with st.expander("System Information", expanded=False):
            system_info = results.get('metrics', {}).get('system_info', {})
            st.markdown(f"""
                - **Python Version:** {system_info.get('python_version', 'N/A')}
                - **Platform:** {system_info.get('platform', 'N/A')}
                - **CPU Cores:** {system_info.get('cpu_cores', 'N/A')}
            """)

        # Function Profiles
        if 'profiled_functions' in results.get('metrics', {}):
            st.markdown("### Function Profiles")
            for func_name, func_data in results['metrics']['profiled_functions'].items():
                with st.expander(f"Function: {func_name}", expanded=False):
                    st.markdown(f"**Duration:** {func_data.get('duration', 0):.3f}s")
                    st.markdown(f"**Memory Delta:** {func_data.get('memory_delta', 0) / (1024*1024):.2f} MB")
                    
                    # Line Profile Data
                    if 'line_profile' in func_data:
                        st.markdown("#### Line-by-Line Profile")
                        profile_data = []
                        for line_num, line_data in func_data['line_profile'].items():
                            profile_data.append({
                                'Line': int(line_num),
                                'Hits': line_data['hits'],
                                'Time (μs)': line_data['time_microseconds'],
                                'Time/Hit (μs)': line_data['time_microseconds'] / line_data['hits'] if line_data['hits'] > 0 else 0
                            })
                        
                        if profile_data:
                            st.dataframe(
                                profile_data,
                                hide_index=True,
                                use_container_width=True
                            )

        # LLM Analysis (if available)
        if 'llm' in results:
            st.markdown("### AI Analysis")
            
            # Narrative Analysis
            if 'narrative' in results['llm']:
                st.markdown("#### Key Findings")
                st.markdown(results['llm']['narrative'])
            
            # Performance Issues
            if 'profile_report' in results['llm']:
                report = results['llm']['profile_report']
                if 'issues' in report:
                    st.markdown("#### Performance Issues")
                    for issue in report['issues']:
                        with st.expander(f"{issue['issue_type']} ({issue['priority_level'].upper()})", expanded=False):
                            st.markdown(f"**Location:** {issue['location']}")
                            if 'solution' in issue:
                                st.markdown("**Original Code:**")
                                st.code(issue['solution']['original'], language="python")
                                st.markdown("**Optimized Code:**")
                                st.code(issue['solution']['optimized'], language="python")
                                st.markdown(f"**Impact:** {issue['solution']['impact_summary']}")
                                if issue['solution'].get('reasoning'):
                                    st.markdown(f"**Reasoning:** {issue['solution']['reasoning']}")

        # Download Options
        st.markdown("### Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON download
            st.download_button(
                "Download JSON Report",
                data=results['llm']['reports']['json'],
                file_name="profiler_report.json",
                mime="application/json"
            )
        
        with col2:
            # Markdown report if available
            if 'llm' in results and isinstance(results['llm'], dict):
                markdown_data = results['llm'].get('reports', {}).get('md', '')
                if markdown_data:
                    st.download_button(
                        "Download Markdown Report",
                        data=markdown_data,
                        file_name="profiler_report.md",
                    )

    def _get_example_code(self) -> str:
            """Return example code for demonstration."""
            return '''def fibonacci(n: int) -> int:
        """
        Calculate the nth Fibonacci number using recursion.
        Warning: This is an intentionally inefficient implementation
        to demonstrate performance profiling.
        """
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)

def calculate_fibonacci_sequence(length: int) -> list:
    """Calculate a sequence of Fibonacci numbers."""
    return [fibonacci(i) for i in range(length)]

def main():
    # Create a list to store results
    results = []

    # Calculate first 30 Fibonacci numbers
    sequence = calculate_fibonacci_sequence(30)

    # Perform some memory-intensive operations
    for i in range(20):
        # Create large temporary lists
        temp_list = sequence * 1000
        results.append(sum(temp_list))

    print(f"Final results: {results}")

if __name__ == "__main__":
    main()'''

    def _format_time(self, seconds: float) -> str:
        """Format time duration in a human-readable format."""
        if seconds < 0.001:
            return f"{seconds * 1000000:.0f}μs"
        elif seconds < 1:
            return f"{seconds * 1000:.1f}ms"
        else:
            return f"{seconds:.2f}s"

    def _format_memory(self, memory_mb: float) -> str:
        """Format memory size in a human-readable format."""
        if memory_mb < 1:
            return f"{memory_mb * 1024:.1f}KB"
        elif memory_mb < 1024:
            return f"{memory_mb:.1f}MB"
        else:
            return f"{memory_mb / 1024:.2f}GB"

    def _get_severity_color(self, severity: str) -> str:
        """Get the appropriate color for a severity level."""
        colors = {
            "critical": "red",
            "high": "orange",
            "medium": "yellow",
            "low": "blue",
            "info": "green"
        }
        return colors.get(severity.lower(), "gray")

    def _create_line_profiling_table(self, line_stats: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create a DataFrame for line-by-line profiling results."""
        if not line_stats:
            return pd.DataFrame()
            
        df = pd.DataFrame(line_stats)
        df['time_percentage'] = df['time_ns'] / df['time_ns'].sum() * 100
        df['memory_percentage'] = df['memory_bytes'] / df['memory_bytes'].sum() * 100
        
        # Format columns
        df['time'] = df['time_ns'].apply(lambda x: self._format_time(x / 1e9))
        df['memory'] = df['memory_bytes'].apply(lambda x: self._format_memory(x / (1024 * 1024)))
        
        return df[['line_number', 'line_content', 'time', 'time_percentage', 
                'memory', 'memory_percentage', 'hits']]

    def _handle_profiling_error(self, error: Exception):
        """Handle and display profiling errors in a user-friendly way."""
        error_type = type(error).__name__
        error_message = str(error)
        
        st.error(f"Profiling Error: {error_type}")
        
        with st.expander("Error Details"):
            st.markdown(f"**Error Type:** {error_type}")
            st.markdown(f"**Error Message:** {error_message}")
            if hasattr(error, '__traceback__'):
                st.markdown("**Traceback:**")
                st.code(error.__traceback__)
            
        st.markdown("""
            **Possible Solutions:**
            1. Check if your code has syntax errors
            2. Ensure all required modules are imported
            3. Verify the code doesn't exceed the execution timeout
            4. Make sure the code doesn't use restricted operations
        """)

    def _cache_results(self, results: Dict[str, Any]):
        """Cache profiling results with timestamp."""
        results['timestamp'] = datetime.now().isoformat()
        results['code_hash'] = hash(st.session_state.profiler_code)
        
        # Store only the last 5 results
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        st.session_state.history.append(results)
        if len(st.session_state.history) > 5:
            st.session_state.history.pop(0)

    def _compare_with_previous(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current results with previous run if available."""
        if not st.session_state.history or len(st.session_state.history) < 2:
            return {}
            
        previous = st.session_state.history[-2]
        
        comparison = {
            'execution_time_change': (
                (current_results['metrics']['execution_time'] - 
                previous['metrics']['execution_time']) / 
                previous['metrics']['execution_time'] * 100
            ),
            'memory_usage_change': (
                (current_results['metrics']['peak_memory_mb'] - 
                previous['metrics']['peak_memory_mb']) / 
                previous['metrics']['peak_memory_mb'] * 100
            ),
            'cpu_usage_change': (
                (current_results['metrics']['cpu_percent'] - 
                previous['metrics']['cpu_percent']) / 
                previous['metrics']['cpu_percent'] * 100
            )
        }
        
        return comparison
