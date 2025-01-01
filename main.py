# File: main.py
import streamlit as st
from app import CodeAssistantApp
from models.code_assistant import CodeAssistant
from models.profiler.config import Config, ExecutionConfig, ProfilerConfig, ReportConfig
from models.profiler.profiler_handler import ProfileHandler

def main():
    config = Config(
        execution=ExecutionConfig(
            max_execution_time=30.0,
            max_memory_mb=1024,
            sandbox_enabled=True,
            cleanup_temp=True,
        ),
        profiler=ProfilerConfig(
            enable_cpu_profiling=True,
            enable_memory_profiling=True,
        ),
        report=ReportConfig(
            include_recommendations=True,
            detailed_memory_analysis=True,
            include_code_snippets=True,
            include_performance_graphs=True,
        ),
    )
    # Initialize the CodeAssistant if not already in session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = CodeAssistant()

    if 'config' not in st.session_state:
        st.session_state.config = config

    if 'profiler' not in st.session_state:
        st.session_state.profiler = ProfileHandler(config)

    # Create and run the application
    app = CodeAssistantApp()
    app.run()

if __name__ == "__main__":
    main()