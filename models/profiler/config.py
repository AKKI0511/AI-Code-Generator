import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
import json
import yaml

@dataclass
class ExecutionConfig:
    """Configuration for code execution"""
    max_execution_time: float = 30.0  # seconds
    max_memory_mb: int = 512  # MB
    timeout_enabled: bool = True
    allowed_modules: Set[str] = field(default_factory=lambda: set())
    restricted_builtins: Set[str] = field(default_factory=lambda: set())
    sandbox_enabled: bool = True
    max_cpu_percent: float = 90.0
    max_thread_count: int = 10
    enable_async: bool = True
    temp_directory: str = 'temp'
    cleanup_temp: bool = True

@dataclass
class ProfilerConfig:
    """Configuration for performance profiling"""
    sampling_interval: float = 0.1  # seconds
    stack_depth: int = 25  # for tracemalloc
    enable_cpu_profiling: bool = True
    enable_memory_profiling: bool = True
    enable_io_profiling: bool = True
    trace_async_operations: bool = True
    function_call_tracking: bool = True
    loop_detection: bool = True
    metrics_history_size: int = 1000  # number of snapshots to keep
    memory_growth_tracking: bool = True
    hotspot_detection_threshold: float = 0.1  # 10% CPU usage threshold
    max_stack_frames: int = 50  # for recursive call detection
    min_loop_iterations: int = 10  # minimum iterations to track a loop
    thread_tracking: bool = True
    trace_malloc: bool = True  # for memory allocation tracking
    io_operation_tracking: bool = True
    snapshot_interval: float = 0.5  # seconds between memory snapshots

@dataclass
class DependencyConfig:
    """Configuration for dependency management"""
    allow_install: bool = False
    trusted_sources: List[str] = field(default_factory=lambda: ['pypi'])
    max_install_time: int = 300  # seconds
    verify_checksums: bool = True
    allowed_package_patterns: List[str] = field(default_factory=list)

@dataclass
class ReportConfig:
    """Configuration for report generation"""
    report_formats: List[str] = field(default_factory=lambda: ['md', 'json'])
    include_system_info: bool = True
    include_recommendations: bool = True
    max_bottlenecks: int = 5
    max_suggestions: int = 3
    detailed_memory_analysis: bool = True
    save_reports: bool = True
    report_output_dir: str = 'reports'
    template_dir: str = 'templates'
    llm_config: Dict[str, Any] = field(default_factory=lambda: {
        'model': 'llama-3.1-8b-instant',
        'temperature': 0.7,
        'max_tokens': 4096
    })
    metrics_retention_days: int = 30
    report_format_version: str = "1.0"
    include_code_snippets: bool = True
    include_performance_graphs: bool = True

@dataclass
class Config:
    """Main configuration class"""
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    dependency: DependencyConfig = field(default_factory=DependencyConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    env_vars: Dict[str, str] = field(default_factory=dict)
    log_level: str = 'INFO'
    log_file: Optional[str] = None

    def setup_logging(self) -> None:
        """Configure logging based on settings"""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        
        handlers = [logging.StreamHandler()]
        if self.log_file:
            handlers.append(logging.FileHandler(self.log_file))
        
        logging.basicConfig(
            format=log_format,
            level=log_level,
            handlers=handlers
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'execution': {
                'max_execution_time': self.execution.max_execution_time,
                'max_memory_mb': self.execution.max_memory_mb,
                'timeout_enabled': self.execution.timeout_enabled,
                'allowed_modules': list(self.execution.allowed_modules),
                'restricted_builtins': list(self.execution.restricted_builtins),
                'sandbox_enabled': self.execution.sandbox_enabled,
                'max_cpu_percent': self.execution.max_cpu_percent,
                'max_thread_count': self.execution.max_thread_count,
                'enable_async': self.execution.enable_async,
                'temp_directory': self.execution.temp_directory,
                'cleanup_temp': self.execution.cleanup_temp
            },
            'profiler': {
                'sampling_interval': self.profiler.sampling_interval,
                'stack_depth': self.profiler.stack_depth,
                'enable_cpu_profiling': self.profiler.enable_cpu_profiling,
                'enable_memory_profiling': self.profiler.enable_memory_profiling,
                'enable_io_profiling': self.profiler.enable_io_profiling,
                'trace_async_operations': self.profiler.trace_async_operations,
                'function_call_tracking': self.profiler.function_call_tracking,
                'loop_detection': self.profiler.loop_detection,
                'metrics_history_size': self.profiler.metrics_history_size,
                'memory_growth_tracking': self.profiler.memory_growth_tracking,
                'hotspot_detection_threshold': self.profiler.hotspot_detection_threshold,
                'max_stack_frames': self.profiler.max_stack_frames,
                'min_loop_iterations': self.profiler.min_loop_iterations,
                'thread_tracking': self.profiler.thread_tracking,
                'trace_malloc': self.profiler.trace_malloc,
                'io_operation_tracking': self.profiler.io_operation_tracking,
                'snapshot_interval': self.profiler.snapshot_interval
            },
            'dependency': {
                'allow_install': self.dependency.allow_install,
                'trusted_sources': self.dependency.trusted_sources,
                'max_install_time': self.dependency.max_install_time,
                'verify_checksums': self.dependency.verify_checksums,
                'allowed_package_patterns': self.dependency.allowed_package_patterns
            },
            'report': {
                'report_formats': self.report.report_formats,
                'include_system_info': self.report.include_system_info,
                'include_recommendations': self.report.include_recommendations,
                'max_bottlenecks': self.report.max_bottlenecks,
                'max_suggestions': self.report.max_suggestions,
                'detailed_memory_analysis': self.report.detailed_memory_analysis,
                'save_reports': self.report.save_reports,
                'report_output_dir': self.report.report_output_dir,
                'template_dir': self.report.template_dir,
                'llm_config': self.report.llm_config,
                'metrics_retention_days': self.report.metrics_retention_days,
                'report_format_version': self.report.report_format_version,
                'include_code_snippets': self.report.include_code_snippets,
                'include_performance_graphs': self.report.include_performance_graphs
            },
            'env_vars': self.env_vars,
            'log_level': self.log_level,
            'log_file': self.log_file
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary"""
        execution_config = ExecutionConfig(**config_dict.get('execution', {}))
        profiler_config = ProfilerConfig(**config_dict.get('profiler', {}))
        dependency_config = DependencyConfig(**config_dict.get('dependency', {}))
        report_config = ReportConfig(**config_dict.get('report', {}))
        
        return cls(
            execution=execution_config,
            profiler=profiler_config,
            dependency=dependency_config,
            report=report_config,
            env_vars=config_dict.get('env_vars', {}),
            log_level=config_dict.get('log_level', 'INFO'),
            log_file=config_dict.get('log_file')
        )

def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from file or use defaults"""
    if not config_path:
        return Config()
    
    try:
        with open(config_path) as f:
            if config_path.suffix == '.json':
                config_dict = json.load(f)
            elif config_path.suffix in {'.yml', '.yaml'}:
                config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return Config.from_dict(config_dict)
        
    except Exception as e:
        logging.error(f"Error loading config from {config_path}: {e}")
        return Config()

def get_config(config_path: Optional[Path] = None) -> Config:
    """Get configuration singleton"""
    if not hasattr(get_config, '_config'):
        get_config._config = load_config(config_path)
    return get_config._config 