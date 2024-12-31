import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
import hashlib
from datetime import datetime
import json
from .utils import create_error_report, safe_delete, format_size, measure_execution_time
from .config import get_config, ProfilerConfig

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

class FileHandler:
    """
    Handles file operations for the profiler system, including:
    - Saving uploaded files
    - Managing temporary files
    - Cleaning up after execution
    - Tracking file metadata
    """

    def __init__(self, config: Optional[ProfilerConfig] = None):
        """
        Initialize the FileHandler with configuration.

        Args:
            config (Optional[ProfilerConfig]): Configuration object for the profiler.
        """
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize from config
        self.temp_dir = self.config.file.temp_dir
        self.max_file_size = self.config.file.max_file_size_mb * 1024 * 1024
        self.allowed_extensions = self.config.file.allowed_extensions
        self.cleanup_enabled = self.config.file.cleanup_enabled
        
        # Create directories
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.temp_dirs: List[Path] = []
        self.file_registry: Dict[str, Dict] = {}
        
        # Load existing file registry if it exists
        self._load_registry()

    def _load_registry(self) -> None:
        """
        Load the file registry from disk if it exists.
        """
        registry_path = self.temp_dir / "file_registry.json"
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    self.file_registry = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading file registry: {e}")
                self.file_registry = {}

    def _save_registry(self) -> None:
        """
        Save the current file registry to disk.
        """
        registry_path = self.temp_dir / "file_registry.json"
        try:
            with open(registry_path, 'w') as f:
                json.dump(self.file_registry, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving file registry: {e}")

    def create_temp_directory(self) -> Path:
        """
        Create a new temporary directory for file operations.

        Returns:
            Path: Path to the created temporary directory.
        """
        temp_dir = Path(tempfile.mkdtemp(dir=str(self.temp_dir)))
        self.temp_dirs.append(temp_dir)
        return temp_dir

    @measure_execution_time
    def save_files_locally(self, 
                         files: List[Union[str, Path, Dict]],
                         temp_dir: Optional[Path] = None) -> List[Path]:
        """
        Save files to a local directory.

        Args:
            files (List[Union[str, Path, Dict]]): List of file paths, Path objects, or dicts with file info.
            temp_dir (Optional[Path]): Optional temporary directory to use.

        Returns:
            List[Path]: List of paths to the saved files.
        """
        if not temp_dir:
            temp_dir = self.create_temp_directory()

        saved_paths = []
        
        for file_item in files:
            try:
                # Handle different input types
                if isinstance(file_item, dict):
                    file_path = file_item.get('path')
                    content = file_item.get('content')
                    if not file_path:
                        continue
                else:
                    file_path = Path(file_item)
                    content = None

                # Generate a unique filename
                file_hash = self._generate_file_hash(str(file_path))
                dest_path = temp_dir / f"{file_hash}_{file_path.name}"

                # Save the file
                if content:
                    with open(dest_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                else:
                    shutil.copy2(file_path, dest_path)

                # Update registry
                self._update_registry(file_path, dest_path)
                saved_paths.append(dest_path)

            except Exception as e:
                self.logger.error(f"Error saving file {file_item}: {e}")
                continue

        return saved_paths

    def _generate_file_hash(self, file_path: str) -> str:
        """
        Generate a unique hash for a file path.

        Args:
            file_path (str): The file path to hash.

        Returns:
            str: The generated hash.
        """
        return hashlib.md5(f"{file_path}_{datetime.now()}".encode()).hexdigest()[:8]

    def _update_registry(self, original_path: Path, saved_path: Path) -> None:
        """
        Update the file registry with new file information.

        Args:
            original_path (Path): The original path of the file.
            saved_path (Path): The path where the file is saved.
        """
        try:
            file_info = {
                'original_path': str(original_path),
                'saved_path': str(saved_path),
                'timestamp': datetime.now().isoformat(),
                'size': format_size(saved_path.stat().st_size),
                'profiling_status': 'pending',
                'last_profiled': None,
                'dependencies': [],  # Will be populated during execution
                'dependency_status': 'unchecked'
            }
            self.file_registry[str(saved_path)] = file_info
            self._save_registry()
        except Exception as e:
            self.logger.error(f"Error updating registry: {e}")

    def update_dependency_info(self, file_path: Path, dependencies: List[str], status: str) -> None:
        """
        Update the dependency information for a file.

        Args:
            file_path (Path): The path to the file.
            dependencies (List[str]): List of dependencies.
            status (str): The status of the dependencies.
        """
        try:
            file_info = self.file_registry.get(str(file_path))
            if file_info:
                file_info['dependencies'] = dependencies
                file_info['dependency_status'] = status
                self._save_registry()
        except Exception as e:
            self.logger.error(f"Error updating dependency info: {e}")

    def update_profiling_status(self, file_path: Path, status: str, metrics: Optional[Dict] = None) -> None:
        """
        Update the profiling status and metrics for a file.
        
        Args:
            file_path (Path): Path to the file.
            status (str): Current profiling status.
            metrics (Optional[Dict]): Optional profiling metrics to store.
        """
        try:
            file_info = self.file_registry.get(str(file_path))
            if file_info:
                file_info['profiling_status'] = status
                file_info['last_profiled'] = datetime.now().isoformat()
                if metrics:
                    file_info['profiling_metrics'] = metrics
                self._save_registry()
        except Exception as e:
            self.logger.error(f"Error updating profiling status: {e}")

    def get_file_info(self, file_path: Union[str, Path]) -> Optional[Dict]:
        """
        Get information about a saved file.

        Args:
            file_path (Union[str, Path]): Path to the file.

        Returns:
            Optional[Dict]: File information if found.
        """
        return self.file_registry.get(str(file_path))

    def cleanup_files(self, paths: Optional[List[Path]] = None) -> Tuple[int, int]:
        """
        Clean up temporary files and directories.

        Args:
            paths (Optional[List[Path]]): Optional list of specific paths to clean up.
                                          If None, cleans up all temporary directories.

        Returns:
            Tuple[int, int]: (number of files deleted, number of directories deleted).
        """
        files_deleted = 0
        dirs_deleted = 0

        try:
            if paths:
                for path in paths:
                    if safe_delete(path):
                        files_deleted += 1
            else:
                for temp_dir in self.temp_dirs:
                    if temp_dir.exists():
                        files_deleted += sum(1 for _ in temp_dir.glob('*'))
                        safe_delete(temp_dir)
                        dirs_deleted += 1
            
            return files_deleted, dirs_deleted
            
        except Exception as e:
            error_report = create_error_report(e, {'paths': [str(p) for p in paths or []]})
            self.logger.error(f"Cleanup failed: {error_report}")
            return files_deleted, dirs_deleted

    def __enter__(self):
        """
        Context manager entry.
        
        Returns:
            FileHandler: The current instance of FileHandler.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit with automatic cleanup.

        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Exception traceback.
        """
        self.cleanup_files()

    def update_report_info(self, file_path: Path, report: Dict[str, Any]) -> None:
        """
        Update file registry with report information.

        Args:
            file_path (Path): Path to the file.
            report (Dict[str, Any]): Report information to update.
        """
        try:
            file_info = self.file_registry.get(str(file_path))
            if file_info:
                file_info['report'] = {
                    'summary': report['json']['summary'],
                    'timestamp': report['json']['metadata']['timestamp'],
                    'performance_analysis': report['json']['performance_analysis']
                }
                self._save_registry()
        except Exception as e:
            self.logger.error(f"Error updating report info: {e}")

    def cleanup(self) -> None:
        """
        Cleanup temporary files based on configuration.
        """
        if self.cleanup_enabled:
            try:
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info("Temporary files cleaned up")
            except Exception as e:
                self.logger.error(f"Cleanup failed: {e}")

    def validate_file(self, file_path: Path) -> bool:
        """
        Validate file before processing.

        Args:
            file_path (Path): Path to the file.

        Returns:
            bool: True if the file is valid, False otherwise.
        """
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            if not file_path.suffix in self.allowed_extensions:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
                
            if file_path.stat().st_size > self.max_file_size:
                raise ValueError(f"File too large: {file_path}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"File validation failed: {e}")
            return False
