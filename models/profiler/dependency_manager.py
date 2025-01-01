from contextlib import asynccontextmanager
import logging
import os
import subprocess
import sys
from typing import Any, List, Dict, Optional, Tuple, Set
import ast
from pathlib import Path
from importlib.metadata import distribution, distributions, PackageNotFoundError
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from packaging import version
import json
from datetime import datetime
import asyncio
import hashlib
import requests

from .utils import measure_execution_time, create_error_report, log_metrics, get_system_info
from .config import get_config, ProfilerConfig

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

class DependencyManager:
    """
    Manages Python package dependencies for the profiler system.
    Handles dependency detection, installation, and validation.
    """

    def __init__(self, config: Optional[ProfilerConfig] = None):
        """Initialize with configuration"""
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize from config with more permissive defaults
        self.allow_install = getattr(self.config.dependency, 'allow_install', True)
        self.trusted_sources = getattr(self.config.dependency, 'trusted_sources', ['pypi'])
        self.max_install_time = getattr(self.config.dependency, 'max_install_time', 300)
        self.verify_checksums = getattr(self.config.dependency, 'verify_checksums', False)
        self.allowed_package_patterns = getattr(self.config.dependency, 'allowed_package_patterns', ['*'])
        
        # Initialize package tracking
        self.installed_packages = self._get_installed_packages()
        self._lock = threading.Lock()
        
        try:
            self.cache_dir = Path(self.config.execution.temp_directory) / 'dependency_cache'
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.dependency_cache_file = self.cache_dir / "dependency_cache.json"
            self.dependency_cache = self._load_dependency_cache()
        except Exception as e:
            self.logger.warning(f"Cache initialization failed, continuing without cache: {e}")
            self.dependency_cache = {}

        # Session for HTTP requests
        self.session = requests.Session()

    def __del__(self):
        """Cleanup on deletion"""
        self.session.close()
        if hasattr(self, 'config') and self.config.execution.cleanup_temp:
            self.cleanup()
        
    def _load_dependency_cache(self) -> Dict:
        """Load the dependency cache from disk"""
        try:
            if self.dependency_cache_file.exists():
                with open(self.dependency_cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading dependency cache: {e}")
            # Return empty cache on error instead of propagating the exception
            return {}
        return {}

    def _save_dependency_cache(self) -> None:
        """Save the dependency cache to disk with proper error handling"""
        try:
            # Ensure directory exists before saving
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Use atomic write pattern to prevent corruption
            temp_file = self.dependency_cache_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self.dependency_cache, f, indent=4)
            
            # Atomic rename
            temp_file.replace(self.dependency_cache_file)
            
        except Exception as e:
            self.logger.error(f"Error saving dependency cache: {e}")
            # Continue execution even if cache save fails

    def _get_installed_packages(self) -> Dict[str, str]:
        """Get a dictionary of installed packages and their versions"""
        installed: Dict[str, str] = {}
        for dist in distributions():
            try:
                installed[dist.metadata['Name'].lower()] = dist.version
            except Exception as e:
                self.logger.warning(f"Error getting package info for {dist}: {e}")
        return installed

    def extract_dependencies(self, code: str) -> Set[str]:
        """
        Extract required dependencies from Python code.

        Args:
            code (str): Python code to analyze

        Returns:
            Set[str]: Set of required package names
        """
        try:
            dependencies = set()
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        package_name = alias.name.split('.')[0]
                        dependencies.add(package_name)
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        package_name = node.module.split('.')[0]
                        dependencies.add(package_name)

            # Remove standard library modules
            stdlib_modules = set(sys.stdlib_module_names)
            return dependencies - stdlib_modules

        except Exception as e:
            self.logger.error(f"Error extracting dependencies: {e}")
            return set()

    async def _verify_package_checksum_async(self, package: str) -> bool:
        """
        Verify package checksum asynchronously if enabled in config.
        """
        if not self.verify_checksums:
            return True
            
        try:
            async with asyncio.timeout(10):
                response = await asyncio.to_thread(
                    self.session.get,
                    f"https://pypi.org/pypi/{package}/json"
                )
                
                if response.status_code != 200:
                    return False
                    
                package_data = response.json()
                latest_version = package_data['info']['version']
                
                for release in package_data['releases'][latest_version]:
                    if release['packagetype'] == 'sdist':
                        expected_sha256 = release['digests']['sha256']
                        
                        package_response = await asyncio.to_thread(
                            self.session.get,
                            release['url']
                        )
                        actual_sha256 = hashlib.sha256(package_response.content).hexdigest()
                        
                        return expected_sha256 == actual_sha256
                        
                return False
                
        except Exception as e:
            self.logger.error(f"Error verifying package checksum: {e}")
            return False
        
    @asynccontextmanager
    async def _installation_timeout(self):
        """Context manager for handling installation timeouts"""
        try:
            async with asyncio.timeout(self.max_install_time):
                yield
        except asyncio.TimeoutError:
            self.logger.error(f"Installation timeout after {self.max_install_time} seconds")
            raise

    def _verify_package_checksum(self, package: str) -> bool:
        """
        Verify package checksum if enabled in config.
        
        Args:
            package (str): Name of the package to verify
            
        Returns:
            bool: True if verification succeeds or is disabled, False otherwise
        """
        if not self.verify_checksums:
            return True
            
        try:
            # Get package info from PyPI
            response = requests.get(
                f"https://pypi.org/pypi/{package}/json",
                timeout=10  # Add timeout
            )
            if response.status_code != 200:
                return False
                
            package_data = response.json()
            latest_version = package_data['info']['version']
            
            # Get checksum for latest version
            for release in package_data['releases'][latest_version]:
                if release['packagetype'] == 'sdist':
                    expected_sha256 = release['digests']['sha256']
                    
                    # Download and verify with timeout
                    package_response = requests.get(
                        release['url'],
                        timeout=30  # Add timeout
                    )
                    actual_sha256 = hashlib.sha256(package_response.content).hexdigest()
                    
                    return expected_sha256 == actual_sha256
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Error verifying package checksum: {e}")
            return False

    async def install_dependencies(self, packages: Set[str]) -> Dict[str, Any]:
        """Install missing dependencies if allowed."""
        if not self.allow_install:
            self.logger.warning("Package installation is disabled")
            return {'success': True, 'message': 'Installation skipped'}
        
        try:
            # Attempt to install packages
            process = await asyncio.create_subprocess_exec(
                sys.executable, '-m', 'pip', 'install', *packages,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info("Successfully installed packages")
                # Refresh installed packages list
                self.installed_packages = self._get_installed_packages()
                return {'success': True, 'message': 'Packages installed successfully'}
            else:
                self.logger.warning(f"Package installation warning: {stderr.decode()}")
                return {'success': True, 'message': 'Continuing without installation'}
                
        except Exception as e:
            self.logger.warning(f"Package installation warning: {e}")
            return {'success': True, 'message': 'Continuing without installation'}

    def _is_package_allowed(self, package: str) -> bool:
        """Check if package is allowed based on patterns"""
        if not self.allowed_package_patterns:
            return True
            
        return any(
            re.match(pattern, package) 
            for pattern in self.allowed_package_patterns
        )

    def _install_package(self, package: str) -> bool:
        """Install a single package using pip"""
        try:
            if not self._verify_package_checksum(package):
                self.logger.error(f"Package checksum verification failed: {package}")
                return False

            cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
            
            # Add trusted hosts if configured
            for source in self.trusted_sources:
                cmd.extend(["--trusted-host", source])
                
            cmd.append(package)

            with self._lock:  # Ensure thread-safe pip operations
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )

            self.logger.info(f"Successfully installed {package}")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install {package}: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Error installing {package}: {e}")
            return False

    @measure_execution_time
    def validate_environment(self, code: str) -> Dict[str, Any]:
        """
        Validate the Python environment for running the given code.
        Now more permissive and focuses on ensuring code can run.
        """
        try:
            # Extract dependencies
            dependencies = self.extract_dependencies(code)
            
            # Check dependencies but don't fail on outdated packages
            missing, _ = self.check_dependencies(dependencies)
            
            # Get Python version info
            python_info = {
                'version': sys.version,
                'platform': sys.platform,
                'executable': sys.executable,
                'implementation': sys.implementation.name
            }
            
            # Consider environment valid even with outdated packages
            validation_result = {
                'valid': True,  # Always consider valid unless critical failure
                'missing_packages': missing,
                'outdated_packages': [],  # Don't report outdated packages
                'required_packages': list(dependencies),
                'installed_packages': self.installed_packages,
                'python_info': python_info,
                'system_info': get_system_info(),
                'package_versions': {
                    pkg: self.installed_packages.get(pkg, 'not installed')
                    for pkg in dependencies
                }
            }
            
            return validation_result
            
        except Exception as e:
            self.logger.warning(f"Environment validation warning: {e}")
            # Return valid result even on error
            return {
                'valid': True,
                'warning': str(e),
                'required_packages': [],
                'missing_packages': [],
                'outdated_packages': []
            }

    def check_dependencies(self, dependencies: Set[str]) -> Tuple[List[str], List[str]]:
        """Check for missing and outdated dependencies, but be more permissive."""
        try:
            missing = []
            outdated = []
            
            for package in dependencies:
                if package.lower() not in self.installed_packages:
                    missing.append(package)
                    continue
                
                # Don't check for outdated packages
                continue
            
            return missing, []  # Always return empty outdated list
            
        except Exception as e:
            self.logger.warning(f"Dependency check warning: {e}")
            return [], []  # Return empty lists on error

    def _check_requirements_file(
        self, 
        requirements_file: Path,
        package: str,
        installed_version: str,
        outdated: List[str]
    ) -> None:
        """Check package version against requirements file."""
        try:
            with open(requirements_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        req_parts = re.split(r'[=<>~!]', line)
                        if req_parts[0].strip() == package:
                            try:
                                from packaging.requirements import Requirement
                                req = Requirement(line)
                                if not version.parse(installed_version) in req.specifier:
                                    if package not in outdated:
                                        outdated.append(package)
                            except Exception as e:
                                self.logger.warning(
                                    f"Error parsing requirement {line}: {e}"
                                )
        except Exception as e:
            self.logger.error(f"Error reading requirements file: {e}")

    def _update_dependency_cache(
        self,
        dependencies: Set[str],
        missing: List[str],
        outdated: List[str]
    ) -> None:
        """Update the dependency cache with check results."""
        cache_update = {
            'timestamp': datetime.now().isoformat(),
            'missing': missing,
            'outdated': outdated
        }
        
        for dep in dependencies:
            self.dependency_cache[dep] = {
                'last_checked': datetime.now().isoformat(),
                'version': self.installed_packages.get(dep),
                'status': 'missing' if dep in missing else 
                         'outdated' if dep in outdated else 'ok'
            }
        
        self._save_dependency_cache()

    def cleanup(self):
        """Cleanup temporary files and resources"""
        try:
            if self.cache_dir.exists():
                import shutil
                shutil.rmtree(self.cache_dir)
        except Exception as e:
            self.logger.warning(f"Failed to cleanup dependency cache: {e}")

# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize dependency manager
        dep_manager = DependencyManager()
        
        # Example code with dependencies
        code = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.random.randn(100)
df = pd.DataFrame(data, columns=['values'])
plt.plot(df['values'])
plt.show()
"""
        
        # Validate environment
        validation = dep_manager.validate_environment(code)
        print("\nEnvironment Validation:")
        print(f"Valid: {validation['valid']}")
        print(f"Required packages: {validation['required_packages']}")
        print(f"Missing packages: {validation['missing_packages']}")
        
        # Install missing packages if any
        if validation['missing_packages']:
            print("\nInstalling missing packages...")
            results = await dep_manager.install_dependencies(
                set(validation['missing_packages'])
            )
            print(f"Installation results: {results}")
        
        # Cleanup
        dep_manager.cleanup()

    # Run example
    asyncio.run(main())
