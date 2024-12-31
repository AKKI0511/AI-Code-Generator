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
        
        # Initialize from config
        self.allow_install = self.config.dependency.allow_install
        self.trusted_sources = self.config.dependency.trusted_sources
        self.max_install_time = self.config.dependency.max_install_time
        self.verify_checksums = self.config.dependency.verify_checksums
        self.allowed_package_patterns = self.config.dependency.allowed_package_patterns
        
        # Create cache directory
        self.cache_dir = Path(self.config.execution.temp_directory) / 'dependency_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize package tracking
        self.installed_packages = self._get_installed_packages()
        self._lock = threading.Lock()
        
        # Load dependency cache
        self.dependency_cache_file = self.cache_dir / "dependency_cache.json"
        self.dependency_cache = self._load_dependency_cache()

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
        return {}

    def _save_dependency_cache(self) -> None:
        """Save the dependency cache to disk"""
        try:
            with open(self.dependency_cache_file, 'w') as f:
                json.dump(self.dependency_cache, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving dependency cache: {e}")

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

    async def install_dependencies(self, dependencies: Set[str]) -> Dict[str, bool]:
        """
        Install the given dependencies if allowed by configuration.
        
        Args:
            dependencies (Set[str]): Set of dependencies to install
            
        Returns:
            Dict[str, bool]: Dictionary with package names as keys and installation success as values
        """
        if not self.allow_install:
            self.logger.warning("Dependency installation is disabled in config")
            return {dep: False for dep in dependencies}
            
        if 'pypi' not in self.trusted_sources:
            self.logger.warning("PyPI is not in trusted sources")
            return {dep: False for dep in dependencies}

        # Filter out any packages that aren't allowed
        allowed_deps = {
            dep for dep in dependencies 
            if self._is_package_allowed(dep)
        }
        
        if len(allowed_deps) < len(dependencies):
            self.logger.warning(
                f"Skipping installation of unauthorized packages: "
                f"{', '.join(dependencies - allowed_deps)}"
            )

        results: Dict[str, bool] = {}
        max_workers = min(len(allowed_deps), os.cpu_count() or 1)
        
        try:
            async with self._installation_timeout:
                # Verify checksums first
                if self.verify_checksums:
                    verification_tasks = [
                        self._verify_package_checksum_async(pkg)
                        for pkg in allowed_deps
                    ]
                    verifications = await asyncio.gather(*verification_tasks)
                    allowed_deps = {
                        pkg for pkg, verified in zip(allowed_deps, verifications)
                        if verified
                    }

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_pkg = {
                        executor.submit(self._install_package, pkg): pkg 
                        for pkg in allowed_deps
                    }
                    
                    for future in as_completed(future_to_pkg):
                        pkg = future_to_pkg[future]
                        try:
                            success = future.result()
                            results[pkg] = success
                        except Exception as e:
                            self.logger.error(f"Error installing {pkg}: {e}")
                            results[pkg] = False

        except asyncio.TimeoutError:
            self.logger.error(f"Installation timeout after {self.max_install_time} seconds")
            return {dep: False for dep in dependencies}

        # Update installed packages cache
        self.installed_packages = self._get_installed_packages()

        # Log installation metrics
        log_metrics({
            'installed_packages': results,
            'success_rate': sum(results.values()) / len(results) if results else 0,
            'skipped_packages': list(dependencies - allowed_deps)
        })
        
        return results

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

        Args:
            code (str): Python code to validate

        Returns:
            Dict containing validation results and environment information
        """
        try:
            # Check cache first
            cache_key = hash(code)
            if cache_key in self.dependency_cache:
                cached_result = self.dependency_cache[cache_key]
                if (datetime.now() - datetime.fromisoformat(cached_result['timestamp'])).days < 1:
                    self.logger.info("Using cached dependency validation result")
                    return cached_result['result']
            
            # Extract dependencies
            dependencies = self.extract_dependencies(code)
            
            # Check dependencies
            missing, outdated = self.check_dependencies(dependencies)
            
            # Get Python version info
            python_info = {
                'version': sys.version,
                'platform': sys.platform,
                'executable': sys.executable,
                'implementation': sys.implementation.name
            }
            
            validation_result = {
                'valid': not (missing or outdated),
                'missing_packages': missing,
                'outdated_packages': outdated,
                'required_packages': list(dependencies),
                'installed_packages': self.installed_packages,
                'python_info': python_info,
                'system_info': get_system_info(),
                'package_versions': {
                    pkg: self.installed_packages.get(pkg, 'not installed')
                    for pkg in dependencies
                }
            }
            
            # Cache the result
            self.dependency_cache[cache_key] = {
                'timestamp': datetime.now().isoformat(),
                'result': validation_result
            }
            self._save_dependency_cache()
            
            return validation_result

        except Exception as e:
            error_report = create_error_report(e, {'code': code})
            self.logger.error(f"Environment validation failed: {error_report}")
            return {
                'valid': False,
                'error': str(e),
                'error_report': error_report
            }

    def check_dependencies(self, dependencies: Set[str]) -> Tuple[List[str], List[str]]:
        """
        Check which dependencies are missing or need updating.

        Args:
            dependencies (Set[str]): Set of required dependencies

        Returns:
            Tuple[List[str], List[str]]: Lists of missing and outdated packages
        """
        try:
            missing: List[str] = []
            outdated: List[str] = []

            for dep in dependencies:
                try:
                    dist = distribution(dep)

                    # If package is installed, check version requirements
                    if self.verify_checksums:
                        response = self.session.get(
                            f"https://pypi.org/pypi/{dep}/json",
                            timeout=10
                        )
                        if response.status_code == 200:
                            pypi_data = response.json()
                            latest_version = version.parse(pypi_data['info']['version'])
                            installed_version = version.parse(dist.version)
                            
                            if installed_version < latest_version:
                                outdated.append(dep)
                
                    # Check against requirements file if it exists
                    requirements_file = Path('requirements.txt')
                    if requirements_file.exists():
                        self._check_requirements_file(
                            requirements_file, dep, dist.version, outdated
                        )

                except PackageNotFoundError:
                    missing.append(dep)
                except Exception as e:
                    self.logger.error(f"Error checking dependency {dep}: {e}")
                    missing.append(dep)

            # Log results
            if missing:
                self.logger.warning(f"Missing dependencies: {', '.join(missing)}")
            if outdated:
                self.logger.info(f"Outdated dependencies: {', '.join(outdated)}")

            self._update_dependency_cache(dependencies, missing, outdated)
            return missing, outdated

        except Exception as e:
            self.logger.error(f"Error checking dependencies: {e}")
            return list(dependencies), []  # Assume all missing on error

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

    def cleanup(self) -> None:
        """Clean up temporary files and caches"""
        try:
            if self.cache_dir.exists():
                import shutil
                shutil.rmtree(self.cache_dir)
            self.dependency_cache = {}
        except Exception as e:
            self.logger.error(f"Error cleaning up dependency manager: {e}")

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
