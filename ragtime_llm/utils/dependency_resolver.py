"""
Automatic dependency resolver for test and runtime environments.
Handles dynamic dependency installation and resolution using uv.
"""

import importlib
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

import rich
from rich.console import Console

console = Console()

class DependencyResolver:
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the dependency resolver.
        
        Args:
            project_root: Root directory of the project. If None, will try to find it.
        """
        self.project_root = project_root or self._find_project_root()
        self.requirements_file = self.project_root / "requirements.txt"
        self.requirements_in_file = self.project_root / "requirements.in"
        self.installed_packages: Set[str] = set()
        self._load_installed_packages()

    def _find_project_root(self) -> Path:
        """Find the project root by looking for pyproject.toml or setup.py."""
        current = Path.cwd()
        while current != current.parent:
            if (current / "pyproject.toml").exists() or (current / "setup.py").exists():
                return current
            current = current.parent
        return Path.cwd()

    def _load_installed_packages(self) -> None:
        """Load the list of currently installed packages."""
        try:
            result = subprocess.run(
                ["uv", "pip", "list", "--format=freeze"],
                capture_output=True,
                text=True,
                check=True
            )
            self.installed_packages = {
                line.split("==")[0].lower()
                for line in result.stdout.splitlines()
                if line and not line.startswith("-")
            }
        except subprocess.CalledProcessError:
            console.print("[yellow]Warning: Could not load installed packages list[/yellow]")
            self.installed_packages = set()

    def _parse_import_error(self, error_msg: str) -> Optional[str]:
        """Parse an ImportError message to extract the missing package name."""
        patterns = [
            r"No module named '([\w\._-]+)'",
            r"cannot import name '(.+)' from '([\w\._-]+)'",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_msg)
            if match:
                # Get the base package name (first component)
                package = match.group(1).split('.')[0]
                # Handle common package name variations
                package_map = {
                    'cv2': 'opencv-python',
                    'PIL': 'pillow',
                }
                return package_map.get(package, package)
        return None

    def _get_package_version_from_requirements(self, package: str) -> Optional[str]:
        """Get the required version of a package from requirements files."""
        for req_file in [self.requirements_file, self.requirements_in_file]:
            if not req_file.exists():
                continue
            with open(req_file) as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split('>=')
                        if parts[0].strip() == package:
                            return parts[1] if len(parts) > 1 else None
        return None

    def install_package(self, package: str) -> bool:
        """Install a package using uv."""
        try:
            version = self._get_package_version_from_requirements(package)
            package_spec = f"{package}>={version}" if version else package
            
            console.print(f"[cyan]Installing {package_spec}...[/cyan]")
            subprocess.run(
                ["uv", "pip", "install", package_spec],
                check=True,
                capture_output=True,
                text=True
            )
            self.installed_packages.add(package.lower())
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Failed to install {package}: {e.stderr}[/red]")
            return False

    def resolve_import_errors(self, error_output: str) -> List[str]:
        """Resolve import errors from test output."""
        resolved_packages = []
        
        # Extract all import errors
        for line in error_output.splitlines():
            if "ModuleNotFoundError" in line or "ImportError" in line:
                package = self._parse_import_error(line)
                if package and package.lower() not in self.installed_packages:
                    if self.install_package(package):
                        resolved_packages.append(package)
        
        return resolved_packages

    def run_with_auto_resolve(self, command: List[str], max_iterations: int = 3) -> Tuple[int, str]:
        """Run a command with automatic dependency resolution.
        
        Args:
            command: Command to run as a list of strings
            max_iterations: Maximum number of resolution iterations
            
        Returns:
            Tuple of (exit_code, output)
        """
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            
            # Run the command
            console.print(f"\n[cyan]Running iteration {iteration}...[/cyan]")
            process = subprocess.run(command, capture_output=True, text=True)
            
            # If successful or no import errors, return
            if process.returncode == 0 or "ModuleNotFoundError" not in process.stderr:
                return process.returncode, process.stdout + process.stderr
            
            # Try to resolve dependencies
            resolved = self.resolve_import_errors(process.stderr)
            if not resolved:
                console.print("[yellow]No more dependencies to resolve[/yellow]")
                return process.returncode, process.stdout + process.stderr
            
            console.print(f"[green]Resolved dependencies: {', '.join(resolved)}[/green]")
            
            # Reload modules that might have been partially loaded
            for package in resolved:
                try:
                    if package in sys.modules:
                        importlib.reload(sys.modules[package])
                except:
                    pass
        
        console.print(f"[yellow]Reached maximum iterations ({max_iterations})[/yellow]")
        return process.returncode, process.stdout + process.stderr

def auto_resolve_dependencies(command: List[str], max_iterations: int = 3, return_output: bool = False) -> int | Tuple[int, str]:
    """Main entry point for automatic dependency resolution.
    
    Args:
        command: Command to run as a list of strings
        max_iterations: Maximum number of resolution iterations
        return_output: Whether to return the command output along with exit code
        
    Returns:
        Exit code from the command, or tuple of (exit_code, output) if return_output is True
    """
    resolver = DependencyResolver()
    exit_code, output = resolver.run_with_auto_resolve(command, max_iterations)
    
    # Print output
    print(output)
    
    return (exit_code, output) if return_output else exit_code 