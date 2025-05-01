#!/usr/bin/env python3
"""
Script to fork and consolidate dependencies into the vendor directory.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pkg_resources
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

class DependencyForker:
    """Handles forking and consolidating dependencies."""
    
    def __init__(self, vendor_dir: Path):
        """Initialize the dependency forker.
        
        Args:
            vendor_dir: Path to the vendor directory
        """
        self.vendor_dir = vendor_dir
        self.dependency_map = {
            'ml': [
                'huggingface-hub',
                'transformers',
                'sentence-transformers',
                'faiss-cpu',
                'accelerate',
                'bitsandbytes',
                'safetensors',
                'einops'
            ],
            'audio': [
                'moviepy',
                'pydub',
                'librosa',
                'soundfile'
            ],
            'web': [
                'fastapi',
                'uvicorn',
                'requests',
                'python-dotenv',
                'pytube',
                'yt-dlp'
            ],
            'utils': [
                'rich',
                'tqdm',
                'python-magic',
                'psutil',
                'opencv-python',
                'matplotlib',
                'seaborn'
            ],
            'distributed': [
                'ray'
            ],
            'storage': [
                'ipfshttpclient',
                'aiofiles',
                'cachetools'
            ]
        }

    def get_package_path(self, package_name: str) -> Optional[Path]:
        """Get the path to a package's source code.
        
        Args:
            package_name: Name of the package
            
        Returns:
            Path to the package's source code or None if not found
        """
        try:
            return Path(pkg_resources.get_distribution(package_name).location)
        except pkg_resources.DistributionNotFound:
            return None

    def fork_package(self, package_name: str, category: str) -> bool:
        """Fork a package into the vendor directory.
        
        Args:
            package_name: Name of the package to fork
            category: Category directory to fork into
            
        Returns:
            True if successful, False otherwise
        """
        source_path = self.get_package_path(package_name)
        if not source_path:
            console.print(f"[red]Package {package_name} not found[/red]")
            return False
            
        target_dir = self.vendor_dir / category / package_name.replace('-', '_')
        
        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy package files
        try:
            if source_path.is_dir():
                shutil.copytree(source_path, target_dir, dirs_exist_ok=True)
            else:
                shutil.copy2(source_path, target_dir)
                
            # Create __init__.py if it doesn't exist
            init_file = target_dir / '__init__.py'
            if not init_file.exists():
                init_file.write_text(f'"""Forked version of {package_name}."""\n')
                
            return True
        except Exception as e:
            console.print(f"[red]Error forking {package_name}: {str(e)}[/red]")
            return False

    def fork_all_dependencies(self) -> None:
        """Fork all dependencies into the vendor directory."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            for category, packages in self.dependency_map.items():
                task = progress.add_task(f"Forking {category} packages...", total=len(packages))
                for package in packages:
                    if self.fork_package(package, category):
                        progress.advance(task)
                    else:
                        console.print(f"[yellow]Skipping {package} due to errors[/yellow]")

    def create_init_files(self) -> None:
        """Create __init__.py files in all vendor directories."""
        for category in self.dependency_map.keys():
            init_file = self.vendor_dir / category / '__init__.py'
            if not init_file.exists():
                init_file.write_text(f'"""Vendor directory for {category} packages."""\n')

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    vendor_dir = project_root / 'ragtime_llm' / 'vendor'
    
    # Create forker
    forker = DependencyForker(vendor_dir)
    
    # Fork dependencies
    console.print("[bold]Starting dependency forking process...[/bold]")
    forker.fork_all_dependencies()
    
    # Create init files
    console.print("[bold]Creating vendor directory structure...[/bold]")
    forker.create_init_files()
    
    console.print("[bold green]Dependency forking complete![/bold green]")
    console.print("\nNext steps:")
    console.print("1. Review the forked dependencies in ragtime_llm/vendor/")
    console.print("2. Update imports in your code to use the forked versions")
    console.print("3. Test the application to ensure everything works as expected")

if __name__ == "__main__":
    main() 