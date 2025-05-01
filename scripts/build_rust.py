#!/usr/bin/env python3
"""
Script to help build and install Rust components.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
import json
import subprocess
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

class RustBuildManager:
    """Manages building and installation of Rust components."""
    
    def __init__(self, project_root: Path):
        """Initialize the build manager.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = project_root
        self.rust_dir = project_root / "rust"
        self.build_dir = project_root / "build"
        self.dist_dir = project_root / "dist"
        
        # Create build and dist directories if they don't exist
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.dist_dir.mkdir(parents=True, exist_ok=True)
    
    def build_release(self) -> bool:
        """Build Rust components in release mode."""
        try:
            console.print("[bold]Building Rust components in release mode...[/bold]")
            
            # Change to Rust directory
            os.chdir(self.rust_dir)
            
            # Build release
            result = subprocess.run(
                ["cargo", "build", "--release"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                console.print("[green]Build successful![/green]")
                return True
            else:
                console.print("[red]Build failed:[/red]")
                console.print(result.stdout)
                console.print(result.stderr)
                return False
        except Exception as e:
            console.print(f"[red]Error building components: {e}[/red]")
            return False
    
    def install_python_bindings(self) -> bool:
        """Install Python bindings for Rust components."""
        try:
            console.print("[bold]Installing Python bindings...[/bold]")
            
            # Change to project root
            os.chdir(self.project_root)
            
            # Install in development mode
            result = subprocess.run(
                ["pip", "install", "-e", "."],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                console.print("[green]Python bindings installed successfully![/green]")
                return True
            else:
                console.print("[red]Installation failed:[/red]")
                console.print(result.stdout)
                console.print(result.stderr)
                return False
        except Exception as e:
            console.print(f"[red]Error installing Python bindings: {e}[/red]")
            return False
    
    def create_wheel(self) -> bool:
        """Create a wheel package for distribution."""
        try:
            console.print("[bold]Creating wheel package...[/bold]")
            
            # Change to project root
            os.chdir(self.project_root)
            
            # Build wheel
            result = subprocess.run(
                ["python", "setup.py", "bdist_wheel"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                console.print("[green]Wheel package created successfully![/green]")
                return True
            else:
                console.print("[red]Wheel creation failed:[/red]")
                console.print(result.stdout)
                console.print(result.stderr)
                return False
        except Exception as e:
            console.print(f"[red]Error creating wheel package: {e}[/red]")
            return False
    
    def build_all(self) -> None:
        """Build and install all Rust components."""
        # Build release
        if not self.build_release():
            console.print("\n[bold red]Build failed![/bold red]")
            return
        
        # Install Python bindings
        if not self.install_python_bindings():
            console.print("\n[bold red]Installation failed![/bold red]")
            return
        
        # Create wheel
        if not self.create_wheel():
            console.print("\n[bold red]Wheel creation failed![/bold red]")
            return
        
        console.print("\n[bold green]Build and installation complete![/bold green]")
        console.print("\nNext steps:")
        console.print("1. Test the installed components")
        console.print("2. Verify Python bindings are working")
        console.print("3. Check wheel package in dist/ directory")
        console.print("4. Update documentation if needed")

def main():
    """Main entry point for the build script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build and install Rust components")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Path to project root directory"
    )
    
    args = parser.parse_args()
    
    # Initialize and run build manager
    manager = RustBuildManager(Path(args.project_root))
    manager.build_all()

if __name__ == "__main__":
    main() 