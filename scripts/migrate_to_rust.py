#!/usr/bin/env python3
"""
Script to help migrate Python components to Rust.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import json
import subprocess
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

class RustMigrationManager:
    """Manages migration of Python components to Rust."""
    
    def __init__(self, project_root: Path):
        """Initialize the migration manager.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = project_root
        self.rust_dir = project_root / "rust"
        self.migration_map = {
            "core": {
                "distributed_processor": {
                    "priority": 1,
                    "dependencies": ["ray", "numpy", "torch"],
                    "description": "Distributed processing system"
                },
                "vector_store": {
                    "priority": 2,
                    "dependencies": ["faiss", "numpy"],
                    "description": "Vector store implementation"
                },
                "debug_engine": {
                    "priority": 3,
                    "dependencies": ["pytest", "git"],
                    "description": "Debug-iteration-merge engine"
                }
            },
            "utils": {
                "resource_manager": {
                    "priority": 1,
                    "dependencies": ["psutil"],
                    "description": "Resource management system"
                },
                "storage_manager": {
                    "priority": 2,
                    "dependencies": ["ipfshttpclient"],
                    "description": "Storage management system"
                }
            }
        }
    
    def setup_rust_project(self) -> bool:
        """Set up Rust project structure."""
        try:
            # Create Rust directory
            self.rust_dir.mkdir(exist_ok=True)
            
            # Initialize Cargo.toml
            cargo_toml = self.rust_dir / "Cargo.toml"
            if not cargo_toml.exists():
                cargo_toml.write_text("""[package]
name = "ragtime_llm"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
rayon = "1.7"
ndarray = "0.15"
faiss = "0.12"
pyo3 = { version = "0.19", features = ["extension-module"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
thiserror = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
""")
            
            # Create module structure
            for category in self.migration_map.keys():
                (self.rust_dir / "src" / category).mkdir(parents=True, exist_ok=True)
                (self.rust_dir / "src" / category / "mod.rs").touch()
            
            # Create lib.rs
            lib_rs = self.rust_dir / "src" / "lib.rs"
            if not lib_rs.exists():
                lib_rs.write_text("""use pyo3::prelude::*;

pub mod core;
pub mod utils;

/// Python module initialization
#[pymodule]
fn ragtime_llm(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<core::distributed_processor::DistributedProcessor>()?;
    m.add_class::<core::vector_store::VectorStore>()?;
    m.add_class::<core::debug_engine::DebugEngine>()?;
    m.add_class::<utils::resource_manager::ResourceManager>()?;
    m.add_class::<utils::storage_manager::StorageManager>()?;
    Ok(())
}
""")
            
            return True
        except Exception as e:
            console.print(f"[red]Error setting up Rust project: {e}[/red]")
            return False
    
    def create_rust_module(self, 
                          category: str,
                          module_name: str,
                          module_info: Dict) -> bool:
        """Create a Rust module for a Python component.
        
        Args:
            category: Module category (core/utils)
            module_name: Name of the module
            module_info: Module information from migration map
        """
        try:
            module_dir = self.rust_dir / "src" / category / module_name
            module_dir.mkdir(exist_ok=True)
            
            # Create mod.rs
            mod_rs = module_dir / "mod.rs"
            if not mod_rs.exists():
                mod_rs.write_text(f"""//! {module_info['description']}

use pyo3::prelude::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

mod implementation;

pub use implementation::*;
""")
            
            # Create implementation.rs
            impl_rs = module_dir / "implementation.rs"
            if not impl_rs.exists():
                impl_rs.write_text(f"""use pyo3::prelude::*;
use anyhow::Result;
use serde::{Serialize, Deserialize};

#[pyclass]
pub struct {module_name.title().replace('_', '')} {{
    // TODO: Add fields
}}

#[pymethods]
impl {module_name.title().replace('_', '')} {{
    #[new]
    pub fn new() -> Self {{
        Self {{
            // TODO: Initialize fields
        }}
    }}
    
    // TODO: Add methods
}}
""")
            
            return True
        except Exception as e:
            console.print(f"[red]Error creating Rust module: {e}[/red]")
            return False
    
    def migrate_component(self, 
                         category: str,
                         module_name: str,
                         module_info: Dict) -> bool:
        """Migrate a Python component to Rust.
        
        Args:
            category: Module category (core/utils)
            module_name: Name of the module
            module_info: Module information from migration map
        """
        try:
            console.print(f"[bold]Migrating {category}.{module_name}...[/bold]")
            
            # Create Rust module
            if not self.create_rust_module(category, module_name, module_info):
                return False
            
            # Create Python bindings
            python_bindings = self.project_root / "ragtime_llm" / category / f"{module_name}_rust.py"
            if not python_bindings.exists():
                python_bindings.write_text(f"""\"\"\"
Rust implementation of {module_name}.
\"\"\"

from ragtime_llm.rust import {module_name.title().replace('_', '')}

class {module_name.title().replace('_', '')}Wrapper:
    \"\"\"Python wrapper for Rust implementation.\"\"\"
    
    def __init__(self):
        self._rust_impl = {module_name.title().replace('_', '')}()
    
    # TODO: Add wrapper methods
""")
            
            return True
        except Exception as e:
            console.print(f"[red]Error migrating component: {e}[/red]")
            return False
    
    def migrate_all(self) -> None:
        """Migrate all components to Rust."""
        # Set up Rust project
        if not self.setup_rust_project():
            return
        
        # Sort components by priority
        components = []
        for category, modules in self.migration_map.items():
            for module_name, module_info in modules.items():
                components.append((category, module_name, module_info))
        
        components.sort(key=lambda x: x[2]["priority"])
        
        # Migrate components
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Migrating components...", total=len(components))
            
            for category, module_name, module_info in components:
                if self.migrate_component(category, module_name, module_info):
                    progress.advance(task)
                else:
                    console.print(f"[yellow]Skipping {category}.{module_name}[/yellow]")
        
        console.print("\n[bold green]Migration complete![/bold green]")
        console.print("\nNext steps:")
        console.print("1. Review the generated Rust code in the rust/ directory")
        console.print("2. Implement the Rust modules")
        console.print("3. Update Python bindings")
        console.print("4. Run tests to verify functionality")

def main():
    """Main entry point for the migration script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate Python components to Rust")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Path to project root directory"
    )
    
    args = parser.parse_args()
    
    # Initialize and run migration manager
    manager = RustMigrationManager(Path(args.project_root))
    manager.migrate_all()

if __name__ == "__main__":
    main() 