#!/usr/bin/env python3
"""
Script to help test Rust components.
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

class RustTestManager:
    """Manages testing of Rust components."""
    
    def __init__(self, project_root: Path):
        """Initialize the test manager.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = project_root
        self.rust_dir = project_root / "rust"
        self.tests_dir = project_root / "tests" / "rust"
        
        # Create tests directory if it doesn't exist
        self.tests_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize test templates
        self._init_test_templates()
    
    def _init_test_templates(self) -> None:
        """Initialize Rust test templates."""
        templates = {
            "distributed_processor": """#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;
    
    #[test]
    fn test_initialize_workers() {
        let processor = DistributedProcessor::new();
        processor.initialize_workers().unwrap();
        
        let workers = processor.workers.lock().unwrap();
        assert_eq!(workers.len(), num_cpus::get());
        
        for worker in workers.iter() {
            assert_eq!(worker.status, WorkerStatus::Idle);
        }
    }
    
    #[test]
    fn test_process_batch() {
        let processor = DistributedProcessor::new();
        processor.initialize_workers().unwrap();
        
        let items = vec![
            PyObject::from_str("test1").unwrap(),
            PyObject::from_str("test2").unwrap(),
            PyObject::from_str("test3").unwrap(),
        ];
        
        let results = processor.process_batch(items).unwrap();
        assert_eq!(results.len(), 3);
    }
}""",
            
            "vector_store": """#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new_vector_store() {
        let store = VectorStore::new(128).unwrap();
        assert_eq!(store.dimension, 128);
    }
    
    #[test]
    fn test_add_and_search_vectors() {
        let mut store = VectorStore::new(3).unwrap();
        
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        
        store.add_vectors(vectors).unwrap();
        
        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = store.search(query, 2).unwrap();
        
        assert_eq!(indices.len(), 2);
        assert_eq!(distances.len(), 2);
        assert_eq!(indices[0], 0);
    }
}""",
            
            "debug_engine": """#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new_debug_engine() {
        let engine = DebugEngine::new();
        assert_eq!(engine.iteration_count, 0);
        assert!(engine.test_results.is_empty());
    }
    
    #[test]
    fn test_run_test() {
        let mut engine = DebugEngine::new();
        
        let result = engine.run_test("test1", "assert True").unwrap();
        assert!(result.passed);
        assert!(result.error_message.is_none());
        
        assert_eq!(engine.iteration_count, 1);
        assert_eq!(engine.test_results.len(), 1);
    }
}""",
            
            "resource_manager": """#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new_resource_manager() {
        let manager = ResourceManager::new();
        assert!(!manager.monitoring);
    }
    
    #[test]
    fn test_monitoring() {
        let mut manager = ResourceManager::new();
        
        manager.start_monitoring().unwrap();
        assert!(manager.monitoring);
        
        let stats = manager.get_stats().unwrap();
        assert!(stats.cpu_usage >= 0.0);
        assert!(stats.cpu_usage <= 100.0);
        assert!(stats.memory_usage >= 0.0);
        assert!(stats.memory_usage <= 100.0);
        
        manager.stop_monitoring().unwrap();
        assert!(!manager.monitoring);
    }
}""",
            
            "storage_manager": """#[cfg(test)]
mod tests {
    use super::*;
    use tokio::runtime::Runtime;
    
    #[test]
    fn test_new_storage_manager() {
        let manager = StorageManager::new("test_data").unwrap();
        assert_eq!(manager.cache.len(), 0);
    }
    
    #[test]
    fn test_store_and_retrieve() {
        let rt = Runtime::new().unwrap();
        let mut manager = StorageManager::new("test_data").unwrap();
        
        let data = vec![1, 2, 3, 4, 5];
        
        rt.block_on(async {
            manager.store("test1", data.clone()).await.unwrap();
            let retrieved = manager.retrieve("test1").await.unwrap();
            assert_eq!(retrieved, data);
        });
        
        let stats = manager.get_stats().unwrap();
        assert_eq!(stats.cache_size, 1);
    }
}"""
        }
        
        for name, content in templates.items():
            test_path = self.tests_dir / f"{name}_tests.rs"
            if not test_path.exists():
                test_path.write_text(content)
    
    def create_test_file(self, 
                        category: str,
                        module_name: str) -> bool:
        """Create a test file for a Rust component.
        
        Args:
            category: Module category (core/utils)
            module_name: Name of the module
        """
        try:
            console.print(f"[bold]Creating tests for {category}.{module_name}...[/bold]")
            
            # Get template
            template_path = self.tests_dir / f"{module_name}_tests.rs"
            if not template_path.exists():
                console.print(f"[yellow]No test template found for {module_name}[/yellow]")
                return False
            
            template = template_path.read_text()
            
            # Create test file
            test_path = self.rust_dir / "src" / category / module_name / "tests.rs"
            if not test_path.exists():
                test_path.write_text(template)
            
            return True
        except Exception as e:
            console.print(f"[red]Error creating test file: {e}[/red]")
            return False
    
    def run_tests(self) -> bool:
        """Run Rust tests."""
        try:
            console.print("[bold]Running Rust tests...[/bold]")
            
            # Change to Rust directory
            os.chdir(self.rust_dir)
            
            # Run tests
            result = subprocess.run(
                ["cargo", "test"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                console.print("[green]All tests passed![/green]")
                return True
            else:
                console.print("[red]Tests failed:[/red]")
                console.print(result.stdout)
                console.print(result.stderr)
                return False
        except Exception as e:
            console.print(f"[red]Error running tests: {e}[/red]")
            return False
    
    def test_all(self) -> None:
        """Test all Rust components."""
        components = [
            ("core", "distributed_processor"),
            ("core", "vector_store"),
            ("core", "debug_engine"),
            ("utils", "resource_manager"),
            ("utils", "storage_manager"),
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Creating test files...", total=len(components))
            
            for category, module_name in components:
                if self.create_test_file(category, module_name):
                    progress.advance(task)
                else:
                    console.print(f"[yellow]Skipping tests for {category}.{module_name}[/yellow]")
        
        # Run tests
        if self.run_tests():
            console.print("\n[bold green]Testing complete![/bold green]")
        else:
            console.print("\n[bold red]Testing failed![/bold red]")
        
        console.print("\nNext steps:")
        console.print("1. Review test results")
        console.print("2. Fix any failing tests")
        console.print("3. Add more test cases if needed")
        console.print("4. Update implementation based on test results")

def main():
    """Main entry point for the test script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Rust components")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Path to project root directory"
    )
    
    args = parser.parse_args()
    
    # Initialize and run test manager
    manager = RustTestManager(Path(args.project_root))
    manager.test_all()

if __name__ == "__main__":
    main() 