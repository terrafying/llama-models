#!/usr/bin/env python3
"""
Script to help analyze Python code for migration to Rust.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Set
import ast
import json
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

console = Console()

class PythonAnalyzer:
    """Analyzes Python code for migration to Rust."""
    
    def __init__(self, project_root: Path):
        """Initialize the analyzer.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = project_root
        self.analysis_file = project_root / "python_analysis.json"
        
        # Load or initialize analysis
        self.analysis = self._load_analysis()
    
    def _load_analysis(self) -> Dict:
        """Load analysis from file."""
        if self.analysis_file.exists():
            try:
                return json.loads(self.analysis_file.read_text())
            except Exception as e:
                console.print(f"[yellow]Error loading analysis: {e}[/yellow]")
        
        # Initialize default analysis
        return {
            "components": {},
            "dependencies": {},
            "complexity": {},
            "migration_notes": {}
        }
    
    def _save_analysis(self) -> None:
        """Save analysis to file."""
        try:
            self.analysis_file.write_text(json.dumps(self.analysis, indent=2))
        except Exception as e:
            console.print(f"[red]Error saving analysis: {e}[/red]")
    
    def analyze_file(self, file_path: Path) -> Dict:
        """Analyze a Python file.
        
        Args:
            file_path: Path to Python file
        
        Returns:
            Analysis results
        """
        try:
            with open(file_path, "r") as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Analyze imports
            imports = self._analyze_imports(tree)
            
            # Analyze classes
            classes = self._analyze_classes(tree)
            
            # Analyze functions
            functions = self._analyze_functions(tree)
            
            # Analyze complexity
            complexity = self._analyze_complexity(tree)
            
            return {
                "imports": imports,
                "classes": classes,
                "functions": functions,
                "complexity": complexity
            }
        except Exception as e:
            console.print(f"[red]Error analyzing file {file_path}: {e}[/red]")
            return {}
    
    def _analyze_imports(self, tree: ast.AST) -> Dict:
        """Analyze imports in a Python file.
        
        Args:
            tree: AST of Python file
        
        Returns:
            Import analysis
        """
        imports = {
            "standard_library": set(),
            "third_party": set(),
            "local": set()
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.name.startswith("."):
                        imports["local"].add(name.name)
                    elif name.name in self._get_standard_library_modules():
                        imports["standard_library"].add(name.name)
                    else:
                        imports["third_party"].add(name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    imports["local"].add(".")
                elif node.module.startswith("."):
                    imports["local"].add(node.module)
                elif node.module in self._get_standard_library_modules():
                    imports["standard_library"].add(node.module)
                else:
                    imports["third_party"].add(node.module)
        
        return {k: list(v) for k, v in imports.items()}
    
    def _analyze_classes(self, tree: ast.AST) -> List[Dict]:
        """Analyze classes in a Python file.
        
        Args:
            tree: AST of Python file
        
        Returns:
            Class analysis
        """
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "bases": [self._get_name(base) for base in node.bases],
                    "methods": [],
                    "attributes": [],
                    "complexity": self._calculate_complexity(node)
                }
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_info = {
                            "name": item.name,
                            "args": self._get_function_args(item),
                            "returns": self._get_function_returns(item),
                            "complexity": self._calculate_complexity(item)
                        }
                        class_info["methods"].append(method_info)
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                class_info["attributes"].append(target.id)
                
                classes.append(class_info)
        
        return classes
    
    def _analyze_functions(self, tree: ast.AST) -> List[Dict]:
        """Analyze functions in a Python file.
        
        Args:
            tree: AST of Python file
        
        Returns:
            Function analysis
        """
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_info = {
                    "name": node.name,
                    "args": self._get_function_args(node),
                    "returns": self._get_function_returns(node),
                    "complexity": self._calculate_complexity(node)
                }
                functions.append(function_info)
        
        return functions
    
    def _analyze_complexity(self, tree: ast.AST) -> Dict:
        """Analyze code complexity.
        
        Args:
            tree: AST of Python file
        
        Returns:
            Complexity analysis
        """
        complexity = {
            "cyclomatic": self._calculate_cyclomatic_complexity(tree),
            "cognitive": self._calculate_cognitive_complexity(tree),
            "halstead": self._calculate_halstead_metrics(tree)
        }
        
        return complexity
    
    def _get_standard_library_modules(self) -> Set[str]:
        """Get list of standard library modules.
        
        Returns:
            Set of standard library module names
        """
        import sys
        import builtins
        
        stdlib_modules = set()
        
        # Add built-in modules
        for name in sys.stdlib_module_names:
            stdlib_modules.add(name)
        
        # Add built-in types
        for name in dir(builtins):
            if not name.startswith("_"):
                stdlib_modules.add(name)
        
        return stdlib_modules
    
    def _get_name(self, node: ast.AST) -> str:
        """Get name from AST node.
        
        Args:
            node: AST node
        
        Returns:
            Name string
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)
    
    def _get_function_args(self, node: ast.FunctionDef) -> List[Dict]:
        """Get function arguments.
        
        Args:
            node: Function definition node
        
        Returns:
            List of argument information
        """
        args = []
        
        for arg in node.args.args:
            arg_info = {
                "name": arg.arg,
                "annotation": self._get_name(arg.annotation) if arg.annotation else None
            }
            args.append(arg_info)
        
        if node.args.vararg:
            args.append({
                "name": node.args.vararg.arg,
                "annotation": self._get_name(node.args.vararg.annotation) if node.args.vararg.annotation else None,
                "vararg": True
            })
        
        if node.args.kwarg:
            args.append({
                "name": node.args.kwarg.arg,
                "annotation": self._get_name(node.args.kwarg.annotation) if node.args.kwarg.annotation else None,
                "kwarg": True
            })
        
        return args
    
    def _get_function_returns(self, node: ast.FunctionDef) -> Optional[str]:
        """Get function return type.
        
        Args:
            node: Function definition node
        
        Returns:
            Return type string or None
        """
        if node.returns:
            return self._get_name(node.returns)
        return None
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate complexity of a node.
        
        Args:
            node: AST node
        
        Returns:
            Complexity score
        """
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity.
        
        Args:
            tree: AST of Python file
        
        Returns:
            Cyclomatic complexity score
        """
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity.
        
        Args:
            tree: AST of Python file
        
        Returns:
            Cognitive complexity score
        """
        complexity = 0
        nesting = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1 + nesting
                nesting += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1 + nesting
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _calculate_halstead_metrics(self, tree: ast.AST) -> Dict:
        """Calculate Halstead metrics.
        
        Args:
            tree: AST of Python file
        
        Returns:
            Halstead metrics
        """
        operators = set()
        operands = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                operators.add(type(node.op).__name__)
            elif isinstance(node, ast.UnaryOp):
                operators.add(type(node.op).__name__)
            elif isinstance(node, ast.Compare):
                operators.add(type(node.ops[0]).__name__)
            elif isinstance(node, ast.Name):
                operands.add(node.id)
            elif isinstance(node, ast.Num):
                operands.add(str(node.n))
            elif isinstance(node, ast.Str):
                operands.add(node.s)
        
        n1 = len(operators)  # Number of unique operators
        n2 = len(operands)   # Number of unique operands
        
        return {
            "vocabulary": n1 + n2,
            "length": len(list(ast.walk(tree))),
            "difficulty": (n1 * n2) / (2 * n2) if n2 > 0 else 0,
            "effort": (n1 * n2 * (n1 + n2)) / (2 * n2) if n2 > 0 else 0
        }
    
    def analyze_component(self, category: str, module_name: str) -> None:
        """Analyze a component for migration.
        
        Args:
            category: Module category (core/utils)
            module_name: Name of the module
        """
        try:
            console.print(f"[bold]Analyzing {category}.{module_name}...[/bold]")
            
            # Find Python files
            python_files = []
            for ext in [".py", ".pyi"]:
                python_files.extend(
                    self.project_root.glob(f"ragtime_llm/{category}/{module_name}*{ext}")
                )
            
            if not python_files:
                console.print(f"[yellow]No Python files found for {category}.{module_name}[/yellow]")
                return
            
            # Analyze each file
            component_analysis = {
                "files": {},
                "dependencies": set(),
                "complexity": {
                    "cyclomatic": 0,
                    "cognitive": 0,
                    "halstead": {
                        "vocabulary": 0,
                        "length": 0,
                        "difficulty": 0,
                        "effort": 0
                    }
                },
                "migration_notes": []
            }
            
            for file_path in python_files:
                file_analysis = self.analyze_file(file_path)
                component_analysis["files"][file_path.name] = file_analysis
                
                # Update dependencies
                for import_type, imports in file_analysis["imports"].items():
                    component_analysis["dependencies"].update(imports)
                
                # Update complexity
                component_analysis["complexity"]["cyclomatic"] += file_analysis["complexity"]["cyclomatic"]
                component_analysis["complexity"]["cognitive"] += file_analysis["complexity"]["cognitive"]
                
                for metric, value in file_analysis["complexity"]["halstead"].items():
                    component_analysis["complexity"]["halstead"][metric] += value
            
            # Add migration notes
            component_analysis["migration_notes"].extend(
                self._generate_migration_notes(component_analysis)
            )
            
            # Convert sets to lists for JSON serialization
            component_analysis["dependencies"] = list(component_analysis["dependencies"])
            
            # Save analysis
            self.analysis["components"][f"{category}.{module_name}"] = component_analysis
            self._save_analysis()
            
            console.print(f"[green]Analysis complete for {category}.{module_name}[/green]")
        except Exception as e:
            console.print(f"[red]Error analyzing component: {e}[/red]")
    
    def _generate_migration_notes(self, analysis: Dict) -> List[str]:
        """Generate migration notes based on analysis.
        
        Args:
            analysis: Component analysis
        
        Returns:
            List of migration notes
        """
        notes = []
        
        # Check complexity
        if analysis["complexity"]["cyclomatic"] > 10:
            notes.append("High cyclomatic complexity - consider breaking down into smaller functions")
        if analysis["complexity"]["cognitive"] > 15:
            notes.append("High cognitive complexity - consider simplifying control flow")
        
        # Check dependencies
        for dep in analysis["dependencies"]:
            if dep.startswith("numpy") or dep.startswith("pandas"):
                notes.append(f"Consider using ndarray for {dep} operations")
            elif dep.startswith("torch"):
                notes.append("Consider using tch-rs for PyTorch operations")
            elif dep.startswith("ray"):
                notes.append("Consider using rayon for parallel processing")
        
        # Check for Python-specific features
        for file_analysis in analysis["files"].values():
            for class_info in file_analysis["classes"]:
                if any(method["name"].startswith("__") for method in class_info["methods"]):
                    notes.append("Contains Python magic methods - need to implement Rust traits")
            
            for function_info in file_analysis["functions"]:
                if function_info["args"] and any(arg.get("vararg") or arg.get("kwarg") for arg in function_info["args"]):
                    notes.append("Contains variable arguments - need to use Rust macros or traits")
        
        return notes
    
    def analyze_all(self) -> None:
        """Analyze all components."""
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
            task = progress.add_task("Analyzing components...", total=len(components))
            
            for category, module_name in components:
                self.analyze_component(category, module_name)
                progress.advance(task)
        
        console.print("\n[bold green]Analysis complete![/bold green]")
        console.print("\nNext steps:")
        console.print("1. Review analysis results in python_analysis.json")
        console.print("2. Plan migration based on complexity and dependencies")
        console.print("3. Start with simpler components")
        console.print("4. Address migration notes for each component")

def main():
    """Main entry point for the analysis script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Python code for migration to Rust")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Path to project root directory"
    )
    parser.add_argument(
        "--component",
        nargs=2,
        metavar=("CATEGORY", "MODULE"),
        help="Specific component to analyze"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = PythonAnalyzer(Path(args.project_root))
    
    if args.component:
        category, module_name = args.component
        analyzer.analyze_component(category, module_name)
    else:
        analyzer.analyze_all()

if __name__ == "__main__":
    main() 