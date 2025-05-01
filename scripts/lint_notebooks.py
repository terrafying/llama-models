#!/usr/bin/env python3
"""
Notebook linter for ensuring consistency and best practices.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import nbformat
from nbformat import NotebookNode

def check_cell_order(notebook: NotebookNode) -> List[str]:
    """Check if cells follow the recommended order."""
    issues = []
    current_section = None
    sections = ['markdown', 'code', 'markdown']  # Expected order
    
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            # Check if this is a section header
            if cell.source.startswith('#'):
                current_section = 'markdown'
        elif cell.cell_type == 'code':
            current_section = 'code'
            
        if current_section not in sections:
            issues.append(f"Unexpected cell type {cell.cell_type} after {current_section}")
    
    return issues

def check_imports(notebook: NotebookNode) -> List[str]:
    """Check if imports are properly organized."""
    issues = []
    imports_found = False
    
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            if 'import' in cell.source:
                if not imports_found:
                    imports_found = True
                else:
                    issues.append("Imports should be in the first code cell")
    
    return issues

def check_markdown_formatting(notebook: NotebookNode) -> List[str]:
    """Check markdown formatting."""
    issues = []
    
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            # Check for proper heading hierarchy
            lines = cell.source.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('#'):
                    if i > 0 and not lines[i-1].strip() == '':
                        issues.append(f"Heading should be preceded by empty line: {line}")
    
    return issues

def check_code_quality(notebook: NotebookNode) -> List[str]:
    """Check code quality."""
    issues = []
    
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            # Check for print statements
            if 'print(' in cell.source:
                issues.append("Consider using logger instead of print statements")
            
            # Check for proper error handling
            if 'try:' in cell.source and 'except:' in cell.source:
                if 'except Exception as e:' not in cell.source:
                    issues.append("Use specific exception handling")
    
    return issues

def lint_notebook(notebook_path: Path) -> Dict[str, List[str]]:
    """Lint a single notebook."""
    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)
    
    issues = {
        'cell_order': check_cell_order(notebook),
        'imports': check_imports(notebook),
        'markdown': check_markdown_formatting(notebook),
        'code': check_code_quality(notebook)
    }
    
    return issues

def main():
    """Main function to lint all notebooks."""
    notebooks_dir = Path('notebooks')
    if not notebooks_dir.exists():
        print("Notebooks directory not found")
        sys.exit(1)
    
    all_issues = {}
    for notebook_path in notebooks_dir.glob('*.ipynb'):
        print(f"\nLinting {notebook_path.name}...")
        issues = lint_notebook(notebook_path)
        
        # Filter out empty issues
        issues = {k: v for k, v in issues.items() if v}
        if issues:
            all_issues[notebook_path.name] = issues
            
            print(f"Issues found in {notebook_path.name}:")
            for category, category_issues in issues.items():
                print(f"\n{category.upper()}:")
                for issue in category_issues:
                    print(f"- {issue}")
    
    if all_issues:
        sys.exit(1)
    else:
        print("\nAll notebooks passed linting!")

if __name__ == '__main__':
    main() 