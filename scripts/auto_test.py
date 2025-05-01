#!/usr/bin/env python3
"""
Auto-resolving test runner that handles dependencies automatically and provides intelligent error analysis.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from ragtime_llm.utils.dependency_resolver import auto_resolve_dependencies
from ragtime_llm.utils.error_handler import ErrorAnalyzer
from ragtime_llm.utils.auto_fix_manager import AutoFixManager

console = Console()

def setup_environment() -> None:
    """Set up the environment for testing."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for required environment variables
    required_vars = {
        "HUGGINGFACE_CACHE_DIR": "HuggingFace cache directory for local models",
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var} ({description})")
    
    if missing_vars:
        console.print("[yellow]Warning: The following environment variables are not set:[/yellow]")
        for var in missing_vars:
            console.print(f"  - {var}")
        console.print("\n[yellow]Some features may be limited. Consider setting these variables in a .env file.[/yellow]")

def main():
    parser = argparse.ArgumentParser(description="Run tests with automatic dependency resolution and error analysis")
    parser.add_argument(
        "test_path",
        nargs="?",
        default="tests/",
        help="Path to test file or directory (default: tests/)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of dependency resolution iterations"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run tests with coverage"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Verbosity level (can be used multiple times)"
    )
    parser.add_argument(
        "--no-env-check",
        action="store_true",
        help="Skip environment variable checks"
    )
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Enable automatic fixes with human approval"
    )
    parser.add_argument(
        "--approval-threshold",
        type=int,
        default=3,
        help="Number of successful approvals needed for medium confidence"
    )
    parser.add_argument(
        "--trusted-threshold",
        type=int,
        default=10,
        help="Number of successful approvals needed for trusted status"
    )
    
    args = parser.parse_args()
    
    # Set up environment
    if not args.no_env_check:
        setup_environment()
    
    # Build the pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    cmd.extend(["-" + "v" * args.verbose]) if args.verbose else None
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=ragtime_llm"])
    
    # Add test path
    cmd.append(args.test_path)
    
    # Initialize auto-fix manager if enabled
    auto_fix_manager = None
    if args.auto_fix:
        auto_fix_manager = AutoFixManager(
            approval_threshold=args.approval_threshold,
            trusted_threshold=args.trusted_threshold
        )
    
    # Initialize error analyzer
    error_analyzer = ErrorAnalyzer(auto_fix_manager=auto_fix_manager)
    
    # Run with auto-resolution
    exit_code, output = auto_resolve_dependencies(cmd, args.max_iterations, return_output=True)
    
    # If there were errors, analyze them
    if exit_code != 0:
        error_contexts = error_analyzer.extract_error_context(output)
        for context in error_contexts:
            analysis = error_analyzer.analyze_error(context)
            error_analyzer.display_analysis(context, analysis)
        
        # Display fix statistics if auto-fix is enabled
        if auto_fix_manager:
            stats = auto_fix_manager.get_fix_statistics()
            console.print(Panel(
                f"[bold]Auto-fix Statistics[/bold]\n"
                f"Total fixes attempted: {stats['total_fixes']}\n"
                f"Successful fixes: {stats['successful_fixes']}\n"
                f"Failed fixes: {stats['failed_fixes']}\n"
                f"Trusted fixes: {stats['trusted_fixes']}",
                title="Fix Statistics"
            ))
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 