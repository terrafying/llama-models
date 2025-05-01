#!/usr/bin/env python3
"""
Auto-resolving test runner that handles dependencies automatically and provides intelligent error analysis.
"""

import argparse
import os
import sys
from pathlib import Path

from ragtime_llm.utils.dependency_resolver import auto_resolve_dependencies
from ragtime_llm.utils.error_handler import ErrorAnalyzer
from ragtime_llm.utils.llm_client import get_llm_client

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
        "--llm-provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider to use for error analysis"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM-based error analysis"
    )
    
    args = parser.parse_args()
    
    # Build the pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    cmd.extend(["-" + "v" * args.verbose]) if args.verbose else None
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=ragtime_llm"])
    
    # Add test path
    cmd.append(args.test_path)
    
    # Initialize error analyzer with LLM if enabled
    llm_client = None if args.no_llm else get_llm_client(args.llm_provider)
    error_analyzer = ErrorAnalyzer(llm_client)
    
    # Run with auto-resolution
    exit_code, output = auto_resolve_dependencies(cmd, args.max_iterations, return_output=True)
    
    # If there were errors, analyze them
    if exit_code != 0:
        error_contexts = error_analyzer.extract_error_context(output)
        for context in error_contexts:
            analysis = error_analyzer.analyze_error(context)
            error_analyzer.display_analysis(context, analysis)
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 