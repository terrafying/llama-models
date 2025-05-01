"""
Enhanced Debug-Iteration-Merge Engine for intelligent test and code iteration.

This module provides:
1. Advanced test iteration and debugging with intelligent prioritization
2. Comprehensive error analysis and auto-resolution
3. Smart code merging and conflict resolution
4. Performance optimization and resource management
5. Code quality tracking and improvement
6. Regression detection and prevention
7. Auto-fix learning and adaptation

Key Components:
- DebugEngine: Core debugging and iteration logic
- TestIterator: Manages test execution and iteration
- MergeManager: Handles code merging and conflict resolution
- PerformanceTracker: Monitors and optimizes performance
- QualityAnalyzer: Tracks and improves code quality
- RegressionDetector: Detects and prevents regressions
- AutoFixLearner: Learns from successful fixes
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Set
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor
import subprocess
import git
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from ragtime_llm.utils.error_handler import ErrorAnalyzer
from ragtime_llm.utils.auto_fix_manager import AutoFixManager
from ragtime_llm.utils.resource_manager import ResourceManager

logger = logging.getLogger(__name__)
console = Console()

@dataclass
class TestResult:
    """Results from a test iteration."""
    test_name: str
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    fix_applied: bool = False
    fix_description: Optional[str] = None
    code_coverage: float = 0.0
    complexity_score: float = 0.0
    quality_score: float = 0.0
    regression_risk: float = 0.0

@dataclass
class IterationMetrics:
    """Metrics for a debug iteration."""
    iteration_id: str
    start_time: float
    end_time: float
    test_results: List[TestResult]
    resource_usage: Dict[str, float]
    fix_count: int
    success_rate: float
    quality_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    regression_metrics: Dict[str, float]

class QualityAnalyzer:
    """Analyzes and tracks code quality."""
    
    def __init__(self):
        """Initialize the quality analyzer."""
        self.metrics_history: List[Dict[str, float]] = []
        self.thresholds = {
            "complexity": 10.0,
            "coverage": 80.0,
            "quality": 0.8
        }
    
    def analyze_code(self, file_path: Path) -> Dict[str, float]:
        """Analyze code quality metrics.
        
        Args:
            file_path: Path to code file
        
        Returns:
            Dictionary of quality metrics
        """
        try:
            # Run code analysis tools
            complexity = self._analyze_complexity(file_path)
            coverage = self._analyze_coverage(file_path)
            quality = self._analyze_quality(file_path)
            
            metrics = {
                "complexity": complexity,
                "coverage": coverage,
                "quality": quality
            }
            
            self.metrics_history.append(metrics)
            return metrics
        except Exception as e:
            logger.error(f"Error analyzing code quality: {e}")
            return {}
    
    def _analyze_complexity(self, file_path: Path) -> float:
        """Analyze code complexity."""
        # TODO: Implement complexity analysis
        return 0.0
    
    def _analyze_coverage(self, file_path: Path) -> float:
        """Analyze code coverage."""
        # TODO: Implement coverage analysis
        return 0.0
    
    def _analyze_quality(self, file_path: Path) -> float:
        """Analyze code quality."""
        # TODO: Implement quality analysis
        return 0.0

class RegressionDetector:
    """Detects and prevents regressions."""
    
    def __init__(self):
        """Initialize the regression detector."""
        self.test_history: List[Dict[str, Any]] = []
        self.regression_patterns: Set[str] = set()
    
    def analyze_test_results(self, results: List[TestResult]) -> Dict[str, float]:
        """Analyze test results for regression patterns.
        
        Args:
            results: List of test results
        
        Returns:
            Dictionary of regression metrics
        """
        try:
            # Update test history
            self.test_history.append({
                "timestamp": time.time(),
                "results": results
            })
            
            # Analyze patterns
            patterns = self._analyze_patterns(results)
            self.regression_patterns.update(patterns)
            
            # Calculate regression risk
            risk = self._calculate_regression_risk(results)
            
            return {
                "regression_risk": risk,
                "pattern_count": len(patterns),
                "history_size": len(self.test_history)
            }
        except Exception as e:
            logger.error(f"Error analyzing regression patterns: {e}")
            return {}
    
    def _analyze_patterns(self, results: List[TestResult]) -> Set[str]:
        """Analyze test results for patterns."""
        patterns = set()
        # TODO: Implement pattern analysis
        return patterns
    
    def _calculate_regression_risk(self, results: List[TestResult]) -> float:
        """Calculate regression risk score."""
        # TODO: Implement risk calculation
        return 0.0

class AutoFixLearner:
    """Learns from successful fixes."""
    
    def __init__(self):
        """Initialize the auto-fix learner."""
        self.fix_history: List[Dict[str, Any]] = []
        self.success_patterns: Dict[str, float] = {}
    
    def learn_from_fix(self, 
                      error_context: Dict[str, Any],
                      fix_action: Dict[str, Any],
                      success: bool) -> None:
        """Learn from a fix attempt.
        
        Args:
            error_context: Error context
            fix_action: Fix action taken
            success: Whether the fix was successful
        """
        try:
            # Record fix attempt
            self.fix_history.append({
                "timestamp": time.time(),
                "error_context": error_context,
                "fix_action": fix_action,
                "success": success
            })
            
            # Update success patterns
            if success:
                pattern = self._extract_pattern(error_context, fix_action)
                self.success_patterns[pattern] = self.success_patterns.get(pattern, 0) + 1
        except Exception as e:
            logger.error(f"Error learning from fix: {e}")
    
    def suggest_fix(self, error_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Suggest a fix based on learned patterns.
        
        Args:
            error_context: Error context
        
        Returns:
            Suggested fix action or None
        """
        try:
            # Find matching patterns
            patterns = self._find_matching_patterns(error_context)
            
            if patterns:
                # Get most successful pattern
                best_pattern = max(patterns.items(), key=lambda x: x[1])[0]
                return self._generate_fix_action(best_pattern, error_context)
            
            return None
        except Exception as e:
            logger.error(f"Error suggesting fix: {e}")
            return None
    
    def _extract_pattern(self, 
                        error_context: Dict[str, Any],
                        fix_action: Dict[str, Any]) -> str:
        """Extract pattern from error and fix."""
        # TODO: Implement pattern extraction
        return ""
    
    def _find_matching_patterns(self, 
                              error_context: Dict[str, Any]) -> Dict[str, float]:
        """Find patterns matching error context."""
        # TODO: Implement pattern matching
        return {}
    
    def _generate_fix_action(self, 
                           pattern: str,
                           error_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fix action from pattern."""
        # TODO: Implement fix generation
        return {}

class TestIterator:
    """Manages test execution and iteration."""
    
    def __init__(self, 
                 test_path: str,
                 max_iterations: int = 10,
                 max_workers: int = 4):
        """Initialize the test iterator.
        
        Args:
            test_path: Path to test file or directory
            max_iterations: Maximum number of iterations
            max_workers: Maximum number of parallel workers
        """
        self.test_path = Path(test_path)
        self.max_iterations = max_iterations
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.resource_manager = ResourceManager(str(self.test_path.parent))
        self.quality_analyzer = QualityAnalyzer()
        self.regression_detector = RegressionDetector()
        
    def run_test(self, test_name: str) -> TestResult:
        """Run a single test and collect metrics."""
        start_time = time.time()
        try:
            # Run test with resource monitoring
            result = subprocess.run(
                ["python", "-m", "pytest", test_name, "-v"],
                capture_output=True,
                text=True
            )
            
            # Collect metrics
            execution_time = time.time() - start_time
            resource_usage = self.resource_manager.get_resource_usage()
            
            # Analyze code quality
            test_file = Path(test_name)
            quality_metrics = self.quality_analyzer.analyze_code(test_file)
            
            # Calculate regression risk
            regression_metrics = self.regression_detector.analyze_test_results([TestResult(
                test_name=test_name,
                success=result.returncode == 0,
                error_message=result.stderr if result.returncode != 0 else None,
                execution_time=execution_time,
                memory_usage=resource_usage["memory_percent"],
                cpu_usage=resource_usage["cpu_percent"]
            )])
            
            return TestResult(
                test_name=test_name,
                success=result.returncode == 0,
                error_message=result.stderr if result.returncode != 0 else None,
                execution_time=execution_time,
                memory_usage=resource_usage["memory_percent"],
                cpu_usage=resource_usage["cpu_percent"],
                code_coverage=quality_metrics.get("coverage", 0.0),
                complexity_score=quality_metrics.get("complexity", 0.0),
                quality_score=quality_metrics.get("quality", 0.0),
                regression_risk=regression_metrics.get("regression_risk", 0.0)
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def prioritize_tests(self, test_files: List[Path]) -> List[Path]:
        """Prioritize tests based on various factors.
        
        Args:
            test_files: List of test files
        
        Returns:
            Prioritized list of test files
        """
        try:
            # Collect test metrics
            metrics = []
            for test_file in test_files:
                quality_metrics = self.quality_analyzer.analyze_code(test_file)
                metrics.append({
                    "file": test_file,
                    "complexity": quality_metrics.get("complexity", 0.0),
                    "coverage": quality_metrics.get("coverage", 0.0),
                    "quality": quality_metrics.get("quality", 0.0)
                })
            
            # Convert to numpy array for clustering
            X = np.array([[m["complexity"], m["coverage"], m["quality"]] for m in metrics])
            X = StandardScaler().fit_transform(X)
            
            # Cluster tests
            clustering = DBSCAN(eps=0.3, min_samples=2).fit(X)
            
            # Sort by cluster and metrics
            prioritized = []
            for label in sorted(set(clustering.labels_)):
                cluster_tests = [m["file"] for i, m in enumerate(metrics) 
                               if clustering.labels_[i] == label]
                prioritized.extend(sorted(cluster_tests))
            
            return prioritized
        except Exception as e:
            logger.error(f"Error prioritizing tests: {e}")
            return test_files
    
    def iterate_tests(self) -> List[IterationMetrics]:
        """Run tests iteratively with debugging."""
        metrics = []
        test_files = list(self.test_path.glob("**/test_*.py"))
        
        # Prioritize tests
        prioritized_files = self.prioritize_tests(test_files)
        
        for i in range(self.max_iterations):
            iteration_id = f"iter_{i+1}"
            start_time = time.time()
            
            # Run tests in parallel
            futures = []
            for test_file in prioritized_files:
                future = self.executor.submit(self.run_test, str(test_file))
                futures.append(future)
            
            # Collect results
            test_results = []
            for future in futures:
                test_results.append(future.result())
            
            # Calculate metrics
            success_count = sum(1 for r in test_results if r.success)
            fix_count = sum(1 for r in test_results if r.fix_applied)
            
            # Calculate quality metrics
            quality_metrics = {
                "avg_complexity": np.mean([r.complexity_score for r in test_results]),
                "avg_coverage": np.mean([r.code_coverage for r in test_results]),
                "avg_quality": np.mean([r.quality_score for r in test_results])
            }
            
            # Calculate performance metrics
            performance_metrics = {
                "avg_execution_time": np.mean([r.execution_time for r in test_results]),
                "avg_memory_usage": np.mean([r.memory_usage for r in test_results]),
                "avg_cpu_usage": np.mean([r.cpu_usage for r in test_results])
            }
            
            # Calculate regression metrics
            regression_metrics = self.regression_detector.analyze_test_results(test_results)
            
            metrics.append(IterationMetrics(
                iteration_id=iteration_id,
                start_time=start_time,
                end_time=time.time(),
                test_results=test_results,
                resource_usage=self.resource_manager.get_resource_usage(),
                fix_count=fix_count,
                success_rate=success_count / len(test_results),
                quality_metrics=quality_metrics,
                performance_metrics=performance_metrics,
                regression_metrics=regression_metrics
            ))
            
            # Check if we've achieved stability
            if success_count == len(test_results):
                break
        
        return metrics

class MergeManager:
    """Handles code merging and conflict resolution."""
    
    def __init__(self, repo_path: Optional[str] = None):
        """Initialize the merge manager.
        
        Args:
            repo_path: Path to git repository
        """
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.repo = git.Repo(self.repo_path)
        self.auto_fix_learner = AutoFixLearner()
    
    def create_fix_branch(self, fix_id: str) -> str:
        """Create a new branch for a fix."""
        branch_name = f"fix/{fix_id}"
        current = self.repo.active_branch
        
        # Create and checkout new branch
        new_branch = self.repo.create_head(branch_name)
        new_branch.checkout()
        
        return branch_name
    
    def apply_fix(self, 
                 fix_action: Dict[str, Any],
                 branch_name: str) -> bool:
        """Apply a fix and commit changes."""
        try:
            # Apply changes
            if "file_changes" in fix_action:
                for file_path, content in fix_action["file_changes"].items():
                    with open(file_path, 'w') as f:
                        f.write(content)
            
            # Commit changes
            self.repo.index.add("*")
            self.repo.index.commit(f"Auto-fix: {fix_action['description']}")
            
            # Learn from fix
            self.auto_fix_learner.learn_from_fix(
                fix_action.get("error_context", {}),
                fix_action,
                True
            )
            
            return True
        except Exception as e:
            logger.error(f"Error applying fix: {e}")
            
            # Learn from failed fix
            self.auto_fix_learner.learn_from_fix(
                fix_action.get("error_context", {}),
                fix_action,
                False
            )
            
            return False
    
    def merge_fix(self, branch_name: str) -> bool:
        """Merge a fix branch into main."""
        try:
            main = self.repo.heads.main
            fix_branch = self.repo.heads[branch_name]
            
            # Merge fix branch
            main.checkout()
            self.repo.index.merge_tree(main, base=main, other=fix_branch)
            
            # Commit merge
            self.repo.index.commit(f"Merge fix: {branch_name}")
            
            # Cleanup
            self.repo.delete_head(branch_name)
            
            return True
        except Exception as e:
            logger.error(f"Error merging fix: {e}")
            return False

class DebugEngine:
    """Core debugging and iteration engine."""
    
    def __init__(self,
                 test_path: str,
                 repo_path: Optional[str] = None,
                 max_iterations: int = 10):
        """Initialize the debug engine.
        
        Args:
            test_path: Path to test file or directory
            repo_path: Path to git repository
            max_iterations: Maximum number of iterations
        """
        self.test_iterator = TestIterator(test_path, max_iterations)
        self.merge_manager = MergeManager(repo_path)
        self.error_analyzer = ErrorAnalyzer()
        self.auto_fix_manager = AutoFixManager()
        self.quality_analyzer = QualityAnalyzer()
        self.regression_detector = RegressionDetector()
        self.auto_fix_learner = AutoFixLearner()
    
    def run_debug_cycle(self) -> List[IterationMetrics]:
        """Run a complete debug cycle."""
        console.print("[bold]Starting debug cycle...[/bold]")
        
        # Run test iterations
        metrics = self.test_iterator.iterate_tests()
        
        # Process results
        for metric in metrics:
            console.print(f"\n[bold]Iteration {metric.iteration_id}[/bold]")
            console.print(f"Success rate: {metric.success_rate:.1%}")
            console.print(f"Fixes applied: {metric.fix_count}")
            
            # Display quality metrics
            console.print("\n[bold]Quality Metrics:[/bold]")
            for name, value in metric.quality_metrics.items():
                console.print(f"{name}: {value:.2f}")
            
            # Display performance metrics
            console.print("\n[bold]Performance Metrics:[/bold]")
            for name, value in metric.performance_metrics.items():
                console.print(f"{name}: {value:.2f}")
            
            # Display regression metrics
            console.print("\n[bold]Regression Metrics:[/bold]")
            for name, value in metric.regression_metrics.items():
                console.print(f"{name}: {value:.2f}")
            
            # Analyze failures
            for result in metric.test_results:
                if not result.success:
                    # Analyze error
                    error_context = self.error_analyzer.extract_error_context(
                        result.error_message or ""
                    )
                    
                    if error_context:
                        # Get fix suggestion
                        analysis = self.error_analyzer.analyze_error(error_context[0])
                        
                        # Try to learn from previous fixes
                        suggested_fix = self.auto_fix_learner.suggest_fix(error_context)
                        if suggested_fix:
                            analysis.update(suggested_fix)
                        
                        # Create fix branch
                        branch_name = self.merge_manager.create_fix_branch(
                            f"fix_{result.test_name}"
                        )
                        
                        # Apply fix
                        if self.merge_manager.apply_fix(analysis, branch_name):
                            result.fix_applied = True
                            result.fix_description = analysis.get("description")
                            
                            # Merge fix
                            if self.merge_manager.merge_fix(branch_name):
                                console.print(f"[green]Applied fix for {result.test_name}[/green]")
        
        return metrics

def main():
    """Main entry point for the debug engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug-Iteration-Merge Engine")
    parser.add_argument(
        "test_path",
        help="Path to test file or directory"
    )
    parser.add_argument(
        "--repo-path",
        help="Path to git repository"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum number of iterations"
    )
    
    args = parser.parse_args()
    
    # Initialize and run debug engine
    engine = DebugEngine(
        test_path=args.test_path,
        repo_path=args.repo_path,
        max_iterations=args.max_iterations
    )
    
    metrics = engine.run_debug_cycle()
    
    # Print summary
    console.print("\n[bold]Debug Cycle Summary[/bold]")
    console.print(f"Total iterations: {len(metrics)}")
    console.print(f"Final success rate: {metrics[-1].success_rate:.1%}")
    console.print(f"Total fixes applied: {sum(m.fix_count for m in metrics)}")
    
    # Print quality summary
    console.print("\n[bold]Quality Summary[/bold]")
    for name, value in metrics[-1].quality_metrics.items():
        console.print(f"{name}: {value:.2f}")
    
    # Print performance summary
    console.print("\n[bold]Performance Summary[/bold]")
    for name, value in metrics[-1].performance_metrics.items():
        console.print(f"{name}: {value:.2f}")
    
    # Print regression summary
    console.print("\n[bold]Regression Summary[/bold]")
    for name, value in metrics[-1].regression_metrics.items():
        console.print(f"{name}: {value:.2f}")

if __name__ == "__main__":
    main() 