"""
Auto-fix manager for handling automatic fixes with safety checks and human approval pipeline.
"""

import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import time
from datetime import datetime
import difflib

import git
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

console = Console()

class FixConfidence(Enum):
    """Confidence levels for automatic fixes."""
    LOW = "low"  # Always requires human approval
    MEDIUM = "medium"  # Requires human approval for first N occurrences
    HIGH = "high"  # Can be auto-applied after N successful approvals
    TRUSTED = "trusted"  # Fully automated, no approval needed

@dataclass
class FixAction:
    """Represents a fix action to be taken."""
    action_type: str
    description: str
    command: Optional[List[str]] = None
    file_changes: Optional[Dict[str, str]] = None
    confidence: FixConfidence = FixConfidence.LOW
    success_count: int = 0
    failure_count: int = 0
    last_attempt: Optional[datetime] = None
    context: Optional[Dict[str, Any]] = None
    ml_confidence: float = 0.0
    historical_success_rate: float = 0.0

class AutoFixManager:
    """Manages automatic fixes with safety checks and human approval pipeline."""
    
    def __init__(
        self,
        repo_path: Optional[Path] = None,
        approval_threshold: int = 3,
        trusted_threshold: int = 10,
        ml_model_path: Optional[str] = None
    ):
        """Initialize the auto-fix manager.
        
        Args:
            repo_path: Path to git repository. If None, will use current directory.
            approval_threshold: Number of successful approvals needed for medium confidence.
            trusted_threshold: Number of successful approvals needed for trusted status.
            ml_model_path: Path to ML model for confidence scoring.
        """
        self.repo_path = repo_path or Path.cwd()
        self.approval_threshold = approval_threshold
        self.trusted_threshold = trusted_threshold
        self.fix_history: Dict[str, FixAction] = {}
        self.ml_model_path = ml_model_path
        
        # Initialize git repository
        try:
            self.repo = git.Repo(self.repo_path)
        except git.InvalidGitRepositoryError:
            console.print("[red]Not a git repository. Auto-fix features will be limited.[/red]")
            self.repo = None

    def _calculate_ml_confidence(self, fix_action: FixAction) -> float:
        """Calculate ML-based confidence score for a fix."""
        # TODO: Implement actual ML model prediction
        # For now, use a simple heuristic based on historical data
        if fix_action.context and 'error_confidence' in fix_action.context:
            base_confidence = fix_action.context['error_confidence']
        else:
            base_confidence = 0.5
            
        # Adjust based on historical success rate
        if fix_action.success_count + fix_action.failure_count > 0:
            historical_rate = fix_action.success_count / (fix_action.success_count + fix_action.failure_count)
            return (base_confidence + historical_rate) / 2
        return base_confidence

    def _calculate_historical_success_rate(self, fix_action: FixAction) -> float:
        """Calculate historical success rate for similar fixes."""
        similar_fixes = [
            f for f in self.fix_history.values()
            if f.action_type == fix_action.action_type
        ]
        
        if not similar_fixes:
            return 0.5  # Default to medium confidence
            
        total_attempts = sum(f.success_count + f.failure_count for f in similar_fixes)
        total_successes = sum(f.success_count for f in similar_fixes)
        
        return total_successes / total_attempts if total_attempts > 0 else 0.5

    def create_fix_branch(self, fix_id: str) -> str:
        """Create a new branch for the fix.
        
        Args:
            fix_id: Unique identifier for the fix
            
        Returns:
            Branch name
        """
        if not self.repo:
            return "no-branch"
            
        branch_name = f"fix/{fix_id}"
        current = self.repo.active_branch
        
        # Create and checkout new branch
        new_branch = self.repo.create_head(branch_name)
        new_branch.checkout()
        
        return branch_name

    def _show_changes_preview(self, fix_action: FixAction) -> None:
        """Show a preview of the changes that will be made."""
        table = Table(title="Fix Preview")
        table.add_column("Type", style="cyan")
        table.add_column("Details", style="green")
        
        if fix_action.command:
            table.add_row("Command", " ".join(fix_action.command))
            
        if fix_action.file_changes:
            for file_path, content in fix_action.file_changes.items():
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        original = f.read()
                    # Create a simple diff using difflib
                    diff = difflib.unified_diff(
                        original.splitlines(),
                        content.splitlines(),
                        fromfile=file_path,
                        tofile=file_path,
                        lineterm=''
                    )
                    diff_text = '\n'.join(diff)
                    # Use rich syntax highlighting for the diff
                    syntax = Syntax(diff_text, "diff", theme="monokai")
                    table.add_row("File Change", f"{file_path}\n{syntax}")
                else:
                    table.add_row("New File", file_path)
        
        console.print(table)

    def _show_confidence_analysis(self, fix_action: FixAction) -> None:
        """Show confidence analysis for the fix."""
        table = Table(title="Confidence Analysis")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("ML Confidence", f"{fix_action.ml_confidence:.2%}")
        table.add_row("Historical Success", f"{fix_action.historical_success_rate:.2%}")
        table.add_row("Overall Confidence", f"{fix_action.confidence.value}")
        
        console.print(table)

    def apply_fix(self, fix_action: FixAction) -> Tuple[bool, str]:
        """Apply a fix with enhanced safety checks and approval pipeline."""
        # Calculate confidence scores
        fix_action.ml_confidence = self._calculate_ml_confidence(fix_action)
        fix_action.historical_success_rate = self._calculate_historical_success_rate(fix_action)
        
        # Create unique fix ID
        fix_id = f"{fix_action.action_type}-{hash(str(fix_action))}"
        
        # Show preview and confidence analysis
        self._show_changes_preview(fix_action)
        self._show_confidence_analysis(fix_action)
        
        # Determine if approval is needed
        needs_approval = (
            fix_action.confidence == FixConfidence.LOW or
            (fix_action.confidence == FixConfidence.MEDIUM and 
             fix_action.success_count < self.approval_threshold) or
            fix_action.ml_confidence < 0.8
        )
        
        if needs_approval:
            # Create fix branch
            branch_name = self.create_fix_branch(fix_id)
            
            # Apply changes
            success = self._apply_changes(fix_action)
            if not success:
                return False, "Failed to apply changes"
            
            # Commit changes
            if self.repo:
                self.repo.index.add("*")
                self.repo.index.commit(f"Auto-fix: {fix_action.description}")
            
            # Enhanced approval request
            console.print(Panel(
                f"[bold yellow]Fix requires human approval[/bold yellow]\n"
                f"Type: {fix_action.action_type}\n"
                f"Description: {fix_action.description}\n"
                f"Branch: {branch_name}\n"
                f"ML Confidence: {fix_action.ml_confidence:.2%}\n"
                f"Historical Success: {fix_action.historical_success_rate:.2%}\n\n"
                f"Please review the changes and approve or reject.",
                title="Fix Approval Required"
            ))
            
            # Get human approval with staged review
            approved = self._get_staged_approval(fix_action)
            if not approved:
                if self.repo:
                    self.repo.git.reset("--hard", "HEAD^")
                    self.repo.active_branch.checkout()
                return False, "Fix rejected by human reviewer"
            
            # Update fix history
            fix_action.success_count += 1
            fix_action.last_attempt = datetime.now()
            self.fix_history[fix_id] = fix_action
            
            return True, "Fix applied and approved"
        
        # For trusted fixes, apply directly
        if fix_action.confidence == FixConfidence.TRUSTED:
            success = self._apply_changes(fix_action)
            if success:
                fix_action.success_count += 1
                fix_action.last_attempt = datetime.now()
                self.fix_history[fix_id] = fix_action
                return True, "Fix applied automatically"
            else:
                fix_action.failure_count += 1
                return False, "Failed to apply trusted fix"
        
        return False, "Invalid fix confidence level"

    def _get_staged_approval(self, fix_action: FixAction) -> bool:
        """Get human approval with staged review process."""
        stages = [
            ("Review changes", "Would you like to review the changes in detail?"),
            ("Verify impact", "Have you verified the impact of these changes?"),
            ("Final approval", "Do you approve these changes?")
        ]
        
        for stage_name, prompt in stages:
            console.print(f"\n[bold cyan]{stage_name}[/bold cyan]")
            console.print(f"[yellow]{prompt}[/yellow]")
            
            while True:
                response = input("(y/n/q): ").lower()
                if response == 'q':
                    return False
                elif response in ['y', 'yes']:
                    break
                elif response in ['n', 'no']:
                    return False
                console.print("[yellow]Please enter 'y', 'n', or 'q' to quit[/yellow]")
        
        return True

    def _apply_changes(self, fix_action: FixAction) -> bool:
        """Apply the actual changes for a fix.
        
        Args:
            fix_action: The fix action to apply
            
        Returns:
            True if changes were applied successfully
        """
        try:
            # Apply command if specified
            if fix_action.command:
                result = subprocess.run(
                    fix_action.command,
                    capture_output=True,
                    text=True,
                    check=True
                )
                console.print(result.stdout)
            
            # Apply file changes if specified
            if fix_action.file_changes:
                for file_path, content in fix_action.file_changes.items():
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'w') as f:
                        f.write(content)
            
            return True
        except Exception as e:
            console.print(f"[red]Error applying fix: {str(e)}[/red]")
            return False

    def update_fix_confidence(self, fix_id: str) -> None:
        """Update the confidence level of a fix based on its history.
        
        Args:
            fix_id: The ID of the fix to update
        """
        if fix_id not in self.fix_history:
            return
            
        fix_action = self.fix_history[fix_id]
        
        # Update confidence based on success count and ML confidence
        if (fix_action.success_count >= self.trusted_threshold and 
            fix_action.ml_confidence >= 0.9):
            fix_action.confidence = FixConfidence.TRUSTED
        elif (fix_action.success_count >= self.approval_threshold and 
              fix_action.ml_confidence >= 0.8):
            fix_action.confidence = FixConfidence.MEDIUM
        
        # Update history
        self.fix_history[fix_id] = fix_action

    def get_fix_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about fix history."""
        stats = {
            "total_fixes": len(self.fix_history),
            "successful_fixes": sum(1 for f in self.fix_history.values() if f.success_count > 0),
            "failed_fixes": sum(1 for f in self.fix_history.values() if f.failure_count > 0),
            "trusted_fixes": sum(1 for f in self.fix_history.values() 
                               if f.confidence == FixConfidence.TRUSTED),
            "average_ml_confidence": sum(f.ml_confidence for f in self.fix_history.values()) / 
                                   len(self.fix_history) if self.fix_history else 0,
            "average_success_rate": sum(f.historical_success_rate for f in self.fix_history.values()) /
                                  len(self.fix_history) if self.fix_history else 0,
            "fix_types": {}
        }
        
        # Calculate statistics per fix type
        for fix in self.fix_history.values():
            if fix.action_type not in stats["fix_types"]:
                stats["fix_types"][fix.action_type] = {
                    "count": 0,
                    "success_rate": 0,
                    "avg_confidence": 0
                }
            
            type_stats = stats["fix_types"][fix.action_type]
            type_stats["count"] += 1
            type_stats["success_rate"] = (type_stats["success_rate"] * (type_stats["count"] - 1) +
                                        fix.historical_success_rate) / type_stats["count"]
            type_stats["avg_confidence"] = (type_stats["avg_confidence"] * (type_stats["count"] - 1) +
                                          fix.ml_confidence) / type_stats["count"]
        
        return stats 