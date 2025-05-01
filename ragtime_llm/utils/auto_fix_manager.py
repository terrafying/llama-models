"""
Auto-fix manager for handling automatic fixes with safety checks and human approval pipeline.
"""

import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import git
from rich.console import Console
from rich.panel import Panel

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

class AutoFixManager:
    """Manages automatic fixes with safety checks and human approval pipeline."""
    
    def __init__(
        self,
        repo_path: Optional[Path] = None,
        approval_threshold: int = 3,
        trusted_threshold: int = 10
    ):
        """Initialize the auto-fix manager.
        
        Args:
            repo_path: Path to git repository. If None, will use current directory.
            approval_threshold: Number of successful approvals needed for medium confidence.
            trusted_threshold: Number of successful approvals needed for trusted status.
        """
        self.repo_path = repo_path or Path.cwd()
        self.approval_threshold = approval_threshold
        self.trusted_threshold = trusted_threshold
        self.fix_history: Dict[str, FixAction] = {}
        
        # Initialize git repository
        try:
            self.repo = git.Repo(self.repo_path)
        except git.InvalidGitRepositoryError:
            console.print("[red]Not a git repository. Auto-fix features will be limited.[/red]")
            self.repo = None

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

    def apply_fix(self, fix_action: FixAction) -> Tuple[bool, str]:
        """Apply a fix with appropriate safety checks.
        
        Args:
            fix_action: The fix action to apply
            
        Returns:
            Tuple of (success, message)
        """
        # Create unique fix ID
        fix_id = f"{fix_action.action_type}-{hash(str(fix_action))}"
        
        # Check if we need human approval
        needs_approval = (
            fix_action.confidence == FixConfidence.LOW or
            (fix_action.confidence == FixConfidence.MEDIUM and 
             fix_action.success_count < self.approval_threshold)
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
            
            # Request human approval
            console.print(Panel(
                f"[bold yellow]Fix requires human approval[/bold yellow]\n"
                f"Type: {fix_action.action_type}\n"
                f"Description: {fix_action.description}\n"
                f"Branch: {branch_name}\n\n"
                f"Please review the changes and approve or reject.",
                title="Fix Approval Required"
            ))
            
            # Wait for human approval
            approved = self._get_human_approval()
            if not approved:
                if self.repo:
                    self.repo.git.reset("--hard", "HEAD^")
                    self.repo.active_branch.checkout()
                return False, "Fix rejected by human reviewer"
            
            # Update fix history
            fix_action.success_count += 1
            self.fix_history[fix_id] = fix_action
            
            return True, "Fix applied and approved"
        
        # For trusted fixes, apply directly
        if fix_action.confidence == FixConfidence.TRUSTED:
            success = self._apply_changes(fix_action)
            if success:
                fix_action.success_count += 1
                self.fix_history[fix_id] = fix_action
                return True, "Fix applied automatically"
            else:
                fix_action.failure_count += 1
                return False, "Failed to apply trusted fix"
        
        return False, "Invalid fix confidence level"

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
                    with open(file_path, 'w') as f:
                        f.write(content)
            
            return True
        except Exception as e:
            console.print(f"[red]Error applying fix: {str(e)}[/red]")
            return False

    def _get_human_approval(self) -> bool:
        """Get human approval for a fix.
        
        Returns:
            True if approved, False if rejected
        """
        while True:
            response = input("Approve fix? (y/n): ").lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            console.print("[yellow]Please enter 'y' or 'n'[/yellow]")

    def update_fix_confidence(self, fix_id: str) -> None:
        """Update the confidence level of a fix based on its history.
        
        Args:
            fix_id: The ID of the fix to update
        """
        if fix_id not in self.fix_history:
            return
            
        fix_action = self.fix_history[fix_id]
        
        # Update confidence based on success count
        if fix_action.success_count >= self.trusted_threshold:
            fix_action.confidence = FixConfidence.TRUSTED
        elif fix_action.success_count >= self.approval_threshold:
            fix_action.confidence = FixConfidence.MEDIUM
        
        # Update history
        self.fix_history[fix_id] = fix_action

    def get_fix_statistics(self) -> Dict[str, int]:
        """Get statistics about fix history.
        
        Returns:
            Dictionary of fix statistics
        """
        stats = {
            "total_fixes": len(self.fix_history),
            "successful_fixes": sum(1 for f in self.fix_history.values() if f.success_count > 0),
            "failed_fixes": sum(1 for f in self.fix_history.values() if f.failure_count > 0),
            "trusted_fixes": sum(1 for f in self.fix_history.values() 
                               if f.confidence == FixConfidence.TRUSTED),
        }
        return stats 