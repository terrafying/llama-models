"""
Intelligent error handling using LLM for test failures and dependency issues.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json

from rich.console import Console
from rich.panel import Panel

console = Console()

@dataclass
class ErrorContext:
    """Context information about an error."""
    error_type: str
    error_message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    traceback: Optional[str] = None
    test_name: Optional[str] = None
    module_name: Optional[str] = None

class ErrorAnalyzer:
    """Analyzes test errors using LLM to provide intelligent solutions."""
    
    def __init__(self, llm_client=None):
        """Initialize the error analyzer.
        
        Args:
            llm_client: Optional LLM client for error analysis
        """
        self.llm_client = llm_client
        self.error_patterns = {
            'import_error': r"ImportError.*?No module named '([^']+)'",
            'module_not_found': r"ModuleNotFoundError.*?No module named '([^']+)'",
            'attribute_error': r"AttributeError.*?'([^']+)'",
            'type_error': r"TypeError.*?([^']+)",
            'value_error': r"ValueError.*?([^']+)",
            'assertion_error': r"AssertionError.*?([^']+)",
            'api_key_error': r"api_key.*?must be set",
            'huggingface_error': r"cannot import name '([^']+)' from 'huggingface_hub'",
        }

    def extract_error_context(self, error_output: str) -> List[ErrorContext]:
        """Extract structured error context from test output."""
        contexts = []
        current_error = None
        current_traceback = []
        
        for line in error_output.splitlines():
            # Match test collection errors
            if "ERROR collecting" in line:
                test_name = re.search(r"ERROR collecting ([^\s]+)", line)
                if test_name:
                    current_error = ErrorContext(
                        error_type="collection_error",
                        error_message=line,
                        test_name=test_name.group(1)
                    )
                    contexts.append(current_error)
                    current_traceback = []
            
            # Match traceback lines
            elif current_error and line.strip().startswith("File"):
                file_match = re.search(r'File "([^"]+)", line (\d+)', line)
                if file_match:
                    current_error.file_path = file_match.group(1)
                    current_error.line_number = int(file_match.group(2))
                    current_traceback.append(line)
            
            # Match error messages
            elif current_error and any(err in line for err in ["Error:", "Exception:", "E   "]):
                current_error.error_message = line.strip()
                current_traceback.append(line)
            
            # Match module imports
            elif "import" in line.lower() and "error" in line.lower():
                module_match = re.search(r"from ([^\s]+) import", line)
                if module_match:
                    current_error = ErrorContext(
                        error_type="import_error",
                        error_message=line,
                        module_name=module_match.group(1)
                    )
                    contexts.append(current_error)
                    current_traceback = []
            
            # Match API key errors
            elif "api_key" in line.lower() and "must be set" in line.lower():
                current_error = ErrorContext(
                    error_type="api_key_error",
                    error_message=line,
                    module_name="openai" if "openai" in line.lower() else "anthropic"
                )
                contexts.append(current_error)
                current_traceback = []
            
            # Match HuggingFace errors
            elif "huggingface_hub" in line and "cannot import name" in line:
                match = re.search(r"cannot import name '([^']+)' from 'huggingface_hub'", line)
                if match:
                    current_error = ErrorContext(
                        error_type="huggingface_error",
                        error_message=line,
                        module_name=match.group(1)
                    )
                    contexts.append(current_error)
                    current_traceback = []
            
            # Collect traceback
            elif current_error and line.strip():
                current_traceback.append(line)
        
        # Add traceback to the last error if we have one
        if current_error and current_traceback:
            current_error.traceback = "\n".join(current_traceback)
        
        return contexts

    def analyze_error(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Analyze an error using LLM and return suggested solutions."""
        if not self.llm_client:
            return self._default_analysis(error_context)
        
        prompt = self._create_analysis_prompt(error_context)
        response = self.llm_client.analyze(prompt)
        return self._parse_llm_response(response)

    def _create_analysis_prompt(self, error_context: ErrorContext) -> str:
        """Create a prompt for the LLM to analyze the error."""
        return f"""
        Analyze the following test error and provide a solution:
        
        Error Type: {error_context.error_type}
        Error Message: {error_context.error_message}
        File: {error_context.file_path}
        Line: {error_context.line_number}
        Test: {error_context.test_name}
        Module: {error_context.module_name}
        Traceback: {error_context.traceback}
        
        Please provide:
        1. Root cause analysis
        2. Suggested fixes
        3. Prevention strategies
        4. Priority level (high/medium/low)
        """

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured format."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "analysis": response,
                "priority": "medium",
                "suggested_fixes": [],
                "prevention_strategies": []
            }

    def _default_analysis(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Provide default analysis when LLM is not available."""
        if error_context.error_type in ["import_error", "module_not_found"]:
            return {
                "analysis": "Missing dependency",
                "priority": "high",
                "suggested_fixes": [
                    f"Install missing package: {error_context.module_name}",
                    "Check requirements.txt for correct version",
                    "Run 'pip install -r requirements.txt' to install all dependencies"
                ],
                "prevention_strategies": [
                    "Maintain up-to-date requirements.txt",
                    "Use dependency management tools",
                    "Run tests in a clean virtual environment"
                ]
            }
        elif error_context.error_type == "api_key_error":
            return {
                "analysis": "Missing API key",
                "priority": "high",
                "suggested_fixes": [
                    f"Set {error_context.module_name.upper()}_API_KEY environment variable",
                    f"Add API key to .env file",
                    "Check API key configuration in settings"
                ],
                "prevention_strategies": [
                    "Use environment variables for API keys",
                    "Implement API key validation on startup",
                    "Add API key checks to CI/CD pipeline"
                ]
            }
        elif error_context.error_type == "huggingface_error":
            return {
                "analysis": "HuggingFace Hub compatibility issue",
                "priority": "high",
                "suggested_fixes": [
                    "Update huggingface-hub package",
                    f"Check compatibility of {error_context.module_name} with current version",
                    "Consider using a different version of the package"
                ],
                "prevention_strategies": [
                    "Pin specific versions in requirements.txt",
                    "Test with multiple package versions",
                    "Monitor package updates and compatibility"
                ]
            }
        return {
            "analysis": "Unknown error type",
            "priority": "medium",
            "suggested_fixes": ["Review error message for details"],
            "prevention_strategies": ["Add more error handling"]
        }

    def display_analysis(self, error_context: ErrorContext, analysis: Dict[str, Any]) -> None:
        """Display the error analysis in a formatted way."""
        console.print(Panel(
            f"[bold red]Error Analysis[/bold red]\n"
            f"Type: {error_context.error_type}\n"
            f"Message: {error_context.error_message}\n"
            f"Priority: {analysis['priority']}\n\n"
            f"[bold green]Suggested Fixes:[/bold green]\n"
            + "\n".join(f"- {fix}" for fix in analysis['suggested_fixes']) + "\n\n"
            f"[bold yellow]Prevention Strategies:[/bold yellow]\n"
            + "\n".join(f"- {strategy}" for strategy in analysis['prevention_strategies']),
            title="Error Analysis Report"
        )) 