"""
Unified logging system for the RAG application.
"""

import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

class StructuredLogger:
    """Structured logging with JSON formatting and multiple handlers."""
    
    def __init__(self, name: str, log_dir: str = "logs"):
        """Initialize the logger.
        
        Args:
            name: Name of the logger
            log_dir: Directory to store log files
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create handlers
        self._setup_console_handler()
        self._setup_file_handler(log_path / f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
        
    def _setup_console_handler(self):
        """Setup console handler with colored output."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
    def _setup_file_handler(self, log_file: Path):
        """Setup file handler with JSON formatting."""
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
    def _format_log(self, level: str, message: str, **kwargs) -> str:
        """Format log message as JSON."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            **kwargs
        }
        return json.dumps(log_entry)
        
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(self._format_log("DEBUG", message, **kwargs))
        
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(self._format_log("INFO", message, **kwargs))
        
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(self._format_log("WARNING", message, **kwargs))
        
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(self._format_log("ERROR", message, **kwargs))
        
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(self._format_log("CRITICAL", message, **kwargs))

# Create default logger instance
logger = StructuredLogger("rag_system") 