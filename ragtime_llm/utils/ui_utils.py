"""
UI utilities module for enhanced visual experience.

This module provides:
1. Loading animations and spinners
2. Emoji helpers
3. Color formatting
4. Progress indicators
"""

import sys
import time
import random
from typing import Optional, List, Callable
from datetime import datetime

# Emoji collections
EMOJIS = {
    "success": ["✨", "🎉", "🌟", "💫", "🎊"],
    "error": ["💥", "❌", "⚠️", "🚫", "💢"],
    "loading": ["⏳", "⌛", "⏰", "🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘", "🕙", "🕚", "🕛"],
    "processing": ["⚡", "🔮", "🎯", "🎲", "🎮", "🎨", "🎭", "🎪", "🎡", "🎢"],
    "storage": ["💾", "📦", "🗄️", "📚", "📑", "🔖", "📎", "📌", "📍"],
    "video": ["🎥", "📹", "🎬", "🎞️", "📽️", "🎭", "🎪"],
    "audio": ["🎵", "🎶", "🎼", "🎧", "🎤", "🎹", "🎸", "🎺", "🎻"],
    "text": ["📝", "📄", "📜", "📋", "📑", "🔤", "📚"],
    "search": ["🔍", "🔎", "🔐", "🔑", "🔒", "🔓"],
    "network": ["🌐", "🌍", "🌎", "🌏", "📡", "📶", "📱", "💻", "🖥️"],
    "ai": ["🤖", "👾", "👽", "👻", "🎮", "🎲", "🎯", "🎨", "🎭"],
}

# ANSI color codes
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "italic": "\033[3m",
    "underline": "\033[4m",
    "blink": "\033[5m",
    "reverse": "\033[7m",
    "hidden": "\033[8m",
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bg_black": "\033[40m",
    "bg_red": "\033[41m",
    "bg_green": "\033[42m",
    "bg_yellow": "\033[43m",
    "bg_blue": "\033[44m",
    "bg_magenta": "\033[45m",
    "bg_cyan": "\033[46m",
    "bg_white": "\033[47m",
}

class Spinner:
    """Loading spinner with customizable frames."""
    
    def __init__(
        self,
        frames: Optional[List[str]] = None,
        delay: float = 0.1,
        message: str = "Loading",
        color: str = "cyan"
    ):
        """Initialize spinner.
        
        Args:
            frames: List of frame characters
            delay: Delay between frames
            message: Loading message
            color: ANSI color name
        """
        self.frames = frames or ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.delay = delay
        self.message = message
        self.color = COLORS.get(color, COLORS["cyan"])
        self.stop_running = False
        self.spin_thread = None

    def __enter__(self):
        """Start spinner on context enter."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop spinner on context exit."""
        self.stop()

    def start(self):
        """Start the spinner."""
        self.stop_running = False
        self._spin()

    def stop(self):
        """Stop the spinner."""
        self.stop_running = True
        if self.spin_thread:
            self.spin_thread.join()
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
        sys.stdout.flush()

    def _spin(self):
        """Spin animation."""
        while not self.stop_running:
            for frame in self.frames:
                if self.stop_running:
                    break
                sys.stdout.write(f"\r{self.color}{frame} {self.message}{COLORS['reset']}")
                sys.stdout.flush()
                time.sleep(self.delay)

def get_emoji(category: str) -> str:
    """Get random emoji from category."""
    return random.choice(EMOJIS.get(category, ["✨"]))

def colorize(text: str, color: str) -> str:
    """Colorize text with ANSI colors."""
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"

def format_progress(
    current: int,
    total: int,
    width: int = 40,
    prefix: str = "",
    suffix: str = "",
    fill: str = "█",
    empty: str = "░"
) -> str:
    """Format progress bar.
    
    Args:
        current: Current progress
        total: Total progress
        width: Bar width
        prefix: Prefix text
        suffix: Suffix text
        fill: Fill character
        empty: Empty character
        
    Returns:
        Formatted progress bar
    """
    percent = current / total
    filled_length = int(width * percent)
    bar = fill * filled_length + empty * (width - filled_length)
    return f"{prefix}[{bar}] {percent:.1%}{suffix}"

def print_header(text: str, emoji: Optional[str] = None):
    """Print formatted header."""
    if emoji:
        text = f"{emoji} {text}"
    print(f"\n{colorize('=' * 60, 'cyan')}")
    print(f"{colorize(text.center(60), 'bold')}")
    print(f"{colorize('=' * 60, 'cyan')}\n")

def print_success(text: str):
    """Print success message."""
    emoji = get_emoji("success")
    print(f"{colorize(f'{emoji} {text}', 'green')}")

def print_error(text: str):
    """Print error message."""
    emoji = get_emoji("error")
    print(f"{colorize(f'{emoji} {text}', 'red')}")

def print_info(text: str):
    """Print info message."""
    emoji = get_emoji("processing")
    print(f"{colorize(f'{emoji} {text}', 'cyan')}")

def print_warning(text: str):
    """Print warning message."""
    emoji = get_emoji("error")
    print(f"{colorize(f'{emoji} {text}', 'yellow')}")

def print_table(
    headers: List[str],
    rows: List[List[str]],
    title: Optional[str] = None
):
    """Print formatted table."""
    if title:
        print_header(title)
    
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Print header
    header = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(colorize(header, "bold"))
    print(colorize("-" * len(header), "dim"))
    
    # Print rows
    for row in rows:
        print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))

def print_tree(
    items: List[dict],
    indent: str = "  ",
    prefix: str = "└── ",
    show_emoji: bool = True
):
    """Print tree structure."""
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = prefix if is_last else "├── "
        
        # Get emoji if enabled
        emoji = get_emoji(item.get("type", "text")) if show_emoji else ""
        
        # Print item
        print(f"{indent}{current_prefix}{emoji} {item['name']}")
        
        # Print children
        if "children" in item:
            new_indent = indent + ("    " if is_last else "│   ")
            print_tree(item["children"], new_indent, prefix, show_emoji)

def print_timestamp():
    """Print current timestamp."""
    now = datetime.now()
    print(colorize(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}]", "dim"))

def print_separator(char: str = "─", length: int = 60):
    """Print separator line."""
    print(colorize(char * length, "dim"))

def print_box(
    text: str,
    title: Optional[str] = None,
    width: int = 60,
    color: str = "cyan"
):
    """Print text in a box."""
    lines = text.split("\n")
    max_len = max(len(line) for line in lines)
    box_width = min(max_len + 4, width)
    
    # Print top border
    print(colorize("┌" + "─" * (box_width - 2) + "┐", color))
    
    # Print title if provided
    if title:
        title_line = f"│ {title.center(box_width - 4)} │"
        print(colorize(title_line, color))
        print(colorize("├" + "─" * (box_width - 2) + "┤", color))
    
    # Print content
    for line in lines:
        wrapped_lines = [line[i:i + box_width - 4] 
                        for i in range(0, len(line), box_width - 4)]
        for wrapped_line in wrapped_lines:
            print(colorize(f"│ {wrapped_line.ljust(box_width - 4)} │", color))
    
    # Print bottom border
    print(colorize("└" + "─" * (box_width - 2) + "┘", color)) 