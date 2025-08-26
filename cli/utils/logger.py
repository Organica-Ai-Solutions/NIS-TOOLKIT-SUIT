"""
NIS CLI Logging Utilities
Enhanced logging with colors, formatting, and structured output
"""

import logging
import sys
from typing import Optional

# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Regular colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and emojis"""
    
    def __init__(self, use_color: bool = True):
        self.use_color = use_color
        super().__init__()
    
    def format(self, record):
        # Base format without colors
        log_format = "%(message)s"
        
        if self.use_color:
            # Color and emoji mapping by log level
            level_formats = {
                logging.DEBUG: f"{Colors.DIM}🔍 [DEBUG]{Colors.RESET} %(message)s",
                logging.INFO: f"{Colors.BRIGHT_BLUE}ℹ️  [INFO]{Colors.RESET} %(message)s", 
                logging.WARNING: f"{Colors.BRIGHT_YELLOW}⚠️  [WARNING]{Colors.RESET} %(message)s",
                logging.ERROR: f"{Colors.BRIGHT_RED}❌ [ERROR]{Colors.RESET} %(message)s",
                logging.CRITICAL: f"{Colors.BOLD}{Colors.BRIGHT_RED}🚨 [CRITICAL]{Colors.RESET} %(message)s"
            }
            log_format = level_formats.get(record.levelno, log_format)
        else:
            # Plain format without colors
            level_formats = {
                logging.DEBUG: "[DEBUG] %(message)s",
                logging.INFO: "[INFO] %(message)s",
                logging.WARNING: "[WARNING] %(message)s", 
                logging.ERROR: "[ERROR] %(message)s",
                logging.CRITICAL: "[CRITICAL] %(message)s"
            }
            log_format = level_formats.get(record.levelno, log_format)
        
        formatter = logging.Formatter(log_format)
        return formatter.format(record)

def setup_logging(level: int = logging.INFO, use_color: bool = True):
    """Setup enhanced logging for the NIS CLI"""
    
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter(use_color))
    
    # Configure root logger
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Suppress verbose third-party logs unless in debug mode
    if level > logging.DEBUG:
        for logger_name in ['urllib3', 'requests', 'docker', 'kubernetes']:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name"""
    return logging.getLogger(name)

# Convenience functions for common log patterns
def success(message: str):
    """Log a success message with green checkmark"""
    logging.info(f"{Colors.BRIGHT_GREEN}✅ {message}{Colors.RESET}")

def info(message: str):
    """Log an info message with blue icon"""
    logging.info(f"{Colors.BRIGHT_BLUE}ℹ️  {message}{Colors.RESET}")

def warning(message: str):
    """Log a warning message with yellow icon"""
    logging.warning(f"{Colors.BRIGHT_YELLOW}⚠️  {message}{Colors.RESET}")

def error(message: str):
    """Log an error message with red X"""
    logging.error(f"{Colors.BRIGHT_RED}❌ {message}{Colors.RESET}")

def step(message: str):
    """Log a step in a process with arrow"""
    logging.info(f"{Colors.BRIGHT_CYAN}🚀 {message}{Colors.RESET}")

def progress(message: str, current: int, total: int):
    """Log progress with percentage"""
    percentage = (current / total) * 100
    bar_length = 20
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    logging.info(f"{Colors.BRIGHT_MAGENTA}📊 {message} [{bar}] {percentage:.1f}% ({current}/{total}){Colors.RESET}")

def header(message: str):
    """Log a section header"""
    separator = "=" * len(message)
    logging.info(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}{separator}")
    logging.info(f"{message}")
    logging.info(f"{separator}{Colors.RESET}\n")
