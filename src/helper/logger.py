"""A generic automatic logging module for tracking application operations.

This module provides a flexible logging system that can be used across different projects
and applications. It automatically captures function names for better traceability and
supports both file and console logging with configurable log levels.

Features:
- Daily log file rotation
- Automatic function name detection
- Configurable log levels and formats
- Support for both file and console output
- Exception tracking with stack traces
"""
import logging
import inspect
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Optional, Callable, Any, Union

class AutoLogger:
    """A generic logger class with automatic context detection.
    
    This class provides structured logging capabilities with automatic function name detection,
    daily log file rotation, and support for both file and console output. It's designed to be
    reusable across different projects and applications.

    Features:
        - Automatic function name detection in log messages
        - Daily log file rotation
        - Configurable log levels and formats
        - Support for both file and console output
        - Exception tracking with stack traces
        
    Example:
        >>> logger = AutoLogger("my_project")
        >>> logger.info("Starting process")  # Will include calling function name
        >>> logger.log_step("Important milestone reached")
        >>> try:
        ...     raise ValueError("Something went wrong")
        ... except Exception as e:
        ...     logger.exception("Error occurred")
    """
    
    logger: logging.Logger
    
    def __init__(self, 
                 name: str = "app",
                 log_dir: str = "logs",
                 log_level: int = logging.DEBUG) -> None:
        """Initialize the logger.
        
        Args:
            name: Name of the logger and prefix for log files
            log_dir: Directory to store log files, will be created if it doesn't exist
            log_level: Minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Create log directory if it doesn't exist
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(exist_ok=True)
        
        # Create log file with daily timestamp
        timestamp: str = datetime.now().strftime('%Y%m%d')
        log_file: Path = log_dir_path / f"{name}_{timestamp}.log"
        
        # File handler with detailed format
        file_handler: logging.FileHandler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter: logging.Formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler with simpler format
        console_handler: logging.StreamHandler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter: logging.Formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Add handlers if they haven't been added already
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def _get_caller_name(self) -> str:
        """Get the name of the calling function.
        
        Returns:
            str: Name of the calling function
        """
        frame: Optional[FrameType] = inspect.currentframe()
        if frame is None:
            return "unknown"
        
        caller_frame: Optional[FrameType] = frame.f_back
        if caller_frame is None:
            return "unknown"
        
        # Go back one more frame as this is called from a logging method
        caller_frame = caller_frame.f_back
        if caller_frame is None:
            return "unknown"
        
        # Get function name
        func_name: str = caller_frame.f_code.co_name
        
        # Clean up
        del frame
        del caller_frame
        
        return func_name

    def _log_with_caller(self, 
                        level: Callable[..., None], 
                        message: str, 
                        *args: Any, 
                        **kwargs: Any) -> None:
        """Log a message with the caller's function name automatically added.
        
        Args:
            level: Logging function to use
            message: Message to log
            *args: Additional positional arguments for the logging function
            **kwargs: Additional keyword arguments for the logging function
        """
        caller: str = self._get_caller_name()
        level(f"[{caller}] {message}", *args, **kwargs)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message with automatic function name detection.
        
        Args:
            message: Message to log
            *args: Additional positional arguments for the logging function
            **kwargs: Additional keyword arguments for the logging function
        """
        self._log_with_caller(self.logger.debug, message, *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an info message with automatic function name detection.
        
        Args:
            message: Message to log
            *args: Additional positional arguments for the logging function
            **kwargs: Additional keyword arguments for the logging function
        """
        self._log_with_caller(self.logger.info, message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message with automatic function name detection.
        
        Args:
            message: Message to log
            *args: Additional positional arguments for the logging function
            **kwargs: Additional keyword arguments for the logging function
        """
        self._log_with_caller(self.logger.warning, message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message with automatic function name detection.
        
        Args:
            message: Message to log
            *args: Additional positional arguments for the logging function
            **kwargs: Additional keyword arguments for the logging function
        """
        self._log_with_caller(self.logger.error, message, *args, **kwargs)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a critical message with automatic function name detection.
        
        Args:
            message: Message to log
            *args: Additional positional arguments for the logging function
            **kwargs: Additional keyword arguments for the logging function
        """
        self._log_with_caller(self.logger.critical, message, *args, **kwargs)

    def log_step(self, message: str) -> None:
        """Log a workflow step with automatic function name detection.
        
        Args:
            message: Message to log
        """
        self._log_with_caller(self.logger.info, message)

    def exception(self, message: str, *args: Any, exc_info: bool = True, **kwargs: Any) -> None:
        """Log an exception with automatic function name detection.
        
        Args:
            message: Message to log
            *args: Additional positional arguments for the logging function
            exc_info: Whether to include exception info, defaults to True
            **kwargs: Additional keyword arguments for the logging function
        """
        self._log_with_caller(self.logger.error, message, *args, exc_info=exc_info, **kwargs)

_default_logger: Optional[AutoLogger] = None

def get_logger(name: str = "app") -> AutoLogger:
    """Get or create a logger instance.
    
    Args:
        name: Name of the logger
        
    Returns:
        AutoLogger instance
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = AutoLogger(name=name)
    return _default_logger
