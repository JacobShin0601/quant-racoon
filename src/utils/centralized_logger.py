#!/usr/bin/env python3
"""
Centralized logging module for Quant Racoon
Provides unified logging format and configuration for all components
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any
import json
from pathlib import Path
import colorlog


class QuantRacoonLogger:
    """Centralized logger for Quant Racoon system"""
    
    # ANSI color codes for terminal output
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'STEP': '\033[34m',       # Blue
        'SUCCESS': '\033[92m',    # Bright Green
        'RESET': '\033[0m'
    }
    
    # Emoji indicators for different log levels
    EMOJIS = {
        'DEBUG': '🔍',
        'INFO': 'ℹ️ ',
        'WARNING': '⚠️ ',
        'ERROR': '❌',
        'CRITICAL': '🚨',
        'STEP': '📌',
        'SUCCESS': '✅',
        'START': '🚀',
        'COMPLETE': '🎉',
        'PROCESS': '⚙️ ',
        'DATA': '📊',
        'MODEL': '🧠',
        'TRADE': '💹',
        'PORTFOLIO': '💼'
    }
    
    def __init__(self, name: str, time_horizon: str = "trader", 
                 log_level: str = "INFO", enable_file_logging: bool = True,
                 enable_console_logging: bool = True):
        """
        Initialize centralized logger
        
        Args:
            name: Logger name (usually module name)
            time_horizon: Time horizon for log directory
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_file_logging: Whether to log to file
            enable_console_logging: Whether to log to console
        """
        self.name = name
        self.time_horizon = time_horizon
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        self.logger.handlers.clear()  # Clear existing handlers
        
        # Add step level
        logging.STEP = 25
        logging.addLevelName(logging.STEP, 'STEP')
        
        # Setup file logging
        if enable_file_logging:
            self._setup_file_handler()
        
        # Setup console logging
        if enable_console_logging:
            self._setup_console_handler()
    
    def _setup_file_handler(self):
        """Setup file handler with JSON formatting"""
        # Create log directory
        log_dir = Path(f"log/{self.time_horizon}")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{self.name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.log_level)
        
        # JSON formatter for file logs
        file_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"module": "%(name)s", "message": "%(message)s", '
            '"filename": "%(filename)s", "line": %(lineno)d}'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def _setup_console_handler(self):
        """Setup console handler with colored output"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        
        # Use colorlog for colored console output
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s[%(levelname)-8s]%(reset)s %(asctime)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
                'STEP': 'blue'
            }
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def step(self, message: str, **kwargs):
        """Log step message (custom level)"""
        self._log_with_context(logging.STEP, message, **kwargs)
    
    def success(self, message: str, **kwargs):
        """Log success message"""
        emoji = self.EMOJIS.get('SUCCESS', '')
        self.info(f"{emoji} {message}", **kwargs)
    
    def start(self, process: str, **kwargs):
        """Log process start"""
        emoji = self.EMOJIS.get('START', '')
        self.info(f"{emoji} Starting {process}", **kwargs)
    
    def complete(self, process: str, **kwargs):
        """Log process completion"""
        emoji = self.EMOJIS.get('COMPLETE', '')
        self.info(f"{emoji} Completed {process}", **kwargs)
    
    def data_info(self, message: str, **kwargs):
        """Log data-related info"""
        emoji = self.EMOJIS.get('DATA', '')
        self.info(f"{emoji} {message}", **kwargs)
    
    def model_info(self, message: str, **kwargs):
        """Log model-related info"""
        emoji = self.EMOJIS.get('MODEL', '')
        self.info(f"{emoji} {message}", **kwargs)
    
    def trade_info(self, message: str, **kwargs):
        """Log trade-related info"""
        emoji = self.EMOJIS.get('TRADE', '')
        self.info(f"{emoji} {message}", **kwargs)
    
    def portfolio_info(self, message: str, **kwargs):
        """Log portfolio-related info"""
        emoji = self.EMOJIS.get('PORTFOLIO', '')
        self.info(f"{emoji} {message}", **kwargs)
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with additional context"""
        # Add extra context to log record
        extra = {}
        if kwargs:
            extra['context'] = kwargs
        
        self.logger.log(level, message, extra=extra)
    
    def log_metrics(self, metrics: Dict[str, Any], prefix: str = "Metrics"):
        """Log metrics in a structured format"""
        self.info(f"{prefix}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")
    
    def log_table(self, data: list, headers: list = None):
        """Log data in table format"""
        if not data:
            return
        
        # Calculate column widths
        if headers:
            widths = [len(str(h)) for h in headers]
            for row in data:
                for i, cell in enumerate(row):
                    widths[i] = max(widths[i], len(str(cell)))
        else:
            widths = [max(len(str(cell)) for cell in col) for col in zip(*data)]
        
        # Print header
        if headers:
            header_str = " | ".join(str(h).ljust(w) for h, w in zip(headers, widths))
            self.info(header_str)
            self.info("-" * len(header_str))
        
        # Print rows
        for row in data:
            row_str = " | ".join(str(cell).ljust(w) for cell, w in zip(row, widths))
            self.info(row_str)


def get_logger(name: str, config_path: Optional[str] = None, **kwargs) -> QuantRacoonLogger:
    """
    Get or create a logger instance
    
    Args:
        name: Logger name
        config_path: Path to config file (optional)
        **kwargs: Additional logger parameters
    
    Returns:
        QuantRacoonLogger instance
    """
    # Load config if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Extract logging config
        log_config = config.get('logging', {})
        kwargs.update({
            'time_horizon': config.get('time_horizon', 'trader'),
            'log_level': log_config.get('level', 'INFO'),
            'enable_file_logging': log_config.get('enable_file', True),
            'enable_console_logging': log_config.get('enable_console', True)
        })
    
    return QuantRacoonLogger(name, **kwargs)


# Bash script logger helper
class BashLogger:
    """Helper class for consistent bash script logging"""
    
    @staticmethod
    def log(message: str, level: str = "INFO"):
        """Print log message in bash script format"""
        colors = {
            'INFO': '\033[0;32m',
            'WARN': '\033[1;33m',
            'ERROR': '\033[0;31m',
            'STEP': '\033[0;34m',
            'SUCCESS': '\033[0;92m'
        }
        reset = '\033[0m'
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color = colors.get(level, colors['INFO'])
        
        print(f"{color}[{level:<8}]{reset} {timestamp} - {message}")
    
    @staticmethod
    def info(message: str):
        BashLogger.log(message, "INFO")
    
    @staticmethod
    def warn(message: str):
        BashLogger.log(message, "WARN")
    
    @staticmethod
    def error(message: str):
        BashLogger.log(message, "ERROR")
    
    @staticmethod
    def step(message: str):
        BashLogger.log(message, "STEP")
    
    @staticmethod
    def success(message: str):
        BashLogger.log(message, "SUCCESS")


# Example usage
if __name__ == "__main__":
    # Example 1: Basic usage
    logger = get_logger("test_module", time_horizon="trader")
    
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    logger.step("Step 1: Initialize system")
    logger.success("Operation completed successfully")
    
    # Example 2: Context logging
    logger.data_info("Loaded 1000 data points", symbols=["AAPL", "GOOGL"], timeframe="1d")
    logger.model_info("Training neural network", epochs=100, batch_size=32)
    
    # Example 3: Metrics logging
    metrics = {
        "sharpe_ratio": 1.2345,
        "max_drawdown": -0.1523,
        "total_return": 0.2567
    }
    logger.log_metrics(metrics, "Backtest Results")
    
    # Example 4: Table logging
    data = [
        ["AAPL", 150.25, 0.0234],
        ["GOOGL", 2750.50, -0.0156],
        ["MSFT", 325.75, 0.0089]
    ]
    logger.log_table(data, headers=["Symbol", "Price", "Return"])