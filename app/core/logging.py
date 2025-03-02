# app/core/logging.py

import logging
import sys
from pathlib import Path
from datetime import datetime
import os

class TaskLogger:
    """Custom logger for video generation tasks"""

    def __init__(self, task_id, debug_mode=False):
        self.task_id = task_id
        self.debug_mode = debug_mode
        self.logs = []
        self.progress = 0
        self.status = "initializing"
        self.current_step = 0
        self.total_steps = 0
        
        # Set up file logging
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = logs_dir / f"task_{task_id}_{timestamp}.log"
        
        # Configure logger
        self.logger = logging.getLogger(f"task_{task_id}")
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        
        # Clear any existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(file_format)
        self.logger.addHandler(console_handler)
        
    def info(self, message):
        """Log info message and add to log history"""
        self.logger.info(message)
        self.logs.append({"level": "info", "message": message, "time": datetime.now().isoformat()})
        return message
        
    def debug(self, message):
        """Log debug message and add to log history if debug mode is on"""
        self.logger.debug(message)
        if self.debug_mode:  # 使用 debug_mode 而不是 debug
            self.logs.append({"level": "debug", "message": message, "time": datetime.now().isoformat()})
        return message
    
    def warning(self, message):
        """Log warning message and add to log history"""
        self.logger.warning(message)
        self.logs.append({"level": "warning", "message": message, "time": datetime.now().isoformat()})
        return message
        
    def error(self, message):
        """Log error message and add to log history"""
        self.logger.error(message)
        self.logs.append({"level": "error", "message": message, "time": datetime.now().isoformat()})
        self.status = "error"
        return message
    
    def update_progress(self, step, total_steps):
        """Update progress of the task"""
        self.current_step = step
        self.total_steps = total_steps
        self.progress = int((step / total_steps) * 100) if total_steps > 0 else 0
        self.logger.info(f"Progress: {self.progress}% (Step {step}/{total_steps})")
        return self.progress
    
    def update_status(self, status):
        """Update task status"""
        self.status = status
        self.logger.info(f"Status updated: {status}")
        return status
    
    def get_logs(self, max_entries=None):
        """Get log history, optionally limited to last N entries"""
        if max_entries:
            return self.logs[-max_entries:]
        return self.logs
    
    def get_task_info(self):
        """Get current task information"""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "progress": self.progress,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "logs": self.logs[-10:]  # Return the last 10 log entries
        }

def setup_app_logger(debug=False):
    """Set up application-wide logger"""
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Generate timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = f"logs/app_{timestamp}.log"
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    
    # Create logger instance
    logger = logging.getLogger("videoGenPlatform")
    logger.setLevel(log_level)
    
    # Add a rotating file handler
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Log startup information
    logger.info(f"Starting application with debug={debug}")
    logger.info(f"Log file: {log_file}")
    
    return logger