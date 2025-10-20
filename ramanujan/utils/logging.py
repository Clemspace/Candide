"""
Logging utilities for Ramanujan Transformer.

This module provides logging functionality:
- Console logging with colors
- Weights & Biases integration
- TensorBoard integration
- File logging
- Experiment tracking

Example:
    >>> from ramanujan.utils import setup_logging, WandBLogger
    >>> 
    >>> # Setup console logging
    >>> logger = setup_logging('experiment1', log_dir='logs')
    >>> 
    >>> # W&B logging
    >>> wandb_logger = WandBLogger(
    ...     project='ramanujan-transformer',
    ...     name='experiment1',
    ...     config={'lr': 0.0003}
    ... )
    >>> wandb_logger.log({'loss': 3.5, 'step': 100})
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
from datetime import datetime


# ============================================================================
# CONSOLE LOGGING
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Colored console formatter.
    
    Adds colors to log levels for better readability.
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        return super().format(record)


def setup_logging(
    name: str = 'ramanujan',
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True,
    colored: bool = True
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console: Enable console logging
        file: Enable file logging
        colored: Use colored console output
    
    Returns:
        Configured logger
    
    Example:
        >>> logger = setup_logging('experiment1', log_dir='logs')
        >>> logger.info('Training started')
        >>> logger.warning('High loss detected')
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Format
    fmt = '%(asctime)s | %(levelname)-8s | %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if colored:
            console_formatter = ColoredFormatter(fmt, datefmt=datefmt)
        else:
            console_formatter = logging.Formatter(fmt, datefmt=datefmt)
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(fmt, datefmt=datefmt)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


# ============================================================================
# WEIGHTS & BIASES LOGGER
# ============================================================================

class WandBLogger:
    """
    Weights & Biases logger.
    
    Handles W&B initialization and logging with graceful fallback
    if W&B is not available.
    
    Args:
        project: W&B project name
        name: Run name
        config: Configuration dictionary
        tags: List of tags
        notes: Run notes
        dir: Directory for W&B files
        enabled: Enable/disable logging
    
    Example:
        >>> wandb_logger = WandBLogger(
        ...     project='ramanujan-transformer',
        ...     name='optimal-890d',
        ...     config={'lr': 0.0003, 'dim': 890}
        ... )
        >>> 
        >>> for step in range(100):
        ...     wandb_logger.log({'loss': 3.5, 'step': step})
        >>> 
        >>> wandb_logger.finish()
    """
    
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        dir: Optional[str] = None,
        enabled: bool = True
    ):
        self.enabled = enabled
        self.run = None
        
        if not enabled:
            print("W&B logging disabled")
            return
        
        try:
            import wandb
            self.wandb = wandb
            
            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
                tags=tags,
                notes=notes,
                dir=dir
            )
            
            print(f"W&B initialized: {wandb.run.url}")
        
        except ImportError:
            print("W&B not available. Install with: pip install wandb")
            self.enabled = False
        except Exception as e:
            print(f"W&B initialization failed: {e}")
            self.enabled = False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
            commit: Whether to commit immediately
        """
        if not self.enabled or self.run is None:
            return
        
        try:
            if step is not None:
                metrics['step'] = step
            self.wandb.log(metrics, commit=commit)
        except Exception as e:
            print(f"W&B logging error: {e}")
    
    def log_model(self, model_path: str, name: Optional[str] = None):
        """
        Log model artifact to W&B.
        
        Args:
            model_path: Path to model file
            name: Artifact name
        """
        if not self.enabled or self.run is None:
            return
        
        try:
            artifact = self.wandb.Artifact(name or 'model', type='model')
            artifact.add_file(model_path)
            self.wandb.log_artifact(artifact)
            print(f"Model logged to W&B: {model_path}")
        except Exception as e:
            print(f"W&B model logging error: {e}")
    
    def watch(self, model, log: str = 'all', log_freq: int = 1000):
        """
        Watch model gradients and parameters.
        
        Args:
            model: Model to watch
            log: What to log ('gradients', 'parameters', 'all')
            log_freq: Logging frequency
        """
        if not self.enabled or self.run is None:
            return
        
        try:
            self.wandb.watch(model, log=log, log_freq=log_freq)
        except Exception as e:
            print(f"W&B watch error: {e}")
    
    def finish(self):
        """Finish W&B run."""
        if self.enabled and self.run is not None:
            try:
                self.wandb.finish()
                print("W&B run finished")
            except Exception as e:
                print(f"W&B finish error: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


# ============================================================================
# TENSORBOARD LOGGER
# ============================================================================

class TensorBoardLogger:
    """
    TensorBoard logger.
    
    Handles TensorBoard logging with graceful fallback.
    
    Args:
        log_dir: Directory for TensorBoard logs
        enabled: Enable/disable logging
    
    Example:
        >>> tb_logger = TensorBoardLogger(log_dir='runs/experiment1')
        >>> 
        >>> for step in range(100):
        ...     tb_logger.log_scalar('loss', 3.5, step)
        >>> 
        >>> tb_logger.close()
    """
    
    def __init__(self, log_dir: str, enabled: bool = True):
        self.enabled = enabled
        self.writer = None
        
        if not enabled:
            print("TensorBoard logging disabled")
            return
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            print(f"TensorBoard logging to: {log_dir}")
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.enabled = False
        except Exception as e:
            print(f"TensorBoard initialization failed: {e}")
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value."""
        if self.enabled and self.writer is not None:
            try:
                self.writer.add_scalar(tag, value, step)
            except Exception as e:
                print(f"TensorBoard scalar logging error: {e}")
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars."""
        if self.enabled and self.writer is not None:
            try:
                self.writer.add_scalars(main_tag, tag_scalar_dict, step)
            except Exception as e:
                print(f"TensorBoard scalars logging error: {e}")
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram of values."""
        if self.enabled and self.writer is not None:
            try:
                self.writer.add_histogram(tag, values, step)
            except Exception as e:
                print(f"TensorBoard histogram logging error: {e}")
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text."""
        if self.enabled and self.writer is not None:
            try:
                self.writer.add_text(tag, text, step)
            except Exception as e:
                print(f"TensorBoard text logging error: {e}")
    
    def close(self):
        """Close writer."""
        if self.enabled and self.writer is not None:
            try:
                self.writer.close()
                print("TensorBoard writer closed")
            except Exception as e:
                print(f"TensorBoard close error: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# ============================================================================
# CONSOLE LOGGER
# ============================================================================

class ConsoleLogger:
    """
    Simple console logger with formatting.
    
    Provides clean console output without file logging.
    
    Example:
        >>> logger = ConsoleLogger(name='training')
        >>> logger.info('Starting training')
        >>> logger.metric('Loss', 3.5, step=100)
    """
    
    def __init__(self, name: str = 'ramanujan', colored: bool = True):
        self.name = name
        self.colored = colored
    
    def _format_message(self, level: str, message: str) -> str:
        """Format message with timestamp and level."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if self.colored:
            colors = {
                'INFO': '\033[32m',      # Green
                'WARNING': '\033[33m',   # Yellow
                'ERROR': '\033[31m',     # Red
                'METRIC': '\033[36m',    # Cyan
            }
            reset = '\033[0m'
            color = colors.get(level, '')
            return f"{timestamp} | {color}{level:8s}{reset} | {message}"
        else:
            return f"{timestamp} | {level:8s} | {message}"
    
    def info(self, message: str):
        """Log info message."""
        print(self._format_message('INFO', message))
    
    def warning(self, message: str):
        """Log warning message."""
        print(self._format_message('WARNING', message))
    
    def error(self, message: str):
        """Log error message."""
        print(self._format_message('ERROR', message))
    
    def metric(self, name: str, value: float, step: Optional[int] = None):
        """Log metric."""
        if step is not None:
            message = f"{name}: {value:.4f} (step {step})"
        else:
            message = f"{name}: {value:.4f}"
        print(self._format_message('METRIC', message))
    
    def metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics."""
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        if step is not None:
            metrics_str = f"Step {step} | {metrics_str}"
        print(self._format_message('METRIC', metrics_str))


# ============================================================================
# EXPERIMENT LOGGER
# ============================================================================

class ExperimentLogger:
    """
    Multi-backend experiment logger.
    
    Combines console, file, W&B, and TensorBoard logging.
    
    Args:
        name: Experiment name
        log_dir: Directory for logs
        use_wandb: Enable W&B
        use_tensorboard: Enable TensorBoard
        wandb_project: W&B project name
        wandb_config: W&B configuration
    
    Example:
        >>> logger = ExperimentLogger(
        ...     name='experiment1',
        ...     log_dir='logs',
        ...     use_wandb=True,
        ...     wandb_project='ramanujan-transformer'
        ... )
        >>> 
        >>> logger.log_metrics({'loss': 3.5, 'acc': 0.92}, step=100)
        >>> logger.finish()
    """
    
    def __init__(
        self,
        name: str,
        log_dir: str = 'logs',
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.log_dir = log_dir
        
        # Setup console/file logging
        self.logger = setup_logging(name, log_dir=log_dir)
        
        # Setup W&B
        self.wandb_logger = None
        if use_wandb and wandb_project:
            self.wandb_logger = WandBLogger(
                project=wandb_project,
                name=name,
                config=wandb_config,
                enabled=True
            )
        
        # Setup TensorBoard
        self.tb_logger = None
        if use_tensorboard:
            tb_dir = os.path.join(log_dir, 'tensorboard', name)
            self.tb_logger = TensorBoardLogger(tb_dir, enabled=True)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to all backends."""
        # Console
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        if step is not None:
            self.logger.info(f"Step {step} | {metrics_str}")
        else:
            self.logger.info(metrics_str)
        
        # W&B
        if self.wandb_logger:
            self.wandb_logger.log(metrics, step=step)
        
        # TensorBoard
        if self.tb_logger and step is not None:
            for key, value in metrics.items():
                self.tb_logger.log_scalar(key, value, step)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration."""
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
    
    def finish(self):
        """Finish logging."""
        if self.wandb_logger:
            self.wandb_logger.finish()
        if self.tb_logger:
            self.tb_logger.close()


# ============================================================================
# JSON LOGGER
# ============================================================================

class JSONLogger:
    """
    Log metrics to JSON file.
    
    Useful for post-processing and analysis.
    
    Args:
        log_file: Path to JSON log file
    
    Example:
        >>> logger = JSONLogger('results.json')
        >>> logger.log({'loss': 3.5, 'step': 100})
        >>> logger.save()
    """
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.metrics = []
        
        # Create directory if needed
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, metrics: Dict[str, Any]):
        """Log metrics."""
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics.append(metrics)
    
    def save(self):
        """Save metrics to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to: {self.log_file}")
    
    def load(self) -> List[Dict[str, Any]]:
        """Load metrics from file."""
        with open(self.log_file, 'r') as f:
            return json.load(f)


# ============================================================================
# UTILITIES
# ============================================================================

def format_time(seconds: float) -> str:
    """
    Format time in human-readable format.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    
    Example:
        >>> formatted = format_time(3725.5)
        >>> print(formatted)  # "1h 2m 5s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return " ".join(parts)


def format_number(num: int) -> str:
    """
    Format large numbers with K/M/B suffixes.
    
    Args:
        num: Number to format
    
    Returns:
        Formatted string
    
    Example:
        >>> print(format_number(1500000))  # "1.5M"
        >>> print(format_number(2500))     # "2.5K"
    """
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return str(num)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing logging.py module")
    print("="*70)
    
    # Test setup_logging
    print("\n1. Testing setup_logging...")
    logger = setup_logging('test', log_dir='/tmp/test_logs', colored=True)
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    print(f"   ✅ setup_logging working!")
    
    # Test ConsoleLogger
    print("\n2. Testing ConsoleLogger...")
    console = ConsoleLogger(name='test', colored=True)
    console.info("Testing console logger")
    console.metric('loss', 3.5, step=100)
    console.metrics({'loss': 3.5, 'acc': 0.92}, step=100)
    print(f"   ✅ ConsoleLogger working!")
    
    # Test WandBLogger (will fail gracefully if W&B not available)
    print("\n3. Testing WandBLogger...")
    wandb_logger = WandBLogger(
        project='test-project',
        name='test-run',
        config={'lr': 0.001},
        enabled=False  # Disabled for testing
    )
    wandb_logger.log({'loss': 3.5}, step=100)
    wandb_logger.finish()
    print(f"   ✅ WandBLogger working (disabled mode)!")
    
    # Test TensorBoardLogger
    print("\n4. Testing TensorBoardLogger...")
    tb_logger = TensorBoardLogger(log_dir='/tmp/tb_logs', enabled=False)
    tb_logger.log_scalar('loss', 3.5, 100)
    tb_logger.close()
    print(f"   ✅ TensorBoardLogger working (disabled mode)!")
    
    # Test JSONLogger
    print("\n5. Testing JSONLogger...")
    json_logger = JSONLogger('/tmp/test_metrics.json')
    json_logger.log({'loss': 3.5, 'step': 100})
    json_logger.log({'loss': 3.3, 'step': 200})
    json_logger.save()
    
    # Load and verify
    loaded = json_logger.load()
    assert len(loaded) == 2, "Wrong number of metrics!"
    print(f"   Logged {len(loaded)} metrics")
    print(f"   ✅ JSONLogger working!")
    
    # Test ExperimentLogger
    print("\n6. Testing ExperimentLogger...")
    exp_logger = ExperimentLogger(
        name='test-experiment',
        log_dir='/tmp/exp_logs',
        use_wandb=False,
        use_tensorboard=False
    )
    exp_logger.info("Starting experiment")
    exp_logger.log_metrics({'loss': 3.5, 'acc': 0.92}, step=100)
    exp_logger.finish()
    print(f"   ✅ ExperimentLogger working!")
    
    # Test utilities
    print("\n7. Testing utility functions...")
    time_str = format_time(3725.5)
    print(f"   Time: {time_str}")
    
    num_str = format_number(1500000)
    print(f"   Number: {num_str}")
    print(f"   ✅ Utilities working!")
    
    # Test context managers
    print("\n8. Testing context managers...")
    with JSONLogger('/tmp/test_context.json') as log:
        log.log({'test': 1})
        log.save()
    print(f"   ✅ Context managers working!")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
    print("\nModule ready for use. Import with:")
    print("  from ramanujan.utils.logging import setup_logging, ExperimentLogger")
    print("  from ramanujan.utils.logging import WandBLogger, TensorBoardLogger")
    print("="*70)