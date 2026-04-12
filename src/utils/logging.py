"""
Logging utilities for Phase 3 experiments.

Provides centralized logging configuration for all experiment scripts.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Setup a logger with optional file output.

    Args:
        name: Logger name (usually __name__ or script name)
        log_file: Optional path to log file
        level: Logging level (default: INFO)

    Returns:
        logging.Logger instance

    Example:
        >>> logger = setup_logger('my_experiment', 'logs/experiment.log')
        >>> logger.info('Starting experiment...')
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler (always)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_config(logger, config_dict):
    """
    Log configuration dictionary in a formatted way.

    Args:
        logger: Logger instance
        config_dict: Dictionary of configuration parameters
    """
    logger.info("=" * 60)
    logger.info("CONFIGURATION")
    logger.info("=" * 60)

    for key, value in config_dict.items():
        logger.info(f"{key}: {value}")

    logger.info("=" * 60)


def log_results(logger, results_dict):
    """
    Log results dictionary in a formatted way.

    Args:
        logger: Logger instance
        results_dict: Dictionary of results
    """
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    for key, value in results_dict.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")

    logger.info("=" * 60)


def create_experiment_logger(experiment_name, output_dir):
    """
    Create a logger for an experiment with timestamp.

    Args:
        experiment_name: Name of experiment
        output_dir: Directory to save logs

    Returns:
        logging.Logger instance
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path(output_dir) / f"{experiment_name}_{timestamp}.log"

    logger = setup_logger(
        experiment_name,
        log_file=log_file,
        level=logging.INFO
    )

    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")

    return logger


__all__ = [
    'setup_logger',
    'log_config',
    'log_results',
    'create_experiment_logger',
]
