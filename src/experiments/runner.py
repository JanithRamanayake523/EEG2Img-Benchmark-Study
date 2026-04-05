"""
Experiment runner with logging and tracking.

Runs complete experiments with models, data, and configurations.
Handles training, evaluation, and result logging.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, TensorDataset

from .config import ExperimentConfig, ModelConfig, OptimizerConfig
from src.models import get_model
from src.training import Trainer, create_data_loaders, EEGDataset
from src.evaluation import compute_metrics, compute_confusion_matrix, MetricsTracker


class ExperimentLogger:
    """Logger for experiment runs."""

    def __init__(self, log_dir: Path, name: str = 'experiment'):
        """
        Initialize logger.

        Args:
            log_dir: Directory to save logs
            name: Name for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # File handler
        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.log_file = log_file

    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)

    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)

    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)


class ExperimentRunner:
    """Run complete experiments."""

    def __init__(self, config: ExperimentConfig, output_dir: Optional[Path] = None):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration
            output_dir: Output directory for results (uses config.save_dir if None)
        """
        self.config = config
        self.output_dir = Path(output_dir or config.save_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = ExperimentLogger(self.output_dir, name=config.name)

        # Results tracking
        self.results = {
            'config': config.to_dict(),
            'models': {},
            'timestamp': datetime.now().isoformat()
        }

        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.logger.info(f"Initialized ExperimentRunner: {config.name}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Device: {config.device}")
        self.logger.info(f"Seed: {config.seed}")

    def run(self, train_data: np.ndarray, train_labels: np.ndarray,
            val_data: Optional[np.ndarray] = None, val_labels: Optional[np.ndarray] = None,
            test_data: Optional[np.ndarray] = None, test_labels: Optional[np.ndarray] = None):
        """
        Run complete experiment.

        Args:
            train_data: Training data, shape (n_samples, channels, height, width)
            train_labels: Training labels, shape (n_samples,)
            val_data: Validation data, optional
            val_labels: Validation labels, optional
            test_data: Test data, optional
            test_labels: Test labels, optional

        Returns:
            Dictionary with all results
        """
        self.logger.info("="*80)
        self.logger.info(f"Starting Experiment: {self.config.name}")
        self.logger.info("="*80)

        # Convert to tensors
        train_data = torch.FloatTensor(train_data)
        train_labels = torch.LongTensor(train_labels)

        if val_data is not None:
            val_data = torch.FloatTensor(val_data)
            val_labels = torch.LongTensor(val_labels)

        if test_data is not None:
            test_data = torch.FloatTensor(test_data)
            test_labels = torch.LongTensor(test_labels)

        # Run for each model
        for model_config in self.config.models:
            self.logger.info(f"\n{'-'*80}")
            self.logger.info(f"Training model: {model_config.name} ({model_config.architecture})")
            self.logger.info(f"{'-'*80}\n")

            try:
                model_results = self._train_and_evaluate(
                    model_config,
                    train_data, train_labels,
                    val_data, val_labels,
                    test_data, test_labels
                )

                self.results['models'][model_config.name] = model_results

                self.logger.info(f"Successfully trained {model_config.name}")
                self.logger.info(f"Final test accuracy: {model_results.get('test_metrics', {}).get('accuracy', 'N/A'):.4f}")

            except Exception as e:
                self.logger.error(f"Failed to train {model_config.name}: {e}", exc_info=True)
                continue

        # Save results
        self._save_results()

        self.logger.info("\n" + "="*80)
        self.logger.info("Experiment completed successfully")
        self.logger.info("="*80)

        return self.results

    def _train_and_evaluate(self,
                           model_config: ModelConfig,
                           train_data: torch.Tensor,
                           train_labels: torch.Tensor,
                           val_data: Optional[torch.Tensor] = None,
                           val_labels: Optional[torch.Tensor] = None,
                           test_data: Optional[torch.Tensor] = None,
                           test_labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Train and evaluate a single model.

        Args:
            model_config: Model configuration
            train_data: Training data
            train_labels: Training labels
            val_data: Validation data, optional
            val_labels: Validation labels, optional
            test_data: Test data, optional
            test_labels: Test labels, optional

        Returns:
            Dictionary with model results
        """
        results = {}

        # Create model
        self.logger.info(f"Creating model: {model_config.architecture}")
        model = get_model(
            model_config.architecture,
            num_classes=model_config.num_classes,
            in_channels=model_config.in_channels,
            pretrained=model_config.pretrained
        )
        model = model.to(self.config.device)

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Model parameters: {n_params:,.0f}")

        # Create data loaders
        train_dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0
        )

        val_loader = None
        if val_data is not None:
            val_dataset = TensorDataset(val_data, val_labels)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=0
            )

        # Create optimizer
        optimizer = self._create_optimizer(model, model_config)

        # Create trainer
        trainer = Trainer(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            device=self.config.device,
            mixed_precision=self.config.training.mixed_precision,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps
        )

        # Train
        self.logger.info(f"Training for {self.config.training.epochs} epochs")
        history = trainer.fit(
            train_loader,
            val_loader=val_loader,
            epochs=self.config.training.epochs
        )

        results['history'] = history

        # Evaluate on test set if available
        if test_data is not None:
            self.logger.info("Evaluating on test set...")
            test_dataset = TensorDataset(test_data, test_labels)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=0
            )

            # Get predictions
            all_preds = []
            all_probs = []
            all_labels = []

            model.eval()
            with torch.no_grad():
                for data, labels in test_loader:
                    data = data.to(self.config.device)
                    outputs = model(data)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    preds = outputs.argmax(dim=1).cpu().numpy()

                    all_preds.append(preds)
                    all_probs.append(probs)
                    all_labels.append(labels.numpy())

            all_preds = np.concatenate(all_preds)
            all_probs = np.concatenate(all_probs)
            all_labels = np.concatenate(all_labels)

            # Compute metrics
            test_metrics = compute_metrics(all_labels, all_preds, all_probs)
            cm = compute_confusion_matrix(all_labels, all_preds, normalize='true')

            results['test_metrics'] = test_metrics
            results['confusion_matrix'] = cm.tolist()

            self.logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            self.logger.info(f"Test F1: {test_metrics['f1']:.4f}")
            self.logger.info(f"Test AUC: {test_metrics['auc']:.4f}")

        # Save model
        model_path = self.output_dir / f"model_{model_config.name}.pt"
        torch.save(model.state_dict(), model_path)
        self.logger.info(f"Model saved to {model_path}")
        results['model_path'] = str(model_path)

        return results

    def _create_optimizer(self, model: nn.Module, model_config: ModelConfig):
        """
        Create optimizer.

        Args:
            model: PyTorch model
            model_config: Model configuration

        Returns:
            Optimizer instance
        """
        optimizer_config = self.config.optimizer or OptimizerConfig(name='adam')

        if optimizer_config.name.lower() == 'adam':
            optimizer = Adam(
                model.parameters(),
                lr=optimizer_config.learning_rate,
                weight_decay=optimizer_config.weight_decay
            )
        elif optimizer_config.name.lower() == 'sgd':
            optimizer = SGD(
                model.parameters(),
                lr=optimizer_config.learning_rate,
                momentum=optimizer_config.momentum,
                weight_decay=optimizer_config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config.name}")

        return optimizer

    def _save_results(self):
        """Save results to file."""
        results_file = self.output_dir / 'results.json'

        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        self.logger.info(f"Results saved to {results_file}")

    def __repr__(self) -> str:
        """String representation."""
        return f"ExperimentRunner(name={self.config.name}, models={len(self.config.models)})"
