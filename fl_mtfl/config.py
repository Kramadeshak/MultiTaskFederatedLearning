"""
Configuration settings for the federated learning application.

This module provides a centralized configuration system for federated learning
experiments with FedAvg and MTFL approaches.
"""

import os
import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fl_mtfl.config")


@dataclass
class FederatedConfig:
    """Configuration for federated learning experiments.
    
    This dataclass holds all configurable parameters for federated learning experiments.
    It provides methods to load/save configurations and update settings for different
    experiment types.
    """
    
    # Experiment identification
    experiment_name: str = "fedavg"
    run_id: str = field(default_factory=lambda: f"{os.getpid()}")
    
    # Federated Learning parameters
    weights_dir: str = "./data/weights"
    init_weights_path: str = field(init=False)
    partitions_dir: str = "./data/partitions"
    num_clients: int = 20
    client_fraction: float = 0.1  # Client participation rate per round
    num_rounds: int = 10
    local_epochs: int = 1
    target_accuracy: Optional[float] = None  # Target user accuracy for termination
    
    # Dataset parameters
    dataset_name: str = "cifar10"
    batch_size: int = 32
    test_size: float = 0.2
    shards_per_client: int = 2
    
    # Model parameters
    use_adam: bool = False  # Use Adam optimizer (for MTFL)
    use_private_bn: bool = False  # Use private BatchNorm layers (for MTFL)
    bn_private_type: str = "none"  # Options: "none", "gamma_beta", "mu_sigma", "all"
    learning_rate: float = 0.01
    momentum: float = 0.9  # For SGD optimizer
    weight_decay: float = 1e-4
    beta1: float = 0.9  # For Adam optimizer
    beta2: float = 0.999  # For Adam optimizer
    epsilon: float = 1e-8  # For Adam optimizer
    
    # Optimizer state handling parameters
    optimizer_state_transmission: str = "none"  # Options: "none", "metrics", "parameters"
    optimizer_state_aggregation: str = "none"  # Options: "none", "average", "weighted_average"
    compress_optimizer_state: bool = False  # Whether to compress optimizer state
    version_compatible: bool = True  # Add backward compatibility for older clients/servers
    
    # System parameters
    seed: int = 42
    save_metrics: bool = True
    metrics_dir: str = "./metrics"
    metrics_file: str = field(default="", init=False)
    figures_dir: str = "./figures"
    server_address: str = "localhost:8080"
    num_workers: int = 0  # Number of workers for data loading (0 = main process only)
    log_wandb: bool = False  # Whether to log metrics to Weights & Biases
    project_name: str = "mtfl-replication"
    experiment_tag: Optional[str] = None
    per_step_metrics: bool = False  # Whether to log metrics after each local step
    steps_per_epoch: int = 10  # Number of steps per local epoch (for Result 2)
    checkpoint_interval: int = 0  # Interval (in rounds) at which to save model checkpoints
    
    # Track progress
    current_round: int = 0
    
    def __post_init__(self):
        """Initialize derived fields after initialization."""
        # Set metrics file based on experiment name
        self.metrics_file = f"{self.experiment_name}_metrics.json"
        self.init_weights_path = os.path.join(self.weights_dir, f"{self.experiment_name}_initial_weights.pth")
        
        # Create necessary directories
        self._ensure_directories()
        
        # Validate optimizer state configuration
        self._validate_optimizer_state_config()
    
    def _validate_optimizer_state_config(self) -> None:
        """Validate optimizer state configuration and set sensible defaults."""
        # Only enable optimizer state transmission for Adam optimizer
        if not self.use_adam and self.optimizer_state_transmission != "none":
            logger.warning("Optimizer state transmission is only relevant for Adam optimizer. Setting to 'none'.")
            self.optimizer_state_transmission = "none"
            self.optimizer_state_aggregation = "none"
        
        # Ensure sensible defaults for optimizer state aggregation
        if self.optimizer_state_transmission == "none":
            self.optimizer_state_aggregation = "none"
        elif self.optimizer_state_aggregation == "none" and self.optimizer_state_transmission != "none":
            logger.warning("Setting optimizer_state_aggregation to 'average' as transmission is enabled.")
            self.optimizer_state_aggregation = "average"
    
    def _ensure_directories(self) -> None:
        """Ensure all necessary directories exist."""
        directories = [
            self.weights_dir,
            self.metrics_dir,
            self.figures_dir,
            self.partitions_dir,
            "./data"
        ]
        
        for directory in directories:
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
    
    def update_for_experiment(self, experiment_type: str) -> None:
        """Update configuration based on experiment type.
        
        Parameters
        ----------
        experiment_type : str
            Type of experiment to run. Either "fedavg" or "mtfl".
        """
        self.experiment_name = experiment_type.lower()
        self.metrics_file = f"{self.experiment_name}_metrics.json"
        self.init_weights_path = os.path.join(self.weights_dir, f"{self.experiment_name}_initial_weights.pth")
        
        if experiment_type.lower() == "mtfl":
            self.use_private_bn = True
            
            # For MTFL, enable optimizer state transmission by default if using Adam
            if self.use_adam and self.optimizer_state_transmission == "none":
                self.optimizer_state_transmission = "parameters"
                self.optimizer_state_aggregation = "average"
                logger.info(f"Enabled optimizer state transmission for MTFL with Adam optimizer")
            
            logger.info(f"Configured for MTFL experiment: Using private batch norm layers of type {self.bn_private_type}")
        else:  # fedavg
            self.use_private_bn = False
            self.bn_private_type = "none"
            logger.info(f"Configured for FedAvg experiment: Using shared batch norm layers")
    
    def get_metrics_path(self) -> str:
        """Get the full path to the metrics file.
        
        Returns
        -------
        str
            Full path to the metrics file
        """
        return os.path.join(self.metrics_dir, self.metrics_file)
    
    def get_checkpoint_path(self, round_num: Optional[int] = None) -> str:
        """Get the path for model checkpoint.
        
        Parameters
        ----------
        round_num : Optional[int]
            Round number for the checkpoint. If None, uses the current round.
            
        Returns
        -------
        str
            Full path to the checkpoint file
        """
        if round_num is None:
            round_num = self.current_round
            
        return os.path.join(
            self.weights_dir, 
            f"{self.experiment_name}_checkpoint_round_{round_num}.pth"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the configuration.
        """
        return asdict(self)
    
    def save(self, filepath: Optional[str] = None) -> None:
        """Save configuration to a JSON file.
        
        Parameters
        ----------
        filepath : Optional[str]
            Path to save the configuration to. If None, uses the default path.
        """
        # Add timestamp to the configuration
        config_dict = self.to_dict()
        config_dict["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Use timestamped filename if none provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.metrics_dir, f"{self.experiment_name}_config_{timestamp}.json")
            
            # Also save to the default path for backward compatibility
            default_path = os.path.join(self.metrics_dir, f"{self.experiment_name}_config.json")
            os.makedirs(os.path.dirname(default_path), exist_ok=True)
            with open(default_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FederatedConfig':
        """Load configuration from a JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to load the configuration from.
        
        Returns
        -------
        FederatedConfig
            Loaded configuration
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Handle backward compatibility for older config files without optimizer state fields
        if "optimizer_state_transmission" not in config_dict:
            config_dict["optimizer_state_transmission"] = "none"
        if "optimizer_state_aggregation" not in config_dict:
            config_dict["optimizer_state_aggregation"] = "none"
        if "compress_optimizer_state" not in config_dict:
            config_dict["compress_optimizer_state"] = False
        if "version_compatible" not in config_dict:
            config_dict["version_compatible"] = True
        
        logger.info(f"Configuration loaded from {filepath}")
        return cls(**config_dict)
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'FederatedConfig':
        """Create configuration from command-line arguments.
        
        Parameters
        ----------
        args : argparse.Namespace
            Command-line arguments
        
        Returns
        -------
        FederatedConfig
            Configuration based on command-line arguments
        """
        config = cls()
        
        # Map command-line arguments to configuration
        arg_mapping = {
            "exp": "experiment_name",
            "rounds": "num_rounds",
            "local_epochs": "local_epochs",
            "client_fraction": "client_fraction",
            "num_clients": "num_clients",
            "learning_rate": "learning_rate",
            "batch_size": "batch_size",
            "seed": "seed",
            "server_address": "server_address",
            "weights_dir": "weights_dir",
            "metrics_dir": "metrics_dir",
            "partitions_dir": "partitions_dir",
            "target_accuracy": "target_accuracy",
            "bn_private": "bn_private_type",
            "optimizer": "use_adam",  # Special handling below
            "beta1": "beta1",
            "beta2": "beta2",
            "epsilon": "epsilon",
            "log_wandb": "log_wandb",
            "project_name": "project_name",
            "experiment_tag": "experiment_tag",
            "per_step_metrics": "per_step_metrics",
            "steps_per_epoch": "steps_per_epoch",
            "checkpoint_interval": "checkpoint_interval",
            "optimizer_state_transmission": "optimizer_state_transmission",
            "optimizer_state_aggregation": "optimizer_state_aggregation",
            "compress_optimizer_state": "compress_optimizer_state",
            "version_compatible": "version_compatible",
        }
        
        # Update configuration with command-line arguments
        for arg_name, config_name in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                if arg_name == "optimizer":
                    # Special handling for optimizer
                    optimizer_type = getattr(args, arg_name)
                    if optimizer_type == "adam":
                        setattr(config, "use_adam", True)
                    else:
                        setattr(config, "use_adam", False)
                else:
                    setattr(config, config_name, getattr(args, arg_name))
                logger.info(f"Set {config_name} to {getattr(args, arg_name)} from command line")
        
        # Handle special cases
        if hasattr(args, "exp") and args.exp is not None:
            config.update_for_experiment(args.exp)
        
        # Special flag for MTFL experiment
        if hasattr(args, "mtfl") and args.mtfl:
            config.update_for_experiment("mtfl")
        
        # Validate and potentially adjust optimizer state settings
        config._validate_optimizer_state_config()
        
        return config


# Default configuration
CONFIG = FederatedConfig()


def get_parser() -> argparse.ArgumentParser:
    """Create an argument parser for federated learning experiments.
    
    Returns
    -------
    argparse.ArgumentParser
        Argument parser with common options
    """
    parser = argparse.ArgumentParser(description="Federated Learning with Flower")
    
    # Experiment parameters
    parser.add_argument(
        "--exp", 
        choices=["fedavg", "mtfl", "both"], 
        help="Experiment type: 'fedavg', 'mtfl', or 'both' to run both sequentially"
    )
    parser.add_argument(
        "--rounds", 
        type=int, 
        default=500,
        help="Maximum number of federated learning rounds"
    )
    parser.add_argument(
        "--local-epochs", 
        type=int, 
        help="Number of local training epochs"
    )
    parser.add_argument(
        "--client-fraction", 
        type=float, 
        help="Fraction of clients to sample in each round"
    )
    parser.add_argument(
        "--num-clients", 
        type=int, 
        help="Total number of clients"
    )
    parser.add_argument(
        "--client-id",
        type=int,
        help="Client ID (used to identify the Flower client)"
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        help="Target user accuracy for termination"
    )
    
    # Learning parameters
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        help="Batch size for training"
    )
    parser.add_argument(
        "--bn-private",
        type=str,
        choices=["none", "gamma_beta", "mu_sigma", "all"],
        default="none",
        help="Which batch norm parameters to keep private"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sgd", "adam"],
        default="sgd",
        help="Optimizer to use on clients"
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Adam beta1 parameter"
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Adam beta2 parameter"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Adam epsilon parameter"
    )
    
    # Optimizer state handling parameters
    parser.add_argument(
        "--optimizer-state-transmission",
        type=str,
        choices=["none", "metrics", "parameters"],
        default="none",
        help="Method to transmit optimizer state between rounds"
    )
    parser.add_argument(
        "--optimizer-state-aggregation",
        type=str,
        choices=["none", "average", "weighted_average"],
        default="none",
        help="Method to aggregate optimizer states from multiple clients"
    )
    parser.add_argument(
        "--compress-optimizer-state",
        action="store_true",
        help="Compress optimizer state during transmission"
    )
    parser.add_argument(
        "--version-compatible",
        action="store_true",
        default=True,
        help="Maintain backward compatibility with older clients/servers"
    )
    
    # System parameters
    parser.add_argument(
        "--seed", 
        type=int, 
        help="Random seed"
    )
    parser.add_argument(
        "--server-address", 
        type=str, 
        help="Server address (host:port)"
    )
    parser.add_argument(
        "--save-init",
        action="store_true",
        help="Force saving initial weights"
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        help="Directory to store model weights"
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        help="Directory to store metrics"
    )
    parser.add_argument(
        "--partitions-dir",
        type=str,
        help="Directory to store data partitions"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true", 
        help="Enable visualization of results after running"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--shards", 
        type=int, 
        default=2, 
        help="Shards per client"
    )
    
    # Tracking parameters
    parser.add_argument(
        "--log-wandb",
        action="store_true",
        help="Log metrics to Weights & Biases"
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="mtfl-replication",
        help="W&B project name"
    )
    parser.add_argument(
        "--experiment-tag",
        type=str,
        help="Tag for the experiment (e.g., 'result1')"
    )
    parser.add_argument(
        "--per-step-metrics",
        action="store_true",
        help="Log metrics after each local step (for Result 2)"
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=10,
        help="Number of steps per local epoch (for Result 2)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Interval (in rounds) at which to save model checkpoints. 0 to disable."
    )
    
    return parser