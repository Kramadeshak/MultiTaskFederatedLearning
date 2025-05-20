"""Server application for Flower federated learning with CIFAR-10."""

import argparse
import os
import json
import logging
import time
import glob
import random
import sys
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import numpy as np
from collections import OrderedDict, defaultdict

import torch
import flwr as fl
from flwr.common import (
    EvaluateIns, 
    EvaluateRes, 
    FitIns, 
    FitRes, 
    Parameters, 
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Status
)
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server import start_server
from flwr.server.strategy.aggregate import aggregate

from fl_mtfl.model import CIFAR10CNN
from fl_mtfl.task import seed_everything, save_metrics, load_or_initialize_model
from fl_mtfl.config import CONFIG, get_parser, FederatedConfig
from fl_mtfl.wandb_utils import init_wandb, log_metrics, finish_wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fl_mtfl.server")

# Set up global exception handler
def exception_handler(exc_type, exc_value, exc_traceback):
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = exception_handler


class FlexibleTerminationStrategy(FedAvg):
    """Strategy with flexible termination based on rounds or target accuracy."""
    
    def __init__(
        self,
        target_accuracy: Optional[float] = None,
        max_rounds: int = 500,
        experiment_type: str = "fedavg",
        metrics_dir: str = "./metrics",
        weights_dir: str = "./data/weights",
        checkpoint_interval: int = 0,  # Added checkpoint_interval parameter
        log_wandb: bool = False,
        project_name: str = "mtfl-replication",
        experiment_tag: Optional[str] = None,
        *args,
        **kwargs
    ):
        """Initialize strategy with flexible termination.
        
        Parameters
        ----------
        target_accuracy : Optional[float], optional
            Target accuracy for termination, default is None.
        max_rounds : int, optional
            Maximum number of rounds, default is 500.
        experiment_type : str, optional
            Type of experiment, either "fedavg" or "mtfl", default is "fedavg".
        metrics_dir : str, optional
            Directory to save metrics, default is "./metrics".
        weights_dir : str, optional
            Directory to save model checkpoints, default is "./data/weights".
        checkpoint_interval : int, optional
            Interval (in rounds) at which to save model checkpoints, default is 0 (disabled).
        log_wandb : bool, optional
            Whether to log metrics to Weights & Biases, default is False.
        project_name : str, optional
            Weights & Biases project name, default is "mtfl-replication".
        experiment_tag : Optional[str], optional
            Tag for the experiment, default is None.
        *args, **kwargs : Any
            Additional arguments to pass to FedAvg.
        """
        # Remove 'config' from kwargs before passing to parent
        fedavg_kwargs = {k: v for k, v in kwargs.items() if k != 'config'}
        super().__init__(*args, **fedavg_kwargs)
        
        self.target_accuracy = target_accuracy
        self.max_rounds = max_rounds
        self.experiment_type = experiment_type
        self.metrics_dir = metrics_dir
        self.weights_dir = weights_dir
        self.checkpoint_interval = checkpoint_interval
        self.log_wandb = log_wandb
        self.project_name = project_name
        self.experiment_tag = experiment_tag
        self.metrics = defaultdict(lambda: {"accuracies": [], "losses": []})
        self.current_best_accuracy = 0.0
        self.current_parameters = None
        self.current_optimizer_state = None
        
        # Store the config separately if it was provided
        self.strategy_config = kwargs.get('config', {})
        
        # Initialize Weights & Biases run if requested
        self.wandb_run = None
        if log_wandb:
            self.wandb_run = init_wandb(
                project=project_name,
                experiment_name=experiment_type,
                config=self.strategy_config,
                experiment_tag=experiment_tag
            )
        
        # Create necessary directories
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(weights_dir, exist_ok=True)
        
        logger.info(f"Initialized {self.experiment_type} strategy")
        logger.info(f"Termination criteria: max_rounds={max_rounds}, target_accuracy={target_accuracy}")
        logger.info(f"Metrics will be saved to {self.metrics_dir}")
        if checkpoint_interval > 0:
            logger.info(f"Model checkpoints will be saved every {checkpoint_interval} rounds to {self.weights_dir}")
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results and check termination criteria."""
        if not results:
            return None, {}
        
        # Call aggregate_evaluate from base class
        aggregated_loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        
        # Track metrics from this round
        accuracies = []
        losses = []
        total_examples = 0
        weighted_accuracy = 0.0
        weighted_loss = 0.0
        
        for client_proxy, evaluate_res in results:
            client_metrics = evaluate_res.metrics
            client_acc = client_metrics.get("accuracy", client_metrics.get("test_accuracy", 0.0))
            client_loss = client_metrics.get("loss", client_metrics.get("test_loss", 0.0))
            client_examples = evaluate_res.num_examples
            client_id = client_metrics.get("client_id", "unknown")
            
            # Store individual client metrics
            accuracies.append((client_id, client_acc))
            losses.append((client_id, client_loss))
            
            # Calculate weighted metrics
            weighted_accuracy += client_acc * client_examples
            weighted_loss += client_loss * client_examples
            total_examples += client_examples
        
        # Calculate average metrics
        avg_accuracy = weighted_accuracy / total_examples if total_examples > 0 else 0.0
        avg_loss = weighted_loss / total_examples if total_examples > 0 else 0.0
        
        # Store metrics for this round in the metrics dict
        self.metrics[str(server_round)]["accuracies"] = accuracies
        self.metrics[str(server_round)]["losses"] = losses
        self.metrics[str(server_round)]["avg_accuracy"] = avg_accuracy
        self.metrics[str(server_round)]["avg_loss"] = avg_loss
        self.metrics[str(server_round)]["num_clients"] = len(results)
        
        # Save metrics after each round
        self._save_metrics()
        
        # Check if we need to save a checkpoint
        if self.checkpoint_interval > 0 and server_round % self.checkpoint_interval == 0:
            self._save_checkpoint(server_round)
            
        # Check if this is the best model so far
        if avg_accuracy > self.current_best_accuracy and self.current_parameters is not None:
            self.current_best_accuracy = avg_accuracy
            self._save_best_model(server_round)
        
        # Log the metrics
        logger.info(f"[Round {server_round}] Average accuracy: {avg_accuracy:.4f}, Loss: {avg_loss:.4f}")
        
        # Log to Weights & Biases if enabled
        if self.log_wandb and self.wandb_run:
            log_metrics({
                "round": server_round,
                "test_accuracy": avg_accuracy,  # Use test_ prefix for evaluation metrics
                "test_loss": avg_loss,          # Use test_ prefix for evaluation metrics
                "num_clients": len(results),
                "total_examples": total_examples
            })
        
        # Check termination criteria
        if self.target_accuracy is not None and avg_accuracy >= self.target_accuracy:
            logger.info(f"🎉 Reached target accuracy {self.target_accuracy:.4f} in round {server_round}")
            metrics["terminate"] = True
            
            # Save final checkpoint
            self._save_checkpoint(server_round, final=True)
            
            # Finish Weights & Bianes run if enabled
            if self.log_wandb and self.wandb_run:
                finish_wandb()
                
            return aggregated_loss, metrics
        
        # If we've reached max rounds, metrics will include that indicator
        if server_round >= self.max_rounds:
            logger.info(f"Reached maximum number of rounds ({self.max_rounds})")
            metrics["terminate"] = True
            
            # Save final checkpoint
            self._save_checkpoint(server_round, final=True)
            
            # Finish Weights & Biases run if enabled
            if self.log_wandb and self.wandb_run:
                finish_wandb()
        
        return aggregated_loss, metrics
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, BaseException], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results.
        
        This method extends the base FedAvg implementation to support
        averaging optimizer states for Adam based on client sample sizes.
        """
        if not results:
            return None, {}
        
        # Extract parameters from results
        weights_results = []
        optimizer_states = []
        sample_counts = []
        optimizer_state_in_params = False
        optimizer_state_params_indices = []
        
        for client_proxy, fit_res in results:
            # Get client metrics
            client_metrics = fit_res.metrics
            client_samples = fit_res.num_examples
            
            # Check if optimizer state is in parameters
            if "optimizer_state_param_idx" in client_metrics and client_metrics["optimizer_state_param_idx"] is not None:
                optimizer_state_in_params = True
                try:
                    # Extract parameters and optimizer state index
                    params = parameters_to_ndarrays(fit_res.parameters)
                    idx = int(client_metrics["optimizer_state_param_idx"])
                    
                    if 0 <= idx < len(params):
                        # Extract optimizer state from parameters
                        opt_state_bytes = params[idx].tobytes()
                        opt_state_json = opt_state_bytes.decode('utf-8')
                        opt_state = json.loads(opt_state_json)
                        
                        # Store optimizer state and sample count
                        optimizer_states.append(opt_state)
                        sample_counts.append(client_samples)
                        
                        # Record the index for this client
                        optimizer_state_params_indices.append(idx)
                        
                        # Remove optimizer state from parameters before aggregation
                        clean_params = params[:idx] + (params[idx+1:] if idx+1 < len(params) else [])
                        
                        # Create a new FitRes with cleaned parameters
                        clean_fit_res = FitRes(
                            parameters=ndarrays_to_parameters(clean_params),
                            num_examples=fit_res.num_examples,
                            metrics=fit_res.metrics,
                            status=fit_res.status
                        )
                        
                        # Add to weights results for aggregation
                        weights_results.append((client_proxy, clean_fit_res))
                        
                        logger.info(f"Extracted optimizer state from parameters at index {idx}")
                    else:
                        logger.warning(f"Invalid optimizer state index: {idx}, params length: {len(params)}")
                        weights_results.append((client_proxy, fit_res))
                except Exception as e:
                    logger.error(f"Failed to extract optimizer state from parameters: {str(e)}")
                    logger.error(traceback.format_exc())
                    weights_results.append((client_proxy, fit_res))
            # Check for optimizer state in metrics (legacy approach)
            elif "optimizer_state" in client_metrics and client_metrics["optimizer_state"]:
                try:
                    optimizer_states.append(client_metrics["optimizer_state"])
                    sample_counts.append(client_samples)
                except Exception as e:
                    logger.error(f"Failed to extract optimizer state from metrics: {str(e)}")
                weights_results.append((client_proxy, fit_res))
            else:
                # No optimizer state, just add the result as is
                weights_results.append((client_proxy, fit_res))
        
        # If no valid results remain after extraction, return None
        if not weights_results:
            return None, {}
        
        # Call parent method to aggregate parameters
        if optimizer_state_in_params:
            # Use our cleaned results for parameter aggregation
            aggregated_parameters, metrics = super().aggregate_fit(
                server_round, weights_results, failures
            )
        else:
            # Use original results if no parameter-based optimizer states
            aggregated_parameters, metrics = super().aggregate_fit(
                server_round, results, failures
            )
        
        # Store the current parameters for checkpoint saving
        if aggregated_parameters is not None:
            self.current_parameters = aggregated_parameters
        
        # If we have optimizer states, aggregate them
        if optimizer_states:
            try:
                # Calculate weighted average optimizer state
                aggregated_optimizer_state = self._aggregate_optimizer_states(optimizer_states, sample_counts)
                
                # Store the current optimizer state
                self.current_optimizer_state = aggregated_optimizer_state
                
                # Add to metrics for legacy support
                metrics["optimizer_state"] = aggregated_optimizer_state
                
                # Serialize optimizer state and append to parameters for new approach
                if optimizer_state_in_params:
                    try:
                        # Convert parameters to numpy arrays
                        params = parameters_to_ndarrays(aggregated_parameters)
                        
                        # Serialize optimizer state
                        opt_state_bytes = json.dumps(aggregated_optimizer_state).encode('utf-8')
                        opt_state_array = np.frombuffer(opt_state_bytes, dtype=np.uint8)
                        
                        # Append to parameters
                        params.append(opt_state_array)
                        
                        # Convert back to Parameters
                        aggregated_parameters = ndarrays_to_parameters(params)
                        
                        # Add indicator to metrics
                        metrics["optimizer_state_param_idx"] = len(params) - 1
                        
                        # Store optimizer state index in round metrics for later retrieval
                        self.metrics[str(server_round)]["optimizer_state_param_idx"] = len(params) - 1
                        
                        logger.info(f"Appended aggregated optimizer state to parameters at index {len(params) - 1}")
                    except Exception as e:
                        logger.error(f"Failed to append optimizer state to parameters: {str(e)}")
                        logger.error(traceback.format_exc())
            except Exception as e:
                logger.error(f"Failed to aggregate optimizer states: {str(e)}")
                logger.error(traceback.format_exc())
        
        return aggregated_parameters, metrics
    
    def _aggregate_optimizer_states(self, optimizer_states: List[Dict], sample_counts: List[int]) -> Dict:
        """Aggregate Adam optimizer states from clients using weighted averaging.
        
        Parameters
        ----------
        optimizer_states : List[Dict]
            List of optimizer states from clients.
        sample_counts : List[int]
            List of sample counts for each client.
            
        Returns
        -------
        Dict
            Aggregated optimizer state.
        """
        if not optimizer_states or not sample_counts:
            return {}
        
        # Get all keys from the first state
        keys = optimizer_states[0].keys()
        
        # Initialize aggregated state
        aggregated_state = {}
        
        # Normalize sample counts to weights
        total_samples = sum(sample_counts)
        weights = [count / total_samples for count in sample_counts] if total_samples > 0 else [1.0 / len(sample_counts)] * len(sample_counts)
        
        for key in keys:
            if key == "num_examples":
                # Just sum up num_examples
                aggregated_state[key] = sum(state.get("num_examples", 0) for state in optimizer_states)
                continue
                
            # Check if the value is a numpy array
            if isinstance(optimizer_states[0][key], np.ndarray):
                # For arrays (e.g. Adam moments), compute weighted average
                weighted_arrays = []
                for state, weight in zip(optimizer_states, weights):
                    if key in state:
                        # Multiply array by weight and add to list
                        weighted_arrays.append(state[key] * weight)
                
                if weighted_arrays:
                    # Sum weighted arrays
                    aggregated_state[key] = sum(weighted_arrays)
            elif isinstance(optimizer_states[0][key], dict):
                # For nested dictionaries, recursively aggregate
                nested_states = [state[key] for state in optimizer_states if key in state]
                if nested_states:
                    aggregated_state[key] = self._aggregate_nested_state(nested_states, weights)
            else:
                # For scalars (e.g. step count), take maximum
                scalars = [state[key] for state in optimizer_states if key in state]
                if scalars:
                    aggregated_state[key] = max(scalars)
        
        return aggregated_state
    
    def _aggregate_nested_state(self, nested_states: List[Dict], weights: List[float]) -> Dict:
        """Recursively aggregate nested optimizer state dictionaries.
        
        Parameters
        ----------
        nested_states : List[Dict]
            List of nested state dictionaries.
        weights : List[float]
            Weights for each state.
            
        Returns
        -------
        Dict
            Aggregated nested state.
        """
        if not nested_states:
            return {}
        
        # Get all keys from the first state
        keys = nested_states[0].keys()
        
        # Initialize aggregated state
        aggregated_state = {}
        
        for key in keys:
            # Check if all states have this key
            if all(key in state for state in nested_states):
                # Check type of the value
                values = [state[key] for state in nested_states]
                
                if all(isinstance(val, np.ndarray) for val in values):
                    # For arrays, compute weighted average
                    weighted_arrays = [val * weight for val, weight in zip(values, weights)]
                    aggregated_state[key] = sum(weighted_arrays)
                elif all(isinstance(val, dict) for val in values):
                    # For nested dictionaries, recursively aggregate
                    aggregated_state[key] = self._aggregate_nested_state(values, weights)
                elif all(isinstance(val, (int, float)) for val in values):
                    # For scalars, take weighted average
                    aggregated_state[key] = sum(val * weight for val, weight in zip(values, weights))
                else:
                    # For mixed types, take the first value
                    aggregated_state[key] = values[0]
        
        return aggregated_state
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Get fit_ins from parent class
        client_fitins = super().configure_fit(server_round, parameters, client_manager)
        
        # Check if we have an optimizer state to include
        if hasattr(self, 'current_optimizer_state') and self.current_optimizer_state:
            # Convert parameters for inspection
            params = parameters_to_ndarrays(parameters)
            
            # Check if the optimizer state is already in parameters
            prev_round = str(server_round-1) if server_round > 0 else "0"
            if prev_round in self.metrics and "optimizer_state_param_idx" in self.metrics[prev_round]:
                # It's already in the parameters, just need to include the index in config
                for _, fit_ins in client_fitins:
                    # Add the optimizer state index to config
                    config = fit_ins.config
                    config["optimizer_state_param_idx"] = str(self.metrics[prev_round]["optimizer_state_param_idx"])
                    logger.info(f"Added optimizer state index {config['optimizer_state_param_idx']} to config")
            else:
                # Add optimizer state to config for legacy support
                for _, fit_ins in client_fitins:
                    fit_ins.config["optimizer_state"] = self.current_optimizer_state
                    logger.info("Added optimizer state to config for legacy support")
        
        # Add current round to config
        for _, fit_ins in client_fitins:
            fit_ins.config["round"] = str(server_round)
        
        return client_fitins
    
    def _save_metrics(self) -> None:
        """Save metrics to file."""
        metrics_file = os.path.join(self.metrics_dir, f"{self.experiment_type}_metrics.json")
        save_metrics(self.metrics, metrics_file)
    
    def _save_checkpoint(self, server_round: int, final: bool = False) -> None:
        """Save model checkpoint.
        
        Parameters
        ----------
        server_round : int
            Current round number.
        final : bool, optional
            Whether this is the final checkpoint, default is False.
        """
        if self.current_parameters is None:
            logger.warning("No parameters available for checkpoint saving")
            return
        
        # Convert parameters to numpy arrays
        model_params = parameters_to_ndarrays(self.current_parameters)
        
        # Remove optimizer state if it's the last parameter
        if hasattr(self, 'current_optimizer_state') and self.current_optimizer_state:
            # Check if the last parameter might be the optimizer state
            if len(model_params) > 0 and model_params[-1].dtype == np.uint8:
                try:
                    # Try to decode as JSON to verify it's the optimizer state
                    opt_state_bytes = model_params[-1].tobytes()
                    opt_state_json = opt_state_bytes.decode('utf-8')
                    _ = json.loads(opt_state_json)  # Just try to parse it
                    
                    # If we got here, it's likely the optimizer state, remove it for saving
                    model_params = model_params[:-1]
                    logger.info("Removed optimizer state from parameters before saving checkpoint")
                except Exception:
                    # Not the optimizer state, keep all parameters
                    pass
        
        # Create checkpoint filename
        if final:
            checkpoint_path = os.path.join(self.weights_dir, f"{self.experiment_type}_final_model.pth")
        else:
            checkpoint_path = os.path.join(self.weights_dir, f"{self.experiment_type}_round_{server_round:04d}.pth")
        
        # Create model instance
        model = CIFAR10CNN(bn_private=self.strategy_config.get("bn_private_type", "none"))
        
        # Populate shared parameters
        shared_params = list(model.get_shared_parameters())
        
        # Ensure we have the right number of parameters
        if len(model_params) != len(shared_params):
            logger.warning(f"Parameter count mismatch: model expects {len(shared_params)}, got {len(model_params)}")
            # Adjust parameters if needed
            model_params = model_params[:len(shared_params)]
        
        for param, value in zip(shared_params, model_params):
            param.data = torch.from_numpy(np.copy(value))
        
        # Save state dict
        torch.save(model.state_dict(), checkpoint_path)
        
        # If we have optimizer state, save it separately
        if hasattr(self, 'current_optimizer_state') and self.current_optimizer_state:
            opt_state_path = checkpoint_path.replace(".pth", "_optimizer_state.json")
            with open(opt_state_path, 'w') as f:
                json.dump(self.current_optimizer_state, f)
            logger.info(f"Saved optimizer state to {opt_state_path}")
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _save_best_model(self, server_round: int) -> None:
        """Save the best model so far.
        
        Parameters
        ----------
        server_round : int
            Current round number.
        """
        if self.current_parameters is None:
            return
        
        # Create checkpoint filename
        best_model_path = os.path.join(self.weights_dir, f"{self.experiment_type}_best_model.pth")
        
        # Just call _save_checkpoint with appropriate parameters
        self._save_checkpoint(server_round, final=False)
        
        # Rename the checkpoint file to the best model path
        checkpoint_path = os.path.join(self.weights_dir, f"{self.experiment_type}_round_{server_round:04d}.pth")
        if os.path.exists(checkpoint_path):
            os.replace(checkpoint_path, best_model_path)
            
            # Also move optimizer state if it exists
            opt_checkpoint_path = checkpoint_path.replace(".pth", "_optimizer_state.json")
            opt_best_path = best_model_path.replace(".pth", "_optimizer_state.json")
            if os.path.exists(opt_checkpoint_path):
                os.replace(opt_checkpoint_path, opt_best_path)
                
            logger.info(f"Saved best model (accuracy: {self.current_best_accuracy:.4f}) to {best_model_path}")


def get_on_fit_config(
    local_epochs: int, 
    learning_rate: float,
    round_num: int,
    optimizer_state: Optional[Dict] = None,
    optimizer_state_param_idx: Optional[int] = None,
) -> Dict[str, Scalar]:
    """Return training configuration for a given round.
    
    Parameters
    ----------
    local_epochs : int
        Number of local training epochs.
    learning_rate : float
        Learning rate.
    round_num : int
        Current round number.
    optimizer_state : Optional[Dict], optional
        Optimizer state, default is None.
    optimizer_state_param_idx : Optional[int], optional
        Index of optimizer state in parameters, default is None.
        
    Returns
    -------
    Dict[str, Scalar]
        Training configuration.
    """
    config = {
        "epochs": str(local_epochs),
        "learning_rate": str(learning_rate),
        "round": str(round_num)
    }
    
    # Add optimizer state if provided (for backward compatibility)
    if optimizer_state:
        config["optimizer_state"] = optimizer_state
    
    # Add optimizer state index if provided (for new approach)
    if optimizer_state_param_idx is not None:
        config["optimizer_state_param_idx"] = str(optimizer_state_param_idx)
    
    return config


def get_on_evaluate_config(round_num: int) -> Dict[str, Scalar]:
    """Return evaluation configuration for a given round.
    
    Parameters
    ----------
    round_num : int
        Current round number.
        
    Returns
    -------
    Dict[str, Scalar]
        Evaluation configuration.
    """
    return {"round": str(round_num)}


def create_comparative_metrics(metrics_dir: str = "./metrics") -> None:
    """Create and save comparative metrics between FedAvg and MTFL.
    
    Parameters
    ----------
    metrics_dir : str, optional
        Directory containing metrics files, default is "./metrics".
    """
    # Create metrics directory if it doesn't exist
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Load metrics
    try:
        fedavg_file = os.path.join(metrics_dir, "fedavg_metrics.json")
        mtfl_file = os.path.join(metrics_dir, "mtfl_metrics.json")
        
        with open(fedavg_file, "r") as f:
            fedavg_metrics = json.load(f)
        
        with open(mtfl_file, "r") as f:
            mtfl_metrics = json.load(f)
        
        # Create comparative metrics
        comparative = {
            "rounds": [],
            "fedavg_accuracy": [],
            "mtfl_accuracy": [],
            "fedavg_loss": [],
            "mtfl_loss": []
        }
        
        # Find common rounds
        common_rounds = sorted(
            [int(r) for r in set(fedavg_metrics.keys()) & set(mtfl_metrics.keys())]
        )
        
        for round_idx in common_rounds:
            round_str = str(round_idx)
            comparative["rounds"].append(round_idx)
            comparative["fedavg_accuracy"].append(fedavg_metrics[round_str]["avg_accuracy"])
            comparative["mtfl_accuracy"].append(mtfl_metrics[round_str]["avg_accuracy"])
            comparative["fedavg_loss"].append(fedavg_metrics[round_str]["avg_loss"])
            comparative["mtfl_loss"].append(mtfl_metrics[round_str]["avg_loss"])
        
        # Save comparative metrics
        comp_file = os.path.join(metrics_dir, "comparative_metrics.json")
        with open(comp_file, "w") as f:
            json.dump(comparative, f, indent=4)
        
        logger.info(f"Comparative metrics saved to {comp_file}")
    except Exception as e:
        logger.error(f"Failed to create comparative metrics: {str(e)}")


def server_app() -> None:
    """Start the server application."""
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Create a configuration from command-line arguments
    config = FederatedConfig.from_args(args)
    
    # Set seed for reproducibility
    seed_everything(config.seed)
    
    # Check if data partitions exist and create them if not
    # First look for timestamped partition files
    pattern = f"{config.partitions_dir}/noniid_partitions_{config.num_clients}_*.pt"
    timestamped_files = glob.glob(pattern)
    
    # Default partition file as fallback
    default_partition_file = f"{config.partitions_dir}/noniid_partitions_{config.num_clients}.pt"
    
    # Determine which partition file to use
    partition_file = None
    if timestamped_files:
        # Sort by timestamp (descending)
        timestamped_files.sort(reverse=True)
        partition_file = timestamped_files[0]
        logger.info(f"Found timestamped partition files. Using most recent: {partition_file}")
    elif os.path.exists(default_partition_file):
        partition_file = default_partition_file
        logger.info(f"Using default partition file: {default_partition_file}")
    
    if not partition_file:
        logger.info(f"Partitions not found. Creating partitions...")
        try:
            # Import here to avoid circular import
            from preprocess_data import create_partitions
            
            # Generate timestamp for this run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create and save partitions with timestamp
            partition_file, stats_file = create_partitions(
                num_clients=config.num_clients,
                shards_per_client=config.shards_per_client,
                seed=config.seed,
                output_dir=config.partitions_dir
            )
            
            # Log the created files
            logger.info(f"Created new partitions: {partition_file}")
            logger.info(f"Created statistics: {stats_file}")
            
        except Exception as e:
            logger.error(f"Failed to create partitions: {str(e)}")
            logger.warning("Continuing without partitions. Clients will create partitions independently.")
    else:
        logger.info(f"Using existing partitions from {partition_file}")
         
    # Get server address
    server_address = args.server_address or config.server_address
    host, port_str = server_address.split(":")
    port = int(port_str)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Starting {config.experiment_name.upper()} experiment server")
    logger.info(f"{'='*50}")
    logger.info(f"Max rounds: {config.num_rounds}")
    if config.target_accuracy:
        logger.info(f"Target accuracy: {config.target_accuracy}")
    logger.info(f"Expected clients: {config.num_clients}")
    logger.info(f"Participation fraction: {config.client_fraction}")
    logger.info(f"Local epochs: {config.local_epochs}")
    logger.info(f"Server address: {server_address}")
    logger.info(f"Batch normalization privacy: {config.bn_private_type}")
    logger.info(f"Optimizer: {('Adam' if config.use_adam else 'SGD')}")
    
    # Get checkpoint interval
    checkpoint_interval = args.checkpoint_interval
    if checkpoint_interval > 0:
        logger.info(f"Checkpoint interval: {checkpoint_interval} rounds")
    
    # Initialize model
    model = load_or_initialize_model(
        weights_path=config.init_weights_path,
        bn_private=config.bn_private_type,
        save_init=args.save_init
    )
    
    # Get initial weights as NumPy arrays with detach()
    initial_parameters = [param.detach().cpu().numpy() for param in model.get_shared_parameters()]
    
    # Define strategy configuration
    strategy_config = {
        "num_clients": config.num_clients,
        "client_fraction": config.client_fraction,
        "bn_private_type": config.bn_private_type,
        "use_adam": config.use_adam,
        "learning_rate": config.learning_rate,
        "seed": config.seed
    }
    
    # Define strategy
    strategy = FlexibleTerminationStrategy(
        target_accuracy=config.target_accuracy,
        max_rounds=config.num_rounds,
        experiment_type=config.experiment_name,
        metrics_dir=config.metrics_dir,
        weights_dir=config.weights_dir,
        checkpoint_interval=checkpoint_interval,
        log_wandb=config.log_wandb,
        project_name=config.project_name,
        experiment_tag=config.experiment_tag,
        fraction_fit=config.client_fraction,
        fraction_evaluate=config.client_fraction,
        min_fit_clients=max(2, int(config.num_clients * config.client_fraction)),
        min_evaluate_clients=max(2, int(config.num_clients * config.client_fraction)),
        min_available_clients=int(config.num_clients * 0.75),  # Allow for some clients to be unavailable
        on_fit_config_fn=lambda round_num: get_on_fit_config(
            config.local_epochs, 
            config.learning_rate, 
            round_num
        ),
        on_evaluate_config_fn=get_on_evaluate_config,
        initial_parameters=ndarrays_to_parameters(initial_parameters),
        # Don't pass config directly to parent constructor
    )
    
    # Manually set strategy_config attribute
    strategy.strategy_config = strategy_config
    
    # Define stopping criteria
    def stopping_criteria(server_round, result):
        # Check if "terminate" is in the result metrics
        if "metrics" in result and "terminate" in result["metrics"]:
            return result["metrics"]["terminate"]
        # Default to continuing until max_rounds
        return server_round >= config.num_rounds
    
    # Start server with robust error handling
    try:
        server = start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=config.num_rounds),
            grpc_max_message_length = 100 * 1024 * 1024,
            strategy=strategy
        )
    except Exception as e:
        logger.error(f"Server failed to start: {str(e)}")
        traceback.print_exc()
        sys.exit(1)  # Exit with error code
    
    # Save configuration
    config.save()
    
    # Save final metrics
    if args.exp == "both":
        create_comparative_metrics(config.metrics_dir)


if __name__ == "__main__":
    server_app()