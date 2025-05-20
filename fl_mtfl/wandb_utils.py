"""
Weights & Biases utilities for tracking experiments.

This module provides utilities for logging metrics to Weights & Biases.
"""

import os
import logging
import json
import copy
from typing import Dict, List, Optional, Union, Any, Tuple
import torch
import numpy as np
import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fl_mtfl.wandb_utils")


def init_wandb(
    project: str = "mtfl-replication",
    experiment_name: str = "fedavg",
    config: Optional[Dict[str, Any]] = None,
    experiment_tag: Optional[str] = None,
) -> Optional[wandb.run]:
    """Initialize Weights & Biases run.
    
    Parameters
    ----------
    project : str, optional
        Weights & Biases project name, default is "mtfl-replication".
    experiment_name : str, optional
        Name of the experiment, default is "fedavg".
    config : Optional[Dict[str, Any]], optional
        Configuration for the experiment, default is None.
    experiment_tag : Optional[str], optional
        Tag for the experiment, default is None.
        
    Returns
    -------
    Optional[wandb.run]
        Weights & Biases run object if initialization was successful, None otherwise.
    """
    # Check if wandb is installed
    try:
        import wandb
    except ImportError:
        logger.warning("Weights & Biases not installed. Skipping initialization.")
        return None
    
    # Ensure config doesn't contain complex objects that can't be serialized
    if config:
        config = sanitize_config(config)
    
    # Create a descriptive name for the run
    run_name = f"{experiment_name}"
    if config and "bn_private_type" in config:
        run_name += f"_{config['bn_private_type']}"
    if config and "num_clients" in config and "client_fraction" in config:
        run_name += f"_W{config['num_clients']}_C{config['client_fraction']}"
    if config and "optimizer_state_transmission" in config and config["optimizer_state_transmission"] != "none":
        run_name += f"_OST{config['optimizer_state_transmission']}"
    
    # Prepare tags
    tags = []
    if experiment_tag:
        tags.append(experiment_tag)
    
    # Initialize wandb run
    try:
        # End any existing run
        if wandb.run:
            wandb.finish()
            
        run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            tags=tags or None,
            reinit=True
        )
        logger.info(f"Initialized Weights & Biases run: {run_name}")
        return run
    except Exception as e:
        logger.error(f"Failed to initialize Weights & Biases: {str(e)}")
        return None


def sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize configuration dictionary for WandB by converting any complex objects to serializable types.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary.
        
    Returns
    -------
    Dict[str, Any]
        Sanitized configuration dictionary.
    """
    sanitized = {}
    
    # Helper function to sanitize values
    def sanitize_value(value):
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, (list, tuple)):
            return [sanitize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: sanitize_value(v) for k, v in value.items()}
        elif isinstance(value, torch.Tensor):
            return value.cpu().numpy().tolist()
        elif isinstance(value, np.ndarray):
            return value.tolist()
        else:
            # For any other complex objects, convert to string representation
            return str(value)
    
    # Process each key-value pair
    for key, value in config.items():
        sanitized[key] = sanitize_value(value)
    
    return sanitized


def log_metrics(
    metrics: Dict[str, Any], 
    step: Optional[int] = None,
    handle_optimizer_state: bool = True
) -> None:
    """Log metrics to Weights & Biases.
    
    Parameters
    ----------
    metrics : Dict[str, Any]
        Metrics to log.
    step : Optional[int], optional
        Step to associate with the metrics, default is None.
    handle_optimizer_state : bool, optional
        Whether to handle optimizer state in metrics, default is True.
    """
    try:
        # Make sure wandb is initialized
        if not wandb.run:
            logger.error("Cannot log metrics: wandb not initialized. Call init_wandb first.")
            return
        
        # Handle optimizer state if present
        if handle_optimizer_state:
            processed_metrics = process_optimizer_state_metrics(metrics)
        else:
            processed_metrics = metrics
            
        # Ensure we're using consistent metric names
        normalized_metrics = normalize_metric_names(processed_metrics)
        
        # Sanitize metrics before logging
        sanitized_metrics = sanitize_metrics(normalized_metrics)
        
        # Log metrics
        wandb.log(sanitized_metrics, step=step)
    except Exception as e:
        logger.error(f"Failed to log metrics to Weights & Biases: {str(e)}")


def process_optimizer_state_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Process metrics to handle optimizer state properly for logging.
    
    Parameters
    ----------
    metrics : Dict[str, Any]
        Original metrics dictionary.
        
    Returns
    -------
    Dict[str, Any]
        Processed metrics dictionary.
    """
    # Make a deep copy to avoid modifying the original
    result = copy.deepcopy(metrics)
    
    # Check if optimizer state is present in metrics
    if "optimizer_state" in result:
        try:
            # Extract summary information from optimizer state
            optimizer_state = result.pop("optimizer_state")
            
            # Add a flag to indicate optimizer state was present
            result["has_optimizer_state"] = True
            
            # Extract helpful information from optimizer state if available
            if isinstance(optimizer_state, dict):
                # Extract learning rate
                if "param_groups" in optimizer_state and len(optimizer_state["param_groups"]) > 0:
                    param_groups = optimizer_state["param_groups"]
                    if isinstance(param_groups, list) and len(param_groups) > 0:
                        if "lr" in param_groups[0]:
                            result["learning_rate"] = param_groups[0]["lr"]
                
                # Extract statistics about state variables (e.g., exp_avg, exp_avg_sq in Adam)
                if "state" in optimizer_state:
                    state_stats = extract_optimizer_state_stats(optimizer_state["state"])
                    for key, value in state_stats.items():
                        result[f"optimizer_{key}"] = value
        except Exception as e:
            logger.error(f"Error processing optimizer state metrics: {str(e)}")
    
    # Process nested dictionaries
    for key, value in list(result.items()):
        if isinstance(value, dict):
            result[key] = process_optimizer_state_metrics(value)
    
    return result


def extract_optimizer_state_stats(state: Dict) -> Dict[str, float]:
    """Extract statistical information from optimizer state variables.
    
    Parameters
    ----------
    state : Dict
        Optimizer state dictionary.
        
    Returns
    -------
    Dict[str, float]
        Dictionary of statistical information.
    """
    stats = {}
    
    # Check if state is empty
    if not state:
        return stats
    
    # Get all variable names used in state
    var_names = set()
    for param_id, param_state in state.items():
        var_names.update(param_state.keys())
    
    # Process each variable type separately
    for var_name in var_names:
        # Collect values across all parameters
        values = []
        for param_id, param_state in state.items():
            if var_name in param_state and isinstance(param_state[var_name], (torch.Tensor, np.ndarray)):
                try:
                    # Convert to numpy array
                    if isinstance(param_state[var_name], torch.Tensor):
                        values.append(param_state[var_name].cpu().numpy())
                    else:
                        values.append(param_state[var_name])
                except Exception:
                    # Skip if conversion fails
                    pass
        
        # Calculate statistics if we have values
        if values:
            try:
                # Flatten all values
                flattened = np.concatenate([v.flatten() for v in values])
                
                # Calculate statistics
                stats[f"{var_name}_mean"] = float(np.mean(flattened))
                stats[f"{var_name}_std"] = float(np.std(flattened))
                stats[f"{var_name}_min"] = float(np.min(flattened))
                stats[f"{var_name}_max"] = float(np.max(flattened))
                stats[f"{var_name}_median"] = float(np.median(flattened))
            except Exception as e:
                logger.warning(f"Failed to calculate statistics for {var_name}: {str(e)}")
    
    return stats


def sanitize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize metrics dictionary for WandB by converting any complex objects to serializable types.
    
    Parameters
    ----------
    metrics : Dict[str, Any]
        Metrics dictionary.
        
    Returns
    -------
    Dict[str, Any]
        Sanitized metrics dictionary.
    """
    sanitized = {}
    
    # Helper function to sanitize values
    def sanitize_value(value):
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, (list, tuple)):
            if all(isinstance(item, (str, int, float, bool, type(None))) for item in value):
                return value
            return [sanitize_value(item) for item in value]
        elif isinstance(value, dict):
            return sanitize_metrics(value)
        elif isinstance(value, torch.Tensor):
            # Convert tensor to scalar if possible, otherwise to list
            if value.numel() == 1:
                return value.item()
            return value.cpu().numpy().tolist()
        elif isinstance(value, np.ndarray):
            # Convert array to scalar if possible, otherwise to list
            if value.size == 1:
                return value.item()
            return value.tolist()
        elif isinstance(value, np.number):
            # Convert numpy number to Python scalar
            return value.item()
        else:
            # For any other complex objects, convert to string representation
            try:
                # Try converting to primitive types first
                result = json.dumps(value)
                return json.loads(result)
            except Exception:
                return str(value)
    
    # Process each key-value pair
    for key, value in metrics.items():
        sanitized[key] = sanitize_value(value)
    
    return sanitized


def normalize_metric_names(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize metric names to ensure consistency with WandB.
    
    Parameters
    ----------
    metrics : Dict[str, Any]
        Original metrics dictionary.
        
    Returns
    -------
    Dict[str, Any]
        Normalized metrics dictionary with consistent naming.
    """
    # Define mapping for key standardization
    key_mapping = {
        "test_accuracy": "accuracy",
        "avg_accuracy": "accuracy",
        "test_loss": "loss",
        "avg_loss": "loss",
        "train_acc": "train_accuracy",
    }
    
    # Create a new dictionary with normalized keys
    normalized = {}
    
    # Copy all metrics, standardizing keys where needed
    for key, value in metrics.items():
        # Handle nested dictionaries
        if isinstance(value, dict):
            normalized[key] = normalize_metric_names(value)
        # Handle lists of dictionaries
        elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
            normalized[key] = [normalize_metric_names(item) for item in value]
        else:
            # Use mapped key if it exists, otherwise keep original
            if key in key_mapping:
                normalized[key_mapping[key]] = value
                # Keep the original key as well for backward compatibility
                normalized[key] = value
            else:
                normalized[key] = value
    
    return normalized


def log_local_metrics(
    round_num: int,
    step: int,
    loss: float,
    accuracy: float,
    prefix: str = "train",
    num_examples: Optional[int] = None,
    optimizer_info: Optional[Dict] = None
) -> None:
    """Log local training/testing metrics to Weights & Biases.
    
    Parameters
    ----------
    round_num : int
        Current round number.
    step : int
        Current step within the round.
    loss : float
        Loss value.
    accuracy : float
        Accuracy value.
    prefix : str, optional
        Prefix for metric names (train/test), default is "train".
    num_examples : Optional[int], optional
        Number of examples processed, default is None.
    optimizer_info : Optional[Dict], optional
        Additional optimizer information to log, default is None.
    """
    try:
        # Check if wandb is initialized
        if not wandb.run:
            logger.error("Cannot log local metrics: wandb not initialized. Call init_wandb first.")
            return
            
        metrics = {
            f"{prefix}_loss": loss,
            f"{prefix}_accuracy": accuracy,
            "round": round_num,
            "step": step,
            "global_step": round_num * 1000 + step  # Unique step ID
        }
        
        # Add num_examples if provided
        if num_examples is not None:
            metrics["num_examples"] = num_examples
            metrics[f"{prefix}_examples"] = num_examples
        
        # Add optimizer information if provided
        if optimizer_info is not None:
            for key, value in optimizer_info.items():
                metrics[f"optimizer_{key}"] = value
        
        # Log metrics
        wandb.log(metrics)
    except Exception as e:
        logger.error(f"Failed to log local metrics to Weights & Biases: {str(e)}")


def log_model(model, name: str = "model") -> None:
    """Log model to Weights & Biases.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to log.
    name : str, optional
        Name for the model, default is "model".
    """
    try:
        if not wandb.run:
            logger.error("Cannot log model: wandb not initialized. Call init_wandb first.")
            return
            
        wandb.watch(model, log="all", log_freq=10, log_graph=True)
    except Exception as e:
        logger.error(f"Failed to log model to Weights & Biases: {str(e)}")


def finish_wandb() -> None:
    """Finish the current Weights & Biases run."""
    try:
        if wandb.run:
            wandb.finish()
    except Exception as e:
        logger.error(f"Failed to finish Weights & Biases run: {str(e)}")


def log_ua_curve(
    rounds: List[int],
    fedavg_accuracies: List[float],
    mtfl_accuracies: Optional[List[float]] = None,
    experiment_tag: Optional[str] = None
) -> None:
    """Log User Accuracy curve to Weights & Biases.
    
    Parameters
    ----------
    rounds : List[int]
        List of round numbers.
    fedavg_accuracies : List[float]
        List of FedAvg accuracies.
    mtfl_accuracies : Optional[List[float]], optional
        List of MTFL accuracies, default is None.
    experiment_tag : Optional[str], optional
        Tag for the experiment, default is None.
    """
    try:
        # Check if wandb is initialized
        if not wandb.run:
            logger.error("Cannot log UA curve: wandb not initialized. Call init_wandb first.")
            return
            
        # Create a table for the accuracy curves
        columns = ["round", "fedavg_accuracy"]
        if mtfl_accuracies is not None:
            columns.append("mtfl_accuracy")
        
        data = []
        for i, r in enumerate(rounds):
            row = [r, fedavg_accuracies[i]]
            if mtfl_accuracies is not None:
                row.append(mtfl_accuracies[i])
            data.append(row)
        
        table = wandb.Table(data=data, columns=columns)
        
        # Log as a line chart
        title = "User Accuracy Curves"
        if experiment_tag:
            title += f" - {experiment_tag}"
        
        # Ensure consistent naming with standard metrics
        y_fields = []
        for col in columns[1:]:  # Skip 'round'
            if col == "fedavg_accuracy":
                y_fields.append("accuracy")
            elif col == "mtfl_accuracy":
                y_fields.append("mtfl_accuracy")
            else:
                y_fields.append(col)
        
        wandb.log({
            "ua_curves": wandb.plot.line(
                table, 
                "round", 
                y_fields,
                title=title
            )
        })
    except Exception as e:
        logger.error(f"Failed to log UA curve to Weights & Biases: {str(e)}")


def log_optimizer_state_summary(
    optimizer: torch.optim.Optimizer,
    round_num: int,
    client_id: Optional[int] = None
) -> None:
    """Log a summary of the optimizer state to Weights & Biases.
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer.
    round_num : int
        Current round number.
    client_id : Optional[int], optional
        Client ID, default is None.
    """
    try:
        # Check if wandb is initialized
        if not wandb.run:
            logger.error("Cannot log optimizer state: wandb not initialized. Call init_wandb first.")
            return
        
        # Extract state dictionary
        state_dict = optimizer.state_dict()
        
        # Create summary metrics
        metrics = {
            "round": round_num,
            "optimizer_type": type(optimizer).__name__,
        }
        
        # Add client ID if provided
        if client_id is not None:
            metrics["client_id"] = client_id
        
        # Extract hyperparameters from param_groups
        if "param_groups" in state_dict and len(state_dict["param_groups"]) > 0:
            for key, value in state_dict["param_groups"][0].items():
                if key != "params" and not isinstance(value, (list, dict, torch.Tensor)):
                    metrics[f"optimizer_{key}"] = value
        
        # Extract statistics from state variables
        if "state" in state_dict:
            state_stats = extract_optimizer_state_stats(state_dict["state"])
            for key, value in state_stats.items():
                metrics[key] = value
        
        # Log metrics
        wandb.log(metrics)
    except Exception as e:
        logger.error(f"Failed to log optimizer state summary: {str(e)}")


def extract_optimizer_state_info(
    optimizer: torch.optim.Optimizer
) -> Dict[str, Union[float, int, str]]:
    """Extract key information from the optimizer state for logging.
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer.
        
    Returns
    -------
    Dict[str, Union[float, int, str]]
        Dictionary of optimizer information.
    """
    # Create a dict to store optimizer info
    info = {
        "type": type(optimizer).__name__
    }
    
    try:
        # Get state dict
        state_dict = optimizer.state_dict()
        
        # Extract hyperparameters from param_groups
        if "param_groups" in state_dict and len(state_dict["param_groups"]) > 0:
            for key, value in state_dict["param_groups"][0].items():
                if key != "params" and not isinstance(value, (list, dict, torch.Tensor)):
                    info[key] = value
        
        # Extract basic statistics about state size
        if "state" in state_dict:
            # Count number of parameters with state
            info["num_params_with_state"] = len(state_dict["state"])
            
            # Get state variable names
            var_names = set()
            for param_id, param_state in state_dict["state"].items():
                var_names.update(param_state.keys())
            
            info["state_variables"] = list(var_names)
    except Exception as e:
        logger.error(f"Failed to extract optimizer state info: {str(e)}")
    
    return info