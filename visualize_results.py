"""Visualization utilities for FL-MTFL experiments."""

import os
import argparse
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any

from fl_mtfl.config import get_parser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fl_mtfl.visualize_results")


def load_metrics(file_path: str) -> Dict:
    """Load metrics from a JSON file.
    
    Parameters
    ----------
    file_path : str
        Path to the metrics file.
        
    Returns
    -------
    Dict
        Dictionary of metrics.
    """
    try:
        with open(file_path, "r") as f:
            metrics = json.load(f)
        
        logger.info(f"Loaded metrics from {file_path}")
        return metrics
    except FileNotFoundError:
        logger.warning(f"Metrics file not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in metrics file: {file_path}")
        return {}


def standardize_metric_keys(metrics: Dict) -> Dict:
    """Standardize metric keys for consistency with WandB.
    
    Parameters
    ----------
    metrics : Dict
        Original metrics dictionary.
        
    Returns
    -------
    Dict
        Metrics dictionary with standardized key names.
    """
    # Define mapping for key standardization - preserve test/train distinction
    key_mapping = {
        "test_acc": "test_accuracy",
        "avg_accuracy": "test_accuracy",  # Server-side metrics are typically evaluation metrics
        "avg_loss": "test_loss",          # Server-side metrics are typically evaluation metrics
        "train_acc": "train_accuracy"
    }
    
    # Handle special case for nested dictionaries in metrics
    if isinstance(metrics, dict):
        standardized = {}
        for key, value in metrics.items():
            # Process dictionary values recursively
            if isinstance(value, dict):
                standardized[key] = standardize_metric_keys(value)
            # Process list values that might contain dictionaries
            elif isinstance(value, list):
                if all(isinstance(item, dict) for item in value if isinstance(item, dict)):
                    standardized[key] = [standardize_metric_keys(item) if isinstance(item, dict) else item for item in value]
                else:
                    standardized[key] = value
            # Apply key mapping if key in mapping
            elif key in key_mapping:
                standardized_key = key_mapping[key]
                if standardized_key not in standardized:  # Avoid duplicates
                    standardized[standardized_key] = value
                # Keep original key for backward compatibility
                standardized[key] = value
            else:
                standardized[key] = value
        return standardized
    return metrics


def plot_experiment_metrics(
    metrics: Dict, 
    experiment_name: str,
    output_dir: str = "./figures"
) -> None:
    """Plot metrics for a specific experiment.
    
    Parameters
    ----------
    metrics : Dict
        Dictionary containing the metrics.
    experiment_name : str
        Name of the experiment.
    output_dir : str, optional
        Directory to save the plots, default is "./figures".
    """
    # Ensure consistent metric names
    metrics = standardize_metric_keys(metrics)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract round numbers
    rounds = sorted([int(r) for r in metrics.keys()])
    
    # Check if metrics exist
    if not rounds:
        logger.warning(f"No metrics found for {experiment_name.upper()}")
        return
    
    # Extract metrics for each round
    accuracies = []
    losses = []
    
    for r in rounds:
        round_str = str(r)
        # Use test_accuracy primarily, fall back to avg_accuracy or accuracy
        if "test_accuracy" in metrics[round_str]:
            accuracies.append(metrics[round_str]["test_accuracy"])
        elif "avg_accuracy" in metrics[round_str]:
            accuracies.append(metrics[round_str]["avg_accuracy"])
        elif "accuracy" in metrics[round_str]:
            accuracies.append(metrics[round_str]["accuracy"])
        
        # Use test_loss primarily, fall back to avg_loss or loss
        if "test_loss" in metrics[round_str]:
            losses.append(metrics[round_str]["test_loss"])
        elif "avg_loss" in metrics[round_str]:
            losses.append(metrics[round_str]["avg_loss"])
        elif "loss" in metrics[round_str]:
            losses.append(metrics[round_str]["loss"])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot accuracy
    ax1.plot(rounds, accuracies, 'o-', linewidth=2)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"{experiment_name.upper()} Accuracy vs. Round")
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(rounds, losses, 'o-', linewidth=2)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Loss")
    ax2.set_title(f"{experiment_name.upper()} Loss vs. Round")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Adjust layout and save figure
    plt.tight_layout()
    output_file = f"{output_dir}/{experiment_name}_metrics.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved {experiment_name.upper()} metrics plot to {output_file}")
    
    # Plot client-specific metrics if available
    if len(rounds) > 0 and "accuracies" in metrics[str(rounds[0])]:
        plot_client_metrics(metrics, experiment_name, output_dir)


def plot_client_metrics(
    metrics: Dict, 
    experiment_name: str,
    output_dir: str = "./figures"
) -> None:
    """Plot client-specific metrics.
    
    Parameters
    ----------
    metrics : Dict
        Dictionary containing the metrics.
    experiment_name : str
        Name of the experiment.
    output_dir : str, optional
        Directory to save the plots, default is "./figures".
    """
    # Ensure consistent metric names
    metrics = standardize_metric_keys(metrics)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract round numbers
    rounds = sorted([int(r) for r in metrics.keys()])
    
    # Extract client IDs from the first round
    first_round = str(rounds[0])
    if "accuracies" not in metrics[first_round]:
        return  # No client-specific metrics available
    
    client_accuracies = metrics[first_round]["accuracies"]
    client_ids = sorted(set([int(client_id) for client_id, _ in client_accuracies]))
    
    # Create a dictionary to store client metrics across rounds
    client_metrics = {client_id: {"rounds": [], "accuracies": []} for client_id in client_ids}
    
    # Extract client metrics for each round
    for round_num in rounds:
        round_str = str(round_num)
        if "accuracies" not in metrics[round_str]:
            continue
        
        for client_id, accuracy in metrics[round_str]["accuracies"]:
            client_id = int(client_id)
            if client_id in client_metrics:
                client_metrics[client_id]["rounds"].append(round_num)
                client_metrics[client_id]["accuracies"].append(accuracy)
    
    # Plot client accuracies
    plt.figure(figsize=(12, 8))
    
    # Plot individual client accuracies
    for client_id, data in client_metrics.items():
        if len(data["rounds"]) > 0:
            plt.plot(data["rounds"], data["accuracies"], 'o-', alpha=0.2, linewidth=1, label=f"Client {client_id}")
    
    # Plot average accuracy - prioritize test_accuracy
    avg_accuracies = []
    for r in rounds:
        round_str = str(r)
        if "test_accuracy" in metrics[round_str]:
            avg_accuracies.append(metrics[round_str]["test_accuracy"])
        elif "avg_accuracy" in metrics[round_str]:
            avg_accuracies.append(metrics[round_str]["avg_accuracy"])
        elif "accuracy" in metrics[round_str]:
            avg_accuracies.append(metrics[round_str]["accuracy"])
    
    plt.plot(rounds, avg_accuracies, 'k-', linewidth=3, label="Average")
    
    # Add labels and legend
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title(f"{experiment_name.upper()} Client Accuracies")
    plt.grid(True, alpha=0.3)
    
    # Use a more efficient legend
    plt.legend([f"Client (individual)", "Average"], loc='lower right')
    
    # Adjust layout and save figure
    plt.tight_layout()
    output_file = f"{output_dir}/{experiment_name}_client_accuracies.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved {experiment_name.upper()} client accuracies plot to {output_file}")


def plot_comparative_metrics(
    comparative_metrics: Dict,
    output_dir: str = "./figures",
    confidence_interval: Optional[float] = None,
    seeds: Optional[int] = None
) -> None:
    """Plot comparative metrics between FedAvg and MTFL.
    
    Parameters
    ----------
    comparative_metrics : Dict
        Dictionary containing the comparative metrics.
    output_dir : str, optional
        Directory to save the plots, default is "./figures".
    confidence_interval : Optional[float], optional
        Confidence interval to show (e.g., 0.95 for 95%), default is None.
    seeds : Optional[int], optional
        Number of seeds used for the experiments, default is None.
    """
    # Ensure consistent metric names
    comparative_metrics = standardize_metric_keys(comparative_metrics)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics - handle both standard and legacy names with proper test/train distinction
    rounds = comparative_metrics.get("rounds", [])
    
    # Get accuracy metrics with fallbacks - prioritize test_accuracy
    fedavg_accuracy = (
        comparative_metrics.get("fedavg_test_accuracy", []) or 
        comparative_metrics.get("fedavg_accuracy", []) or 
        comparative_metrics.get("fedavg_avg_accuracy", [])
    )
    mtfl_accuracy = (
        comparative_metrics.get("mtfl_test_accuracy", []) or 
        comparative_metrics.get("mtfl_accuracy", []) or 
        comparative_metrics.get("mtfl_avg_accuracy", [])
    )
    
    # Get loss metrics with fallbacks - prioritize test_loss
    fedavg_loss = (
        comparative_metrics.get("fedavg_test_loss", []) or 
        comparative_metrics.get("fedavg_loss", []) or 
        comparative_metrics.get("fedavg_avg_loss", [])
    )
    mtfl_loss = (
        comparative_metrics.get("mtfl_test_loss", []) or 
        comparative_metrics.get("mtfl_loss", []) or 
        comparative_metrics.get("mtfl_avg_loss", [])
    )
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot accuracy comparison
    ax1.plot(rounds, fedavg_accuracy, 'o-', color='red', linewidth=2, label="FedAvg")
    ax1.plot(rounds, mtfl_accuracy, 's-', color='blue', linewidth=2, label="MTFL")
    
    # Add confidence intervals if requested
    if confidence_interval is not None and seeds is not None:
        # Calculate confidence intervals (simplified version)
        alpha = 1 - confidence_interval
        z_score = 1.96  # For 95% CI
        
        # Standard error of the mean = std / sqrt(n)
        fedavg_std = np.std(fedavg_accuracy) / np.sqrt(seeds)
        mtfl_std = np.std(mtfl_accuracy) / np.sqrt(seeds)
        
        # Confidence interval = z * std_error
        fedavg_ci = z_score * fedavg_std
        mtfl_ci = z_score * mtfl_std
        
        # Plot confidence intervals
        ax1.fill_between(rounds, 
                        np.array(fedavg_accuracy) - fedavg_ci, 
                        np.array(fedavg_accuracy) + fedavg_ci, 
                        color='red', alpha=0.2)
        ax1.fill_between(rounds, 
                        np.array(mtfl_accuracy) - mtfl_ci, 
                        np.array(mtfl_accuracy) + mtfl_ci, 
                        color='blue', alpha=0.2)
    
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy Comparison: FedAvg vs. MTFL")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot loss comparison
    ax2.plot(rounds, fedavg_loss, 'o-', color='red', linewidth=2, label="FedAvg")
    ax2.plot(rounds, mtfl_loss, 's-', color='blue', linewidth=2, label="MTFL")
    
    # Add confidence intervals if requested
    if confidence_interval is not None and seeds is not None:
        # Standard error of the mean = std / sqrt(n)
        fedavg_loss_std = np.std(fedavg_loss) / np.sqrt(seeds)
        mtfl_loss_std = np.std(mtfl_loss) / np.sqrt(seeds)
        
        # Confidence interval = z * std_error
        fedavg_loss_ci = z_score * fedavg_loss_std
        mtfl_loss_ci = z_score * mtfl_loss_std
        
        # Plot confidence intervals
        ax2.fill_between(rounds, 
                        np.array(fedavg_loss) - fedavg_loss_ci, 
                        np.array(fedavg_loss) + fedavg_loss_ci, 
                        color='red', alpha=0.2)
        ax2.fill_between(rounds, 
                        np.array(mtfl_loss) - mtfl_loss_ci, 
                        np.array(mtfl_loss) + mtfl_loss_ci, 
                        color='blue', alpha=0.2)
    
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss Comparison: FedAvg vs. MTFL")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Adjust layout and save figure
    plt.tight_layout()
    output_file = f"{output_dir}/comparative_metrics.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparative metrics plot to {output_file}")


def plot_per_step_metrics(
    step_metrics: Dict,
    experiment_name: str,
    output_dir: str = "./figures"
) -> None:
    """Plot per-step training and testing metrics.
    
    Parameters
    ----------
    step_metrics : Dict
        Dictionary containing the per-step metrics.
    experiment_name : str
        Name of the experiment.
    output_dir : str, optional
        Directory to save the plots, default is "./figures".
    """
    # Ensure consistent metric names
    step_metrics = standardize_metric_keys(step_metrics)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics with fallbacks to ensure backward compatibility
    steps = step_metrics.get("steps", [])
    
    train_loss = (
        step_metrics.get("train_loss", []) or 
        step_metrics.get("loss", [])
    )
    
    train_accuracy = (
        step_metrics.get("train_accuracy", []) or 
        step_metrics.get("accuracy", [])
    )
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot training loss
    ax1.plot(steps, train_loss, '-', alpha=0.6, linewidth=1, color='blue')
    ax1.plot(steps, train_loss, 'o', markersize=2, alpha=0.5, color='blue')
    
    # Add a smoothed version for clarity
    window_size = min(len(train_loss) // 20 + 1, 100)  # Dynamic window size
    smoothed_loss = np.convolve(train_loss, np.ones(window_size)/window_size, mode='valid')
    smoothed_steps = steps[window_size-1:]
    ax1.plot(smoothed_steps, smoothed_loss, linewidth=2, color='red', label=f'Smoothed (window={window_size})')
    
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{experiment_name.upper()} Training Loss per Step")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot training accuracy
    ax2.plot(steps, train_accuracy, '-', alpha=0.6, linewidth=1, color='blue')
    ax2.plot(steps, train_accuracy, 'o', markersize=2, alpha=0.5, color='blue')
    
    # Add a smoothed version for clarity
    smoothed_acc = np.convolve(train_accuracy, np.ones(window_size)/window_size, mode='valid')
    ax2.plot(smoothed_steps, smoothed_acc, linewidth=2, color='red', label=f'Smoothed (window={window_size})')
    
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{experiment_name.upper()} Training Accuracy per Step")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Adjust layout and save figure
    plt.tight_layout()
    output_file = f"{output_dir}/{experiment_name}_per_step_metrics.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved {experiment_name.upper()} per-step metrics plot to {output_file}")


def main():
    """Main function to visualize experiment results."""
    # Parse command line arguments
    parser = get_parser()
    
    # Add visualization-specific arguments
    parser.add_argument(
        "--fedavg-only",
        action="store_true",
        help="Only visualize FedAvg results"
    )
    parser.add_argument(
        "--mtfl-only",
        action="store_true",
        help="Only visualize MTFL results"
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="./figures",
        help="Directory to save figures"
    )
    parser.add_argument(
        "--confidence-interval",
        type=float,
        default=None,
        help="Confidence interval for plots (e.g., 0.95 for 95%)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Number of seeds used for the experiments"
    )
    parser.add_argument(
        "--include-step-metrics",
        action="store_true",
        help="Include per-step metrics in visualization"
    )
    
    args = parser.parse_args()
    
    # Create a configuration from command-line arguments
    from fl_mtfl.config import CONFIG
    config = CONFIG
    
    # Define paths for metrics files
    metrics_dir = args.metrics_dir if args.metrics_dir else config.metrics_dir
    figures_dir = args.figures_dir if args.figures_dir else config.figures_dir
    
    fedavg_file = os.path.join(metrics_dir, "fedavg_metrics.json")
    mtfl_file = os.path.join(metrics_dir, "mtfl_metrics.json")
    comparative_file = os.path.join(metrics_dir, "comparative_metrics.json")
    
    # Determine which experiments to visualize
    visualize_fedavg = not args.mtfl_only
    visualize_mtfl = not args.fedavg_only
    visualize_comparative = visualize_fedavg and visualize_mtfl
    
    # Load and plot FedAvg metrics
    if visualize_fedavg and os.path.exists(fedavg_file):
        fedavg_metrics = load_metrics(fedavg_file)
        if fedavg_metrics:
            plot_experiment_metrics(fedavg_metrics, "fedavg", figures_dir)
            
            # Check for per-step metrics
            if args.include_step_metrics:
                step_metrics_file = os.path.join(metrics_dir, "fedavg_step_metrics.json")
                if os.path.exists(step_metrics_file):
                    step_metrics = load_metrics(step_metrics_file)
                    if step_metrics:
                        plot_per_step_metrics(step_metrics, "fedavg", figures_dir)
    
    # Load and plot MTFL metrics
    if visualize_mtfl and os.path.exists(mtfl_file):
        mtfl_metrics = load_metrics(mtfl_file)
        if mtfl_metrics:
            plot_experiment_metrics(mtfl_metrics, "mtfl", figures_dir)
            
            # Check for per-step metrics
            if args.include_step_metrics:
                step_metrics_file = os.path.join(metrics_dir, "mtfl_step_metrics.json")
                if os.path.exists(step_metrics_file):
                    step_metrics = load_metrics(step_metrics_file)
                    if step_metrics:
                        plot_per_step_metrics(step_metrics, "mtfl", figures_dir)
    
    # Load and plot comparative metrics
    if visualize_comparative and os.path.exists(comparative_file):
        comparative_metrics = load_metrics(comparative_file)
        if comparative_metrics:
            plot_comparative_metrics(
                comparative_metrics, 
                figures_dir,
                confidence_interval=args.confidence_interval,
                seeds=args.seeds
            )
    
    logger.info("Visualization completed successfully")


if __name__ == "__main__":
    main()