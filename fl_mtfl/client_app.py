"""Client implementation for Flower federated learning with CIFAR-10."""

import argparse
import os
import logging
import json
import atexit
import sys
import traceback
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
from collections import OrderedDict

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
    parameters_to_ndarrays
)

from fl_mtfl.model import CIFAR10CNN
from fl_mtfl.task import (
    load_data, 
    train, 
    test, 
    get_weights, 
    set_weights, 
    seed_everything,
    load_or_initialize_model
)
from fl_mtfl.config import CONFIG, get_parser, FederatedConfig
import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("client")

# Set up global exception handler
def exception_handler(exc_type, exc_value, exc_traceback):
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = exception_handler


class CifarClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification."""

    def __init__(
        self, 
        client_id: int, 
        config: FederatedConfig
    ):
        """Initialize CIFAR-10 client.
        
        Parameters
        ----------
        client_id : int
            Client ID (must be an integer).
        config : FederatedConfig
            Configuration for the client.
        """
        # Set random seed for reproducibility
        seed_everything(config.seed + client_id)  # Use client_id to ensure different seeds
        
        self.client_id = client_id
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = CIFAR10CNN(bn_private=config.bn_private_type).to(self.device)
        
        # Store private BN parameters and buffers
        self.private_bn_params = {}
        self.private_bn_buffers = {}
        
        # Initialize optimizer state if using Adam
        if config.use_adam:
            self.optimizer_state = None
        
        # Local step metrics for detailed training tracking (Result 2)
        self.step_metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "step": [],
            "round": []
        }
        
        # Initialize wandb for this client if logging is enabled
        self.wandb_run = None
        if config.log_wandb:
            try:
                run_name = f"client_{client_id}"
                self.wandb_run = wandb.init(
                    project=config.project_name,
                    name=run_name,
                    group=config.experiment_name,
                    config={
                        "client_id": client_id,
                        "experiment": config.experiment_name,
                        "bn_private": config.bn_private_type,
                        "optimizer": "adam" if config.use_adam else "sgd",
                        "learning_rate": config.learning_rate
                    },
                    reinit=True
                )
            except Exception as e:
                logger.error(f"Failed to initialize Weights & Biases: {str(e)}")
        
        # Load the non-IID data for this client with robust error handling
        self._load_data_with_fallback()
        
        # Print client info
        logger.info(f"Initialized client {self.client_id} with {len(self.trainloader.dataset)} training examples")
        logger.info(f"  - BN privacy type: {config.bn_private_type}")
        logger.info(f"  - Use Adam: {config.use_adam}")
        logger.info(f"  - Learning rate: {config.learning_rate}")
        
        # Register cleanup function
        atexit.register(self.cleanup)

    def cleanup(self):
        """Clean up resources when the client exits."""
        # Finish wandb run if active
        if wandb.run:
            try:
                wandb.finish()
            except:
                pass

    def _load_data_with_fallback(self) -> None:
        """Load dataset for this client with robust error handling and fallbacks."""
        try:
            # First attempt: try to load the assigned partition
            self.trainloader, self.testloader = load_data(
                partition_id=int(self.client_id),
                num_partitions=self.config.num_clients,
                shards_per_client=self.config.shards_per_client,
                batch_size=self.config.batch_size,
                seed=self.config.seed,
                partitions_dir=self.config.partitions_dir
            )
            logger.info(f"Client {self.client_id}: Successfully loaded assigned partition")
            
        except ValueError as e:
            logger.warning(f"Client {self.client_id}: Failed to load assigned partition: {str(e)}")
            
            # Second attempt: try to load a different partition
            for alternative_id in range(self.config.num_clients):
                if alternative_id == self.client_id:
                    continue  # Skip the original partition that failed
                    
                try:
                    logger.info(f"Client {self.client_id}: Attempting to load alternative partition {alternative_id}")
                    self.trainloader, self.testloader = load_data(
                        partition_id=alternative_id,
                        num_partitions=self.config.num_clients,
                        shards_per_client=self.config.shards_per_client,
                        batch_size=self.config.batch_size,
                        seed=self.config.seed,
                        partitions_dir=self.config.partitions_dir
                    )
                    logger.info(f"Client {self.client_id}: Successfully loaded alternative partition {alternative_id}")
                    return
                except Exception:
                    continue  # Try next partition

    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """Get model parameters as a list of NumPy arrays.
        
        This method returns only the shared parameters, not the private ones.
        """
        # Save private parameters before getting weights
        if self.config.use_private_bn:
            # Store private BN parameters
            if self.config.bn_private_type in ["gamma_beta", "all"]:
                self.private_bn_params = {
                    name: param.clone().detach()
                    for name, param in self.model.named_parameters()
                    if "bn" in name and ("affine" in name or "bias" in name)
                }
            
            # Store private BN buffers
            if self.config.bn_private_type in ["mu_sigma", "all"]:
                self.private_bn_buffers = self.model.get_bn_private_buffers()
        
        
        return [param.detach().cpu().numpy() for param in self.model.get_shared_parameters()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from a list of NumPy arrays.
        
        This method sets only the shared parameters, preserving the private ones.
        """
        # Create a dictionary of shared parameters
        shared_params = list(self.model.get_shared_parameters())
        
        # Check if the last parameter might be the optimizer state
        if len(parameters) > len(shared_params) and self.config.use_adam:
            # Extract optimizer state if it exists (appended as the last item)
            try:
                opt_state_bytes = parameters[-1].tobytes()
                opt_state_json = opt_state_bytes.decode('utf-8')
                opt_state = json.loads(opt_state_json)
                
                # Set the optimizer state
                self.set_optimizer_state(opt_state)
                
                # Remove the optimizer state from parameters
                parameters = parameters[:-1]
                
                logger.info(f"Client {self.client_id}: Successfully extracted optimizer state from parameters")
            except Exception as e:
                logger.warning(f"Client {self.client_id}: Failed to extract optimizer state: {str(e)}")
        
        if len(parameters) != len(shared_params):
            logger.warning(f"Expected {len(shared_params)} parameters, got {len(parameters)}")
            # Adjust parameters length if needed (this shouldn't happen in normal operation)
            parameters = parameters[:len(shared_params)]
        
        # Set shared parameters
        for param, value in zip(shared_params, parameters):
            param.data = torch.from_numpy(np.copy(value)).to(self.device)
        
        # Restore private BN parameters
        if self.config.use_private_bn:
            # Restore private BN parameters
            if self.config.bn_private_type in ["gamma_beta", "all"] and self.private_bn_params:
                for name, param in self.model.named_parameters():
                    if name in self.private_bn_params:
                        param.data = self.private_bn_params[name].to(self.device)
            
            # Restore private BN buffers
            if self.config.bn_private_type in ["mu_sigma", "all"] and self.private_bn_buffers:
                self.model.set_bn_private_buffers(self.private_bn_buffers)

    def get_optimizer_state(self) -> Dict:
        """Get optimizer state for Adam."""
        if not self.config.use_adam or not hasattr(self, 'optimizer_state') or self.optimizer_state is None:
            return {}
        
        # Convert optimizer state to numpy arrays
        state_dict = {}
        for key, value in self.optimizer_state.items():
            if isinstance(value, torch.Tensor):
                state_dict[key] = value.detach().cpu().numpy()  # Add detach() for tensors
            else:
                state_dict[key] = value
        
        # Add number of examples for weighted aggregation
        if hasattr(self, 'trainloader') and self.trainloader is not None:
            state_dict["num_examples"] = len(self.trainloader.dataset)
        
        return state_dict

    def set_optimizer_state(self, state_dict: Dict) -> None:
        """Set optimizer state for Adam."""
        if not self.config.use_adam or not state_dict:
            return
        
        # Remove num_examples key if present (not part of optimizer state)
        if "num_examples" in state_dict:
            state_dict = {k: v for k, v in state_dict.items() if k != "num_examples"}
        
        # Convert numpy arrays back to tensors
        state_dict_tensor = {}
        for key, value in state_dict.items():
            if isinstance(value, np.ndarray):
                state_dict_tensor[key] = torch.from_numpy(value).to(self.device)
            else:
                state_dict_tensor[key] = value
        
        self.optimizer_state = state_dict_tensor

    def log_local_metrics(self, epoch, step, loss, accuracy, prefix="train", num_examples=None):
        """Log local training metrics to Weights & Biases."""
        if not self.config.log_wandb:
            return
        
        # Make sure wandb is initialized
        if not wandb.run:
            try:
                run_name = f"client_{self.client_id}"
                wandb.init(
                    project=self.config.project_name,
                    name=run_name,
                    group=self.config.experiment_name,
                    config={
                        "client_id": self.client_id,
                        "experiment": self.config.experiment_name,
                        "bn_private": self.config.bn_private_type,
                        "optimizer": "adam" if self.config.use_adam else "sgd",
                        "learning_rate": self.config.learning_rate
                    },
                    reinit=True
                )
            except Exception as e:
                logger.error(f"Failed to initialize Weights & Biases: {str(e)}")
                return
        
        # Log metrics
        try:
            metrics = {
                f"{prefix}_loss": loss,
                f"{prefix}_accuracy": accuracy,
                "epoch": epoch,
                "step": step,
                "global_step": epoch * 10000 + step  # Give a unique step ID
            }
            
            # Add num_examples if provided
            if num_examples is not None:
                metrics["num_examples"] = num_examples
                metrics[f"{prefix}_examples"] = num_examples
            
            wandb.log(metrics)
        except Exception as e:
            logger.error(f"Failed to log metrics to Weights & Biases: {str(e)}")

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Train the model on the local dataset.
        
        Parameters
        ----------
        parameters : List[np.ndarray]
            Current model parameters.
        config : Dict[str, Scalar]
            Configuration parameters for training.
            
        Returns
        -------
        Tuple[List[np.ndarray], int, Dict[str, Scalar]]
            Updated model parameters, number of training examples, metrics.
        """
        # Update local model parameters
        self.set_parameters(parameters)
        
        # Get optimizer state if provided
        if "optimizer_state_param_idx" in config and self.config.use_adam:
            # The optimizer state is in the parameters
            try:
                idx = int(config["optimizer_state_param_idx"])
                if 0 <= idx < len(parameters):
                    opt_state_bytes = parameters[idx].tobytes()
                    opt_state_json = opt_state_bytes.decode('utf-8')
                    opt_state = json.loads(opt_state_json)
                    self.set_optimizer_state(opt_state)
                    logger.info(f"Client {self.client_id}: Retrieved optimizer state from parameters")
            except Exception as e:
                logger.warning(f"Client {self.client_id}: Failed to parse optimizer state: {str(e)}")
        elif "optimizer_state" in config and self.config.use_adam:
            self.set_optimizer_state(config["optimizer_state"])
        
        # Get hyperparameters for this round
        epochs = int(config.get("epochs", self.config.local_epochs))
        lr = float(config.get("learning_rate", self.config.learning_rate))
        round_num = int(config.get("round", 0))
        
        logger.info(f"Client {self.client_id}: Starting training for round {round_num}, "
                   f"epochs={epochs}, lr={lr}")
        
        # Define callback function for per-step metrics if needed
        callback = None
        if self.config.per_step_metrics:
            def log_step_metrics(epoch, step, loss, accuracy):
                self.step_metrics["train_loss"].append(loss)
                self.step_metrics["train_accuracy"].append(accuracy)
                self.step_metrics["step"].append(step)
                self.step_metrics["round"].append(round_num)
                
                # Log to wandb if enabled
                self.log_local_metrics(epoch, step, loss, accuracy, prefix="train", num_examples=len(self.trainloader.dataset))
            
            callback = log_step_metrics
        
        # Train the model with appropriate optimizer
        loss, step_metrics, _ = train(
            self.model, 
            self.trainloader, 
            epochs, 
            self.device, 
            use_adam=self.config.use_adam, 
            lr=lr,
            beta1=self.config.beta1,
            beta2=self.config.beta2,
            epsilon=self.config.epsilon,
            log_per_step_metrics=self.config.per_step_metrics,
            callback=callback
        )
        
        # Save optimizer state if using Adam
        if self.config.use_adam:
            # Create a simple optimizer to extract state
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.epsilon
            )
            # Get state dict
            self.optimizer_state = optimizer.state_dict()
        
        # Evaluate model on local test set
        test_loss, test_accuracy = test(self.model, self.testloader, self.device)
        
        # Return updated model parameters and metrics
        updated_parameters = self.get_parameters(config)
        

        metrics = {
            "train_loss": float(loss),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "client_id": int(self.client_id),
            "num_examples": int(len(self.trainloader.dataset)),
            "round": int(round_num)
        }
        
        # For Adam optimizer state, serialize essential parts as primitive types
        if self.config.use_adam and self.optimizer_state:
            # Flatten optimizer state - extract only essential scalar values
            # First, extract param_groups info
            for i, param_group in enumerate(self.optimizer_state.get("param_groups", [])):
                metrics[f"opt_lr_{i}"] = float(param_group.get("lr", 0.0))
                metrics[f"opt_beta1_{i}"] = float(param_group.get("betas", (0.0, 0.0))[0])
                metrics[f"opt_beta2_{i}"] = float(param_group.get("betas", (0.0, 0.0))[1])
                metrics[f"opt_eps_{i}"] = float(param_group.get("eps", 0.0))
                metrics[f"opt_weight_decay_{i}"] = float(param_group.get("weight_decay", 0.0))
            
            # Add a flag indicating optimizer state is included
            metrics["has_optimizer_state"] = True
            
            # Store the full optimizer state for server to access via parameters
            # We'll append it to the parameters list as a serialized JSON string
            # This will be handled specially by the server
            try:
                opt_state_bytes = json.dumps(self.get_optimizer_state()).encode('utf-8')
                updated_parameters.append(np.frombuffer(opt_state_bytes, dtype=np.uint8))
                
                # Add indicator of optimizer state's position
                metrics["optimizer_state_param_idx"] = len(updated_parameters) - 1
                logger.info(f"Client {self.client_id}: Appended optimizer state to parameters at index {len(updated_parameters) - 1}")
            except Exception as e:
                logger.error(f"Client {self.client_id}: Failed to serialize optimizer state: {str(e)}")
        
        # Instead of including the whole step_metrics dict, extract key values if needed
        if self.config.per_step_metrics and step_metrics:
            # Extract a few representative values if needed (e.g., final values)
            if "loss" in step_metrics:
                metrics["final_step_loss"] = float(step_metrics["loss"][-1]) if step_metrics["loss"] else 0.0
            if "accuracy" in step_metrics:
                metrics["final_step_accuracy"] = float(step_metrics["accuracy"][-1]) if step_metrics["accuracy"] else 0.0
        
        num_examples = len(self.trainloader.dataset)
        
        logger.info(f"Client {self.client_id}: Completed training with loss={loss:.4f}, "
                   f"test_accuracy={test_accuracy:.4f}")
        
        return updated_parameters, num_examples, metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model on the local test dataset.
        
        Parameters
        ----------
        parameters : List[np.ndarray]
            Model parameters to evaluate.
        config : Dict[str, Scalar]
            Configuration parameters for evaluation.
            
        Returns
        -------
        Tuple[float, int, Dict[str, Scalar]]
            Loss, number of test examples, metrics.
        """
        # Update local model parameters
        self.set_parameters(parameters)
        
        # Get current round number
        round_num = int(config.get("round", 0))
        
        logger.info(f"Client {self.client_id}: Starting evaluation for round {round_num}")
        
        # Evaluate the updated model
        loss, accuracy = test(self.model, self.testloader, self.device)
        
        # Calculate number of test examples
        num_examples = len(self.testloader.dataset)
        
        # Return evaluation results
        metrics = {
            "test_accuracy": float(accuracy),
            "test_loss": float(loss),
            "client_id": int(self.client_id),
            "num_examples": int(num_examples),
            "round": int(round_num)
        }
        
        logger.info(f"Client {self.client_id}: Completed evaluation with "
                   f"loss={loss:.4f}, accuracy={accuracy:.4f}")
        
        # Log to wandb if enabled
        self.log_local_metrics(round_num, 0, loss, accuracy, prefix="test", num_examples=num_examples)
        
        return float(loss), int(num_examples), metrics


def client_app() -> None:
    """Run the CIFAR-10 client application."""
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Set client ID from environment variable if not provided via command line
    client_id = args.client_id
    if client_id is None:
        try:
            client_id = int(os.getenv("FLOWER_CLIENT_ID", "0"))
        except ValueError:
            client_id = 0
    
    # Create a configuration from command-line arguments
    config = FederatedConfig.from_args(args)
    
    # Add some robustness - catch any exceptions during client initialization
    try:
        # Initialize client
        client = CifarClient(
            client_id=client_id,
            config=config
        )
        
        # Start client
        server_address = args.server_address or config.server_address
        logger.info(f"Starting client {client_id} connecting to {server_address}")
        

        # Convert NumPyClient to Client
        client_fn = lambda: client
    
        # Start Flower client
        fl.client.start_client(
            server_address=server_address,
            grpc_max_message_length = 100 * 1024 * 1024,
            client=client_fn().to_client()
            )
    except Exception as e:
        logger.error(f"Client {client_id} failed to initialize or run: {str(e)}")
        logger.error(traceback.format_exc())
        logger.error("This client will not participate in federated learning")
        # Exit with an error code but don't crash
        exit(1)


if __name__ == "__main__":
    client_app()