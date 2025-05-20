"""Implementation of model, training and evaluation functions for FL."""

import os
import numpy as np
import json
import random
import logging
import glob
import base64
import zlib
import copy
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Normalize, ToTensor
import datasets
from datasets import Dataset

from fl_mtfl.model import CIFAR10CNN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fl_mtfl.task")


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Parameters
    ----------
    seed : int, optional
        Random seed value, default is 42.
    """
    logger.info(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def noniid_partition(
    dataset: Union[Tuple, Dict],
    num_partitions: int,
    shards_per_client: int = 2,
    class_key: str = "label",
    seed: Optional[int] = 42,
) -> List[List[int]]:
    """Partition dataset in a non-IID manner following McMahan et al., AISTATS 2017.
    
    This function creates non-IID partitions by:
    1. Sorting the data by class label
    2. Dividing it into shards_per_client * num_partitions shards
    3. Assigning shards_per_client shards to each partition/client
    
    Parameters
    ----------
    dataset : Union[Tuple, Dict]
        The dataset to partition. Can be a tuple containing (features, labels) or 
        a dictionary-like object with a structure compatible with Flower datasets.
    num_partitions : int
        Number of partitions to create (number of clients).
    shards_per_client : int, optional
        Number of shards to assign to each client, default is 2 as in the original paper.
    class_key : str, optional
        If dataset is a dictionary, the key to access class labels. Default is "label".
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    List[List[int]]
        A list of num_partitions lists, where each inner list contains the indices of
        samples that belong to that partition.
    """
    if seed is not None:
        np.random.seed(seed)

    # Extract labels based on the dataset format
    if isinstance(dataset, tuple) and len(dataset) == 2:
        labels = dataset[1]
    elif isinstance(dataset, dict) or hasattr(dataset, '__getitem__'):
        # For dictionary-like or custom dataset objects that support indexing
        if hasattr(dataset, 'targets'):  # For torchvision datasets
            labels = dataset.targets
        elif class_key in dataset:  # For dictionary datasets
            labels = dataset[class_key]
        else:
            # Try to infer labels from the first few samples
            labels = []
            for i in range(min(100, len(dataset))):
                sample = dataset[i]
                if isinstance(sample, tuple) and len(sample) >= 2:
                    labels.append(sample[1])
                elif isinstance(sample, dict) and class_key in sample:
                    labels.append(sample[class_key])
            
            if not labels:
                raise ValueError(
                    "Could not extract labels from dataset. Please provide a dataset with "
                    "a 'targets' attribute, a dictionary with a 'label' key, or samples "
                    "with labels as the second element of tuples."
                )
            
            # Get all labels after inferring the format
            labels = [
                sample[1] if isinstance(sample, tuple) else sample[class_key]
                for sample in dataset
            ]
    else:
        raise ValueError(
            "Dataset format not recognized. Please provide either a tuple (features, labels) "
            "or a dictionary-like object with samples."
        )

    # Convert labels to numpy array if they're not already
    labels = np.array(labels)
    
    # Get dataset size
    n_samples = len(labels)
    
    # Create a list of indices and sort them by label
    indices = np.argsort(labels).tolist()
    
    # Calculate total number of shards
    num_shards = num_partitions * shards_per_client
    
    # Check if we can create enough shards
    if n_samples < num_shards:
        raise ValueError(
            f"Dataset too small ({n_samples} samples) to create {num_shards} shards."
        )
    
    # Calculate approximate size of each shard
    samples_per_shard = n_samples // num_shards
    
    logger.info(f"Creating {num_shards} shards with ~{samples_per_shard} samples per shard")
    
    # Create shards - each shard will contain consecutive samples of mostly the same class
    shards = []
    for i in range(0, n_samples, samples_per_shard):
        if len(shards) < num_shards:  # Only create up to num_shards
            end_idx = min(i + samples_per_shard, n_samples)
            shards.append(indices[i:end_idx])
    
    # If we have samples left, distribute them among the existing shards
    if len(shards) < num_shards:
        remaining = indices[num_shards * samples_per_shard:]
        for i, idx in enumerate(remaining):
            shards[i % len(shards)].append(idx)
    
    # Shuffle the order of shards (but not the contents within shards)
    # This is to randomize the assignment of shards to clients
    shard_indices = list(range(len(shards)))
    np.random.shuffle(shard_indices)
    
    # Assign shards_per_client shards to each client
    partitions = [[] for _ in range(num_partitions)]
    for i in range(num_partitions):
        for j in range(shards_per_client):
            # Calculate the shard index based on client index and shard per client index
            if i * shards_per_client + j < len(shard_indices):
                shard_idx = shard_indices[i * shards_per_client + j]
                partitions[i].extend(shards[shard_idx])
            else:
                logger.warning(f"Warning: Not enough shards for client {i}, j={j}")
    
    # Log partition information
    for i in range(min(5, num_partitions)):  # Log first 5 partitions
        classes = {}
        for idx in partitions[i]:
            label = int(labels[idx])
            classes[label] = classes.get(label, 0) + 1
        logger.info(f"Partition {i}: {len(partitions[i])} samples, class distribution: {classes}")
    
    return partitions


class NonIidPartitioner:
    """Implementation of McMahan et al. non-IID partitioning strategy for Flower.
    
    This partitioner implements the non-IID partitioning strategy described in
    "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    (McMahan et al., AISTATS 2017).
    """
    
    def __init__(self, num_partitions: int, shards_per_client: int = 2, seed: Optional[int] = 42):
        """Initialize the partitioner.
        
        Parameters
        ----------
        num_partitions : int
            Number of partitions to create.
        shards_per_client : int, optional
            Number of shards per client, default is 2 as in the original paper.
        seed : int, optional
            Random seed for reproducibility. Default is 42.
        """
        self.num_partitions = num_partitions
        self.shards_per_client = shards_per_client
        self.seed = seed
        self.partition_indices = None
    
    def __call__(self, dataset: Dataset) -> Dict[int, Dataset]:
        """Partition the dataset and return a dictionary mapping partition IDs to datasets.
        
        Parameters
        ----------
        dataset : Dataset
            The dataset to partition, expected to be a HuggingFace Dataset.
            
        Returns
        -------
        Dict[int, Dataset]
            A dictionary mapping partition IDs to datasets.
        """
        # Convert dataset to a format compatible with noniid_partition
        # For HuggingFace datasets, we need to extract labels
        label_column = "label" if "label" in dataset.column_names else "fine_label"
        
        # Get partition indices using the noniid_partition function
        self.partition_indices = noniid_partition(
            dataset=(None, dataset[label_column]),  # Pass as (None, labels) since we only need labels for partitioning
            num_partitions=self.num_partitions,
            shards_per_client=self.shards_per_client,
            class_key=label_column,
            seed=self.seed
        )
        
        # Create subsets for each partition
        partitions = {}
        for i, indices in enumerate(self.partition_indices):
            # Convert indices to a list if they aren't already
            indices_list = list(indices)
            # Use HuggingFace's select method to create a subset
            partitions[i] = dataset.select(indices_list)
            
        return partitions


def apply_transforms(batch: Dict) -> Dict:
    """Apply transforms to the batch of images.
    
    Parameters
    ----------
    batch : dict
        Batch of data from HuggingFace dataset
        
    Returns
    -------
    dict
        Transformed batch
    """
    # Define transforms for CIFAR-10
    pytorch_transforms = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    images = [pytorch_transforms(img) for img in batch["img"]]
    # Stack instead of list to ensure proper batching
    batch["img"] = torch.stack(images)
    return batch


# Initialize FederatedDataset cache
_fds_cache = {"dataset": None, "partitions": None, "test": None}

def load_data(
    partition_id: int, 
    num_partitions: int, 
    shards_per_client: int = 2,
    dataset_name: str = "cifar10", 
    test_size: float = 0.2,
    batch_size: int = 32, 
    seed: int = 42,
    cache_dir: Optional[str] = None,
    partitions_dir: str = "./data/partitions"
) -> Tuple[DataLoader, DataLoader]:
    """Load partitioned data using non-IID partitioning following McMahan et al.
    
    Parameters
    ----------
    partition_id : int
        ID of the partition to load (0-indexed).
    num_partitions : int
        Total number of partitions.
    shards_per_client : int, optional
        Number of shards per client, default is 2 as in McMahan et al.
    dataset_name : str, optional
        Name of the dataset to load, default is "cifar10".
    test_size : float, optional
        Fraction of data to use for testing, default is 0.2.
    batch_size : int, optional
        Batch size for data loaders, default is 32.
    seed : int, optional
        Random seed for reproducibility, default is 42.
    cache_dir : Optional[str], optional
        Directory to cache dataset files, default is None.
        
    Returns
    -------
    Tuple[DataLoader, DataLoader]
        A tuple containing the training and test data loaders.
    """
    global _fds_cache
        
    # Find the most recent partition file for this number of clients
    def find_latest_partition_file(directory, num_clients):
        # First check if there's a timestamped version
        pattern = f"{directory}/noniid_partitions_{num_clients}_*.pt"
        timestamped_files = glob.glob(pattern)
        
        # If timestamped files exist, return the most recent one
        if timestamped_files:
            # Sort by timestamp (descending)
            timestamped_files.sort(reverse=True)
            logger.info(f"Found {len(timestamped_files)} timestamped partition files")
            logger.info(f"Using most recent: {timestamped_files[0]}")
            return timestamped_files[0]
        
        # If no timestamped files, check for the default file
        default_file = f"{directory}/noniid_partitions_{num_clients}.pt"
        if os.path.exists(default_file):
            logger.info(f"Using default partition file: {default_file}")
            return default_file
        
        return None
    
    # Find the most recent partition file
    partition_file = find_latest_partition_file(partitions_dir, num_partitions)
    use_predefined_partitions = partition_file is not None

    # Initialize dataset if not already cached
    if _fds_cache["dataset"] is None:
        # Set up cache directory for datasets if provided
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            datasets.config.DOWNLOADED_DATASETS_PATH = cache_dir
            logger.info(f"Using cache directory: {cache_dir}")
            
        # Configure dataset source based on the dataset name
        if dataset_name.lower() == "cifar10":
            dataset_source = "uoft-cs/cifar10"
        else:
            # Support for other datasets can be added here
            dataset_source = dataset_name
        
        try:
            # Load dataset
            dataset = datasets.load_dataset(dataset_source)
            
            # Use the train split as our base dataset
            if 'train' in dataset:
                train_dataset = dataset['train']
                test_dataset = dataset.get('test', None)
                # Cache the full dataset
                _fds_cache["dataset"] = dataset
                _fds_cache["test"] = test_dataset
                
                # If predefined partitions exist, load them
                if use_predefined_partitions:
                    logger.info(f"Loading predefined partitions from {partition_file}")
                    try:
                        partition_indices = torch.load(partition_file)
                        
                        # Create partitions using the loaded indices
                        partitions = {}
                        for i, indices in enumerate(partition_indices):
                            if i < num_partitions:  # Only use the requested number of partitions
                                partitions[i] = train_dataset.select(indices)
                        
                        _fds_cache["partitions"] = partitions
                        logger.info(f"Loaded {len(partitions)} predefined partitions from {partition_file}")
                    except Exception as e:
                        logger.error(f"Failed to load predefined partitions: {str(e)}")
                        logger.info("Falling back to creating new partitions")
                        use_predefined_partitions = False
                
                # If no predefined partitions or loading failed, create new ones
                if not use_predefined_partitions:
                    # Initialize the non-IID partitioner
                    partitioner = NonIidPartitioner(
                        num_partitions=num_partitions,
                        shards_per_client=shards_per_client,
                        seed=seed
                    )
                    
                    # Get partitions
                    partitions = partitioner(train_dataset)
                    _fds_cache["partitions"] = partitions
                    
                    logger.info(f"Created {num_partitions} non-IID partitions")
                
                logger.info(f"Loaded dataset {dataset_name} with {len(train_dataset)} training samples")
            else:
                raise ValueError(f"Dataset {dataset_name} does not have a 'train' split")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {dataset_name}: {str(e)}")
    
    # Get the specific partition
    partitions = _fds_cache["partitions"]
    if partition_id not in partitions:
        raise ValueError(f"Partition {partition_id} not found. Available partitions: {list(partitions.keys())}")
    
    partition = partitions[partition_id]
    
    # Divide data on each node: 80% train, 20% test using the local partition
    train_test_split = partition.train_test_split(test_size=test_size, seed=seed)
    
    # Apply transforms
    transformed_data = train_test_split.with_transform(apply_transforms)
    
    # Create data loaders with a single worker to avoid pickling issues
    trainloader = DataLoader(
        transformed_data["train"], 
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 workers to avoid multiprocessing pickling issues
        pin_memory=torch.cuda.is_available()
    )
    
    testloader = DataLoader(
        transformed_data["test"], 
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 workers to avoid multiprocessing pickling issues
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Loaded partition {partition_id} with {len(transformed_data['train'])} training "
               f"and {len(transformed_data['test'])} testing samples")
    
    return trainloader, testloader


def get_weights(
    net: nn.Module, 
    include_optimizer_state: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    compress_state: bool = False
) -> List[np.ndarray]:
    """Get model parameters as a list of NumPy arrays.
    
    Parameters
    ----------
    net : nn.Module
        The neural network model.
    include_optimizer_state : bool, optional
        Whether to include optimizer state in the returned parameters, default is False.
    optimizer : Optional[torch.optim.Optimizer], optional
        The optimizer to extract state from, required if include_optimizer_state is True.
    compress_state : bool, optional
        Whether to compress the optimizer state using zlib, default is False.
        
    Returns
    -------
    List[np.ndarray]
        List of model parameter arrays, with optimizer state appended if requested.
    """
    # Get model weights as numpy arrays
    weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
    
    # If optimizer state not requested, return just the model weights
    if not include_optimizer_state or optimizer is None:
        return weights
    
    try:
        # Get optimizer state
        optimizer_state = optimizer.state_dict()
        
        # Convert optimizer state to a serializable format
        serializable_state = serialize_optimizer_state(optimizer_state, compress=compress_state)
        
        # Add metadata about optimizer state format
        metadata = {
            "version": "1.0",
            "type": type(optimizer).__name__,
            "compressed": compress_state,
            "format": "json_string" if not compress_state else "compressed_json_string",
            "position": "last_parameter"
        }
        
        # Create numpy array from the serialized state and metadata
        optimizer_state_bytes = json.dumps({
            "metadata": metadata,
            "state": serializable_state
        }).encode('utf-8')
        
        # Convert to numpy array and append to weights
        optimizer_array = np.frombuffer(optimizer_state_bytes, dtype=np.uint8)
        weights.append(optimizer_array)
        
        logger.info(f"Added optimizer state to parameters (size: {len(optimizer_array)} bytes)")
        
        return weights
    except Exception as e:
        logger.error(f"Failed to serialize optimizer state: {str(e)}")
        logger.warning("Returning model weights without optimizer state")
        return weights


def serialize_optimizer_state(
    optimizer_state: Dict, 
    compress: bool = False
) -> Union[Dict, str]:
    """Serialize optimizer state to a format suitable for transmission.
    
    Parameters
    ----------
    optimizer_state : Dict
        The optimizer state dictionary from optimizer.state_dict().
    compress : bool, optional
        Whether to compress the serialized state, default is False.
        
    Returns
    -------
    Union[Dict, str]
        Serialized optimizer state, either as a dictionary or compressed string.
    """
    # Create a deep copy to avoid modifying the original
    state_copy = copy.deepcopy(optimizer_state)
    
    # Convert tensors to lists
    def tensor_to_list(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: tensor_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [tensor_to_list(item) for item in obj]
        else:
            return obj
    
    # Convert all tensors in the state to lists
    serializable_state = tensor_to_list(state_copy)
    
    # If compression is requested, compress the JSON string
    if compress:
        # Convert to JSON string
        json_str = json.dumps(serializable_state)
        # Compress using zlib
        compressed_bytes = zlib.compress(json_str.encode('utf-8'))
        # Encode as base64 string for transmission
        return base64.b64encode(compressed_bytes).decode('utf-8')
    else:
        # Return as serializable dictionary
        return serializable_state


def deserialize_optimizer_state(
    serialized_state: Union[Dict, str],
    compressed: bool = False
) -> Dict:
    """Deserialize optimizer state from a transmitted format.
    
    Parameters
    ----------
    serialized_state : Union[Dict, str]
        The serialized optimizer state, either as a dictionary or compressed string.
    compressed : bool, optional
        Whether the state is compressed, default is False.
        
    Returns
    -------
    Dict
        Deserialized optimizer state dictionary.
    """
    # Decompress if state is compressed
    if compressed:
        # Decode base64 string
        compressed_bytes = base64.b64decode(serialized_state)
        # Decompress using zlib
        json_str = zlib.decompress(compressed_bytes).decode('utf-8')
        # Parse JSON string
        state_dict = json.loads(json_str)
    else:
        # Use as is if not compressed and already a dict
        if isinstance(serialized_state, dict):
            state_dict = serialized_state
        else:
            # Parse JSON string if it's a string
            state_dict = json.loads(serialized_state)
    
    # Convert lists back to tensors
    def list_to_tensor(obj):
        if isinstance(obj, list):
            # Try to convert to tensor if it looks like a tensor
            try:
                return torch.tensor(obj)
            except:
                return [list_to_tensor(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: list_to_tensor(v) for k, v in obj.items()}
        else:
            return obj
    
    # Convert all lists that were tensors back to tensors
    return list_to_tensor(state_dict)


def has_optimizer_state(parameters: List[np.ndarray]) -> bool:
    """Check if parameters list contains optimizer state.
    
    Parameters
    ----------
    parameters : List[np.ndarray]
        List of model parameter arrays.
        
    Returns
    -------
    bool
        True if parameters contains optimizer state, False otherwise.
    """
    if not parameters or len(parameters) == 0:
        return False
    
    # Try to interpret the last parameter as optimizer state metadata
    try:
        last_param = parameters[-1]
        # Check if it's a 1D array of uint8
        if last_param.dtype == np.uint8 and last_param.ndim == 1:
            # Try to decode and parse as JSON
            json_str = last_param.tobytes().decode('utf-8')
            state_data = json.loads(json_str)
            # Check if it has the expected structure
            if (isinstance(state_data, dict) and 
                "metadata" in state_data and 
                "state" in state_data and
                "position" in state_data["metadata"] and
                state_data["metadata"]["position"] == "last_parameter"):
                return True
    except Exception:
        # If any error occurs during parsing, assume it's not optimizer state
        pass
    
    return False


def extract_optimizer_state(
    parameters: List[np.ndarray]
) -> Tuple[List[np.ndarray], Optional[Dict]]:
    """Extract optimizer state from parameters list if present.
    
    Parameters
    ----------
    parameters : List[np.ndarray]
        List of model parameter arrays, potentially with optimizer state.
        
    Returns
    -------
    Tuple[List[np.ndarray], Optional[Dict]]
        A tuple containing:
        - List of model parameter arrays without optimizer state
        - Extracted optimizer state, or None if not present
    """
    if not has_optimizer_state(parameters):
        return parameters, None
    
    try:
        # Get the last parameter which contains optimizer state
        state_param = parameters[-1]
        
        # Decode the bytes to a string
        json_str = state_param.tobytes().decode('utf-8')
        
        # Parse the JSON string
        state_data = json.loads(json_str)
        
        # Extract metadata and state
        metadata = state_data["metadata"]
        serialized_state = state_data["state"]
        
        # Deserialize the state based on format
        is_compressed = metadata.get("compressed", False)
        optimizer_state = deserialize_optimizer_state(serialized_state, compressed=is_compressed)
        
        # Return model parameters without optimizer state, and the extracted state
        return parameters[:-1], optimizer_state
    except Exception as e:
        logger.error(f"Failed to extract optimizer state: {str(e)}")
        logger.warning("Returning model parameters without optimizer state")
        return parameters, None


def set_weights(
    net: nn.Module, 
    parameters: List[np.ndarray],
    optimizer: Optional[torch.optim.Optimizer] = None,
    handle_optimizer_state: bool = False
) -> Optional[Dict]:
    """Set model parameters from a list of NumPy arrays.
    
    Parameters
    ----------
    net : nn.Module
        The neural network model.
    parameters : List[np.ndarray]
        List of model parameter arrays, potentially with optimizer state.
    optimizer : Optional[torch.optim.Optimizer], optional
        The optimizer to update with extracted state, default is None.
    handle_optimizer_state : bool, optional
        Whether to handle optimizer state if present in parameters, default is False.
        
    Returns
    -------
    Optional[Dict]
        Extracted optimizer state if handle_optimizer_state is True and state is present,
        None otherwise.
    """
    # If optimizer state handling is enabled, check for and extract it
    optimizer_state = None
    model_params = parameters
    
    if handle_optimizer_state and optimizer is not None:
        model_params, optimizer_state = extract_optimizer_state(parameters)
    
    # Set model parameters
    params_dict = zip(net.state_dict().keys(), model_params)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    
    # Set optimizer state if available and optimizer is provided
    if optimizer_state is not None and optimizer is not None:
        try:
            # Create a new state dict with the optimizer's structure
            optimizer_state_dict = optimizer.state_dict()
            
            # Update state dict with the extracted state
            # Handle potential mismatch in structure
            if "state" in optimizer_state and "state" in optimizer_state_dict:
                optimizer_state_dict["state"] = optimizer_state["state"]
            if "param_groups" in optimizer_state and "param_groups" in optimizer_state_dict:
                # Keep parameter references from current optimizer but update hyperparameters
                for pg_idx, pg in enumerate(optimizer_state["param_groups"]):
                    if pg_idx < len(optimizer_state_dict["param_groups"]):
                        # Update hyperparameters but keep parameter references
                        current_pg = optimizer_state_dict["param_groups"][pg_idx]
                        for k, v in pg.items():
                            if k != "params":  # Don't overwrite parameter references
                                current_pg[k] = v
            
            # Load the updated state dict
            optimizer.load_state_dict(optimizer_state_dict)
            logger.info("Successfully applied optimizer state")
        except Exception as e:
            logger.error(f"Failed to set optimizer state: {str(e)}")
    
    return optimizer_state


def train(
    net: nn.Module, 
    trainloader: DataLoader, 
    epochs: int, 
    device: torch.device = None,
    use_adam: bool = False,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    log_per_step_metrics: bool = False,
    callback: Optional[Callable[[int, int, float, float], None]] = None,
    initial_optimizer_state: Optional[Dict] = None,
    capture_optimizer_state: bool = False
) -> Tuple[float, Optional[Dict], Optional[torch.optim.Optimizer]]:
    """Train the model on the training set.
    
    Parameters
    ----------
    net : nn.Module
        The neural network model to train.
    trainloader : DataLoader
        DataLoader for the training data.
    epochs : int
        Number of epochs to train.
    device : torch.device, optional
        Device to use for training. If None, use CUDA if available, else CPU.
    use_adam : bool, optional
        Whether to use Adam optimizer. If False, use SGD.
    lr : float, optional
        Learning rate for the optimizer.
    beta1 : float, optional
        Beta1 parameter for Adam optimizer.
    beta2 : float, optional
        Beta2 parameter for Adam optimizer.
    epsilon : float, optional
        Epsilon parameter for Adam optimizer.
    log_per_step_metrics : bool, optional
        Whether to log metrics for each step.
    callback : Optional[Callable[[int, int, float, float], None]], optional
        Callback function to call after each step with args (epoch, step, loss, accuracy).
    initial_optimizer_state : Optional[Dict], optional
        Initial optimizer state to use, default is None.
    capture_optimizer_state : bool, optional
        Whether to capture and return optimizer state, default is False.
        
    Returns
    -------
    Tuple[float, Optional[Dict], Optional[torch.optim.Optimizer]]
        A tuple containing:
        - Average training loss over the training set
        - Step metrics if requested, else None
        - Optimizer if capture_optimizer_state is True, else None
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    net.to(device)
    
    # Create optimizer based on configuration
    if use_adam:
        optimizer = torch.optim.Adam(
            net.parameters(), 
            lr=lr, 
            betas=(beta1, beta2), 
            eps=epsilon
        )
        logger.info(f"Using Adam optimizer with learning rate {lr}, beta1={beta1}, beta2={beta2}, eps={epsilon}")
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        logger.info(f"Using SGD optimizer with learning rate {lr} and momentum 0.9")
    
    # Load initial optimizer state if provided
    if initial_optimizer_state is not None and use_adam:
        try:
            # Create a new state dict with the optimizer's structure
            optimizer_state_dict = optimizer.state_dict()
            
            # Update state dict with the provided state
            if "state" in initial_optimizer_state and "state" in optimizer_state_dict:
                optimizer_state_dict["state"] = initial_optimizer_state["state"]
            if "param_groups" in initial_optimizer_state and "param_groups" in optimizer_state_dict:
                # Keep parameter references but update hyperparameters
                for pg_idx, pg in enumerate(initial_optimizer_state["param_groups"]):
                    if pg_idx < len(optimizer_state_dict["param_groups"]):
                        current_pg = optimizer_state_dict["param_groups"][pg_idx]
                        for k, v in pg.items():
                            if k != "params":  # Don't overwrite parameter references
                                current_pg[k] = v
            
            # Load the updated state dict
            optimizer.load_state_dict(optimizer_state_dict)
            logger.info("Successfully loaded initial optimizer state")
        except Exception as e:
            logger.error(f"Failed to set initial optimizer state: {str(e)}")
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    net.train()
    
    running_loss = 0.0
    total_batches = 0
    
    # For per-step metrics
    step_metrics = None
    if log_per_step_metrics:
        step_metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "loss": [],         # Standard key name
            "accuracy": [],     # Standard key name
            "steps": [],
            "num_examples": []  # Number of examples per step
        }
    
    global_step = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(trainloader):
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy for this batch
            _, predicted = torch.max(outputs.data, 1)
            batch_total = labels.size(0)
            batch_correct = (predicted == labels).sum().item()
            
            total += batch_total
            correct += batch_correct
            
            batch_loss = loss.item()
            epoch_loss += batch_loss
            epoch_batches += 1
            global_step += 1
            
            # Log per-step metrics if requested
            if log_per_step_metrics and step_metrics is not None:
                batch_accuracy = batch_correct / batch_total if batch_total > 0 else 0
                # Add standard key names with appropriate prefixes
                step_metrics["train_loss"].append(batch_loss)
                step_metrics["train_accuracy"].append(batch_accuracy)
                step_metrics["loss"].append(batch_loss)  # Standard key
                step_metrics["accuracy"].append(batch_accuracy)  # Standard key
                step_metrics["steps"].append(global_step)
                step_metrics["num_examples"].append(batch_total)
            
            # Call callback if provided
            if callback is not None:
                batch_accuracy = batch_correct / batch_total if batch_total > 0 else 0
                callback(epoch, global_step, batch_loss, batch_accuracy)
        
        # Log per-epoch statistics
        if epoch_batches > 0:
            avg_epoch_loss = epoch_loss / epoch_batches
            avg_epoch_acc = correct / total if total > 0 else 0
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {avg_epoch_loss:.4f}, Accuracy = {avg_epoch_acc:.4f}")
            
        running_loss += epoch_loss
        total_batches += epoch_batches
    
    avg_trainloss = running_loss / total_batches if total_batches > 0 else 0
    
    # Prepare return values based on what was requested
    result = [avg_trainloss]
    
    # Add step metrics if requested
    result.append(step_metrics)
    
    # Add optimizer if requested
    if capture_optimizer_state:
        result.append(optimizer)
    else:
        result.append(None)
    
    # Convert to tuple
    return tuple(result)


def test(
    net: nn.Module, 
    testloader: DataLoader, 
    device: torch.device = None
) -> Tuple[float, float]:
    """Validate the model on the test set.
    
    Parameters
    ----------
    net : nn.Module
        The neural network model to evaluate.
    testloader : DataLoader
        DataLoader for the test data.
    device : torch.device, optional
        Device to use for evaluation. If None, use CUDA if available, else CPU.
        
    Returns
    -------
    Tuple[float, float]
        A tuple containing the average test loss and accuracy.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    net.to(device)
    net.eval()
    
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, running_loss = 0, 0, 0.0
    
    # Track number of examples for proper weighting
    num_examples = 0
    
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            
            outputs = net(images)
            loss = criterion(outputs, labels).item()
            running_loss += loss
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Keep track of number of examples
            num_examples += labels.size(0)
    
    accuracy = correct / total if total > 0 else 0
    avg_loss = running_loss / len(testloader) if len(testloader) > 0 else 0
    
    logger.info(f"Testing: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}, Examples = {num_examples}")
    
    return avg_loss, accuracy


def save_metrics(metrics: Dict, filepath: str) -> None:
    """Save metrics to a JSON file.
    
    Parameters
    ----------
    metrics : Dict
        Metrics to save.
    filepath : str
        Path to save the metrics to.
    """
    # Convert NumPy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_for_json(val) for key, val in obj.items()}
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        else:
            return obj
    
    # Convert metrics for JSON serialization
    metrics_dict = convert_for_json(metrics)
    
    # Ensure consistent metric key names
    standardized_metrics = standardize_metric_keys(metrics_dict)
    
    # Flatten optimizer state metrics if present - added for optimizer state handling
    flattened_metrics = flatten_optimizer_state_metrics(standardized_metrics)
    
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if directory:  # Only try to create directory if path has a directory component
        os.makedirs(directory, exist_ok=True)
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(flattened_metrics, f, indent=4)
    
    logger.info(f"Metrics saved to {filepath}")


def flatten_optimizer_state_metrics(metrics: Dict) -> Dict:
    """Flatten optimizer state metrics for serialization.
    
    Parameters
    ----------
    metrics : Dict
        Metrics dictionary that might contain nested optimizer state.
        
    Returns
    -------
    Dict
        Flattened metrics dictionary.
    """
    # Create a copy to avoid modifying the original
    flattened = copy.deepcopy(metrics)
    
    # Check if optimizer state is present
    if "optimizer_state" in flattened:
        optimizer_state = flattened.pop("optimizer_state")
        
        # Extract and flatten key metrics from optimizer state
        if isinstance(optimizer_state, dict):
            # Extract learning rate if available
            if "param_groups" in optimizer_state and len(optimizer_state["param_groups"]) > 0:
                if "lr" in optimizer_state["param_groups"][0]:
                    flattened["learning_rate"] = optimizer_state["param_groups"][0]["lr"]
            
            # Add a simple flag to indicate optimizer state was present
            flattened["has_optimizer_state"] = True
    
    # Recursively process nested dictionaries
    for key, value in list(flattened.items()):
        if isinstance(value, dict):
            flattened[key] = flatten_optimizer_state_metrics(value)
    
    return flattened


def standardize_metric_keys(metrics: Dict) -> Dict:
    """Standardize metric key names for consistency with WandB.
    
    Parameters
    ----------
    metrics : Dict
        Original metrics dictionary.
        
    Returns
    -------
    Dict
        Metrics dictionary with standardized key names.
    """
    # Define mapping for consistent key naming
    key_mapping = {
        "test_acc": "test_accuracy",
        "avg_accuracy": "test_accuracy",  
        "avg_loss": "test_loss",          
        "train_acc": "train_accuracy",
    }
    
    # Handle special case of nested dictionaries
    if isinstance(metrics, dict):
        standardized = {}
        for key, value in metrics.items():
            # Process dictionary values
            if isinstance(value, dict):
                standardized[key] = standardize_metric_keys(value)
            # Process list values that might contain dictionaries
            elif isinstance(value, list):
                if all(isinstance(item, dict) for item in value):
                    standardized[key] = [standardize_metric_keys(item) for item in value]
                else:
                    standardized[key] = value
            # Map standard keys and keep original for backward compatibility
            elif key in key_mapping:
                standardized[key] = value  # Keep original key
                standardized[key_mapping[key]] = value  # Add standard key
            else:
                standardized[key] = value
        return standardized
    else:
        return metrics


def load_or_initialize_model(
    weights_path: str, 
    bn_private: str = "none",
    save_init: bool = False,
) -> CIFAR10CNN:
    """Load model weights from file or initialize a new model.
    
    Parameters
    ----------
    weights_path : str
        Path to model weights file.
    bn_private : str, optional
        Which BN parameters to keep private, default is "none".
    save_init : bool, optional
        Whether to save initial weights if they don't exist, default is False.
        
    Returns
    -------
    CIFAR10CNN
        The loaded or initialized model.
    """
    # Create model instance
    model = CIFAR10CNN(bn_private=bn_private)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    
    # Try to load weights if they exist
    if os.path.exists(weights_path) and not save_init:
        try:
            logger.info(f"Loading weights from {weights_path}")
            model.load_state_dict(torch.load(weights_path))
        except Exception as e:
            logger.warning(f"Failed to load weights from {weights_path}: {str(e)}")
            logger.warning("Initializing model with random weights")
    # Save initial weights if requested
    elif save_init or not os.path.exists(weights_path):
        logger.info(f"Saving initial weights to {weights_path}")
        torch.save(model.state_dict(), weights_path)
    
    return model


def save_checkpoint(
    net: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    round_num: int,
    metrics: Dict,
    filepath: str,
    include_optimizer_state: bool = True
) -> None:
    """Save a checkpoint of the model and optimizer state.
    
    Parameters
    ----------
    net : nn.Module
        The neural network model.
    optimizer : Optional[torch.optim.Optimizer]
        The optimizer.
    round_num : int
        The current round number.
    metrics : Dict
        Training metrics.
    filepath : str
        Path to save the checkpoint to.
    include_optimizer_state : bool, optional
        Whether to include optimizer state in the checkpoint, default is True.
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    # Create checkpoint dict
    checkpoint = {
        'round': round_num,
        'model_state_dict': net.state_dict(),
        'metrics': metrics,
    }
    
    # Add optimizer state if requested and optimizer is provided
    if include_optimizer_state and optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str,
    net: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Tuple[int, Dict]:
    """Load a checkpoint of the model and optimizer state.
    
    Parameters
    ----------
    filepath : str
        Path to load the checkpoint from.
    net : nn.Module
        The neural network model.
    optimizer : Optional[torch.optim.Optimizer], optional
        The optimizer, default is None.
        
    Returns
    -------
    Tuple[int, Dict]
        A tuple containing:
        - The round number
        - Training metrics
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    # Load checkpoint
    checkpoint = torch.load(filepath)
    
    # Load model state
    net.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load optimizer state: {str(e)}")
    
    # Return round number and metrics
    return checkpoint.get('round', 0), checkpoint.get('metrics', {})


def aggregate_optimizer_states(
    optimizer_states: List[Dict],
    aggregation_method: str = "average",
    weights: Optional[List[float]] = None
) -> Dict:
    """Aggregate optimizer states from multiple clients.
    
    Parameters
    ----------
    optimizer_states : List[Dict]
        List of optimizer state dictionaries.
    aggregation_method : str, optional
        Method to use for aggregation, default is "average".
        Options: "average", "weighted_average"
    weights : Optional[List[float]], optional
        Weights for weighted averaging, required if aggregation_method is "weighted_average".
        
    Returns
    -------
    Dict
        Aggregated optimizer state.
    """
    if not optimizer_states:
        logger.warning("No optimizer states to aggregate")
        return {}
    
    # Use the first optimizer state as a template
    aggregated_state = copy.deepcopy(optimizer_states[0])
    
    # If only one state, return it directly
    if len(optimizer_states) == 1:
        return aggregated_state
    
    # Check if weights are provided for weighted averaging
    if aggregation_method == "weighted_average" and (not weights or len(weights) != len(optimizer_states)):
        logger.warning("Missing or invalid weights for weighted averaging, falling back to simple average")
        aggregation_method = "average"
        weights = None
    
    # If using simple average, create equal weights
    if aggregation_method == "average" or weights is None:
        weights = [1.0 / len(optimizer_states)] * len(optimizer_states)
    else:
        # Normalize weights to sum to 1
        weight_sum = sum(weights)
        if weight_sum == 0:
            logger.warning("Weights sum to zero, using equal weights")
            weights = [1.0 / len(optimizer_states)] * len(optimizer_states)
        else:
            weights = [w / weight_sum for w in weights]
    
    # Only aggregate "state" part which contains the actual optimizer variables
    if "state" in aggregated_state:
        # Iterate through parameter indices
        for param_idx in aggregated_state["state"]:
            # Initialize parameter state if it doesn't exist in all states
            for state_idx, state in enumerate(optimizer_states):
                if "state" not in state or param_idx not in state["state"]:
                    continue
            
            # Iterate through each variable in the parameter state (e.g., exp_avg, exp_avg_sq)
            for var_name in aggregated_state["state"][param_idx]:
                # Skip non-tensor values
                if not isinstance(aggregated_state["state"][param_idx][var_name], torch.Tensor):
                    continue
                
                # Reset the aggregated value
                aggregated_state["state"][param_idx][var_name].zero_()
                
                # Aggregate values from all clients
                for state_idx, state in enumerate(optimizer_states):
                    if ("state" in state and 
                        param_idx in state["state"] and 
                        var_name in state["state"][param_idx] and
                        isinstance(state["state"][param_idx][var_name], torch.Tensor)):
                        # Add weighted contribution
                        aggregated_state["state"][param_idx][var_name].add_(
                            state["state"][param_idx][var_name], 
                            alpha=weights[state_idx]
                        )
    
    # For hyperparameters in param_groups, use the first state's values
    # These should be the same across clients anyway
    
    return aggregated_state