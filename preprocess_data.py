"""
Preprocess data for federated learning experiments.

This script partitions CIFAR-10 data in a non-IID manner following McMahan et al. (2017)
for use in federated learning experiments.
"""

import argparse
import os
import logging
import json
import random
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
import datasets
from datasets import Dataset

# Add the current directory to the path so we can import fl_mtfl
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fl_mtfl.task import noniid_partition, seed_everything
from fl_mtfl.config import CONFIG, get_parser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fl_mtfl.preprocess")


def setup_directories(directories: List[str]) -> None:
    """Create necessary directories for data storage.
    
    Parameters
    ----------
    directories : List[str]
        List of directories to create.
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory ensured: {directory}")


def download_cifar10() -> Tuple[Dataset, Dataset]:
    """Download CIFAR-10 dataset from Hugging Face.
    
    Returns
    -------
    Tuple[Dataset, Dataset]
        Tuple of (train_dataset, test_dataset)
    """
    logger.info("Downloading CIFAR-10 dataset...")
    
    try:
        cifar10 = datasets.load_dataset("uoft-cs/cifar10")
        trainset = cifar10["train"]
        testset = cifar10["test"]
        
        logger.info(f"Downloaded CIFAR-10 dataset: {len(trainset)} training, {len(testset)} test examples")
        return trainset, testset
    
    except Exception as e:
        logger.error(f"Failed to download CIFAR-10 dataset: {str(e)}")
        raise


def analyze_partitions(
    partition_indices: List[List[int]], 
    labels: List[int], 
    num_classes: int = 10
) -> Dict:
    """Analyze the distribution of classes in the partitions.
    
    Parameters
    ----------
    partition_indices : List[List[int]]
        List of partition indices.
    labels : List[int]
        List of class labels for each sample.
    num_classes : int, optional
        Number of classes in the dataset, default is 10.
    
    Returns
    -------
    Dict
        Dictionary containing partition statistics.
    """
    # Initialize statistics
    stats = {
        "num_partitions": len(partition_indices),
        "partition_sizes": [],
        "class_distributions": [],
        "avg_classes_per_partition": 0,
        "class_imbalance": []
    }
    
    # Calculate class distribution for each partition
    unique_classes_counts = []
    
    for i, indices in enumerate(partition_indices):
        # Count number of samples per class
        class_counts = np.zeros(num_classes)
        for idx in indices:
            class_counts[labels[idx]] += 1
        
        # Calculate class distribution (as percentages)
        class_distribution = (class_counts / len(indices) * 100).tolist()
        
        # Count unique classes with at least one sample
        unique_classes = sum(1 for count in class_counts if count > 0)
        unique_classes_counts.append(unique_classes)
        
        # Calculate imbalance (std deviation of class percentages)
        non_zero_percentages = [pct for pct, count in zip(class_distribution, class_counts) if count > 0]
        imbalance = np.std(non_zero_percentages) if non_zero_percentages else 0
        
        # Store statistics
        stats["partition_sizes"].append(len(indices))
        stats["class_distributions"].append(class_counts.tolist())
        stats["class_imbalance"].append(imbalance)
        
        # Display sample statistics for first few partitions
        if i < 5:
            logger.info(f"Partition {i}: {len(indices)} samples, {unique_classes} classes")
            class_counts_dict = {c: int(count) for c, count in enumerate(class_counts) if count > 0}
            logger.info(f"  Class distribution: {class_counts_dict}")
    
    # Calculate average statistics
    stats["avg_classes_per_partition"] = np.mean(unique_classes_counts)
    stats["avg_partition_size"] = np.mean(stats["partition_sizes"])
    stats["avg_class_imbalance"] = np.mean(stats["class_imbalance"])
    
    logger.info(f"Average unique classes per partition: {stats['avg_classes_per_partition']:.2f}")
    logger.info(f"Average partition size: {stats['avg_partition_size']:.2f} samples")
    logger.info(f"Average class imbalance: {stats['avg_class_imbalance']:.2f}%")
    
    return stats


def save_partitions(
    partition_indices: List[List[int]], 
    num_clients: int,
    stats: Dict,
    output_dir: str = "./data/partitions"
) -> Tuple[str, str]:
    """Save partition indices and statistics.
    
    Parameters
    ----------
    partition_indices : List[List[int]]
        List of partition indices.
    num_clients : int
        Number of clients/partitions.
    stats : Dict
        Partition statistics.
    output_dir : str, optional
        Directory to save partitions, default is "./data/partitions".
    Returns
    -------
    Tuple[str, str]
        Tuple of (partition_file_path, stats_file_path)
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for file versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save partition indices with timestamp
    partition_file = f"{output_dir}/noniid_partitions_{num_clients}_{timestamp}.pt"
    torch.save(partition_indices, partition_file)
    logger.info(f"Saved partition indices to {partition_file}")
    
    # Save statistics with timestamp
    stats_file = f"{output_dir}/partition_stats_{num_clients}_{timestamp}.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved partition statistics to {stats_file}")
    
    # Also save as the default filename for backward compatibility
    default_partition_file = f"{output_dir}/noniid_partitions_{num_clients}.pt"
    torch.save(partition_indices, default_partition_file)
    
    default_stats_file = f"{output_dir}/partition_stats_{num_clients}.json"
    with open(default_stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Also saved as default filenames for compatibility")
    
    return partition_file, stats_file


def create_partitions(
    num_clients: int, 
    shards_per_client: int = 2, 
    seed: int = 42,
    output_dir: str = "./data/partitions"
) -> Tuple[str, str]:
    """Create and save non-IID partitions of CIFAR-10.
    
    Parameters
    ----------
    num_clients : int
        Number of clients to create partitions for.
    shards_per_client : int, optional
        Number of shards per client, default is 2.
    seed : int, optional
        Random seed for reproducibility, default is 42.
    output_dir : str, optional
        Directory to save partitions, default is "./data/partitions".
        
    Returns
    -------
    Tuple[str, str]
        Tuple of (partition_file_path, stats_file_path)
    """
    # Set random seed
    seed_everything(seed)
    
    # Create directories
    setup_directories([
        "./data", 
        output_dir, 
        "./data/weights",
        "./metrics",
        "./figures"
    ])
    
    # Check if partitions already exist
    default_partition_file = f"{output_dir}/noniid_partitions_{num_clients}.pt"
    
    if os.path.exists(default_partition_file):
        logger.info(f"Partitions already exist at {default_partition_file}. Creating new version with timestamp.")
    
    # Download dataset
    trainset, testset = download_cifar10()
    
    # Extract labels
    train_labels = trainset["label"]
    
    # Create non-IID partitions
    partition_indices = noniid_partition(
        dataset=(None, train_labels),
        num_partitions=num_clients,
        shards_per_client=shards_per_client,
        class_key="label",
        seed=seed
    )
    
    # Analyze partitions
    stats = analyze_partitions(partition_indices, train_labels)
    
    # Save partitions with timestamp
    partition_file, stats_file = save_partitions(partition_indices, num_clients, stats, output_dir)
    
    logger.info(f"Non-IID partition process completed for {num_clients} clients")

    return partition_file, stats_file


def main():
    """Main function for data preprocessing."""
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create a configuration from command-line arguments
    config = CONFIG
    
    # Update configuration based on command-line arguments
    num_clients = args.num_clients if args.num_clients else config.num_clients
    shards = args.shards if args.shards else config.shards_per_client
    seed = args.seed if args.seed else config.seed
    output_dir = args.partitions_dir if args.partitions_dir else config.partitions_dir
    
    logger.info(f"Starting preprocessing with {num_clients} clients, "
                f"{shards} shards per client, seed {seed}")
    
    try:
        partition_file, stats_file = create_partitions(
            num_clients=num_clients,
            shards_per_client=shards,
            seed=seed,
            output_dir=output_dir
        )
        logger.info(f"Preprocessing completed successfully")
        logger.info(f"Partition file created: {partition_file}")
        logger.info(f"Statistics file created: {stats_file}")
        
        # Create a metadata file for this preprocessing run
        metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_clients": num_clients,
            "shards_per_client": shards,
            "seed": seed,
            "partition_file": partition_file,
            "stats_file": stats_file
        }
        
        metadata_file = f"{output_dir}/preprocess_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Preprocessing metadata saved to: {metadata_file}")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()