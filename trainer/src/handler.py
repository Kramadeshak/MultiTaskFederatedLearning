import os
import json
from logger import get_logger
from huggingface_hub import hf_hub_download
from dataset_manager import mnist_sharding

logger = get_logger()

DATA_DIR = "/data"  # base directory to store dataset files

def handle_message(message):
    """
    Parses messages that have been received from redis.
    """
    try:
        data = json.loads(message)
        if data["type"] == "dataset":
            dataset_handler(data)
        else:
            logger.warning(f"Unsupported message type: {data['type']}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
    except Exception as e:
        logger.error(f"Error parsing message: {e}")

def dataset_handler(data):
    dataset = data["dataset"]
    repo_id = data["repo_id"]
    train_file = data["train_file"]
    test_file = data["test_file"]
    num_clients = data["num_clients"]
    iid = (data["iid"].lower() == "true")

    dataset_dir = os.path.join(DATA_DIR, dataset.lower())
    os.makedirs(dataset_dir, exist_ok=True)

    try:
        logger.info(f"Downloading {dataset} train set...")
        hf_hub_download(repo_id=repo_id, filename=train_file, repo_type="dataset", local_dir=dataset_dir)

        logger.info(f"Downloading {dataset} test set...")
        hf_hub_download(repo_id=repo_id, filename=test_file, repo_type="dataset", local_dir=dataset_dir)

        logger.info(f"{dataset} dataset downloaded successfully and saved to {dataset_dir}")
    except Exception as e:
        logger.error(f"Dataset download failed: {e}")

    logger.info("Sharding dataset")
    if dataset == "MNIST":
        mnist_sharding(iid, num_clients, train_file, test_file)
    elif dataset == "CIFAR10":
        pass
    else:
        logger.error("No dataset mentioned")

