import pandas as pd
import numpy as np
import os
from huggingface_hub import hf_hub_download
from sklearn.utils import shuffle
from logger import get_logger

DATA_DIR = "/data"
DATASET_DIR = "/dataset"

logger = get_logger()

class DatasetRequest:
    def __init__(self, data):
        self.dataset=data["dataset"].lower()
        self.repo_id=data["repo_id"]
        self.train_file=data["train_file"]
        self.test_file=data["test_file"]
        self.iid=str(data["iid"]).lower() == "true"
        self.num_clients=int(data["num_clients"])

    def download_data(self):
        dataset_dir = os.path.join(DATA_DIR, self.dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        try:
            logger.info(f"Downloading {self.dataset} train set...")
            hf_hub_download(repo_id=self.repo_id, filename=self.train_file, repo_type="dataset", local_dir=dataset_dir)

            logger.info(f"Downloading {self.dataset} test set...")
            hf_hub_download(repo_id=self.repo_id, filename=self.test_file, repo_type="dataset", local_dir=dataset_dir)

            logger.info(f"{self.dataset} dataset downloaded successfully and saved to {dataset_dir}")
        except Exception as e:
            logger.error(f"Dataset download failed: {e}")

    def load_parquet_data(self):
        logger.info("Loading data from file...")
        train_df = pd.read_parquet(os.path.join(DATA_DIR, self.dataset, self.train_file))
        test_df = pd.read_parquet(os.path.join(DATA_DIR, self.dataset, self.test_file))

        X_train = np.vstack(train_df["image"].values)
        y_train = train_df["label"].values

        X_test = np.vstack(test_df["image"].values)
        y_test = test_df["label"].values

        return X_train, y_train, X_test, y_test

    def shard_iid(self, X, y):
        logger.info("Executing IID shard")
        X, y = shuffle(X, y, random_state=42)
        indices = np.array_split(np.arange(len(X)), self.num_clients)

        for i, idx in enumerate(indices):
            self.save_client_data(i, X[idx], y[idx])

    def shard_non_iid(self, X, y):
        logger.info("Executing non-IID shard")
        labels = np.unique(y)
        label_indices = {label: np.where(y == label)[0] for label in labels}
        shards_per_client = len(labels) // self.num_clients

        shuffled_labels = list(labels)
        np.random.seed(42)
        np.random.shuffle(shuffled_labels)

        for i in range(self.num_clients):
            selected_labels = shuffled_labels[i * shards_per_client: (i + 1) * shards_per_client]
            idx = np.hstack([label_indices[label] for label in selected_labels])
            self.save_client_data(i, X[idx], y[idx])

    def save_client_data(self, client_id, X, y):
        client_dir = os.path.join(DATASET_DIR, self.dataset, "train")
        os.makedirs(client_dir, exist_ok=True)
        np.save(os.path.join(client_dir, f"client_{client_id}_X.npy"), X)
        np.save(os.path.join(client_dir, f"client_{client_id}_y.npy"), y)
        logger.info("Saved sharded data.")

def mnist_sharding(data):
    X_train, y_train, X_test, y_test = data.load_parquet_data()
    logger.info("Loaded data from file. Proceeding to shard.")

    if data.iid:
        data.shard_iid(X_train, y_train)
    else:
        data.shard_non_iid(X_train, y_train)

    save_test_data(data.dataset, X_test, y_test)

def save_test_data(dataset, X_test, y_test):
    os.makedirs(os.path.join(DATASET_DIR, dataset, "test"), exist_ok=True)
    np.save(os.path.join(DATASET_DIR, dataset, "test", "X_test.npy"), X_test)
    np.save(os.path.join(DATASET_DIR, dataset, "test", "y_test.npy"), y_test)

def dataset_handler(raw_data):

    data = DatasetRequest(raw_data)
    data.download_data()
    logger.info("Sharding dataset")
    if data.dataset == "mnist":
        mnist_sharding(data)
    elif data.dataset == "cifar10":
        pass
    else:
        logger.error("No dataset mentioned")

