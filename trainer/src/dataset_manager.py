import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
from logger import get_logger

DATA_DIR = "/data"
DATASET_DIR = "/dataset"

logger = get_logger()

def load_parquet_data(data_type, train_file, test_file):
    logger.info("Loading data from file...")
    train_df = pd.read_parquet(os.path.join(DATA_DIR, data_type, train_file))
    test_df = pd.read_parquet(os.path.join(DATA_DIR, data_type, test_file))

    X_train = np.vstack(train_df["image"].values)
    y_train = train_df["label"].values

    X_test = np.vstack(test_df["image"].values)
    y_test = test_df["label"].values

    return X_train, y_train, X_test, y_test

def save_test_data(X_test, y_test):
    os.makedirs(os.path.join(DATASET_DIR, "test"), exist_ok=True)
    np.save(os.path.join(DATASET_DIR, "test", "X_test.npy"), X_test)
    np.save(os.path.join(DATASET_DIR, "test", "y_test.npy"), y_test)

def shard_iid(data_type, X, y, num_clients):
    logger.info("Executing IID shard")
    X, y = shuffle(X, y, random_state=42)
    indices = np.array_split(np.arange(len(X)), num_clients)

    for i, idx in enumerate(indices):
        save_client_data(data_type, i, X[idx], y[idx])

def shard_non_iid(data_type, X, y, num_clients):
    logger.info("Executing non-IID shard")
    labels = np.unique(y)
    label_indices = {label: np.where(y == label)[0] for label in labels}
    shards_per_client = len(labels) // num_clients

    shuffled_labels = list(labels)
    np.random.seed(42)
    np.random.shuffle(shuffled_labels)

    for i in range(num_clients):
        selected_labels = shuffled_labels[i * shards_per_client: (i + 1) * shards_per_client]
        idx = np.hstack([label_indices[label] for label in selected_labels])
        save_client_data(data_type, i, X[idx], y[idx])

def save_client_data(data_type, client_id, X, y):
    client_dir = os.path.join(DATASET_DIR, data_type, "train")
    os.makedirs(client_dir, exist_ok=True)
    np.save(os.path.join(client_dir, f"client_{client_id}_X.npy"), X)
    np.save(os.path.join(client_dir, f"client_{client_id}_y.npy"), y)
    logger.info("Saved sharded data.")

def mnist_sharding(iid, num_clients, train_file, test_file):
    X_train, y_train, X_test, y_test = load_parquet_data("mnist", train_file, test_file)
    logger.info("Loaded data from file. Proceeding to shard.")

    if iid:
        shard_iid("mnist", X_train, y_train, num_clients)
    else:
        shard_non_iid("mnist", X_train, y_train, num_clients)

    save_test_data(X_test, y_test)
