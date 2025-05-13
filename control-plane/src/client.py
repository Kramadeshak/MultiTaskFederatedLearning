import os
import threading
import time
import pickle
import numpy as np
from dataset_manager import dataset_handler
from logger import get_logger

logger = get_logger()

class DataStreamer:
    def __init__(self, raw_data, redis_client):
        self.redis = redis_client
        self.dataset = raw_data["dataset"].lower()
        self.num_clients = int(raw_data["num_clients"])
        self.base_dir = os.path.join("/dataset", self.dataset, "train")

    def start_streaming(self):
        logger.info("Starting data streaming threads...")

        threads = []
        for client_id in range(self.num_clients):
            t = threading.Thread(target=self._stream_to_client, args=(client_id,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        logger.info("All clients have been streamed.")

    def _stream_to_client(self, client_id):
        key = f"client{client_id}"
        x_path = os.path.join(self.base_dir, f"client_{client_id}_X.npy")
        y_path = os.path.join(self.base_dir, f"client_{client_id}_y.npy")

        X = np.load(x_path)
        Y = np.load(y_path)

        logger.info(f"Streaming {len(X)} samples to {key}...")

        for i in range(len(X)):
            payload = pickle.dumps({
                "image": X[i].tolist(),
                "label": int(Y[i])
            })
            self.redis.rpush(key, payload)
            logger.debug(f"Pushed sample {i} to {key}")
            time.sleep(0.1)  # Optional throttle

        logger.info(f"Completed streaming to {key}")

def data_streamer(raw_data, redis):
    streamer = DataStreamer(raw_data, redis)
    logger.info("Streamer initialized, pushing data")
    streamer.start_streaming()

