import socket
import pickle
import torch
import time
import redis
import numpy as np
from PIL import Image
import io
from model import MNISTModel

SERVER_HOST = 'fedsim-server'
SERVER_PORT = 8080
BUFFER_SIZE = 4096

def decode_image(image_info):
    img_bytes = image_info[0]["bytes"]
    image_pil = Image.open(io.BytesIO(img_bytes)).convert("L").resize((28, 28))
    return np.array(image_pil, dtype=np.float32) / 255.0

class Client:
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.device = torch.device("cpu")
        self.model = MNISTModel(self.device)
        self.redis = redis.Redis(host='redis', port=6379, decode_responses=False)

    def receive_all(self, sock):
        """Receive all data from the socket after reading 8-byte length prefix."""
        length_data = b""
        while len(length_data) < 8:
            part = sock.recv(8 - len(length_data))
            if not part:
                raise ConnectionError("Socket connection closed before receiving length.")
            length_data += part
        total_length = int.from_bytes(length_data, byteorder='big')
        print(f"[Client {self.client_id}] Expecting {total_length} bytes")

        data = b""
        while len(data) < total_length:
            part = sock.recv(min(BUFFER_SIZE, total_length - len(data)))
            if not part:
                raise ConnectionError("Socket connection closed before all data received.")
            data += part

        return data

    def download_model(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((SERVER_HOST, SERVER_PORT))
        print(f"[Client {self.client_id}] Connected to {SERVER_HOST}:{SERVER_PORT}")

        received_data = self.receive_all(s)
        print(f"[Client {self.client_id}] Received {len(received_data)} bytes")
        s.close()

        model_params = pickle.loads(received_data)
        if model_params is None:
            print(f"[Client {self.client_id}] No params received.")
        else:
            print(f"[Client {self.client_id}] Loaded model with {len(model_params.params)} params")
            self.model.set_params(model_params.params)


    def observe_redis_cache(self, batch_size=4):
        print(f"[Client {self.client_id}] Observing Redis cache for new data...")
        key = f"client{self.client_id}"
        x_batch, y_batch = [], []
        while True:
            item = self.redis.lpop(key)
            if item is None:
                time.sleep(1)
                continue
            try:
                data = pickle.loads(item)
                if data["label"] == -222:
                    break
                image = decode_image(data["image"])
                label = data["label"]
                print(f"[Client {self.client_id}] Received sample with label {label}, shape {image.shape}")

                x_batch.append(image.flatten())  # Flatten to [784]
                y_batch.append(label)

                # If batch is full, train
                if len(x_batch) >= batch_size:
                    # x = torch.tensor(x_batch, dtype=torch.float32).view(batch_size, -1)
                    x = torch.tensor(x_batch, dtype=torch.float32).view(batch_size, -1)
                    y = torch.tensor(y_batch, dtype=torch.long)
                    err, acc = self.model.train_step(x, y)
                    print(f"[Client {self.client_id}] Trained on batch: err={err:.4f}, acc={acc:.4f}")
                    x_batch, y_batch = [], []

            except Exception as e:
                print(f"[Client {self.client_id}] Error decoding or training on data: {e}")
        print("Trained all data")
