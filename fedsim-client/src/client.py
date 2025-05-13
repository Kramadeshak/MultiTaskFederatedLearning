import socket
import pickle
import torch
import time
from model import MNISTModel

SERVER_HOST = 'fedsim-server'
SERVER_PORT = 8080
BUFFER_SIZE = 4096

class Client:
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.device = torch.device("cpu")
        self.model = MNISTModel(self.device)

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

    def observe_redis_cache(self):
        print(f"[Client {self.client_id}] Observing Redis cache for new data...")
        # Placeholder for Redis logic
        time.sleep(5)
