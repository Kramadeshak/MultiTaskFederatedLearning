import socket
import pickle
import torch
import time
from model import MNISTModel

SERVER_HOST = 'fedsim-server'
SERVER_PORT = 8080
BUFFER_SIZE = 4096

def receive_all(sock):
    """Receive all data from the socket after reading 8-byte length prefix."""
    # Read exactly 8 bytes for length prefix
    length_data = b""
    while len(length_data) < 8:
        part = sock.recv(8 - len(length_data))
        if not part:
            raise ConnectionError("Socket connection closed before receiving length.")
        length_data += part
    total_length = int.from_bytes(length_data, byteorder='big')

    print(f"received {total_length} data")
    # Now read exactly total_length bytes
    data = b""
    while len(data) < total_length:
        part = sock.recv(min(4096, total_length - len(data)))
        if not part:
            raise ConnectionError("Socket connection closed before all data received.")
        data += part

    return data

def load_model_from_server():
    """Handles model loading from the server."""
    # Create a socket connection to the server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((SERVER_HOST, SERVER_PORT))
    print(f"Connected to {SERVER_HOST}:{SERVER_PORT}")

    # Receive serialized model parameters
    received_data = receive_all(s)
    print("Received", len(received_data), "bytes")
    s.close()

    # Initialize model
    device = torch.device("cpu")
    model = MNISTModel(device)

    # Deserialize model parameters
    model_params = pickle.loads(received_data)
    if model_params is None:
        print("No params received, ignoring.")
    else:
        print("Loaded model with", len(model_params.params), "params")
        model.set_params(model_params.params)
        print("Model parameters set")

    return model

def observe_redis_cache():
    """Placeholder function to observe Redis cache for new data."""
    print("Observing Redis cache for new data...")
    # Implement actual Redis logic here
    time.sleep(10)  # Simulate sleeping while waiting for new data

def main():
    # Load the model from the server
    print("Started fedsim-client")
    time.sleep(5)
    model = load_model_from_server()

    # Start observing Redis cache for new data
    observe_redis_cache()

if __name__ == "__main__":
    main()
