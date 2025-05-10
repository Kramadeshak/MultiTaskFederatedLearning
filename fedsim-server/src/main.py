import socket
import pickle
import torch
from model import MNISTModel

HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 8080       # Port to listen on

def main():
    device = torch.device('cpu')  # Send CPU tensors only
    model = MNISTModel(device)

    # Example: initialize optimizer to make model "complete"
    model.set_optim(torch.optim.SGD(model.parameters(), lr=0.01))

    # Get model parameters
    model_params = model.get_params()

    # Serialize model parameters with pickle
    serialized_params = pickle.dumps(model_params)

    print(f"[INFO] Waiting for connection on port {PORT}...")

    # Start TCP server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        conn, addr = s.accept()
        with conn:
            print(f"[INFO] Connected by {addr}")
            # First, send the length of the payload
            conn.sendall(len(serialized_params).to_bytes(8, byteorder='big'))
            print(f"sent {len(serialized_params)}")
            # Then, send the actual payload
            conn.sendall(serialized_params)
            print("[INFO] Parameters sent.")

if __name__ == "__main__":
    main()
