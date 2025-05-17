import socket
import pickle
import torch
import threading
from model import MNISTModel

HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 8080       # Port to listen on

def handle_client_connection(conn, addr, serialized_params):
    print(f"[INFO] Connected by {addr}")
    conn.sendall(len(serialized_params).to_bytes(8, byteorder='big'))
    print(f"sent {len(serialized_params)}")
    conn.sendall(serialized_params)
    print("[INFO] Parameters sent.")

def get_model_params():
    device = torch.device('cpu')  # Send CPU tensors only
    model = MNISTModel(device)
    model.set_optim(torch.optim.SGD(model.parameters(), lr=0.01))
    model_params = model.get_params()
    serialized_params = pickle.dumps(model_params)
    return serialized_params

def main():

    params = get_model_params()
    print(f"[INFO] Waiting for connection on port {PORT}...")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        while True:
            conn, addr = s.accept()
            thread = threading.Thread(target=handle_client_connection, args=(conn, addr, params))
            thread.start()

if __name__ == "__main__":
    main()
