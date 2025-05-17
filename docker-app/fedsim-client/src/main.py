import os
import threading
import time
from client import Client

def client_worker(client_id):
    client = Client(client_id)
    client.download_model()
    client.observe_redis_cache()

def main():
    print("[Main] Starting fedsim-client")
    time.sleep(5)
    num_clients = int(os.getenv("NUM_CLIENTS", 10))
    threads = []

    for i in range(num_clients):
        thread = threading.Thread(target=client_worker, args=(i,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
