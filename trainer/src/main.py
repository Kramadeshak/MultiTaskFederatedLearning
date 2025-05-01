import os
from logger import get_logger
from observer import Observer

redis_host = os.getenv("REDIS_HOST", "redis")
redis_port = int(os.getenv("REDIS_PORT", "6379"))
logger = get_logger()

def main():
    logger.info("Starting Trainer Main")
    o = Observer(redis_host, redis_port)
    o.watch()

if __name__ == "__main__":
    main()
