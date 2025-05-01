import redis
from logger import get_logger
from handler import handle_message

logger = get_logger()

class Observer:
    def __init__(self, redis_host, redis_port):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        logger.info(f"Observer initialized with Redis at {redis_host}:{redis_port}")

    def watch(self):
        pubsub = self.redis.pubsub()
        pubsub.subscribe("trainer_commands")
        logger.info("Subscribed to 'trainer_commands' channel.")

        for message in pubsub.listen():
            if message["type"] != "message":
                continue
            command = message["data"]
            logger.info(f"Received command: {command}")
            handle_message(command)
