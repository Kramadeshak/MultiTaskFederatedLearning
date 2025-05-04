import redis
import json
from logger import get_logger
from dataset_manager import dataset_handler

logger = get_logger()

class RequestProcessor:
    def __init__(self, redis_host, redis_port):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        logger.info(f"Observer initialized with Redis at {redis_host}:{redis_port}")

    def listen(self):
        pubsub = self.redis.pubsub()
        pubsub.subscribe("commands")
        logger.info("Subscribed to 'commands' channel.")

        for message in pubsub.listen():
            if message["type"] != "message":
                continue
            command = message["data"]
            logger.info(f"Received command: {command}")
            self.route_message(command)

    def route_message(self, command):
        """
        Parses messages that have been received from redis.
        """
        try:
            data = json.loads(command)
            if data["type"] == "dataset":
                dataset_handler(data["message"])
            else:
                logger.warning(f"Unsupported message type: {data['type']}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Error parsing message: {e}")
