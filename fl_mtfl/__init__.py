"""Entry point module for fl_mtfl package."""

import logging
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

from fl_mtfl.config import CONFIG, get_parser
from fl_mtfl.model import CIFAR10CNN
from fl_mtfl.task import (
    seed_everything,
    load_data,
    train,
    test,
    get_weights,
    set_weights,
    load_or_initialize_model,
    save_metrics
)

__version__ = "0.1.0"

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fl_mtfl")

# Export key components
__all__ = [
    "CONFIG",
    "get_parser",
    "CIFAR10CNN",
    "seed_everything",
    "load_data",
    "train",
    "test",
    "get_weights",
    "set_weights",
    "load_or_initialize_model",
    "save_metrics"
]