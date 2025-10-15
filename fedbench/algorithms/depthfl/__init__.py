"""DepthFL implementation for federated learning with heterogeneous clients."""

# Make imports available at package level
from .client import FlowerClientDepthFL
from .server import DepthFLServer
from .strategy import DepthFLStrategy
from .simulation import run_depthfl
from .model_utils import train_depthfl, test_depthfl