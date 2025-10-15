import logging
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from flwr.simulation import start_simulation
from flwr.common import NDArrays, Scalar, Context
from flwr.server.client_manager import SimpleClientManager
from flwr.server import ServerConfig

from fedbench.datasets import FederatedDataset
from hydra.utils import instantiate
from fedbench.algorithms.ditto.client import FlowerClientDitto
from fedbench.algorithms.ditto.model_utils import test
from fedbench.algorithms.ditto.strategy import DittoStrategy
from fedbench.algorithms.ditto.server import DittoServer

# Configure logging
logging.basicConfig(level=logging.INFO)

def ditto_gen_client_fn(
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    epochs: int,
    learning_rate: float,
    model_cfg: DictConfig,
    device: torch.device,
    mu: float,
    client_results_dir: Optional[str] = None,
    momentum: float = 0.9,
    weight_decay: float = 0.00001,
) -> Callable[[str], FlowerClientDitto]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates Ditto Flower clients."""
    
    def client_fn(context: Context) -> FlowerClientDitto:
        """Create a Flower client representing a single organization."""
        # Instantiate global and personalized models
        global_model = instantiate(model_cfg)
        personalized_model = instantiate(model_cfg)

        global_model.to(device)
        personalized_model.to(device)
        
        cid = context.node_config["partition-id"]
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return FlowerClientDitto(
            cid=int(cid),
            global_model=global_model,
            personalized_model=personalized_model,
            trainloader=trainloader,
            valloader=valloader,
            device=device,
            num_epochs=epochs,
            learning_rate=learning_rate,
            client_results_dir=client_results_dir,
            lambda_reg=mu,
            momentum=momentum,
            weight_decay=weight_decay,
        ).to_client()

    return client_fn

def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    model_cfg: DictConfig,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generate centralized evaluation function."""

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model on full test dataset."""
        net = instantiate(model_cfg)
        net.to(device)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        loss, accuracy = test(net, testloader, device)
        return loss, {"test_accuracy": accuracy}

    return evaluate

def run_ditto(
    data_config: DictConfig,
    model_cfg: DictConfig,
    backend_config: Dict[str, int],
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    mu: float,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: Optional[int] = None,
    min_evaluate_clients: Optional[int] = None,
    min_available_clients: Optional[int] = None,
    client_results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the FedDitto server with the provided configuration."""

    # Load federated dataset
    federated_dataset = FederatedDataset(data_config, num_clients=num_clients)
    trainloaders, valloaders, testloader = federated_dataset.get_dataloaders()

    # Generate client and evaluation functions
    fed_ditto_client_fn = ditto_gen_client_fn(
        trainloaders=trainloaders,
        valloaders=valloaders,
        model_cfg=model_cfg,
        device=device,
        epochs=num_epochs,
        learning_rate=learning_rate,
        client_results_dir=client_results_dir,
        mu=mu,
    )
    fed_ditto_evaluate_fn = gen_evaluate_fn(
        testloader=testloader,
        device=device,
        model_cfg=model_cfg,
    )

    # Define Ditto strategy
    fed_ditto_strategy = DittoStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients or num_clients,
        min_evaluate_clients=min_evaluate_clients or num_clients,
        min_available_clients=min_available_clients or num_clients,
        evaluate_fn=fed_ditto_evaluate_fn,
    )

    net = instantiate(model_cfg)
    net.to(device)

    # Initialize server
    fed_ditto_server = DittoServer(
        client_manager=SimpleClientManager(),
        strategy=fed_ditto_strategy,
        net=net,
    )

    # Start simulation
    ditto_history = start_simulation(
        server=fed_ditto_server,
        client_fn=fed_ditto_client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=fed_ditto_strategy,
        client_resources=backend_config,
    )

    return ditto_history
