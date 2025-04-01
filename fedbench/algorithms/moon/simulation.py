import logging
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Dict,Any

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from flwr.simulation import start_simulation
from flwr.common import NDArrays, Scalar, Context
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server import Server, ServerConfig

from fedbench.datasets import FederatedDataset
from hydra.utils import instantiate
from flwr.server.strategy import FedAvg
from fedbench.algorithms.moon.client import FlowerClientMoon
from fedbench.algorithms.moon.model_utils import moon_test
# Configure logging
logging.basicConfig(level=logging.INFO)



def moon_gen_client_fn(
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    device: torch.device,
    model_cfg: DictConfig,
    epochs: int,
    learning_rate: float,
    mu: float,
    temperature: float,
    model_dir: str, 
    momentum: float = 0.9,
    weight_decay: float = 1e-5,
) -> Callable[[str], FlowerClientMoon]:
    """Generate the client function that creates the Flower Clients."""

    def client_fn(context: Context) -> FlowerClientMoon:
        """Create a Flower client representing a single organization."""
        cid =  context.node_config["partition-id"]
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return FlowerClientMoon(
            net_id=int(cid),
            model_cfg = model_cfg,
            trainloader=trainloader,
            valloader=valloader,
            device=device,
            num_epochs=epochs,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            mu=mu,
            temperature=temperature,
            model_dir=model_dir,
        ).to_client()

    return client_fn


def moon_gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    model_cfg: DictConfig,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generate the function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        net = instantiate(model_cfg)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        accuracy, loss = moon_test(net, testloader, device=device)
        return loss, {"test_accuracy": accuracy}

    return evaluate

def run_moon(
    data_config: DictConfig,
    model_cfg: DictConfig,
    backend_config: Dict[str, int],
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    model_dir: str,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: Optional[int] = None,
    min_evaluate_clients: Optional[int] = None,
    min_available_clients: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the FedNova server with the provided configuration.

    Args:
        data_config (DictConfig): Configuration for the dataset (e.g., name, partitioning, alpha, batch_size).
        model_cfg (DictConfig): Configuration for the model (e.g., input_dim, hidden_dims, num_classes).
        backend_config (Dict[str, int]): Configuration for backend resources (e.g., num_cpus, num_gpus).
        num_clients (int): Number of clients participating in the federated learning process.
        num_rounds (int): Number of federated learning rounds.
        num_epochs (int): Number of local training epochs per client.
        learning_rate (float): Learning rate for the SGD optimizer.
        device (torch.device): Device to use for training (e.g., "cpu" or "cuda").
        fraction_fit (float): Fraction of clients to sample for training.
        fraction_evaluate (float): Fraction of clients to sample for evaluation.
        min_fit_clients (Optional[int]): Minimum number of clients to sample for training.
        min_evaluate_clients (Optional[int]): Minimum number of clients to sample for evaluation.
        min_available_clients (Optional[int]): Minimum number of available clients to start the simulation.

    Returns:
        Dict[str, Any]: History of the federated learning process.
    """
    # Load federated dataset
    federated_dataset = FederatedDataset(data_config, num_clients=num_clients)
    trainloaders, valloaders, testloader = federated_dataset.get_dataloaders()

    # Generate client and evaluation functions
    fed_moon_client_fn = moon_gen_client_fn(
        trainloaders=trainloaders,
        valloaders=valloaders,
        model_cfg=model_cfg,
        device=device,
        epochs=num_epochs,
        learning_rate=learning_rate,
        model_dir=model_dir,
        mu=5,
        temperature=0.5,
    )
    fed_moon_evaluate_fn = moon_gen_evaluate_fn(
        testloader=testloader,
        device=device,
        model_cfg=model_cfg,
    )

    # Define FedNova strategy
    fed_moon_strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients or num_clients,
        min_evaluate_clients=min_evaluate_clients or num_clients,
        min_available_clients=min_available_clients or num_clients,
        evaluate_fn=fed_moon_evaluate_fn,
    )

    # Start simulation
    fedmoon_history = start_simulation(
        client_fn=fed_moon_client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=fed_moon_strategy,
        client_resources=backend_config,
    )

    return fedmoon_history