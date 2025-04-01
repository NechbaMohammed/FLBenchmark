import logging
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Dict,Any

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from flwr.simulation import start_simulation
from flwr.common import NDArrays, Scalar, Context
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.strategy import FedProx
from flwr.server import Server, ServerConfig

from fedbench.datasets import FederatedDataset
from fedbench.algorithms.fedavg.client import FlowerClient
from fedbench.algorithms.fedavg.model_utils import test
from hydra.utils import instantiate

# Configure logging
logging.basicConfig(level=logging.INFO)

def gen_client_fn(
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    model_cfg: DictConfig,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    momentum: float = 0.9,
    weight_decay: float = 1e-5,
) -> Callable[[str], FlowerClient]:
    """Generate a function to create FlowerClient instances.

    Args:
        trainloaders (List[DataLoader]): List of training data loaders for each client.
        valloaders (List[DataLoader]): List of validation data loaders for each client.
        model_cfg (DictConfig): Model configuration (e.g., input_dim, hidden_dims, num_classes).
        device (torch.device): Device to use for training (e.g., "cpu" or "cuda").
        epochs (int): Number of local training epochs.
        learning_rate : float
        The learning rate for the SGD  optimizer of clients.
    model : DictConfig
        The model configuration.
    momentum : float
        The momentum for SGD optimizer of clients
    weight_decay : float
        The weight decay for SGD optimizer of clients

    Returns:
        Callable[[str], FlowerClient]: A function to create FlowerClient instances.
    """
    def client_fn(context: Context) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        # Load model
        net = instantiate(model_cfg)
        net.to(device)

        # Get client ID and corresponding data loaders
        cid = context.node_config["partition-id"]
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return FlowerClient(
            net=net,
            trainloader=trainloader,
            valloader=valloader,
            device=device,
            epochs=epochs,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        ).to_client()

    return client_fn

def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    model_cfg: DictConfig,
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]:
    """Generate the function for centralized evaluation.

    Args:
        testloader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to use for evaluation (e.g., "cpu" or "cuda").
        model_cfg (DictConfig): Model configuration (e.g., input_dim, hidden_dims, num_classes).

    Returns:
        Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]:
        A function to evaluate the global model on the test set.
    """
    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the global model on the test set."""
        # Load model
        net = instantiate(model_cfg)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        # Compute loss and accuracy
        loss, accuracy = test(net, testloader, device)
        return loss, {"test_accuracy": accuracy}

    return evaluate

def run_fedprox(
    data_config: DictConfig,
    model_cfg: DictConfig,
    backend_config: Dict[str, int],
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    learning_rate: float,
    mu: float,
    device: torch.device,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: Optional[int] = None,
    min_evaluate_clients: Optional[int] = None,
    min_available_clients: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the fedprox server with the provided configuration.

    Args:
        data_config (DictConfig): Configuration for the dataset (e.g., name, partitioning, alpha, batch_size).
        model_cfg (DictConfig): Configuration for the model (e.g., input_dim, hidden_dims, num_classes).
        backend_config (Dict[str, int]): Configuration for backend resources (e.g., num_cpus, num_gpus).
        num_clients (int): Number of clients participating in the federated learning process.
        num_rounds (int): Number of federated learning rounds.
        num_epochs (int): Number of local training epochs per client.
        learning_rate (float): Learning rate for the SGD optimizer of clients.
        mu (float): Proximal term weight.
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
    fed_prox_client_fn = gen_client_fn(
        trainloaders=trainloaders,
        valloaders=valloaders,
        model_cfg=model_cfg,
        device=device,
        epochs=num_epochs,
        learning_rate=learning_rate,
    )
    fed_prox_evaluate_fn = gen_evaluate_fn(
        testloader=testloader,
        device=device,
        model_cfg=model_cfg,
    )

    # Define FedAvg strategy
    fed_prox_strategy = FedProx(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients or num_clients,
        min_evaluate_clients=min_evaluate_clients or num_clients,
        min_available_clients=min_available_clients or num_clients,
        evaluate_fn=fed_prox_evaluate_fn,
        proximal_mu = mu
    )

    # Initialize server
    fed_prox_server = Server(
        strategy=fed_prox_strategy,
        client_manager=SimpleClientManager(),
    )

    # Start simulation
    fedprox_history = start_simulation(
        server=fed_prox_server,
        client_fn=fed_prox_client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=fed_prox_strategy,
        client_resources=backend_config,
    )

    return fedprox_history