import logging
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Dict, Any
import os
import csv

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from flwr.simulation import start_simulation
from flwr.common import NDArrays, Scalar, Context
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server import Server, ServerConfig

from fedbench.datasets import FederatedDataset
from hydra.utils import instantiate
from fedbench.algorithms.dasha.client import FlowerClientDasha
from fedbench.algorithms.dasha.model_utils import test, DashaCompressor
from fedbench.algorithms.dasha.strategy import DashaStrategy
from fedbench.algorithms.dasha.server import DashaServer

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add CSV metrics recording functionality
""""
def save_metrics_to_csv(metrics_dir, filename, round_num, accuracy):
    #Save metrics to CSV file.
    os.makedirs(metrics_dir, exist_ok=True)
    file_path = os.path.join(metrics_dir, filename)
    
    # If file doesn't exist, create it and write headers
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Round", "Accuracy"])
    
    # Append the metrics
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([round_num, accuracy])
"""

def dasha_gen_client_fn(
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    model_dir: str,
    epochs: int,
    learning_rate: float,
    model_cfg: DictConfig,
    device: torch.device,
    compressor_coordinates: int = 10,
    probability_q: float = 0.5,
    momentum: float = 0.9,
    weight_decay: float = 0.00001,
) -> Callable[[str], FlowerClientDasha]:
    """Generate the client function that creates the DASHA flower clients.

    Parameters
    ----------
    trainloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset training partition
        belonging to a particular client.
    valloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset validation partition
        belonging to a particular client.
    model_dir : str
        The directory where the model parameters are stored (persistent storage).
    epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    learning_rate : float
        The learning rate for the SGD optimizer of clients.
    model_cfg : DictConfig
        The model configuration.
    device : torch.device
        The device to use for training.
    compressor_coordinates : int
        The number of coordinates to use for compression in DASHA.
    probability_q : float
        The probability of using full gradient vs compressed gradient in DASHA.
    momentum : float
        The momentum for SGD optimizer of clients.
    weight_decay : float
        The weight decay for SGD optimizer of clients.

    Returns
    -------
    Callable[[str], FlowerClientDasha]
        The client function that creates the DASHA flower clients.
    """

    def client_fn(context: Context) -> FlowerClientDasha:
        """Create a Flower client representing a single organization."""
        # Load model
        net = instantiate(model_cfg)
        net.to(device)
        cid = context.node_config["partition-id"]
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return FlowerClientDasha(
            int(cid),
            net,
            trainloader,
            valloader,
            device,
            epochs,
            learning_rate,
            momentum,
            weight_decay,
            save_dir=model_dir,
            probability_q=probability_q,
            compressor_coordinates=compressor_coordinates
        ).to_client()
    return client_fn

def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    model_cfg: DictConfig,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generate the function for centralized evaluation.

    Parameters
    ----------
    testloader : DataLoader
        The dataloader to test the model with.
    device : torch.device
        The device to test the model on.
    model_cfg : DictConfig
        The model configuration.
    metrics_dir : str
        Directory to save the metrics CSV file.
    filename : str
        Name of the metrics CSV file.

    Returns
    -------
    Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]
        The centralized evaluation function.
    """

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Use the entire test set for evaluation."""
        net = instantiate(model_cfg)
        net.to(device)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        loss, accuracy = test(net, testloader, device)
        #save_metrics_to_csv(metrics_dir, filename, server_round, accuracy)
        return loss, {"test_accuracy": accuracy}

    return evaluate

def run_dasha(
    data_config: DictConfig,
    model_cfg: DictConfig,
    backend_config: Dict[str, int],
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    model_dir: str,
    compressor_coordinates: int = 10,
    probability_q: float = 0.5,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: Optional[int] = None,
    min_evaluate_clients: Optional[int] = None,
    min_available_clients: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the DASHA algorithm in federated learning setting.

    Args:
        data_config (DictConfig): Configuration for the dataset (e.g., name, partitioning, alpha, batch_size).
        model_cfg (DictConfig): Configuration for the model (e.g., input_dim, hidden_dims, num_classes).
        backend_config (Dict[str, int]): Configuration for backend resources (e.g., num_cpus, num_gpus).
        num_clients (int): Number of clients participating in the federated learning process.
        num_rounds (int): Number of federated learning rounds.
        num_epochs (int): Number of local training epochs per client.
        learning_rate (float): Learning rate for the SGD optimizer.
        model_dir (str): Directory to save the model weights.
        device (torch.device): Device to use for training (e.g., "cpu" or "cuda").
        metrics_dir (str): Directory to save the metrics CSV file.
        metrics_filename (str): Name of the metrics CSV file.
        compressor_coordinates (int): Number of coordinates to use for compression in DASHA.
        probability_q (float): Probability of using full gradient vs compressed gradient in DASHA.
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
    dasha_client_fn = dasha_gen_client_fn(
        trainloaders=trainloaders,
        valloaders=valloaders,
        model_cfg=model_cfg,
        device=device,
        epochs=num_epochs,
        model_dir=model_dir,
        learning_rate=learning_rate,
        compressor_coordinates=compressor_coordinates,
        probability_q=probability_q
    )
    dasha_evaluate_fn = gen_evaluate_fn(
        testloader=testloader,
        device=device,
        model_cfg=model_cfg,
    )

    # Define DASHA strategy
    dasha_strategy = DashaStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients or num_clients,
        min_evaluate_clients=min_evaluate_clients or num_clients,
        min_available_clients=min_available_clients or num_clients,
        evaluate_fn=dasha_evaluate_fn,
    )
    
    # Initialize model
    net = instantiate(model_cfg)
    net.to(device)
    
    # Initialize server
    dasha_server = DashaServer(
        client_manager=SimpleClientManager(),
        strategy=dasha_strategy,
        net=net,
    )

    # Start simulation
    dasha_history = start_simulation(
        server=dasha_server,
        client_fn=dasha_client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=dasha_strategy,
        client_resources=backend_config,
    )

    return dasha_history