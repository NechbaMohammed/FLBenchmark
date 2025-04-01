import logging
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Dict,Any

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from flwr.simulation import start_simulation
from flwr.common import NDArrays, Scalar, Context
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.strategy import FedAvg
from flwr.server import Server, ServerConfig

from fedbench.datasets import FederatedDataset
from fedbench.algorithms.fedbn.client import FedBNFlowerClient
from fedbench.algorithms.fedbn.model_utils import test
from hydra.utils import instantiate

# Configure logging
logging.basicConfig(level=logging.INFO)

def gen_client_fn(
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    model_cfg: DictConfig,    
    epochs:float,
    save_path: str,
    device: torch.device,
    learning_rate: float,
    momentum: float = 0.9,
    weight_decay: float = 1e-5,
) -> Callable[[Context], FedBNFlowerClient]:
    """Generate client function that creates FedBNFlowerClient instances.
    
    Args:
        trainloaders: List of DataLoaders for training data (one per client)
        valloaders: List of DataLoaders for validation data (one per client)
        model_cfg: Model configuration dictionary
        learning_rate: Learning rate for clients
        save_path: Path to save client-specific data
         The model configuration.
        momentum : float
            The momentum for SGD optimizer of clients
        weight_decay : float
            The weight decay for SGD optimizer of clients
        
    Returns:
        A function that creates FedBNFlowerClient instances
    """
    def client_fn(context: Context) -> FedBNFlowerClient:
        """Create a FedBNFlowerClient instance."""
        # Instantiate model
        net = instantiate(model_cfg)
        net.to(device)
        
        # Get client ID from context
        cid = context.node_config["partition-id"]
        
        # Get client-specific data loaders
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        
        return FedBNFlowerClient(
            model=net,
            trainloader=trainloader,
            testloader=valloader,
            save_path=save_path,
            client_id=int(cid),
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            epochs=epochs
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

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]],
               Optional[Tuple[float, Dict[str, Scalar]]] ]
    The centralized evaluation function.
    """

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        """Use the entire Emnist test set for evaluation."""
        net = instantiate(model_cfg)
        net.to(device)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        loss, accuracy = test(net, testloader, device)
        return loss, {"test_accuracy": accuracy}

    return evaluate

def get_on_fit_config():
    """Return a config (a dict) to be sent to clients during fit()."""

    def fit_config_fn(server_round: int):
        # resolve and convert to python dict
        fit_config = {}
        fit_config["round"] = server_round  # add round info
        return fit_config

    return fit_config_fn

def run_fedbn(
    data_config: DictConfig,
    model_cfg: DictConfig,
    backend_config: Dict[str, int],
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    save_path:str,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: Optional[int] = None,
    min_evaluate_clients: Optional[int] = None,
    min_available_clients: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the FedAvg server with the provided configuration.

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
    fed_bn_client_fn = gen_client_fn(
        trainloaders=trainloaders,
        valloaders=valloaders,
        model_cfg=model_cfg,
        device=device,
        epochs=num_epochs,
        learning_rate=learning_rate,
        save_path=save_path,
    )
    fed_bn_evaluate_fn = gen_evaluate_fn(
        testloader=testloader,
        device=device,
        model_cfg=model_cfg,
    )

    # Define FedAvg strategy
    fed_bn_strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients or num_clients,
        min_evaluate_clients=min_evaluate_clients or num_clients,
        min_available_clients=min_available_clients or num_clients,
        on_fit_config_fn = get_on_fit_config(),
        evaluate_fn=fed_bn_evaluate_fn,
    )

    

    # Start simulation
    fedbn_history = start_simulation(
        client_fn=fed_bn_client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=fed_bn_strategy,
        client_resources=backend_config,
    )

    return fedbn_history