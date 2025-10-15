import logging
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Dict, Any

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from flwr.simulation import start_simulation
from flwr.common import NDArrays, Scalar, Context
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server import ServerConfig


from fedbench.datasets import FederatedDataset
from hydra.utils import instantiate
from fedbench.algorithms.fedrep.clinent import FlowerClientFedRep
from fedbench.algorithms.fedrep.server import FedRepServer
from flwr.server.strategy import FedAvg  # Use FedAvg as a base strategy

# Configure logging
logging.basicConfig(level=logging.INFO)

def fedrep_gen_client_fn(
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    client_cv_dir: str,
    epochs: int,
    learning_rate: float,
    model_cfg: DictConfig,
    device: torch.device,
    momentum: float = 0.9,
    weight_decay: float = 0.00001,
    head_epochs: int = 2,
) -> Callable[[str], FlowerClientFedRep]:
    """Generate the client function that creates FedRep flower clients.

    Parameters
    ----------
    trainloaders : List[DataLoader]
        A list of DataLoaders, each pointing to the dataset training partition
        belonging to a particular client.
    valloaders : List[DataLoader]
        A list of DataLoaders, each pointing to the dataset validation partition
        belonging to a particular client.
    client_cv_dir : str
        The directory where the client control variates are stored (persistent storage).
    epochs : int
        The number of local rounds each client should run for training the representation.
    learning_rate : float
        The learning rate for the SGD optimizer of clients.
    model_cfg : DictConfig
        Configuration for the model (e.g., input_dim, hidden_dims, num_classes).
    device : torch.device
        The device to train the model on.
    momentum : float
        The momentum for SGD optimizer of clients.
    weight_decay : float
        The weight decay for SGD optimizer of clients.
    head_epochs : int
        The number of epochs to train the client-specific head per round.

    Returns
    -------
    Callable[[str], FlowerClientFedRep]
        The client function that creates the FedRep flower clients.
    """
    def client_fn(context: Context) -> FlowerClientFedRep:
        """Create a Flower client representing a single organization."""
        # Load model
        net = instantiate(model_cfg)
        import torch

        device = torch.device("cpu")  # Force CPU usage
# OR
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device)
        cid = context.node_config["partition-id"]
        # Get representation and head parameters
        rep_parameters = list(net.feature_extractor.parameters())  # Adjust based on model architecture
        head_parameters = list(net.classifier.parameters())       # Adjust based on model architecture
        # Note: each client gets a different trainloader/valloader
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return FlowerClientFedRep(
            int(cid),
            net,
            trainloader,
            valloader,
            device,
            epochs,
            learning_rate,
            momentum,
            weight_decay,
            rep_parameters=rep_parameters,
            head_parameters=head_parameters,
            head_epochs=head_epochs,
            save_dir=client_cv_dir
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
        Configuration for the model.

    Returns
    -------
    Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]
        The centralized evaluation function.
    """
    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Use the entire test set for evaluation with a default head."""
        net = instantiate(model_cfg)
        net.to(device)
        # Only update representation parameters
        state_dict = net.state_dict()
        rep_keys = [k for k, v in state_dict.items() if any(p is v for p in net.feature_extractor.parameters())]
        params_dict = zip(rep_keys, parameters_ndarrays)
        new_state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        state_dict.update(new_state_dict)
        net.load_state_dict(state_dict, strict=False)

        from fedbench.algorithms.fedrep.model_utils import test
        loss, accuracy = test(net, testloader, device)
        return loss, {"test_accuracy": accuracy}

    return evaluate

def run_fedrep(
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
    head_epochs: int = 2,
) -> Dict[str, Any]:
    """Run the FedRep server with the provided configuration.

    Args:
        data_config (DictConfig): Configuration for the dataset (e.g., name, partitioning, alpha, batch_size).
        model_cfg (DictConfig): Configuration for the model (e.g., input_dim, hidden_dims, num_classes).
        backend_config (Dict[str, int]): Configuration for backend resources (e.g., num_cpus, num_gpus).
        num_clients (int): Number of clients participating in the federated learning process.
        num_rounds (int): Number of federated learning rounds.
        num_epochs (int): Number of local training rounds for the representation per client.
        learning_rate (float): Learning rate for the SGD optimizer.
        device (torch.device): Device to use for training (e.g., "cpu" or "cuda").
        model_dir (str): Directory to save the model weights and client control variates.
        fraction_fit (float): Fraction of clients to sample for training.
        fraction_evaluate (float): Fraction of clients to sample for evaluation.
        min_fit_clients (Optional[int]): Minimum number of clients to sample for training.
        min_evaluate_clients (Optional[int]): Minimum number of clients to sample for evaluation.
        min_available_clients (Optional[int]): Minimum number of available clients to start the simulation.
        head_epochs (int): Number of epochs to train the client-specific head per round.

    Returns:
        Dict[str, Any]: History of the federated learning process.
    """
    # Load federated dataset
    federated_dataset = FederatedDataset(data_config, num_clients=num_clients)
    trainloaders, valloaders, testloader = federated_dataset.get_dataloaders()

    # Generate client and evaluation functions
    fedrep_client_fn = fedrep_gen_client_fn(
        trainloaders=trainloaders,
        valloaders=valloaders,
        model_cfg=model_cfg,
        device=device,
        epochs=num_epochs,
        client_cv_dir=model_dir,
        learning_rate=learning_rate,
        head_epochs=head_epochs,
    )
    fedrep_evaluate_fn = gen_evaluate_fn(
        testloader=testloader,
        device=device,
        model_cfg=model_cfg,
    )

    # Define FedRep strategy (using FedAvg as a base, adapted for SCAFFOLD-style updates)
    fedrep_strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients or num_clients,
        min_evaluate_clients=min_evaluate_clients or num_clients,
        min_available_clients=min_available_clients or num_clients,
        evaluate_fn=fedrep_evaluate_fn,
    )

    # Initialize server
    net = instantiate(model_cfg)
    net.to(device)
    rep_parameters = list(net.feature_extractor.parameters())  # Adjust based on model architecture
    fedrep_server = FedRepServer(
        client_manager=SimpleClientManager(),
        strategy=fedrep_strategy,
        net=net,
        rep_parameters=rep_parameters,
    )

    # Start simulation
    fedrep_history = start_simulation(
        server=fedrep_server,
        client_fn=fedrep_client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=fedrep_strategy,
        client_resources=backend_config,
    )

    return fedrep_history