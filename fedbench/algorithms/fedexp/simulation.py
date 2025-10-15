import logging
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from flwr.common import Metrics, FitIns, EvaluateIns, FitRes, EvaluateRes, Parameters, NDArrays, Context, Scalar
from flwr.server.strategy import Strategy
import numpy as np
import os
from hydra.utils import instantiate
from omegaconf import DictConfig
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server import Server, ServerConfig
from flwr.simulation import start_simulation

from fedbench.datasets import FederatedDataset
from fedbench.algorithms.fedexp.strategy import FedExpStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)

class FlowerClientFedExp(fl.client.NumPyClient):
    """Flower client implementing FedExp."""

    def __init__(
        self,
        cid: str,
        net: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        num_epochs: int,
        device: torch.device,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
    ):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.num_epochs = num_epochs
        self.device = device
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.optimizer = optim.SGD(
            self.net.parameters(), 
            lr=self.learning_rate, 
            momentum=self.momentum
        )
        # Keep track of the previous model for calculating updates
        self.previous_model = None

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Save previous model for calculating updates (delta)
        if self.previous_model is None:
            self.previous_model = OrderedDict(
                {k: v.clone() for k, v in self.net.state_dict().items()}
            )
        else:
            self.previous_model = OrderedDict(
                {k: v.clone() for k, v in self.net.state_dict().items()}
            )

        # Update model with received parameters
        self.set_parameters(parameters)

        # Training
        self.net.to(self.device)
        self.net.train()
        
        criterion = nn.CrossEntropyLoss()
        num_examples = 0
        total_loss = 0.0
        
        for _ in range(self.num_epochs):
            for batch_idx, (x, y) in enumerate(self.trainloader):
                x, y = x.to(self.device), y.to(self.device)
                num_examples += len(x)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.net(x)
                loss = criterion(outputs, y)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * len(x)

        # Calculate model update (delta)
        updated_params = self.get_parameters(config={})
        
        # Compute updates (the differences between updated and initial parameters)
        delta_weights = []
        for i, (name, param) in enumerate(self.net.state_dict().items()):
            prev_param = self.previous_model[name]
            delta = param.cpu() - prev_param.cpu()
            delta_weights.append(delta.numpy())
        
        # Update previous model for next round
        self.previous_model = OrderedDict(
            {k: v.clone() for k, v in self.net.state_dict().items()}
        )

        # Return training statistics
        return delta_weights, num_examples, {"loss": total_loss / num_examples}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.net.to(self.device)
        self.net.eval()
        
        criterion = nn.CrossEntropyLoss()
        loss = 0.0
        correct = 0
        num_examples = 0
        
        with torch.no_grad():
            for x, y in self.valloader:
                x, y = x.to(self.device), y.to(self.device)
                num_examples += len(x)
                outputs = self.net(x)
                loss += criterion(outputs, y).item() * len(x)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == y).sum().item()
        
        accuracy = correct / num_examples
        return float(loss), num_examples, {"accuracy": float(accuracy)}


class FedExpFlowerStrategy(fl.server.strategy.FedAvg):
    """FedExp strategy implementation for Flower"""

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        beta: float = 0.9,  # EMA parameter
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
        )
        self.beta = beta
        self.round_counter = 0
        self.ema_parameters = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model deltas using weighted average and apply EMA"""
        if not results:
            return None, {}
        
        # Standard FedAvg aggregation
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round=server_round,
            results=results,
            failures=failures,
        )
        
        if aggregated_parameters is None:
            return None, {}
        
        # Apply EMA if not first round
        if self.round_counter == 0:
            # First round, just initialize EMA model with current model
            self.ema_parameters = aggregated_parameters
        else:
            # Convert parameters to NDArrays
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
            ema_ndarrays = parameters_to_ndarrays(self.ema_parameters)
            
            # Apply EMA formula
            updated_ema_ndarrays = []
            for current, ema in zip(aggregated_ndarrays, ema_ndarrays):
                updated_ema = self.beta * ema + (1 - self.beta) * current
                updated_ema_ndarrays.append(updated_ema)
            
            # Convert back to Parameters
            self.ema_parameters = ndarrays_to_parameters(updated_ema_ndarrays)
        
        self.round_counter += 1
        
        # Return EMA parameters
        return self.ema_parameters, metrics


def gen_evaluate_fn(
    testloader: torch.utils.data.DataLoader,
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
        """Use the entire test set for evaluation."""
        net = instantiate(model_cfg)
        net.to(device)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        # Evaluate the model
        net.eval()
        loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                logits = net(x)
                loss += criterion(logits, y).item() * len(x)
                _, predicted = torch.max(logits.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        accuracy = correct / total
        loss = loss / total
        return loss, {"test_accuracy": accuracy}

    return evaluate


def fedexp_gen_client_fn(
    trainloaders: List[torch.utils.data.DataLoader],
    valloaders: List[torch.utils.data.DataLoader],
    epochs: int,
    learning_rate: float,
    model_cfg: DictConfig,
    device: torch.device,
    momentum: float = 0.9,
) -> Callable[[Context], FlowerClientFedExp]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates the FedExp flower clients.

    Parameters
    ----------
    trainloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset training partition
        belonging to a particular client.
    valloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset validation partition
        belonging to a particular client.
    epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    learning_rate : float
        The learning rate for the SGD optimizer of clients.
    momentum : float
        The momentum for SGD optimizer of clients.
    model_cfg : DictConfig
        Model configuration.
    device : torch.device
        Device to use for training.

    Returns
    -------
    Callable[[Context], FlowerClientFedExp]
        The client function that creates the FedExp flower clients.
    """

    def client_fn(context: Context) -> FlowerClientFedExp:
        """Create a Flower client representing a single organization."""
        # Load model
        net = instantiate(model_cfg)
        net.to(device)
        # cid = str(context.cid)
        cid = context.node_config["partition-id"]
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return FlowerClientFedExp(
            cid=cid,
            net=net,
            trainloader=trainloader,
            valloader=valloader,
            num_epochs=epochs,
            device=device,
            learning_rate=learning_rate,
            momentum=momentum,
        )
    
    return client_fn


def run_fedexp(
    data_config: DictConfig,
    model_cfg: DictConfig,
    backend_config: Dict,
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    learning_rate: float,
    model_dir: str = None,
    device: torch.device = torch.device("cpu"),
    beta: float = 0.9,  # EMA parameter
):
    """Run FedExp simulation."""
    # Create directory for model checkpoints
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    # Set up the dataset
    feddata = FederatedDataset(data_config, num_clients=num_clients)
    trainloaders, valloaders, testloader = feddata.get_dataloaders()
    
    # Configure client resources
    client_resources = {"num_cpus": backend_config.get("num_cpus", 1)}
    if "num_gpus" in backend_config and backend_config["num_gpus"] > 0:
        client_resources["num_gpus"] = backend_config["num_gpus"]
    
    # Generate client and evaluation functions
    fed_fedexp_client_fn = fedexp_gen_client_fn(
        trainloaders=trainloaders,
        valloaders=valloaders,
        model_cfg=model_cfg,
        device=device,
        epochs=num_epochs,
        learning_rate=learning_rate,
        momentum=0.9,
    )
    
    fed_fedexp_evaluate_fn = gen_evaluate_fn(
        testloader=testloader,
        device=device,
        model_cfg=model_cfg,
    )
    
    # Define FedExp strategy
    fed_fedexp_strategy = FedExpFlowerStrategy(
        fraction_fit=1.0,  
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=fed_fedexp_evaluate_fn,
        beta=beta,  # EMA parameter
    )
    
    # Start simulation
    fedexp_history = start_simulation(
        client_fn=fed_fedexp_client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=fed_fedexp_strategy,
        client_resources=client_resources,
    )
    
    # Save the model if requested
    if model_dir and hasattr(fed_fedexp_strategy, "ema_parameters") and fed_fedexp_strategy.ema_parameters is not None:
        # Get the final model
        final_model = instantiate(model_cfg)
        ema_ndarrays = parameters_to_ndarrays(fed_fedexp_strategy.ema_parameters)
        params_dict = zip(final_model.state_dict().keys(), ema_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        final_model.load_state_dict(state_dict, strict=True)
        
        # Save model
        torch.save(final_model.state_dict(), os.path.join(model_dir, "final_model.pt"))
    
    return fedexp_history