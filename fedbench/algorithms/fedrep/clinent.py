import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
from collections import OrderedDict
import flwr as fl
from flwr.common import Scalar
import numpy as np
import os
from fedbench.algorithms.fedrep.model_utils import train_fedrep, test  # Import from previous artifact

class FlowerClientFedRep(fl.client.NumPyClient):
    """Flower client implementing FedRep with SCAFFOLD structure."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        cid: int,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
        rep_parameters: List[torch.nn.Parameter],
        head_parameters: List[torch.nn.Parameter],
        head_epochs: int = 2,
        save_dir: str = ""
    ) -> None:
        """
        Initialize the FedRep client.

        Parameters
        ----------
        cid : int
            Client ID.
        net : torch.nn.Module
            The neural network (split into representation and head).
        trainloader : DataLoader
            Training set dataloader.
        valloader : DataLoader
            Validation set dataloader.
        device : torch.device
            Device to train on.
        num_epochs : int
            Number of training rounds.
        learning_rate : float
            Learning rate for both representation and head.
        momentum : float
            Momentum for SGD optimizer.
        weight_decay : float
            Weight decay for SGD optimizer.
        rep_parameters : List[torch.nn.Parameter]
            Parameters of the representation part of the network.
        head_parameters : List[torch.nn.Parameter]
            Parameters of the head part of the network.
        head_epochs : int
            Number of epochs to train the head per round.
        save_dir : str
            Directory to save client control variates.
        """
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.rep_parameters = rep_parameters
        self.head_parameters = head_parameters
        self.head_epochs = head_epochs
        # Initialize client control variate for representation parameters only
        self.client_cv = [torch.zeros_like(param) for param in rep_parameters]
        # Save control variates to directory
        if save_dir == "":
            save_dir = "client_cvs"
        self.dir = save_dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """Return the current representation parameters."""
        # Only return representation parameters, as head is client-specific
        state_dict = self.net.state_dict()
        rep_keys = [k for k, v in state_dict.items() if any(p is v for p in self.rep_parameters)]
        return [state_dict[k].cpu().numpy() for k in rep_keys]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set the representation parameters using given ones."""
        # Only update representation parameters
        state_dict = self.net.state_dict()
        rep_keys = [k for k, v in state_dict.items() if any(p is v for p in self.rep_parameters)]
        params_dict = zip(rep_keys, parameters)
        new_state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        state_dict.update(new_state_dict)
        self.net.load_state_dict(state_dict, strict=False)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model using FedRep and return representation updates."""
        # Split parameters into model weights and server control variates
        model_weights = parameters[:len(parameters) // 2]
        server_cv = parameters[len(parameters) // 2:]
        
        # Set representation parameters
        self.set_parameters(model_weights)
        server_cv = [torch.Tensor(cv).to(self.device) for cv in server_cv]
        
        # Train the model using FedRep
        train_fedrep(
            self.net,
            self.trainloader,
            self.device,
            self.num_epochs,
            self.learning_rate,
            self.momentum,
            self.weight_decay,
            server_cv,
            self.client_cv,
            head_epochs=self.head_epochs,
            rep_parameters=self.rep_parameters,
            head_parameters=self.head_parameters
        )
        
        # Compute updates for representation parameters
        updated_weights = self.get_parameters(config={})
        delta_weights = [np.subtract(updated, initial) for updated, initial in zip(updated_weights, model_weights)]
        
        # Update client control variate for representation parameters
        cv_updates = [
            (1.0 / (self.learning_rate * self.num_epochs * len(self.trainloader))) * (updated - initial)
            for updated, initial in zip(updated_weights, model_weights)
        ]
        
        # Update and save client control variates
        self.client_cv = [
            c_i_j - c_j + cv_update
            for c_i_j, c_j, cv_update in zip(self.client_cv, server_cv, cv_updates)
        ]
        torch.save(self.client_cv, f"{self.dir}/client_cv_{self.cid}.pt")
        
        # Return representation weight deltas and control variate updates
        return delta_weights + cv_updates, len(self.trainloader.dataset), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, float]]:
        """Evaluate using given representation parameters."""
        self.set_parameters(parameters)
        loss, acc = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(acc)}