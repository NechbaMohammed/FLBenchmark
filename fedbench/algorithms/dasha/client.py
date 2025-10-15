import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from collections import OrderedDict
import flwr as fl
from flwr.common import Scalar
import numpy as np
import os

from fedbench.algorithms.dasha.model_utils import train_dasha, test, DashaCompressor

class FlowerClientDasha(fl.client.NumPyClient):
    """Flower client implementing DASHA algorithm."""

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
        save_dir: str = "",
        probability_q: float = 0.5,
        compressor_coordinates: int = 10
    ) -> None:
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.probability_q = probability_q
        self.compressor = DashaCompressor(number_of_coordinates=compressor_coordinates)
        
        # Initialize previous model parameters with zeros
        self.prev_model = []
        for param in self.net.parameters():
            self.prev_model.append(torch.zeros_like(param))
            
        # Save directory for model parameters
        if save_dir == "":
            save_dir = "dasha_models"
        self.dir = save_dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Return the current local model parameters."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Set the local model parameters using given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config: Dict[str, Scalar]):
        """Train the model using DASHA algorithm."""
        # Set model parameters
        self.set_parameters(parameters)
        
        # Update previous model parameters
        self.prev_model = [param.clone().detach() for param in self.net.parameters()]
        
        # Train the model using DASHA
        self.net, variance_reduction_direction = train_dasha(
            net=self.net,
            trainloader=self.trainloader,
            device=self.device,
            epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            prev_model=self.prev_model,
            compressor=self.compressor,
            probability_q=self.probability_q
        )
        
        # Get updated model parameters
        updated_weights = self.get_parameters(config={})
        
        # Save model with proper naming convention based on client ID
        # Format: client_cv_X.pt where X is the client ID
        model_filename = f"client_cv_{self.cid}.pt"
        model_path = os.path.join(self.dir, model_filename)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        torch.save(self.net.state_dict(), model_path)
        print(f"Saved model to {model_path}")
        
        # Calculate L2 norm of variance reduction direction for metrics
        vr_norm = 0.0
        for direction in variance_reduction_direction:
            vr_norm += torch.norm(direction).item()
        
        # Return weights and metrics (with no numpy arrays)
        metrics = {
            "vr_norm": float(vr_norm),
            "vr_used": True,
            "client_id": self.cid,
        }
        
        # Store the variance reduction direction locally for debug purposes
        # But don't send it to the server as it's a numpy array
        self.last_vr_direction = [direction.cpu().numpy() for direction in variance_reduction_direction]
        
        return updated_weights, len(self.trainloader.dataset), metrics
    
    def evaluate(self, parameters, config: Dict[str, Scalar]):
        """Evaluate using given parameters."""
        self.set_parameters(parameters)
        loss, acc = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(acc)}