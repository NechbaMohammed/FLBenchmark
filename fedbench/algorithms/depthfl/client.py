"""Client for DepthFL algorithm."""

import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
from collections import OrderedDict
import flwr as fl
from flwr.common import Scalar
import numpy as np
import os
import pickle
import copy


class FlowerClientDepthFL(fl.client.NumPyClient):
    """Flower client implementing DepthFL."""

    def __init__(
        self,
        cid: int,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        learning_rate_decay: float,
        save_dir: str = "",
    ) -> None:
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
            
        # Initialize previous gradients for FedDyn
        self.prev_grads = {}
        for k, param in net.named_parameters():
            self.prev_grads[k] = torch.zeros(param.numel(), device=torch.device("cpu"))
            
        # Setup save directory
        if save_dir == "":
            save_dir = "client_grads"
        self.dir = save_dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            
        # Try to load previous gradients if they exist
        grad_path = f"{self.dir}/client_{self.cid}_grads.pkl"
        if os.path.exists(grad_path):
            try:
                with open(grad_path, "rb") as f:
                    self.prev_grads = pickle.load(f)
            except Exception as e:
                print(f"Error loading previous gradients: {e}")

    def get_parameters(self, config: Dict[str, Scalar]):
        """Return the current local model parameters."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Set the local model parameters using given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config: Dict[str, Scalar]):
        """Train the model using the provided parameters."""
        self.set_parameters(parameters)
        
        # Get current round for learning rate decay and consistency weight
        curr_round = int(config.get("curr_round", 0)) - 1
        
        # Calculate consistency weight for self-distillation
        consistency_weight_constant = 300
        current = np.clip(curr_round, 0.0, consistency_weight_constant)
        phase = 1.0 - current / consistency_weight_constant
        consistency_weight = float(np.exp(-5.0 * phase * phase))
        
        # Calculate decayed learning rate
        current_lr = self.learning_rate * (self.learning_rate_decay ** curr_round)
        
        # Import here to avoid circular imports
        from fedbench.algorithms.depthfl.model_utils import train_depthfl
        
        try:
            train_depthfl(
                self.net,
                self.trainloader,
                self.device,
                self.num_epochs,
                current_lr,
                consistency_weight,
                self.prev_grads,
                feddyn=config.get("feddyn", True),
                alpha=config.get("alpha", 0.01),
                extended=config.get("extended", True),
                kd=config.get("kd", True)
            )
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
        
        # Save updated gradients
        grad_path = f"{self.dir}/client_{self.cid}_grads.pkl"
        try:
            with open(grad_path, "wb") as f:
                pickle.dump(self.prev_grads, f)
        except Exception as e:
            print(f"Error saving gradients: {e}")
        
        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        """Evaluate using given parameters."""
        self.set_parameters(parameters)
        
        # Import here to avoid circular imports
        from fedbench.algorithms.depthfl.model_utils import test_depthfl
        
        try:
            loss, acc = test_depthfl(self.net, self.valloader, self.device)
            return float(loss), len(self.valloader.dataset), {
                "test_accuracy": float(acc)
            }
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, 0, {"test_accuracy": 0.0}