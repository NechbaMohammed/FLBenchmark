import os
import copy
import time
from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import NDArrays, Scalar
from omegaconf import DictConfig
from hydra.utils import instantiate
from fedbench.algorithms.moon.model_utils import train_moon

class FlowerClientMoon(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        net_id: int,
        model_cfg: DictConfig,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        mu: float,
        temperature: float,
        model_dir: str,
        momentum: float = 0.9,
        weight_decay: float = 1e-5,
       
    ):  # pylint: disable=too-many-arguments
        self.model_cfg = model_cfg
        self.net = instantiate(model_cfg)
        self.net_id = net_id
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.mu = mu  # pylint: disable=invalid-name
        self.temperature = temperature
        self.model_dir = model_dir
        self.momentum = momentum
        self.weight_decay = weight_decay

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Change the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        self.set_parameters(parameters)
        prev_net = instantiate(self.model_cfg)
        if not os.path.exists(os.path.join(self.model_dir, str(self.net_id))):
            prev_net = copy.deepcopy(self.net)
        else:
            # load previous model from model_dir
            prev_net.load_state_dict(
                torch.load(
                    os.path.join(self.model_dir, str(self.net_id), "prev_net.pt"),
                     weights_only=True
                )
            )
        global_net = instantiate(self.model_cfg)
        global_net.load_state_dict(self.net.state_dict())

        train_moon(
                net=self.net,
                global_net=global_net,
                previous_net=prev_net,
                train_dataloader=self.trainloader,
                epochs=self.num_epochs,
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                mu = self.mu,
                temperature=self.temperature,
                device=self.device,
            )
        if not os.path.exists(os.path.join(self.model_dir, str(self.net_id))):
            os.makedirs(os.path.join(self.model_dir, str(self.net_id)))
        torch.save(
            self.net.state_dict(),
            os.path.join(self.model_dir, str(self.net_id), "prev_net.pt"),
        )
        return self.get_parameters({}), len(self.trainloader), {"is_straggler": False}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        self.set_parameters(parameters)
        # skip evaluation in the client-side
        loss = 0.0
        accuracy = 0.0
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}