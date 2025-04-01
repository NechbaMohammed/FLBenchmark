
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from collections import OrderedDict
import flwr as fl
from fedbench.algorithms.fednova.model_utils import train_fednova, test
from flwr.common import Scalar

class FlowerClientFedNova(fl.client.NumPyClient):

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
    ) -> None:
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

    def get_parameters(self, config: Dict[str, Scalar]):
        """Return the current local model parameters."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Set the local model parameters using given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config: Dict[str, Scalar]):
        """Implement distributed fit function for a given client for FedNova."""
        self.set_parameters(parameters)
        a_i, g_i = train_fednova(
            self.net,
            self.trainloader,
            self.device,
            self.num_epochs,
            self.learning_rate,
            self.momentum,
            self.weight_decay,
        )
        # final_p_np = self.get_parameters({})
        g_i_np = [param.cpu().numpy() for param in g_i]
        return g_i_np, len(self.trainloader.dataset), {"a_i": a_i}

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        """Evaluate using given parameters."""
        self.set_parameters(parameters)
        loss, acc = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(acc)}