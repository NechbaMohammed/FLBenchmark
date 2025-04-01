
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from fedbench.algorithms.fedavg.model_utils import train, test
from flwr.client import Client, ClientApp, NumPyClient


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, device: torch.device,epochs,
                  learning_rate: float,
        momentum: float,
        weight_decay: float):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        
    def set_parameters(self, parameters):
        """Set the local model parameters using given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters( parameters)
        train(self.net, self.trainloader, epochs=self.epochs, device = self.device, learning_rate=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        return self.get_parameters(config), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters( parameters)
        loss, accuracy = test(self.net, self.valloader, device= self.device)
        return (
            float(loss),
            len(self.valloader.dataset),  # num_examples
            {"accuracy": float(accuracy)}  # metrics
        )