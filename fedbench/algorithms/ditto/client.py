import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict
from collections import OrderedDict
import flwr as fl
from flwr.common import Scalar
from fedbench.algorithms.ditto.model_utils import train_ditto, test  # <-- You will implement train_ditto
import numpy as np
import os
import csv

class FlowerClientDitto(fl.client.NumPyClient):
    """Flower client implementing Ditto."""

    def __init__(
        self,
        cid: int,
        global_model: torch.nn.Module,
        personalized_model: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        client_results_dir: str,
        momentum: float,
        weight_decay: float,
        lambda_reg: float,
    ) -> None:
        self.cid = cid
        self.global_model = global_model
        self.personalized_model = personalized_model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.client_results_dir = client_results_dir
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lambda_reg = lambda_reg

    def get_parameters(self, config: Dict[str, Scalar] = {}):
        """Return the current personalized model parameters."""
        return [val.cpu().numpy() for _, val in self.personalized_model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set both global and personalized models using given parameters."""
        params_dict = zip(self.global_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.global_model.load_state_dict(state_dict, strict=True)
        self.personalized_model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config: Dict[str, Scalar] = {}):
        """Train personalized model with Ditto regularization."""
        self.set_parameters(parameters)

        # Train personalized model
        train_ditto(
            global_net=self.global_model,
            perso_net=self.personalized_model,
            trainloader=self.trainloader,
            device=self.device,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            #momentum=self.momentum,
            weight_decay=self.weight_decay,
            lambda_reg=self.lambda_reg,
        )

        updated_weights = self.get_parameters()
        return updated_weights, len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config: Dict[str, Scalar] = {}):
        """Evaluate the personalized model."""
        print(f"Evaluating the personalized model for client {self.cid}...")
        self.set_parameters(parameters)
        loss, acc = test(self.personalized_model, self.valloader, self.device)
        print(f"Client {self.cid} - Loss: {loss}, Accuracy: {acc}")

        os.makedirs(self.client_results_dir, exist_ok=True)
        client_file = os.path.join(self.client_results_dir, f"client_{self.cid}_results.csv")

        # Check if the file exists to write the header only once
        file_exists = os.path.isfile(client_file)

        with open(client_file, "a", newline="") as f:  # Open in append mode
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Loss", "Accuracy"])  # Write header only if the file is new
            writer.writerow([loss, acc])  # Append the new loss and accuracy

        return float(loss), len(self.valloader.dataset), {"accuracy": float(acc)}