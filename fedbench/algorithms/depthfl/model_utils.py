"""Training and testing functions for DepthFL."""

from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class KLLoss(nn.Module):
    """KL divergence loss for self distillation."""

    def __init__(self):
        super().__init__()
        self.temperature = 1

    def forward(self, pred, label):
        """KL loss forward."""
        predict = F.log_softmax(pred / self.temperature, dim=1)
        target_data = F.softmax(label / self.temperature, dim=1)
        target_data = target_data + 10 ** (-7)
        with torch.no_grad():
            target = target_data.detach().clone()

        loss = (
            self.temperature
            * self.temperature
            * ((target * (target.log() - predict)).sum(1).sum() / target.size()[0])
        )
        return loss


def train_depthfl(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    consistency_weight: float,
    prev_grads: dict,
    feddyn: bool = True,
    alpha: float = 0.01,
    extended: bool = True,
    kd: bool = True
) -> None:
    """Train the network using DepthFL approach.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained.
    epochs : int
        The number of epochs the model should be trained for.
    learning_rate : float
        The learning rate for the SGD optimizer.
    consistency_weight : float
        Weight for self distillation loss (not used in standard model).
    prev_grads : dict
        Previous gradients for FedDyn regularization.
    feddyn : bool
        Whether to use FedDyn regularization.
    alpha : float
        FedDyn regularization parameter.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=1e-3)
    global_params = {
        k: val.detach().clone().flatten() for (k, val) in net.named_parameters()
    }

    for k, _ in net.named_parameters():
        if k in prev_grads:
            prev_grads[k] = prev_grads[k].to(device)
        else:
            prev_grads[k] = torch.zeros(net.state_dict()[k].numel(), device=device)

    net.train()
    config = {
        "feddyn": feddyn,
        "alpha": alpha
    }
    
    for _ in range(epochs):
        _train_one_epoch(
            net,
            global_params,
            trainloader,
            device,
            criterion,
            optimizer,
            config,
            prev_grads,
        )

    # update prev_grads for FedDyn
    if feddyn:
        update_prev_grads(alpha, net, prev_grads, global_params)


def update_prev_grads(alpha, net, prev_grads, global_params):
    """Update prev_grads for FedDyn."""
    for k, param in net.named_parameters():
        curr_param = param.detach().clone().flatten()
        if k in prev_grads:
            prev_grads[k] = prev_grads[k] - alpha * (
                curr_param - global_params[k]
            )
            prev_grads[k] = prev_grads[k].to(torch.device("cpu"))


def _train_one_epoch(
    net: nn.Module,
    global_params: dict,
    trainloader: DataLoader,
    device: torch.device,
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.SGD,
    config: dict,
    prev_grads: dict,
):
    """Train for one epoch with DepthFL approach."""
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        loss = torch.zeros(1).to(device)
        optimizer.zero_grad()
        
        # For standard models (not multi-classifier like ResNet)
        outputs = net(images)
        loss = criterion(outputs, labels)

        # Dynamic regularization in FedDyn
        if config["feddyn"]:
            for k, param in net.named_parameters():
                if k in prev_grads:
                    curr_param = param.flatten()
                    lin_penalty = torch.dot(curr_param, prev_grads[k])
                    loss -= lin_penalty

                    quad_penalty = (
                        config["alpha"]
                        / 2.0
                        * torch.sum(torch.square(curr_param - global_params[k]))
                    )
                    loss += quad_penalty

        loss.backward()
        optimizer.step()


def test_depthfl(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Evaluate the network on the entire test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    device : torch.device
        The device on which the model should be tested.

    Returns
    -------
    Tuple[float, float]
        The loss and accuracy of the model.
    """
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            # For standard model (not multi-classifier like ResNet)
            outputs = net(images)
            
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
        
    loss /= len(testloader.dataset)
    accuracy = correct / total
    
    return loss, accuracy