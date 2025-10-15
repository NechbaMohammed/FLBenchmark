from typing import List, Tuple
import torch
import torch.nn as nn
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader

class ScaffoldOptimizer(SGD):
    """Implements SGD optimizer step function as defined in the SCAFFOLD paper."""

    def __init__(self, params, step_size, momentum, weight_decay):
        super().__init__(
            params, lr=step_size, momentum=momentum, weight_decay=weight_decay
        )

    def step_custom(self, server_cv, client_cv):
        """Implement the custom step function for SCAFFOLD."""
        self.step()
        for group in self.param_groups:
            for par, s_cv, c_cv in zip(group["params"], server_cv, client_cv):
                par.data.add_(s_cv - c_cv, alpha=-group["lr"])

def train_fedrep(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    server_cv: torch.Tensor,
    client_cv: torch.Tensor,
    head_epochs: int = 1,
    rep_parameters: list = None,
    head_parameters: list = None
) -> None:
    """Train the network on the training set using FedRep with SCAFFOLD structure.

    Parameters
    ----------
    net : nn.Module
        The neural network to train (split into representation and head).
    trainloader : DataLoader
        The training set dataloader object.
    device : torch.device
        The device on which to train the network.
    epochs : int
        The number of total training rounds.
    learning_rate : float
        The learning rate for both representation and head.
    momentum : float
        The momentum for SGD optimizer.
    weight_decay : float
        The weight decay for SGD optimizer.
    server_cv : torch.Tensor
        The server's control variate for representation parameters.
    client_cv : torch.Tensor
        The client's control variate for representation parameters.
    head_epochs : int
        Number of epochs to train the head locally per round.
    rep_parameters : list
        List of parameters for the representation part of the network.
    head_parameters : list
        List of parameters for the head part of the network.
    """
    criterion = nn.CrossEntropyLoss()
    rep_optimizer = ScaffoldOptimizer(
        rep_parameters, learning_rate, momentum, weight_decay
    )
    head_optimizer = SGD(
        head_parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
    )
    net.train()
    for _ in range(epochs):
        # Train head for head_epochs
        for _ in range(head_epochs):
            net = _train_one_epoch_fedrep(
                net, trainloader, device, criterion, head_optimizer, head=True
            )
        # Train representation for one epoch using SCAFFOLD
        net = _train_one_epoch_fedrep(
            net, trainloader, device, criterion, rep_optimizer, head=False,
            server_cv=server_cv, client_cv=client_cv
        )

def _train_one_epoch_fedrep(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optimizer,
    head: bool,
    server_cv: torch.Tensor = None,
    client_cv: torch.Tensor = None
) -> nn.Module:
    """Train either the head or representation of the network for one epoch.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The training set dataloader object.
    device : torch.device
        The device on which to train.
    criterion : nn.Module
        The loss function.
    optimizer : Optimizer
        The optimizer (SGD for head, ScaffoldOptimizer for representation).
    head : bool
        If True, train the head; if False, train the representation.
    server_cv : torch.Tensor, optional
        The server's control variate (used for representation only).
    client_cv : torch.Tensor, optional
        The client's control variate (used for representation only).
    """
    for data, target in trainloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], max_norm=1.0)
        if head:
            optimizer.step()
        else:
            optimizer.step_custom(server_cv, client_cv)
    return net

def compute_accuracy(model, dataloader, device):
    """Compute accuracy."""
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    loss = loss / total
    acc = correct / total
    return loss, acc

def test(net, test_dataloader, device):
    """Test function."""
    net.to(device)
    loss, test_acc = compute_accuracy(net, test_dataloader, device=device)
    net.to("cpu")
    return loss, test_acc