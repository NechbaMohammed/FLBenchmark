
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader



def train_fednova(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
) -> Tuple[float, List[torch.Tensor]]:
    # pylint: disable=too-many-arguments
    """Train the network on the training set using FedNova.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The training set dataloader object.
    device : torch.device
        The device on which to train the network.
    epochs : int
        The number of epochs to train the network.
    learning_rate : float
        The learning rate.
    momentum : float
        The momentum for SGD optimizer.
    weight_decay : float
        The weight decay for SGD optimizer.

    Returns
    -------
    tuple[float, List[torch.Tensor]]
        The a_i and g_i values.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
    )
    net.train()
    local_steps = 0
    # clone all the parameters
    prev_net = [param.detach().clone() for param in net.parameters()]
    for _ in range(epochs):
        net, local_steps = _train_one_epoch_fednova(
            net, trainloader, device, criterion, optimizer, local_steps
        )
    # compute ||a_i||_1
    a_i = (
        local_steps - (momentum * (1 - momentum**local_steps) / (1 - momentum))
    ) / (1 - momentum)
    # compute g_i
    g_i = [
        torch.div(prev_param - param.detach(), a_i)
        for prev_param, param in zip(prev_net, net.parameters())
    ]

    return a_i, g_i

def _train_one_epoch_fednova(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optimizer,
    local_steps: int,
) -> Tuple[nn.Module, int]:
    # pylint: disable=too-many-arguments
    """Train the network on the training set for one epoch."""
    for data, target in trainloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        local_steps += 1
    return net, local_steps

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

    return  loss, acc

def test(net, test_dataloader, device):
    """Test function."""
    net.to(device)
    loss, test_acc = compute_accuracy(net, test_dataloader, device=device)
    #print(">> Test accuracy: %f" % test_acc)
    net.to("cpu")
    return loss, test_acc