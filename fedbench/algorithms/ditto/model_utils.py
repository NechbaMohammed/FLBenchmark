from typing import List, Tuple
import torch
import torch.nn as nn
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
import copy

def train_ditto(
    perso_net: nn.Module,
    global_net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    num_epochs : int,
    learning_rate: float,
    weight_decay: float,
    lambda_reg : float,
) -> None:
    """Train the network on the training set using Ditto.

    Parameters
    ----------
    perso_net : nn.Module
        The client's personalized model to train.
    global_net : nn.Module
        The global model (fixed during local training).
    trainloader : DataLoader
        The training set dataloader object.
    device : torch.device
        The device on which to train the network.
    epochs : int
        Number of local epochs.
    learning_rate : float
        Learning rate.
    weight_decay : float
        Weight decay for optimizer.
    lambda_reg : float
        Regularization coefficient.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(perso_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Put models on device
    perso_net.to(device)
    global_net.to(device)

    # Freeze global model
    global_params = copy.deepcopy(list(global_net.parameters()))
    
    perso_net.train()
    for _ in range(num_epochs):
        # Iterate over epoch of training data
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = perso_net(data)
            loss = criterion(output, target)

            # Regularization term: || theta - theta_global ||^2
            personalized_reg = 0.0
            for p_local, p_global in zip(perso_net.parameters(), global_params):
                personalized_reg += torch.norm(p_local - p_global.to(device)) ** 2
            loss += (lambda_reg / 2.0) * personalized_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(perso_net.parameters(), max_norm=1.0)
            optimizer.step()

    #perso_net.to("cpu")

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
    print(">> Test accuracy: %f" % test_acc)
    net.to("cpu")
    return loss, test_acc
