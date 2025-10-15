from typing import List, Tuple
import torch
import torch.nn as nn
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
import numpy as np

class DashaCompressor:
    """Implements compression techniques for DASHA algorithm."""
    
    def __init__(self, number_of_coordinates: int = 10):
        self.number_of_coordinates = number_of_coordinates
        
    def compress(self, vector: torch.Tensor) -> torch.Tensor:
        """Apply random sparsification compression (Random-K)."""
        if self.number_of_coordinates >= vector.numel():
            # If the number of coordinates is >= vector size, return the full vector
            return vector
            
        # Choose random indices
        indices = torch.randperm(vector.numel())[:self.number_of_coordinates].to(vector.device)
        
        # Create compressed vector (sparse representation)
        compressed = torch.zeros_like(vector)
        compressed.view(-1)[indices] = vector.view(-1)[indices] * (vector.numel() / self.number_of_coordinates)
        
        return compressed

class DashaOptimizer(SGD):
    """Implements SGD optimizer step function for DASHA algorithm."""

    def __init__(self, params, step_size, momentum=0, weight_decay=0):
        super().__init__(
            params, lr=step_size, momentum=momentum, weight_decay=weight_decay
        )
        
    def step_dasha(self, variance_reduction_direction):
        """
        Implement the custom step function for DASHA.
        
        The DASHA update rule uses variance reduction direction to reduce 
        communication costs while maintaining convergence.
        """
        self.step()  # Call the standard SGD step first
        
        # Apply the variance reduction direction
        for group in self.param_groups:
            for p, v_dir in zip(group["params"], variance_reduction_direction):
                if p.grad is None:
                    continue
                p.data.add_(v_dir, alpha=-group["lr"])

def train_dasha(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    prev_model: List[torch.Tensor],
    compressor: DashaCompressor,
    probability_q: float = 0.5
) -> Tuple[nn.Module, List[torch.Tensor]]:
    """Train the network on the training set using DASHA.

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
    prev_model : List[torch.Tensor]
        The previous model parameters.
    compressor : DashaCompressor
        The compressor to use for variance reduction.
    probability_q : float
        The probability of computing a full gradient (vs compressed gradient).
    
    Returns
    -------
    Tuple[nn.Module, List[torch.Tensor]]
        The trained network and the variance reduction direction.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = DashaOptimizer(
        net.parameters(), learning_rate, momentum, weight_decay
    )
    
    net.train()
    
    # Get model parameters as a list of tensors
    current_params = [param.clone().detach() for param in net.parameters()]
    
    # Decide whether to use full gradient (with probability q) or compressed gradient (with probability 1-q)
    use_full_gradient = torch.rand(1).item() < probability_q
    
    # Compute previous gradient or direction
    prev_direction = []
    for curr, prev in zip(current_params, prev_model):
        direction = curr - prev
        # Apply compression if not using full gradient
        if not use_full_gradient:
            direction = compressor.compress(direction)
        prev_direction.append(direction)
    
    for _ in range(epochs):
        net = _train_one_epoch_dasha(
            net, trainloader, device, criterion, optimizer, prev_direction
        )
    
    # Return the updated network and the variance reduction direction used
    return net, prev_direction

def _train_one_epoch_dasha(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: DashaOptimizer,
    variance_reduction_direction: List[torch.Tensor],
) -> nn.Module:
    """Train the network on the training set for one epoch using DASHA."""
    for data, target in trainloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        
        # Apply the DASHA step with variance reduction
        optimizer.step_dasha(variance_reduction_direction)
    
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