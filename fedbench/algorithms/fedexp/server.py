import torch
import torch.nn as nn
import copy
from typing import Dict, List, Any, Tuple
import numpy as np


class FedExpServer:
    """
    Server class for the FedExp federated learning algorithm.
    
    FedExp uses exponential moving average to improve convergence in federated learning,
    particularly in heterogeneous (non-IID) data distributions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        beta: float = 0.9,  # EMA parameter
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the FedExp server.
        
        Args:
            model: Initial global model
            beta: Exponential moving average parameter (0 <= beta < 1)
            device: Device to perform computations on
        """
        self.model = copy.deepcopy(model)        # Standard global model
        self.previous_model = copy.deepcopy(model)
        self.beta = beta
        self.device = device
        
        # For tracking metrics
        self.global_loss_history = []
        self.global_acc_history = []
        self.client_losses = {}
        self.client_samples = {}
        
        # Initialize EMA model (for FedExp algorithm)
        self.ema_model = copy.deepcopy(model)
        self.round_counter = 0
        
    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float, int]]) -> nn.Module:
        """
        Aggregate client updates using FedExp algorithm.
        
        Args:
            client_updates: List of tuples containing (model_update, loss, num_samples) from clients
            
        Returns:
            Updated global model
        """
        self.round_counter += 1
        
        # Save the previous model for EMA calculation
        self.previous_model = copy.deepcopy(self.model)
        
        # Extract client updates, losses and sample counts
        updates = [update for update, _, _ in client_updates]
        losses = [loss for _, loss, _ in client_updates]
        sample_counts = [count for _, _, count in client_updates]
        
        # Store client data for metrics
        for i, (_, loss, samples) in enumerate(client_updates):
            self.client_losses[i] = loss
            self.client_samples[i] = samples
        
        # Calculate global weighted update (FedAvg style)
        total_samples = sum(sample_counts)
        scaled_updates = {}
        
        # Initialize with zeros for all parameters
        for name, param in self.model.named_parameters():
            scaled_updates[name] = torch.zeros_like(param.data)
        
        # Perform weighted aggregation of updates
        for client_idx, update in enumerate(updates):
            weight = sample_counts[client_idx] / total_samples
            for name, param in self.model.named_parameters():
                if name in update:
                    scaled_updates[name] += update[name] * weight
        
        # Apply updates to the global model
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in scaled_updates:
                    param.data += scaled_updates[name]
        
        # Apply exponential moving average for FedExp
        if self.round_counter > 1:  # Skip first round as there's no previous data
            with torch.no_grad():
                for ema_param, global_param in zip(self.ema_model.parameters(), self.model.parameters()):
                    ema_param.data = self.beta * ema_param.data + (1 - self.beta) * global_param.data
        else:
            # First round, initialize EMA model with current global model
            self.ema_model = copy.deepcopy(self.model)
        
        # Calculate average loss for this round
        avg_loss = sum(l * s for l, s in zip(losses, sample_counts)) / total_samples
        self.global_loss_history.append(avg_loss)
        
        # Return the EMA model as the new global model
        return self.ema_model
    
    def evaluate(self, test_data):
        """
        Evaluate the global model on test data.
        
        Args:
            test_data: Test dataset or dataloader
            
        Returns:
            accuracy: Test accuracy
            loss: Test loss
        """
        self.model.eval()
        self.model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                test_loss += criterion(output, target).item() * data.size(0)
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        test_loss /= total
        
        self.global_acc_history.append(accuracy)
        
        # Move model back to CPU
        self.model.to(torch.device('cpu'))
        
        return accuracy, test_loss
    
    def get_model(self):
        """
        Get the current global model.
        
        Returns:
            Current global model
        """
        return self.ema_model  # Return the EMA model as per FedExp algorithm