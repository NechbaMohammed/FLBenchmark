import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
import copy
import random
import numpy as np


from fedbench.algorithms.fedexp.client import FedExpClient
from fedbench.algorithms.fedexp.server import FedExpServer


class FedExpStrategy:
    """
    Implementation of the FedExp federated learning strategy.
    
    FedExp applies an exponential moving average (EMA) to model updates to improve
    convergence and model performance in heterogeneous (non-IID) data settings.
    """
    
    def __init__(
        self,
        model: nn.Module,
        client_data_dict: Dict[int, Any],  # Dictionary mapping client ID to client data
        test_data: Any = None,
        num_clients: int = 10,
        client_sample_ratio: float = 1.0,
        local_epochs: int = 10,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        beta: float = 0.9,  # EMA parameter
        device: torch.device = torch.device("cpu"),
        seed: int = 42,
    ):
        """
        Initialize the FedExp strategy.
        
        Args:
            model: Neural network model
            client_data_dict: Dictionary mapping client IDs to their local data
            test_data: Test dataset for evaluation
            num_clients: Total number of clients
            client_sample_ratio: Fraction of clients to select in each round
            local_epochs: Number of local training epochs for each client
            learning_rate: Learning rate for local optimization
            momentum: Momentum for SGD optimizer
            beta: Exponential moving average parameter (0 <= beta < 1)
            device: Device to perform computations on
            seed: Random seed for reproducibility
        """
        self.model = model
        self.client_data_dict = client_data_dict
        self.test_data = test_data
        self.num_clients = num_clients
        self.client_sample_ratio = client_sample_ratio
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta = beta
        self.device = device
        self.seed = seed
        
        # Set seed for reproducibility
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Initialize server
        self.server = FedExpServer(
            model=self.model,
            beta=self.beta,
            device=self.device
        )
        
        # Initialize clients
        self.clients = {}
        for client_id, client_data in self.client_data_dict.items():
            self.clients[client_id] = FedExpClient(
                client_id=client_id,
                local_data=client_data,
                model=self.model,
                device=self.device,
                learning_rate=self.learning_rate,
                momentum=self.momentum,
                local_epochs=self.local_epochs
            )
    
    def select_clients(self, round_idx: int) -> List[int]:
        """
        Select a subset of clients to participate in the current round.
        
        Args:
            round_idx: Current communication round
            
        Returns:
            List of selected client IDs
        """
        num_selected = max(1, int(self.client_sample_ratio * self.num_clients))
        
        # Set seed based on round for reproducibility but different selection each round
        np.random.seed(self.seed + round_idx)
        
        selected_clients = np.random.choice(
            list(self.clients.keys()), 
            size=num_selected, 
            replace=False
        ).tolist()
        
        return selected_clients
    
    def train_round(self, round_idx: int) -> Tuple[nn.Module, float, float]:
        """
        Execute one round of federated learning.
        
        Args:
            round_idx: Current communication round
            
        Returns:
            Tuple containing:
                - Updated global model
                - Average training loss
                - Global test accuracy (if test_data provided)
        """
        # Select clients for this round
        selected_clients = self.select_clients(round_idx)
        
        # Get current global model
        global_model = self.server.get_model()
        
        # Distribute global model to selected clients
        for client_id in selected_clients:
            self.clients[client_id].update_local_model(global_model)
        
        # Collect client updates after local training
        client_updates = []
        for client_id in selected_clients:
            model_update, loss, num_samples = self.clients[client_id].train()
            client_updates.append((model_update, loss, num_samples))
        
        # Aggregate updates at the server using FedExp algorithm
        updated_model = self.server.aggregate(client_updates)
        
        # Evaluate the global model if test data is provided
        accuracy = 0.0
        if self.test_data is not None:
            accuracy, _ = self.server.evaluate(self.test_data)
        
        # Calculate average loss across clients
        total_samples = sum(num_samples for _, _, num_samples in client_updates)
        avg_loss = sum(loss * samples for _, loss, samples in client_updates) / total_samples
        
        return updated_model, avg_loss, accuracy
    
    def train(self, num_rounds: int) -> Dict[str, List[float]]:
        """
        Execute the federated learning process for multiple rounds.
        
        Args:
            num_rounds: Number of communication rounds
            
        Returns:
            Dictionary with training metrics history
        """
        metrics = {
            'train_loss': [],
            'test_accuracy': []
        }
        
        for round_idx in range(num_rounds):
            # Execute one round of training
            _, avg_loss, accuracy = self.train_round(round_idx)
            
            # Record metrics
            metrics['train_loss'].append(avg_loss)
            metrics['test_accuracy'].append(accuracy)
            
            # Print progress
            print(f"Round {round_idx+1}/{num_rounds}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
        
        return metrics
    
    def get_global_model(self) -> nn.Module:
        """
        Get the current global model.
        
        Returns:
            Current global model
        """
        return self.server.get_model()