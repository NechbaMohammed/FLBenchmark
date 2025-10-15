import torch
import torch.optim as optim
import torch.nn as nn
import copy
from typing import Dict, List, Any, Tuple


class FedExpClient:
    """
    Client class for the FedExp federated learning algorithm.
    
    FedExp uses exponential moving average to improve convergence in federated learning,
    particularly in non-IID data settings.
    """
    
    def __init__(
        self,
        client_id: int,
        local_data: Any,  # Dataset or DataLoader
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        local_epochs: int = 10,
        criterion=nn.CrossEntropyLoss(),
    ):
        """
        Initialize the FedExp client.
        
        Args:
            client_id: Unique identifier for the client
            local_data: Client's local dataset or dataloader
            model: Neural network model
            device: Device to train on (cpu or cuda)
            learning_rate: Learning rate for local optimization
            momentum: Momentum for SGD optimizer
            local_epochs: Number of local training epochs
            criterion: Loss function
        """
        self.client_id = client_id
        self.local_data = local_data
        self.model = copy.deepcopy(model)
        self.device = device
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.local_epochs = local_epochs
        self.criterion = criterion
        
        # Initialize optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )
        
        # Initialize previous model (for FedExp algorithm)
        self.previous_model = copy.deepcopy(model)
        
        # For tracking metrics
        self.train_loss_history = []
        self.train_acc_history = []

    def update_local_model(self, global_model: nn.Module) -> None:
        """
        Update the local model with the global model parameters.
        
        Args:
            global_model: The global model from the server
        """
        # Save the previous model for weight update calculations
        self.previous_model = copy.deepcopy(self.model)
        
        # Update local model with global model
        self.model.load_state_dict(copy.deepcopy(global_model.state_dict()))
        
        # Reset optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )

    def train(self) -> Tuple[Dict[str, torch.Tensor], float, int]:
        """
        Perform local training using the client's dataset.
        
        Returns:
            model_update: Dictionary containing weight updates
            loss: Average training loss
            num_samples: Number of samples used for training
        """
        self.model.train()
        self.model.to(self.device)
        
        epoch_losses = []
        num_samples = len(self.local_data.dataset) if hasattr(self.local_data, 'dataset') else len(self.local_data)
        
        for epoch in range(self.local_epochs):
            running_loss = 0.0
            samples_count = 0
            
            for batch_idx, (data, target) in enumerate(self.local_data):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item() * data.size(0)
                samples_count += data.size(0)
            
            epoch_loss = running_loss / samples_count
            epoch_losses.append(epoch_loss)
            self.train_loss_history.append(epoch_loss)
        
        # Calculate model updates (delta)
        model_update = {}
        for name, param in self.model.named_parameters():
            prev_param = dict(self.previous_model.named_parameters())[name].data
            model_update[name] = param.data.clone() - prev_param.clone()
        
        # Calculate average loss
        average_loss = sum(epoch_losses) / len(epoch_losses)
        
        # Move model back to CPU for communication efficiency
        self.model.to(torch.device('cpu'))
        
        return model_update, average_loss, num_samples

    def evaluate(self, test_data=None):
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test dataset or dataloader (uses local data if None)
            
        Returns:
            accuracy: Test accuracy
            loss: Test loss
        """
        if test_data is None:
            test_data = self.local_data
            
        self.model.eval()
        self.model.to(self.device)
        
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                test_loss += self.criterion(output, target).item() * data.size(0)
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        test_loss /= total
        
        # Move model back to CPU
        self.model.to(torch.device('cpu'))
        
        return accuracy, test_loss