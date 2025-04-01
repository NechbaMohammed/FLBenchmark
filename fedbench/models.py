import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class MLP(nn.Module):
    """Implement an MLP model with three hidden layers for tabular datasets.

    Parameters
    ----------
    input_dim : int
        The input dimension (number of features).
    hidden_dims : List[int]
        The hidden dimensions for the MLP.
    num_classes : int
        The number of classes in the dataset.
    """

    def __init__(self, input_dim, hidden_dims, num_classes) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])  # Input layer to first hidden layer
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])  # First to second hidden layer
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])  # Second to third hidden layer
        self.fc4 = nn.Linear(hidden_dims[2], num_classes)  # Third hidden layer to output layer

    def forward(self, x):
        """Implement forward pass."""
        x = F.relu(self.fc1(x))  # First hidden layer
        x = F.relu(self.fc2(x))  # Second hidden layer
        x = F.relu(self.fc3(x))  # Third hidden layer
        x = self.fc4(x)  # Output layer
        return x
    
class MNISTModel(nn.Module):
    """Implement a CNN model for MNIST and Fashion-MNIST.

    Parameters
    ----------
    input_dim : int
        The input dimension for classifier.
    hidden_dims : List[int]
        The hidden dimensions for classifier.
    num_classes : int
        The number of classes in the dataset.
    """

    def __init__(self, input_dim, hidden_dims, num_classes) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], num_classes)

    def forward(self, x):
        """Implement forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class CNN(nn.Module):
    """Implement a CNN model for 3-channel RGB images (e.g., SVHN, CIFAR-10, CIFAR-100).

    Parameters
    ----------
    input_dim : int
        The input dimension for classifier.
    hidden_dims : List[int]
        The hidden dimensions for classifier.
    num_classes : int
        The number of classes in the dataset.
    """

    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], num_classes)

    def forward(self, x):
        """Implement forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNHeader(nn.Module):
    """Simple CNN model."""

    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

    def forward(self, x):
        """Implement forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    

    
class CNNModelMOON(nn.Module):
    """CNNModel for MOON."""

    def __init__(self, input_dim,hidden_dims, output_dim, num_classes):
        super().__init__()

        
        self.features = CNNHeader(
                input_dim=input_dim, hidden_dims=hidden_dims
            )
        num_ftrs = 84 #hidden_dims[1]

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, output_dim)

        # last layer
        self.l3 = nn.Linear(output_dim, num_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            return model
        except KeyError as err:
            raise ValueError("Invalid model name.") from err

    def forward(self, x):
        """Forward."""
        h = self.features(x)
        #h = h.squeeze()
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

        y = self.l3(x)
        return h, x, y
    
class MnistCNNHeader(nn.Module):
    """Simple CNN model."""

    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

    def forward(self, x):
        """Forward."""
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.fc3(x)
        return x
    
class MnistModelMOON(nn.Module):
    """MnistModel for MOON."""

    def __init__(self, input_dim,hidden_dims, output_dim, num_classes):
        super().__init__()

        
        self.features = MnistCNNHeader(
                input_dim=input_dim, hidden_dims=hidden_dims
            )
        num_ftrs = 84

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, output_dim)

        # last layer
        self.l3 = nn.Linear(output_dim, num_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            return model
        except KeyError as err:
            raise ValueError("Invalid model name.") from err

    def forward(self, x):
        """Forward."""
        h = self.features(x)
        #h = h.squeeze()
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

        y = self.l3(x)
        return h, x, y

class MLPHeader(nn.Module):
    """Simple MLP model."""

    def __init__(self, input_dim, hidden_dims) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])  # Input layer to first hidden layer
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])  # First to second hidden layer
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])  # Second to third hidden layer

    def forward(self, x):
        """Implement forward pass."""
        x = F.relu(self.fc1(x))  # First hidden layer
        x = F.relu(self.fc2(x))  # Second hidden layer
        x = F.relu(self.fc3(x))  # Third hidden layer
        return x
class MLPModelMOON(nn.Module):
    """MLPModel for MOON."""

    def __init__(self, input_dim,hidden_dims, output_dim, num_classes):
        super().__init__()

        
        self.features = MLPHeader(
                input_dim=input_dim, hidden_dims=hidden_dims
            )
        num_ftrs = hidden_dims[2]

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, output_dim)

        # last layer
        self.l3 = nn.Linear(output_dim, num_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            return model
        except KeyError as err:
            raise ValueError("Invalid model name.") from err

    def forward(self, x):
        """Forward."""
        h = self.features(x)
        #h = h.squeeze()
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

        y = self.l3(x)
        return h, x, y

class CNNTamuna(nn.Module):
    """Convolutional Neural Network architecture.

    As described in McMahan 2017 paper:

    [Communication-Efficient Learning of Deep Networks from
    Decentralized Data] (https://arxiv.org/pdf/1602.05629.pdf)
    """

    def __init__(self, input_dim, hidden_dim, num_classes) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    