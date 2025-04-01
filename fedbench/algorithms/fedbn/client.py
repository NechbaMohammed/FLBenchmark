import torch
import flwr as fl
import os
import pickle
from collections import OrderedDict
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from flwr.common import NDArrays, Scalar
from fedbench.algorithms.fedbn.model_utils import train, test



class FedBNFlowerClient(fl.client.NumPyClient):
    """Flower client implementing FedBN strategy.
    
    This client excludes BatchNorm layer parameters from being shared with the server
    and maintains them locally across rounds.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        trainloader: DataLoader,
        testloader: DataLoader,
        save_path: str,
        client_id: int,
        epochs: int,
       learning_rate: float,
        momentum: float,
        weight_decay: float,
        **kwargs,
    ) -> None:
        """Initialize FedBNFlowerClient.
        
        Args:
            model: The neural network model to train
            trainloader: DataLoader for training data
            testloader: DataLoader for test/validation data
            save_path: Path to save BatchNorm layer states
            client_id: Unique identifier for this client
            l_r: Learning rate for local training
        """
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.client_id = client_id
        self.epochs = epochs
        
        # Setup device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup BN state persistence
        bn_state_dir = os.path.join(save_path, "bn_states")
        os.makedirs(bn_state_dir, exist_ok=True)
        self.bn_state_pkl = os.path.join(bn_state_dir, f"client_{client_id}.pkl")

    def _save_bn_statedict(self) -> None:
        """Save BatchNorm layer states to disk."""
        bn_state = {
            name: val.cpu().numpy()
            for name, val in self.model.state_dict().items()
            if "bn" in name.lower()  # Case-insensitive check for BN layers
        }

        with open(self.bn_state_pkl, "wb") as handle:
            pickle.dump(bn_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_bn_statedict(self) -> Dict[str, torch.Tensor]:
        """Load BatchNorm layer states from disk."""
        with open(self.bn_state_pkl, "rb") as handle:
            data = pickle.load(handle)
        bn_state_dict = {k: torch.tensor(v) for k, v in data.items()}
        return bn_state_dict

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return model parameters excluding BatchNorm layers."""
        # Save current BN state before returning parameters
        self._save_bn_statedict()
        
        # Exclude BN layer parameters
        return [
            val.cpu().numpy()
            for name, val in self.model.state_dict().items()
            if "bn" not in name.lower()  # Case-insensitive check
        ]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters while preserving local BatchNorm layers."""
        # Get non-BN parameter names
        keys = [k for k in self.model.state_dict().keys() if "bn" not in k.lower()]
        
        # Create state dict for non-BN parameters
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        # Load non-BN parameters
        self.model.load_state_dict(state_dict, strict=False)

        # Load BN parameters if they exist (won't exist in first round)
        if os.path.exists(self.bn_state_pkl):
            bn_state_dict = self._load_bn_statedict()
            self.model.load_state_dict(bn_state_dict, strict=False)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Train the model on local data while preserving BN layers."""
        # Set parameters (excluding BN layers)
        self.set_parameters(parameters)

        # Evaluate global model on local train set before training
        pre_train_loss, pre_train_acc = test(
            self.model, self.trainloader, device=self.device
        )

        # Train model on local dataset
        loss, acc = train(
            self.model,
            self.trainloader,
            epochs= self.epochs,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            device=self.device,
        )

        # Construct metrics
        fl_round = config["round"]
        metrics = {
            "round": fl_round,
            "accuracy": acc,
            "loss": loss,
            "pre_train_loss": pre_train_loss,
            "pre_train_acc": pre_train_acc,
        }

        return (
            self.get_parameters({}),
            len(self.trainloader.dataset),
            metrics,
        )

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Evaluate the model on local test data."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader, device=self.device)
        return (
            float(loss),
            len(self.testloader.dataset),
            {
                "loss": loss,
                "accuracy": accuracy,
            },
        )