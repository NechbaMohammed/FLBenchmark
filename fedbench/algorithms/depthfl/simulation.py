"""Simulation function for DepthFL."""

import logging
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Dict, Any

import torch
from torch.utils.data import DataLoader
import json
from flwr.common import NDArrays, Scalar, Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager, SimpleClientManager

from fedbench.datasets import FederatedDataset
from fedbench.algorithms.depthfl.client import FlowerClientDepthFL
from fedbench.models import MNISTModel, CNN

# Configure logging
logging.basicConfig(level=logging.INFO)


def gen_clients(
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    model_depths: List[int],
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    learning_rate_decay: float,
    save_dir: str,
    global_model_params,  # Add global model parameters
    dataset_name: str = "mnist",
    num_classes: int = 10,
) -> List[FlowerClientDepthFL]:
    """Generate clients for DepthFL."""
    clients = []
    
    for cid in range(len(trainloaders)):
        # Get model depth for this client
        depth_idx = cid % len(model_depths)
        depth = model_depths[depth_idx]
        
        # Adjust complexity based on depth (1 is lowest, 4 is highest)
        scale_factor = depth / 4.0
        
        # Load model based on dataset type
        if dataset_name in ["mnist", "fmnist"]:
            # Adjust hidden dimensions based on depth
            hidden_dim1 = max(30, int(120 * scale_factor))  # min 30, max 120
            hidden_dim2 = max(20, int(84 * scale_factor))   # min 20, max 84
            net = MNISTModel(input_dim=256, hidden_dims=[hidden_dim1, hidden_dim2], num_classes=num_classes)
        else:  # cifar10 or other RGB datasets
            hidden_dim1 = max(30, int(120 * scale_factor))  # min 30, max 120
            hidden_dim2 = max(20, int(84 * scale_factor))   # min 20, max 84
            net = CNN(input_dim=400, hidden_dims=[hidden_dim1, hidden_dim2], num_classes=num_classes)
        
        net.to(device)
        
        # Get client-specific data loaders
        trainloader = trainloaders[cid]
        valloader = valloaders[cid]

        client = FlowerClientDepthFL(
            cid,
            net,
            trainloader,
            valloader,
            device,
            num_epochs,  # This is now correctly passed from the main function
            learning_rate,
            learning_rate_decay,
            save_dir=save_dir,
        )
        clients.append(client)
    
    return clients


def run_depthfl(
    data_config: dict,
    model_cfg: dict,
    backend_config: Dict[str, int],
    num_clients: int,
    model_depths: List[int],
    num_rounds: int,
    num_epochs: int,  # This is now properly received from main.py
    learning_rate: float,
    learning_rate_decay: float,
    model_dir: str,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: Optional[int] = None,
    min_evaluate_clients: Optional[int] = None,
    min_available_clients: Optional[int] = None,
    feddyn: bool = True,
    alpha_feddyn: float = 0.01,
    device: Optional[torch.device] = None,
    ray_init_args: Optional[Dict] = None,  # Keep for compatibility
) -> Dict[str, Any]:
    """Run DepthFL simulation without using Ray."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create necessary directories
    import os
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("client_grads", exist_ok=True)
    
    # Log key parameters for reproducibility
    logging.info(f"DepthFL Simulation Parameters:")
    logging.info(f"- Dataset: {data_config.get('name', 'unknown')}")
    logging.info(f"- Partitioning: {data_config.get('partitioning', 'unknown')}")
    logging.info(f"- Number of clients: {num_clients}")
    logging.info(f"- Number of rounds: {num_rounds}")
    logging.info(f"- Number of epochs: {num_epochs}")
    logging.info(f"- Learning rate: {learning_rate}")
    logging.info(f"- Model depths: {model_depths}")
    
    try:
        # Load federated dataset
        dataset_name = data_config.get("name", "mnist")
        federated_dataset = FederatedDataset(data_config, num_clients=num_clients)
        trainloaders, valloaders, testloader = federated_dataset.get_dataloaders()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Add dataset name to model config for evaluation
    model_cfg["dataset"] = dataset_name
    
    # Create fit config
    fit_config = {
        "feddyn": feddyn, 
        "alpha": alpha_feddyn,
        "extended": True,
        "kd": True
    }
    
    # Use common dimensions for all models to ensure compatibility
    # Use the largest possible model dimensions
    if dataset_name in ["mnist", "fmnist"]:
        global_model = MNISTModel(
            input_dim=256,
            hidden_dims=[120, 84],  # Use fixed dimensions for global model
            num_classes=model_cfg.get("num_classes", 10)
        )
    else:  # cifar10 or other RGB datasets
        global_model = CNN(
            input_dim=400,
            hidden_dims=[120, 84],  # Use fixed dimensions for global model
            num_classes=model_cfg.get("num_classes", 10)
        )
    
    global_model.to(device)
    
    # Get initial global parameters
    global_parameters = [val.cpu().numpy() for _, val in global_model.state_dict().items()]
    
    # Create a separate list of models for each client with the same dimensions as the global model
    client_models = []
    for cid in range(num_clients):
        # Create models with the same architecture as the global model
        if dataset_name in ["mnist", "fmnist"]:
            model = MNISTModel(
                input_dim=256,
                hidden_dims=[120, 84],  # Use same dimensions as global model
                num_classes=model_cfg.get("num_classes", 10)
            )
        else:  # cifar10 or other RGB datasets
            model = CNN(
                input_dim=400,
                hidden_dims=[120, 84],  # Use same dimensions as global model
                num_classes=model_cfg.get("num_classes", 10)
            )
        model.to(device)
        # Apply global parameters to client model
        params_dict = zip(model.state_dict().keys(), global_parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict)
        client_models.append(model)
    
    # Create clients with the same model architecture
    clients = []
    for cid in range(num_clients):
        # Get client-specific data loaders
        trainloader = trainloaders[cid]
        valloader = valloaders[cid]
        
        client = FlowerClientDepthFL(
            cid,
            client_models[cid],
            trainloader,
            valloader,
            device,
            num_epochs,  # Use the num_epochs passed from main
            learning_rate,
            learning_rate_decay,
            save_dir=model_dir,
        )
        clients.append(client)
    
    # Define FedAvg parameters aggregation
    def aggregate_parameters(parameters_list, num_examples_list):
        """Aggregate parameters using weighted averaging."""
        if not parameters_list:
            return None
        
        # Compute total number of examples
        total_examples = sum(num_examples_list)
        if total_examples == 0:
            return parameters_list[0]  # Return any parameters if no examples
        
        # Get the number of parameters
        num_params = len(parameters_list[0])
        
        # Initialize aggregated parameters with zeros
        aggregated_params = [
            torch.zeros_like(torch.tensor(param))
            for param in parameters_list[0]
        ]
        
        # Weighted aggregation
        for i, (params, num_examples) in enumerate(zip(parameters_list, num_examples_list)):
            weight = num_examples / total_examples
            for j, param in enumerate(params):
                weighted_param = torch.tensor(param) * weight
                aggregated_params[j] += weighted_param
                
        return [param.numpy() for param in aggregated_params]
    
    # Initialize metrics history dictionary
    history = {
        "metrics_centralized": {
            "loss": [],
            "test_accuracy": [],
        }
    }
    
    # Run federated learning rounds
    print(f"Starting federated learning with {num_rounds} rounds and {num_epochs} epochs per client")
    
    for round_num in range(1, num_rounds + 1):
        print(f"Round {round_num}/{num_rounds}")
        
        # Update fit config with current round
        round_fit_config = dict(fit_config)
        round_fit_config["curr_round"] = round_num
        
        # Client selection for this round
        num_fit_clients = max(int(fraction_fit * num_clients), 1)
        fit_client_indices = list(range(num_clients))[:num_fit_clients]
        
        # Clients training (fit)
        print(f"Training {len(fit_client_indices)} clients with {num_epochs} epochs each")
        parameters_list = []
        num_examples_list = []
        
        for idx in fit_client_indices:
            client = clients[idx]
            
            # Train client
            updated_parameters, num_examples, _ = client.fit(global_parameters, round_fit_config)
            
            # Collect results
            parameters_list.append(updated_parameters)
            num_examples_list.append(num_examples)
        
        # Aggregate parameters
        aggregated_parameters = aggregate_parameters(parameters_list, num_examples_list)
        global_parameters = aggregated_parameters
        
        # Evaluate on global test set
        # Load parameters to global model
        params_dict = zip(global_model.state_dict().keys(), global_parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        global_model.load_state_dict(state_dict)
        
        # Evaluate model
        global_model.eval()
        total = 0
        correct = 0
        loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = global_model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        loss = loss / len(testloader)
        
        # Save metrics
        history["metrics_centralized"]["loss"].append((round_num, loss))
        history["metrics_centralized"]["test_accuracy"].append((round_num, accuracy))
        
        print(f"Round {round_num}: loss={loss:.4f}, accuracy={accuracy:.4f}")
        
        # Save model weights at each round
        try:
            round_model_path = os.path.join(model_dir, f"global_model_round_{round_num}.pt")
            torch.save(global_model.state_dict(), round_model_path)
            print(f"Saved global model at round {round_num} to {round_model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    # Create a class to mimic Flower's History object
    class History:
        def __init__(self, metrics_centralized):
            self.metrics_centralized = metrics_centralized
    
    # Save final model
    try:
        final_model_path = os.path.join(model_dir, "final_model.pt")
        torch.save(global_model.state_dict(), final_model_path)
        print(f"Saved final global model to {final_model_path}")
        
        # Save metrics to JSON for easier analysis
        metrics_path = os.path.join(model_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump({
                "loss": [(int(r), float(l)) for r, l in history["metrics_centralized"]["loss"]],
                "test_accuracy": [(int(r), float(a)) for r, a in history["metrics_centralized"]["test_accuracy"]]
            }, f, indent=2)
        print(f"Saved metrics to {metrics_path}")
    except Exception as e:
        print(f"Error saving final results: {e}")
    
    return History(history["metrics_centralized"])