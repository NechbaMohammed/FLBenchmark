from typing import Callable, List, Optional, Tuple, Dict, Any
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from flwr.simulation import start_simulation
from flwr.common import NDArrays, Scalar, Context
from flwr.server.client_manager import SimpleClientManager
from flwr.server import ServerConfig

from hydra.utils import instantiate
from fedbench.datasets import FederatedDataset
from fedbench.algorithms.fedpac.client import FedPACClient
from fedbench.algorithms.fedpac.strategy import FedPACStrategy
from fedbench.algorithms.fedpac.server import FedPACServer


def fedpac_gen_client_fn(
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    model_cfg: DictConfig,
    device: torch.device,
    epochs: int,
    learning_rate: float,  # Single learning rate from args
    lambda_reg: float = 0.1,
    batch_size: int = 32,
    centroid_start_round: int = 5,
) -> Callable[[Context], FedPACClient]:
    def client_fn(context: Context) -> FedPACClient:
        cid_int = int(context.node_config["partition-id"])

        model = instantiate(model_cfg).to(device)
        train_dataset = trainloaders[cid_int].dataset
        val_dataset = valloaders[cid_int].dataset

        args = DictConfig({
            "epochs": epochs,
            "lr_f": learning_rate,  # Feature extractor LR
            "lr_g": learning_rate * 10,  # Classifier LR (10x higher)
            "lambda_reg": lambda_reg,
            "batch_size": batch_size,
            "centroid_start_round": centroid_start_round,
            "round": 0,
        })

        return FedPACClient(
            client_id=cid_int,
            train_dataset=train_dataset,
            test_dataset=val_dataset,
            model=model,
            args=args
        ).to_client()

    return client_fn


def gen_evaluate_fn(
    testloader: DataLoader,
    model_cfg: DictConfig,
    device: torch.device,
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]:
    def evaluate(server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]):
        model = instantiate(model_cfg).to(device)
        state_dict = OrderedDict({
            k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters_ndarrays)
        })
        model.load_state_dict(state_dict, strict=True)

        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                total_loss += criterion(preds, y).item() * y.size(0)
                total_correct += (preds.argmax(1) == y).sum().item()
                total_samples += y.size(0)

        loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        print(f"[FedPAC] Round {server_round} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
        return loss, {"test_accuracy": accuracy}

    return evaluate


def run_fedpac(
    data_config: DictConfig,
    model_cfg: DictConfig,
    backend_config: Dict[str, int],
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    batch_size: int = 32,
    lambda_reg: float = 0.1,
    centroid_start_round: int = 5,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: Optional[int] = None,
    min_evaluate_clients: Optional[int] = None,
    min_available_clients: Optional[int] = None,
) -> Dict[str, Any]:
    dataset = FederatedDataset(data_config, num_clients=num_clients)
    trainloaders, valloaders, testloader = dataset.get_dataloaders()

    client_fn = fedpac_gen_client_fn(
        trainloaders, valloaders, model_cfg, device,
        epochs=num_epochs, 
        learning_rate=learning_rate,  # Pass single learning_rate
        lambda_reg=lambda_reg, 
        batch_size=batch_size,
        centroid_start_round=centroid_start_round,
    )

    evaluate_fn = gen_evaluate_fn(testloader, model_cfg, device)

    strategy = FedPACStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients or num_clients,
        min_evaluate_clients=min_evaluate_clients or num_clients,
        min_available_clients=min_available_clients or num_clients,
        evaluate_fn=evaluate_fn,
    )

    net = instantiate(model_cfg).to(device)
    
    server = FedPACServer(
        client_manager=SimpleClientManager(),
        strategy=strategy,
        net=net,
    )

    return start_simulation(
        server=server,
        client_fn=client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        client_resources=backend_config,
    )