"""Server implementation for DepthFL."""

from typing import Dict, Optional, Tuple
from flwr.server.server import FitResultsAndFailures, Server, fit_clients
from flwr.common import Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.common.logger import log
from logging import INFO, DEBUG
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager, SimpleClientManager
import torch
from typing import Callable, Dict, List, Optional, Tuple, Union
import logging
from logging import DEBUG, INFO, WARNING


class DepthFLServer(Server):
    """Server implementation for DepthFL."""

    def __init__(
        self,
        strategy: Strategy,
        client_manager: Optional[ClientManager] = None,
    ):        
        if client_manager is None:
            client_manager = SimpleClientManager()
        super().__init__(client_manager=client_manager, strategy=strategy)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated learning."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=str(server_round)
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate results
        aggregated_result = self.strategy.aggregate_fit(
            server_round, results, failures
        )
        
        parameters_aggregated, metrics_aggregated = aggregated_result
        
        return parameters_aggregated, metrics_aggregated, (results, failures)


def get_evaluate_fn(
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
    model_cfg: dict,
) -> Callable:
    """Generate evaluation function for the server.
    
    Parameters
    ----------
    testloader : DataLoader
        The dataloader for the test dataset
    device : torch.device
        The device to run evaluation on
    model_cfg : dict
        The model configuration
        
    Returns
    -------
    Callable
        The evaluation function
    """
    from collections import OrderedDict
    
    def evaluate(
        server_round: int, parameters, config: Dict[str, Scalar]
    ):
        """Evaluate the model on the test dataset."""
        from fedml.algorithms.depthfl.model_utils import test_depthfl
        from fedml.models import MNISTModel, CNN
        
        try:
            # Determine which model to use based on dataset
            if model_cfg.get("dataset", "mnist") in ["mnist", "fmnist"]:
                net = MNISTModel(
                    input_dim=model_cfg.get("input_dim", 256),
                    hidden_dims=model_cfg.get("hidden_dims", [120, 84]),
                    num_classes=model_cfg.get("num_classes", 10)
                )
            else:  # cifar10 or other RGB datasets
                net = CNN(
                    input_dim=model_cfg.get("input_dim", 400),
                    hidden_dims=model_cfg.get("hidden_dims", [120, 84]),
                    num_classes=model_cfg.get("num_classes", 10)
                )
            
            # Load parameters
            parameters_ndarrays = parameters_to_ndarrays(parameters)
            params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=False)
            net.to(device)
            
            # Evaluate
            loss, accuracy = test_depthfl(net, testloader, device)
            return loss, {"test_accuracy": accuracy}
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, {"test_accuracy": 0.0}
    
    return evaluate