from typing import Dict, Optional, Tuple
from flwr.server.server import FitResultsAndFailures, Server, fit_clients
from flwr.common import Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.common.logger import log
from logging import INFO, DEBUG
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import Strategy
from omegaconf import DictConfig
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.common.logger import log
import torch
from typing import Callable, Dict, List, Optional, Tuple, Union
import logging
from logging import DEBUG, INFO, WARNING

class DashaServer(Server):
    """Implement server for DASHA."""

    def __init__(
        self,
        strategy: Strategy,
        net: torch.nn.Module,
        client_manager: Optional[ClientManager] = None,
    ):        
        if client_manager is None:
            client_manager = SimpleClientManager()
        super().__init__(client_manager=client_manager, strategy=strategy)
        
        # Initialize model parameters
        self.model_params = net
        model_ndarrays = [val.cpu().numpy() for val in self.model_params.state_dict().values()]
        self.parameters = ndarrays_to_parameters(model_ndarrays)

    def _get_initial_parameters(self, timeout: Optional[float], **kwargs) -> Parameters:
        """Return the initial parameters of the global model."""
        return self.parameters

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        log(INFO, f"Dasha FL Server: fit_round {server_round}")
        
        # Get clients and their respective instructions
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        
        if not client_instructions:
            log(INFO, "No clients selected for this round")
            return None
        
        # Collect training results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=str(server_round),  # Using the server round as the group_id
        )
        
        log(INFO, f"{len(results)} clients successfully completed, {len(failures)} failures")
        
        # Aggregate the training results
        aggregated_parameters, metrics_aggregated = self.strategy.aggregate_fit(
            server_round=server_round,
            results=results,
            failures=failures,
        )
        
        # Update global model parameters if new parameters are available
        if aggregated_parameters is not None:
            self.parameters = aggregated_parameters
        
        return aggregated_parameters, metrics_aggregated, (results, failures)