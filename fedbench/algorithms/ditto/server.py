from typing import Dict, Optional, Tuple
from flwr.server.server import FitResultsAndFailures, Server, fit_clients
from flwr.common import Parameters, Scalar
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.common.logger import log
from logging import DEBUG, INFO
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.typing import GetParametersIns
from flwr.server.strategy import Strategy
import torch

class DittoServer(Server):
    """Implement server for Ditto."""

    def __init__(
        self,
        strategy: Strategy,
        net: torch.nn.Module,
        client_manager: Optional[ClientManager] = None,
    ):
        if client_manager is None:
            client_manager = SimpleClientManager()
        super().__init__(client_manager=client_manager, strategy=strategy)
        
        # Initialize server model parameters
        self.model_params = net
        model_ndarrays = [val.cpu().numpy() for val in self.model_params.state_dict().values()]
        self.parameters = ndarrays_to_parameters(model_ndarrays)

    def _get_initial_parameters(self, timeout: Optional[float], **kwargs) -> Parameters:
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout, group_id="default_group")
        
        log(INFO, "Received initial parameters from one random client")
        return get_parameters_res.parameters

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated learning."""
        
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

        # Aggregate training results
        aggregated_result: Tuple[Optional[Parameters], Dict[str, Scalar]] = (
            self.strategy.aggregate_fit(server_round, results, failures)
        )

        if aggregated_result[0] is not None:
            self.parameters = aggregated_result[0]  # Update server parameters

        metrics_aggregated = aggregated_result[1]
        return self.parameters, metrics_aggregated, (results, failures)
