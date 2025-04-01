
from typing import Dict, Optional, Tuple
from flwr.server.server import FitResultsAndFailures, Server, fit_clients
from flwr.common import Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.common.logger import log
from logging import INFO, DEBUG
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common import Parameters, Scalar
from flwr.common.typing import GetParametersIns
from flwr.server.strategy import Strategy
from omegaconf import DictConfig
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.common.logger import log
import torch
from typing import Callable, Dict, List, Optional, Tuple, Union
import logging
from logging import DEBUG, INFO,WARNING

class ScaffoldServer(Server):
    """Implement server for SCAFFOLD."""

    def __init__(
        self,
        strategy: Strategy,
        net: torch.nn.Module,
        client_manager: Optional[ClientManager] = None,
    ):        
        if client_manager is None:
            client_manager = SimpleClientManager()
        super().__init__(client_manager=client_manager, strategy=strategy)
        
        # Initialize server_cv to zeros
        self.model_params = net
        model_ndarrays = [val.cpu().numpy() for val in self.model_params.state_dict().values()]
        self.parameters = ndarrays_to_parameters(model_ndarrays)
        # Initialize server_cv to zeros
        self.server_cv = [
            torch.zeros_like(torch.Tensor(param)) 
            for param in model_ndarrays
        ]

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
        # Server_cv is already initialized to zeros; no need to overwrite
        return get_parameters_res.parameters

    # pylint: disable=too-many-locals
    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strateg
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=update_parameters_with_cv(self.parameters, self.server_cv),
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
            group_id = str(server_round)
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
        aggregated_result_arrays_combined = []
        if aggregated_result[0] is not None:
            aggregated_result_arrays_combined = parameters_to_ndarrays(
                aggregated_result[0]
            )
        aggregated_parameters = aggregated_result_arrays_combined[
            : len(aggregated_result_arrays_combined) // 2
        ]
        aggregated_cv_update = aggregated_result_arrays_combined[
            len(aggregated_result_arrays_combined) // 2 :
        ]
        # Check if lengths match before updating server_cv
        if len(self.server_cv) != len(aggregated_cv_update):
            log(logging.ERROR, "Mismatch in lengths of server_cv and aggregated_cv_update")
            return None

        # convert server cv into ndarrays
        server_cv_np = [cv.numpy() for cv in self.server_cv]
        # update server cv
        total_clients = len(self._client_manager.all())
        cv_multiplier = len(results) / total_clients
        self.server_cv = [
            torch.from_numpy(cv + cv_multiplier * aggregated_cv_update[i])
            for i, cv in enumerate(server_cv_np)
        ]

        # update parameters x = x + 1* aggregated_update
        curr_params = parameters_to_ndarrays(self.parameters)
        updated_params = [
            x + aggregated_parameters[i] for i, x in enumerate(curr_params)
        ]
        parameters_updated = ndarrays_to_parameters(updated_params)

        # metrics
        metrics_aggregated = aggregated_result[1]
        return parameters_updated, metrics_aggregated, (results, failures)


def update_parameters_with_cv(
    parameters: Parameters, s_cv: List[torch.Tensor]
) -> Parameters:
    """Append server control variates to model parameters."""
    parameters_np = parameters_to_ndarrays(parameters)
    cv_np = [cv.numpy() for cv in s_cv]
    parameters_np.extend(cv_np)  # Now parameters include both model weights and server_cv
    return ndarrays_to_parameters(parameters_np)