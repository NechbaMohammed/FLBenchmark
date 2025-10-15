from typing import Dict, Optional, Tuple, List
from flwr.server.server import Server, FitResultsAndFailures, fit_clients
from flwr.common import Parameters, Scalar
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.common.logger import log
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.typing import GetParametersIns
from logging import INFO, DEBUG, ERROR
import torch
import numpy as np
from flwr.server.strategy import Strategy

class FedRepServer(Server):
    """Implement server for FedRep with SCAFFOLD structure."""

    def __init__(
        self,
        strategy: Strategy,
        net: torch.nn.Module,
        rep_parameters: List[torch.nn.Parameter],
        client_manager: Optional[ClientManager] = None,
    ):        
        """
        Initialize the FedRep server.

        Parameters
        ----------
        strategy : Strategy
            The federated learning strategy.
        net : torch.nn.Module
            The neural network (used to initialize representation parameters).
        rep_parameters : List[torch.nn.Parameter]
            Parameters of the representation part of the network.
        client_manager : Optional[ClientManager]
            The client manager (defaults to SimpleClientManager if None).
        """
        if client_manager is None:
            client_manager = SimpleClientManager()
        super().__init__(client_manager=client_manager, strategy=strategy)
        
        # Initialize server with representation parameters only
        self.model_params = net
        state_dict = self.model_params.state_dict()
        rep_keys = [k for k, v in state_dict.items() if any(p is v for p in rep_parameters)]
        model_ndarrays = [state_dict[k].cpu().numpy() for k in rep_keys]
        self.parameters = ndarrays_to_parameters(model_ndarrays)
        # Initialize server control variates to zeros for representation parameters
        self.server_cv = [
            torch.zeros_like(torch.Tensor(param)) 
            for param in model_ndarrays
        ]

    def _get_initial_parameters(self, timeout: Optional[float], **kwargs) -> Parameters:
        """Get initial representation parameters from strategy or a random client."""
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters
        
        log(INFO, "Requesting initial representation parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout, group_id="default_group")
        
        log(INFO, "Received initial representation parameters from one random client")
        return get_parameters_res.parameters

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging for FedRep."""
        # Configure clients with representation parameters and server control variates
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

        # Collect `fit` results from all clients
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
            log(ERROR, "Mismatch in lengths of server_cv and aggregated_cv_update")
            return None

        # Convert server control variates to ndarrays and update
        server_cv_np = [cv.numpy() for cv in self.server_cv]
        total_clients = len(self._client_manager.all())
        cv_multiplier = len(results) / total_clients
        self.server_cv = [
            torch.from_numpy(cv + cv_multiplier * aggregated_cv_update[i])
            for i, cv in enumerate(server_cv_np)
        ]

        # Update representation parameters: x = x + aggregated_parameters
        curr_params = parameters_to_ndarrays(self.parameters)
        updated_params = [
            x + aggregated_parameters[i] for i, x in enumerate(curr_params)
        ]
        parameters_updated = ndarrays_to_parameters(updated_params)

        # Metrics
        metrics_aggregated = aggregated_result[1]
        return parameters_updated, metrics_aggregated, (results, failures)

def update_parameters_with_cv(
    parameters: Parameters, s_cv: List[torch.Tensor]
) -> Parameters:
    """Append server control variates to representation parameters."""
    parameters_np = parameters_to_ndarrays(parameters)
    cv_np = [cv.numpy() for cv in s_cv]
    parameters_np.extend(cv_np)  # Append server control variates to parameters
    return ndarrays_to_parameters(parameters_np)