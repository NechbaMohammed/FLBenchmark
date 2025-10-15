from typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.common import Scalar, Parameters
import flwr as fl
import logging
from logging import DEBUG, INFO, WARNING
from flwr.common.logger import log
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.typing import FitRes
from flwr.server.strategy.aggregate import aggregate
from flwr.server.strategy import FedAvg

class DittoStrategy(FedAvg):
    """Custom strategy for Ditto, extending FedAvg."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate only the global models from client updates."""

        if not results:
            return None, {}
        # If there are failures and we do not accept failures, abort aggregation
        if not self.accept_failures and failures:
            return None, {}

        # Extract parameters (both global and personal) from all client results
        combined_parameters_all = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]

        len_combined_parameter = len(combined_parameters_all[0])
        num_examples_all = [fit_res.num_examples for _, fit_res in results]

        # In Ditto, we assume parameters are concatenated: [global_parameters] + [personal_parameters]
        # We aggregate only the global parts.
        global_updates = [
            (params[: len_combined_parameter], num_examples)
            for params, num_examples in zip(combined_parameters_all, num_examples_all)
        ]

        # Aggregate global parameters
        aggregated_global_parameters = aggregate(global_updates)

        # No aggregation of personal parameters at server (they stay local).

        # Aggregate custom metrics if a fit_metrics_aggregation_fn is provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Return only the updated global parameters
        return (
            ndarrays_to_parameters(aggregated_global_parameters),
            metrics_aggregated
        )
