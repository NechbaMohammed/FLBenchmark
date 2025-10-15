from typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.common import Scalar, Parameters, FitRes
import flwr as fl
from flwr.common.logger import log
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
import numpy as np
from logging import DEBUG, INFO, WARNING

class FedRepStrategy(FedAvg):
    """Implement custom strategy for FedRep based on FedAvg class, with SCAFFOLD structure."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results for representation parameters and control variates using weighted average.

        Parameters
        ----------
        server_round : int
            The current server round.
        results : List[Tuple[ClientProxy, FitRes]]
            List of client results containing representation parameter updates and control variate updates.
        failures : List[Union[Tuple[ClientProxy, FitRes], BaseException]]
            List of client failures, if any.

        Returns
        -------
        Tuple[Optional[Parameters], Dict[str, Scalar]]
            Aggregated representation parameters and control variates, along with aggregated metrics.
        """
        if not results:
            log(WARNING, "No results to aggregate in round %s", server_round)
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            log(WARNING, "Failures detected and not accepted in round %s", server_round)
            return None, {}

        # Convert client results to ndarrays
        combined_parameters_all_updates = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        len_combined_parameter = len(combined_parameters_all_updates[0])
        num_examples_all_updates = [fit_res.num_examples for _, fit_res in results]

        # Aggregate representation parameter updates
        weights_results = [
            (update[: len_combined_parameter // 2], num_examples)
            for update, num_examples in zip(
                combined_parameters_all_updates, num_examples_all_updates
            )
        ]
        parameters_aggregated = aggregate(weights_results)

        # Aggregate control variate updates for representation parameters
        client_cv_updates_and_num_examples = [
            (update[len_combined_parameter // 2 :], num_examples)
            for update, num_examples in zip(
                combined_parameters_all_updates, num_examples_all_updates
            )
        ]
        aggregated_cv_update = aggregate(client_cv_updates_and_num_examples)

        # Aggregate custom metrics if aggregation function is provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Combine aggregated representation parameters and control variates
        return (
            ndarrays_to_parameters(parameters_aggregated + aggregated_cv_update),
            metrics_aggregated
        )