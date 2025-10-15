from typing import Dict, List, Optional, Tuple, Union

from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, Scalar
from flwr.common.typing import FitRes, FitIns
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate

from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.logger import log
from logging import WARNING


class FedPACStrategy(FedAvg):
    """Federated Prototypical Alignment Clustering (FedPAC) strategy using FedAvg base."""

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients for this round
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        # Create fit instructions with empty config (will be updated in server)
        fit_ins = FitIns(parameters, {})

        # Return client/instruction pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates from clients."""

        if not results:
            log(WARNING, "[Server] No results received for aggregation.")
            return None, {}

        if not self.accept_failures and failures:
            log(WARNING, "[Server] Aggregation failed due to client errors.")
            return None, {}

        # Extract weights and number of examples
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Weighted average of client weights
        aggregated_parameters = aggregate(weights_results)
        aggregated_flwr_parameters = ndarrays_to_parameters(aggregated_parameters)

        # Aggregate metrics
        total_examples = sum(res.num_examples for _, res in results)
        metrics = {}

        try:
            if "accuracy" in results[0][1].metrics:
                avg_accuracy = sum(res.metrics["accuracy"] * res.num_examples for _, res in results) / total_examples
                metrics["accuracy"] = avg_accuracy
                
            if "loss" in results[0][1].metrics:
                avg_loss = sum(res.metrics["loss"] * res.num_examples for _, res in results) / total_examples
                metrics["loss"] = avg_loss
                
            if metrics:
                log(WARNING, f"[Server][Round {server_round}] Aggregated - Accuracy: {metrics.get('accuracy', 'N/A'):.4f}, Loss: {metrics.get('loss', 'N/A'):.4f}")
        except (KeyError, IndexError) as e:
            log(WARNING, f"[Server] Metrics missing from some clients: {e}")

        return aggregated_flwr_parameters, metrics
