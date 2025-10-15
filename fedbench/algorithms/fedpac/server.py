from typing import Dict, Optional, Tuple, List
from flwr.server.server import FitResultsAndFailures, Server, fit_clients
from flwr.common import Parameters, Scalar
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.common.typing import GetParametersIns
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import Strategy
from flwr.common.logger import log
from logging import INFO, DEBUG
import torch
import numpy as np


class FedPACServer(Server):
    """FedPAC Server with centroid aggregation and round tracking."""

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
        self.model = net
        model_ndarrays = [val.cpu().numpy() for val in self.model.state_dict().values()]
        self.parameters = ndarrays_to_parameters(model_ndarrays)

        # Initialize empty global centroids
        self.global_centroids: Dict[int, torch.Tensor] = {}

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        # Get client fit instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, f"fit_round {server_round}: no clients selected")
            return None

        # Prepare config with round number and centroids
        config = {"server_round": server_round}
        
        if self.global_centroids:
            # Convert centroids to list format for transmission
            config["global_centroids"] = {
                str(k): v.cpu().numpy().tolist() for k, v in self.global_centroids.items()
            }
            log(INFO, f"Sending {len(self.global_centroids)} global centroids to clients")

        # Update config for each client
        for client, fit_ins in client_instructions:
            fit_ins.config.update(config)

        # Fit clients
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=str(server_round)
        )

        log(DEBUG, f"fit_round {server_round} received {len(results)} results and {len(failures)} failures")

        # Aggregate model updates
        aggregated_result: Tuple[Optional[Parameters], Dict[str, Scalar]] = (
            self.strategy.aggregate_fit(server_round, results, failures)
        )

        if aggregated_result[0] is not None:
            self.parameters = aggregated_result[0]

        # Aggregate centroids after model aggregation
        client_stats = self._extract_client_centroid_stats(results)
        if client_stats:
            self.aggregate_centroids(client_stats)
            log(INFO, f"Aggregated centroids for {len(self.global_centroids)} classes")

        return self.parameters, aggregated_result[1], (results, failures)

    def _extract_client_centroid_stats(self, results) -> List[Dict[str, Dict]]:
        """Extract centroids and sample counts from clients."""
        all_stats = []

        for client_proxy, fit_res in results:
            metrics = fit_res.metrics
            
            # Extract centroids and sample counts from flat metrics structure
            centroids = {}
            sample_counts = {}
            
            for key, value in metrics.items():
                if key.startswith("centroid_"):
                    class_id = int(key.replace("centroid_", ""))
                    # Parse the string representation of the centroid
                    import ast
                    centroid_list = ast.literal_eval(value)
                    centroids[class_id] = torch.tensor(centroid_list)
                elif key.startswith("sample_count_"):
                    class_id = int(key.replace("sample_count_", ""))
                    sample_counts[class_id] = int(value)
            
            if centroids and sample_counts:
                all_stats.append({
                    "centroids": centroids,
                    "sample_counts": sample_counts,
                })
            else:
                log(DEBUG, f"Client did not return centroid stats")

        return all_stats

    def aggregate_centroids(self, client_stats: List[Dict[str, Dict]]):
        """Aggregate local centroids into global centroids using weighted average."""
        weighted_sums = {}
        total_counts = {}

        for stats in client_stats:
            centroids = stats["centroids"]
            counts = stats["sample_counts"]
            
            for cls, centroid in centroids.items():
                if cls not in weighted_sums:
                    weighted_sums[cls] = torch.zeros_like(centroid)
                    total_counts[cls] = 0
                
                # Weighted sum of centroids
                weighted_sums[cls] += centroid * counts[cls]
                total_counts[cls] += counts[cls]

        # Compute weighted average
        self.global_centroids = {
            cls: weighted_sums[cls] / total_counts[cls]
            for cls in weighted_sums if total_counts[cls] > 0
        }
        
        log(INFO, f"Global centroids updated for classes: {list(self.global_centroids.keys())}")

