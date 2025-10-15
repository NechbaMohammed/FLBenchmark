from typing import Dict, List, Optional, Tuple, Union, Callable
from flwr.server.client_proxy import ClientProxy
from flwr.common import Scalar
import torch
from torch.utils.data import DataLoader
import flwr as fl
from collections import OrderedDict
import logging
from logging import DEBUG, INFO, WARNING
from flwr.common.logger import log
from functools import reduce
import numpy as np
import os
from flwr.common import Parameters, Scalar
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.typing import FitRes
from flwr.server.strategy.aggregate import aggregate
from flwr.server.strategy import FedAvg

class DashaStrategy(FedAvg):
    """Implement custom strategy for DASHA based on FedAvg class.
    
    DASHA (Distributed Nonconvex Optimization with Communication Compression and Optimal Oracle Complexity)
    uses random sparsification for communication compression while maintaining convergence guarantees.
    """

    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[[int, Dict[str, List[np.ndarray]]], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        fit_metrics_aggregation_fn: Optional[
            Callable[[List[Tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]
        ] = None,
        evaluate_metrics_aggregation_fn: Optional[
            Callable[[List[Tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]
        ] = None,
        compressor_coordinates: int = 10,
        probability_q: float = 0.5,
        eta: float = 0.1  # Learning rate for server aggregation
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.compressor_coordinates = compressor_coordinates
        self.probability_q = probability_q
        self.eta = eta
        self.global_variance_estimate = None  # Will store the global variance estimate

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Configure the next round of training with DASHA-specific parameters."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        
        # Add DASHA-specific parameters to config
        config["compressor_coordinates"] = self.compressor_coordinates
        config["probability_q"] = self.probability_q
        config["server_round"] = server_round
        config["eta"] = self.eta
        
        # Sample clients 
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        #fit instructions
        fit_ins = fl.common.FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate fit results using weighted average with DASHA compression.
        DASHA applies compressed model updates for improved communication efficiency.
        """
        if not results:
            return None, {}
        
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Extract weights from results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Aggregate parameters using weighted average
        parameters_aggregated = aggregate(weights_results)

        # Process variance reduction metrics from clients
        # Note: We no longer receive the variance reduction direction directly as numpy arrays
        total_examples = 0
        total_vr_norm = 0.0
        
        for client_proxy, fit_res in results:
            if "vr_norm" in fit_res.metrics and "vr_used" in fit_res.metrics:
                if fit_res.metrics["vr_used"]:
                    client_vr_norm = fit_res.metrics["vr_norm"]
                    num_examples = fit_res.num_examples
                    total_vr_norm += client_vr_norm * (num_examples / sum(r.num_examples for _, r in results))
                    total_examples += num_examples
        
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {
            "vr_norm_avg": total_vr_norm,
            "clients": len(results)
        }
        
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            additional_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
            metrics_aggregated.update(additional_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Log DASHA-specific information
        log(INFO, f"DASHA Round {server_round}: Aggregated parameters from {len(results)} clients")
        log(INFO, f"Using compressor with {self.compressor_coordinates} coordinates")
        log(INFO, f"Probability of full gradient: {self.probability_q}")
        
        # Convert parameters back to bytes and return
        return ndarrays_to_parameters(parameters_aggregated), metrics_aggregated