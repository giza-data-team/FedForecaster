# Import aggregator
import time

from server_utils.aggregators.create_aggregator import CreateAggregator
from flwr.server.strategy.aggregate import (
    weighted_loss_avg,
)

import numpy as np
from datetime import datetime, timedelta
import json
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
import flwr as fl
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from .save_results import SaveResults
from .meta_model.meta_model import MetaModel
from .meta_model.new_model.pycaret_metamodel import MetaModelPyCaret
from .meta_model.model_optimizer import ModelOptimizer
WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


class CustomStrategy(Strategy):
    """Federated Averaging strategy.

    Implementation based on https://arxiv.org/abs/1602.05629

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    inplace : bool (default: True)
        Enable (True) or disable (False) in-place aggregation of model updates.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
            self,
            round_number,
            *,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            inplace: bool = True,
            dataset_name,
            n_clients,
            time_budget = 10
    ) -> None:
        super().__init__()

        if (
                min_fit_clients > min_available_clients
                or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace
        self.aggregator = CreateAggregator()
        self.num_client_available = 0
        self.weights = {}
        self.best_round = -1
        self.best_loss = 9999999999
        self.max_round = round_number
        self.best_model = ""
        self.best_parameters = []
        self.loss_flag = 0
        self.index_flag = 0
        self.start_time = 0
        self.n_clients = n_clients
        self.save_result = SaveResults(n_clients=n_clients)
        self.dataset_name = dataset_name
        self.train_loss = -1
        self.selected_models = []
        self.models_probs = []
        self.current_params = {}
        self.model_index = 0
        self.time_budget = time_budget
        self.model_optimizer = ModelOptimizer()
        # self.meta_model = MetaModelPyCaret(prob_threshold=None, top_n=3, model_path='server_utils/meta_model/new_model/final_model.pkl',
        #                    encoder_path='server_utils/meta_model/new_model/label_encoder.pkl')
        self.meta_model = MetaModel(prob_threshold=None, top_n=3, model_path='server_utils/meta_model/new_model/final_model.pkl',
                           encoder_path='server_utils/meta_model/new_model/label_encoder.pkl')

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"CustomAgg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        import time

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        num_clients_connected = client_manager.num_available()
        while num_clients_connected < min_num_clients:
            num_clients_connected = client_manager.num_available()
            if num_clients_connected == min_num_clients:
                time.sleep(20)
        config = {}
        if self.num_client_available < client_manager.num_available():
            server_round = 1
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        self.num_client_available = num_clients_connected
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0 or server_round < 4:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        agg_parameters = fl.common.Parameters(tensors=[], tensor_type="dict")
        client, fit_res = results[0]
        metrics = fit_res.metrics
        aggregator = self.aggregator.createAggregator(
            server_round=server_round, metrics=metrics
        )
        if aggregator:
            if server_round >= 4:
                agg_parameters = aggregator.aggregate(results)
                self.weights[server_round] = agg_parameters
                encoded_model_name = self.selected_models[self.model_index].encode('utf-8')
                encoded_model_parameters = json.dumps(self.current_params).encode('utf-8')
                tensors = [encoded_model_name,encoded_model_parameters]
                tensors.extend(agg_parameters)
                tensor_type = self.selected_models[self.model_index]
            else:
                agg_features = aggregator.aggregate_keys(results=results)
                agg_size = aggregator.aggregate_size(results=results)
                agg_parameters = aggregator.aggregate(agg_features, agg_size)
                agg_parameters["server_round"] = server_round + 1
                agg_features = json.dumps(agg_parameters).encode("utf-8")
                tensors = [agg_features]
                tensor_type = "features_weights"

        if server_round == 3:
            self.start_time = datetime.now()
            features = {}
            features['num_clients'] = self.min_fit_clients
            for k,v in agg_parameters['aggregated features']['meta_features'].items():
                features[k] = v
            self.selected_models, self.models_probs= self.meta_model.predict_best_model(input_features=features)
            self.current_params = self.model_optimizer.initialize_params(self.selected_models[self.model_index])
            encoded_model_name = self.selected_models[self.model_index].encode('utf-8')
            encoded_model_parameters = json.dumps(self.current_params).encode('utf-8')
            tensors = [encoded_model_name,encoded_model_parameters]
            tensor_type = self.selected_models[self.model_index]
        print("end agg fit inside server")
        agg_parameters = fl.common.Parameters(
                tensors=tensors, tensor_type=tensor_type
        )
        return agg_parameters, {}

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
        if self.best_loss > loss_aggregated:
            self.best_model = self.selected_models[self.model_index]
            self.best_parameters = self.current_params
            self.best_loss = loss_aggregated
            self.best_round = server_round
            self.loss_flag = 0
            train_loss = []
            for _, loss in eval_metrics:
                train_loss.append(loss['train_loss'])
            self.train_loss = np.mean(train_loss)

        self.current_params = self.model_optimizer.optimize_hyperparameters(model_name=self.selected_models[self.model_index],
                                                                            prev_params=self.current_params,
                                                                            avg_loss=loss_aggregated)
        time_now = datetime.now()
        time_difference = abs(time_now - self.start_time)
        time_difference_in_seconds = time_difference.total_seconds()
        time_budget_in_seconds = self.time_budget * 60
        if (time_difference_in_seconds/time_budget_in_seconds) >= np.sum(self.models_probs[:self.model_index+1]):
            self.model_index = self.model_index+1

            if self.model_index >= len(self.selected_models):

                self.save_result.save(
                    num_clients=self.min_fit_clients,
                    dataset_name=self.dataset_name,
                    test_loss=self.best_loss,
                    train_loss=self.train_loss,
                    time_taken=time_difference,
                    model=self.best_model,
                    parameters=self.best_parameters,
                    models = self.selected_models
                )
                raise Exception("time budget done")
            else:
                self.current_params = self.model_optimizer.initialize_params(self.selected_models[self.model_index])
        return loss_aggregated, metrics_aggregated
