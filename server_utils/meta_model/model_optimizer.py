import time

import numpy as np
from sklearn.linear_model import Lasso, ElasticNetCV, HuberRegressor, QuantileRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
class ModelOptimizer:
    def __init__(self):
        self.optimizer = None
        self.models = {
            "LASSO": (
                Lasso(random_state=42),
                [
                    Real(1e-5, 2, prior='log-uniform', name='alpha'),
                    Categorical(["cyclic", "random"], name='selection'),
                    # Integer(42,42,name="random_state")
                ]
            ),
            "LinearSVR": (
                LinearSVR(random_state=42),
                [
                    Integer(1, 10, name='C'),
                    Real(0.01, 0.1, name='epsilon'),
                    # Integer(42,42,name="random_state")
                ]
            ),
            "ELASTICNETCV": (
                ElasticNetCV(random_state=42),
                [
                    Real(0.3, 1, name='l1_ratio'),
                    Categorical(["cyclic", "random"], name='selection'),
                    # Integer(42,42,name="random_state")
                ]
            ),
            "XGBRegressor": (
                XGBRegressor(random_state=42),
                [
                    Integer(5, 10, name='n_estimators'),
                    Integer(2, 7, name='max_depth'),
                    Real(0.1, 1, name='learning_rate'),
                    Real(0.8, 10, name='reg_lambda'),
                    Real(0.1, 1, name='subsample'),
                    # Integer(42,42,name="random_state")
                ]
            ),
            'HUBERREGRESSOR': (
                HuberRegressor(),
                [
                    Real(1.0, 1.5, name='epsilon'),
                    Real(1e-3, 1e2, prior='log-uniform', name='alpha'),
                    # Integer(42,42,name="random_state")
                ]
            ),
            'QUANTILEREGRESSOR': (
                QuantileRegressor(),
                [
                    Real(1e-3, 1e2, prior='log-uniform', name='alpha'),
                    Real(0.1, 0.75, name='quantile'),
                    # Integer(42,42,name="random_state")
                ]
            )
        }
        self.model_params = {}
        self.client_losses = {}

    def encode_params(self, params):
        """Encode categorical parameters as integers or provide bounds for continuous parameters."""
        encoded_params = {}
        encoding_map = {}

        for k, v in params.items():
            if isinstance(v, list) or isinstance(v, np.ndarray):  # Categorical or continuous
                if isinstance(v[0], (int, float)):
                    encoded_params[k] = (min(v), max(v))  # Continuous range
                else:
                    encoded_params[k] = (0, len(v) - 1)  # Encode as integer range
                    encoding_map[k] = v  # Save the original values for decoding
            else:
                encoded_params[k] = v  # Single continuous value

        return encoded_params, encoding_map

    def decode_params(self, best_params, encoding_map):
        """Decode integer-encoded parameters back to their original categorical values."""
        decoded_params = {}

        for k, v in best_params.items():
            if k in encoding_map:  # If the parameter was categorical
                decoded_params[k] = encoding_map[k][int(round(v))]  # Decode back to original
            else:
                decoded_params[k] = int(v) if k in ['n_estimators', 'max_depth', 'random_state'] else v  # Cast 'random_state' back to int if necessary
            decoded_params['random_state'] = 42
        return decoded_params

    def initialize_params(self, model_name):
        if model_name in self.models:
            model, param_space = self.models[model_name]
            default_params = {}
            for param in param_space:
                if isinstance(param, Real):
                    default_value = (param.low + param.high) / 2
                elif isinstance(param, Integer):
                    default_value = int((param.low + param.high) / 2)
                elif isinstance(param, Categorical):
                    default_value = param.categories[0]
                default_params[param.name] = default_value

            self.optimizer = BayesianOptimizer(param_space)
            self.model_params[model_name] = default_params
            return default_params
        else:
            raise ValueError(f"Model {model_name} is not defined.")

    def optimize_hyperparameters(self, model_name, prev_params, avg_loss):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not supported.")

        model, params = self.models[model_name]
        if "random_state" in prev_params:
            prev_params.pop("random_state")

        self.optimizer.update(prev_params, avg_loss)
        next_params = self.optimizer.suggest()

        # Create a dictionary of the new parameters
        new_params = {params[i].name: next_params[i] for i in range(len(params))}

        # Add random_state if necessary
        if model_name not in ['QUANTILEREGRESSOR','HUBERREGRESSOR']:
            new_params['random_state'] = 42
        # Ensure conversion of numpy types to standard Python types
        new_params = convert_numpy_types(new_params)

        return new_params

def convert_numpy_types(data):
    if isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(v) for v in data]
    elif isinstance(data, np.generic):
        return data.item()
    else:
        return data

class BayesianOptimizer:
    def __init__(self, space):
        self.optimizer = Optimizer(
            dimensions=space,
            base_estimator='GP',
            acq_func='EI',
            acq_optimizer='auto',
            n_initial_points=20,
            acq_func_kwargs={'xi': 0.001, 'kappa': 5},
            random_state=42
        )
        self.X = []
        self.y = []
        self.iteration = 0

    def suggest(self):
        return self.optimizer.ask()

    def update(self, params, score):
        self.X.append(params)
        self.y.append(score)
        self.optimizer.tell(list(params.values()), score)
        self.iteration += 1

    def get_best(self):
        return self.optimizer.get_result().x
