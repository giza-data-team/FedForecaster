import time
import numpy as np
from sklearn.linear_model import Lasso, ElasticNetCV, HuberRegressor, QuantileRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
from smac.configspace import ConfigurationSpace
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter


class ModelOptimizer:
    def __init__(self):
        self.optimizer = None
        self.models = {
            "LASSO": (
                Lasso(random_state=42),
                [
                    UniformFloatHyperparameter('alpha', 1e-5, 2, log=True),
                    CategoricalHyperparameter('selection', ["cyclic", "random"]),
                ]
            ),
            "LinearSVR": (
                LinearSVR(random_state=42),
                [
                    UniformIntegerHyperparameter('C', 1, 10),
                    UniformFloatHyperparameter('epsilon', 0.01, 0.1),
                ]
            ),
            "ELASTICNETCV": (
                ElasticNetCV(random_state=42),
                [
                    UniformFloatHyperparameter('l1_ratio', 0.3, 1),
                    CategoricalHyperparameter('selection', ["cyclic", "random"]),
                ]
            ),
            "XGBRegressor": (
                XGBRegressor(random_state=42),
                [
                    UniformIntegerHyperparameter('n_estimators', 5, 10),
                    UniformIntegerHyperparameter('max_depth', 2, 7),
                    UniformFloatHyperparameter('learning_rate', 0.1, 1),
                    UniformFloatHyperparameter('reg_lambda', 0.8, 10),
                    UniformFloatHyperparameter('subsample', 0.1, 1),
                ]
            ),
            'HUBERREGRESSOR': (
                HuberRegressor(),
                [
                    UniformFloatHyperparameter('epsilon', 1.0, 1.5),
                    UniformFloatHyperparameter('alpha', 1e-3, 1e2, log=True),
                ]
            ),
            'QUANTILEREGRESSOR': (
                QuantileRegressor(),
                [
                    UniformFloatHyperparameter('alpha', 1e-3, 1e2, log=True),
                    UniformFloatHyperparameter('quantile', 0.1, 0.75),
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

            # Initialize SMACOptimizer
            self.optimizer = SMACOptimizer(param_space, model_name)
            default_params = self.optimizer.get_initial_params()

            self.model_params[model_name] = default_params
            return default_params
        else:
            raise ValueError(f"Model {model_name} is not defined.")

    def optimize_hyperparameters(self, model_name, prev_params, avg_loss):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not supported.")

        model, _ = self.models[model_name]
        if "random_state" in prev_params:
            prev_params.pop("random_state")

        self.optimizer.update(prev_params, avg_loss)
        next_params = self.optimizer.suggest()

        # Create a dictionary of the new parameters
        new_params = {key: next_params[key] for key in prev_params}

        # Add random_state if necessary
        if model_name not in ['QUANTILEREGRESSOR', 'HUBERREGRESSOR']:
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


class SMACOptimizer:
    def __init__(self, param_space, model_name):
        self.param_space = param_space
        self.model_name = model_name
        self.scenario = self.create_scenario()
        self.config_space = self.create_config_space()
        self.smac = None
        self.X = []  # To store configurations
        self.y = []  # To store losses

    def create_scenario(self):
        return Scenario({
            "run_obj": "quality",  # We optimize for quality (minimize loss)
            "runcount-limit": 50,  # Maximum number of function evaluations
            "cs": self.create_config_space(),  # Configuration space
            "deterministic": "true",  # The optimization process is deterministic
            "output_dir": f"smac_output_{self.model_name}"  # Optional: specify output directory
        })

    def create_config_space(self):
        cs = ConfigurationSpace()

        for param in self.param_space:
            if isinstance(param, UniformFloatHyperparameter):
                cs.add_hyperparameter(UniformFloatHyperparameter(param.name, param.lower, param.upper))
            elif isinstance(param, UniformIntegerHyperparameter):
                cs.add_hyperparameter(UniformIntegerHyperparameter(param.name, param.lower, param.upper))
            elif isinstance(param, CategoricalHyperparameter):
                cs.add_hyperparameter(CategoricalHyperparameter(param.name, param.choices))

        return cs

    def get_initial_params(self):
        """Get the initial default parameters."""
        initial_params = {}
        for param in self.param_space:
            if isinstance(param, UniformFloatHyperparameter):
                initial_params[param.name] = (param.lower + param.upper) / 2
            elif isinstance(param, UniformIntegerHyperparameter):
                initial_params[param.name] = int((param.lower + param.upper) / 2)
            elif isinstance(param, CategoricalHyperparameter):
                initial_params[param.name] = param.choices[0]
        return initial_params

    def suggest(self):
        """Return the next suggestion."""
        if not self.smac:
            # Initialize SMAC if it's the first call
            self.smac = SMAC(scenario=self.scenario, tae_runner=self.objective_function)
        next_config = self.smac.suggest_configuration()
        return next_config.get_dictionary()

    def update(self, params, score):
        """Update SMAC with the previous result."""
        self.X.append(params)
        self.y.append(score)
        self.smac.update_config_space(list(params.values()), score)

    def objective_function(self, config):
        """Placeholder function for SMAC. You will need to define a real evaluation function."""
        pass  # This should be defined when evaluating the model


# Example Usage
optimizer = ModelOptimizer()
default_params = optimizer.initialize_params("LASSO")

# Simulate the training loop:
for i in range(10):
    # Evaluate the model and calculate average loss
    avg_loss = np.random.random()  # Replace with actual loss
    next_params = optimizer.optimize_hyperparameters("LASSO", default_params, avg_loss)
    print(f"Round {i + 1} suggested params: {next_params}")
    default_params = next_params
