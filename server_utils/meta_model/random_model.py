import numpy as np
import random
from sklearn.linear_model import Lasso, ElasticNetCV, HuberRegressor, QuantileRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor


models = {
        'LASSO': (
            Lasso(random_state=42),
            {'alpha': np.logspace(np.log10(1e-5), np.log10(2), num=30), 'selection': ['cyclic', 'random']}
        ),
        'SVR': (
            LinearSVR(random_state=42),
            {'C': [1, 2, 3, 5, 10], 'epsilon': [0.01, 0.05, 0.1]}
        ),
        'ELASTICNETCV': (
            ElasticNetCV(random_state=42),
            {'l1_ratio': np.linspace(0.3, 1, 10), 'selection': ['cyclic', 'random']}
        ),
        'XGBOOST_REGRESSOR': (
            XGBRegressor(random_state=42),
            {
                'learning_rate': [0.1, 1],
                'reg_lambda': [0.8, 10],
                'gamma': [0.9, 1.16467595, 2.248149123539492, 3.9963209507789],
                'subsample': [0.1, 1]
            }
        ),
        'HUBERREGRESSOR' : (
            HuberRegressor(),
            {
                "epsilon": [1.0, 1.35, 1.5],
                "alpha": np.logspace(np.log10(1e-3), np.log10(1e2), num=10),
            }
        ),
        'QUANTILEREGRESSOR' : (
            QuantileRegressor(),
            {
                "alpha": np.logspace(np.log10(1e-3), np.log10(1e2), num=10),
                "quantile": [0.1, 0.25, 0.5, 0.75],
            },
        )
    }

class RandomModel:

    def __init__(self):
        self.model = ""
        self.model_name = ""
        self.hyper_parameters = []
        self.model_params_list = {}
        random.seed(42)
    # Define models and their hyperparameters


    def set_model(self):
        # Randomly select a model and its hyperparameters
        selected_model_name = random.choice(list(models.keys()))
        selected_model, param_distributions = models[selected_model_name]

        # Randomly select hyperparameters from the distributions
        random_hyperparameters = {key: random.choice(value) for key, value in param_distributions.items()}
        print(f"Selected model: {selected_model_name}")
        print(f"Random hyperparameters: {random_hyperparameters}")

        # self.model = selected_model
        # self.model_name = selected_model_name
        # self.hyper_parameters = random_hyperparameters
        if selected_model_name not in self.model_params_list:
            self.model_params_list[selected_model_name] = []
        if random_hyperparameters in self.model_params_list[selected_model_name]:
            print(f"Duplicate hyperparameters found for {selected_model_name}, retrying...")
            self.set_model()  # Call the function again if hyperparameters are not unique
        else:
            print(f"Selected model: {selected_model_name}")
            print(f"Random hyperparameters: {random_hyperparameters}")

            self.model = selected_model
            self.model_name = selected_model_name
            self.hyper_parameters = random_hyperparameters
            self.model_params_list[selected_model_name].append(random_hyperparameters)
    def get_model(self):
        # Initialize the model with the random hyperparameters
        model = self.model
        hyper_parameters = self.hyper_parameters
        initialized_model = model.set_params(**hyper_parameters)

        print(f"Initialized model: {initialized_model}")
        return self.model_name, self.hyper_parameters
