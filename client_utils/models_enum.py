from enum import Enum
from sklearn.linear_model import Lasso, ElasticNetCV, HuberRegressor, QuantileRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
import numpy as np


class ModelsEnum(Enum):
    LASSO = (
        Lasso(random_state=42),
        {
            "alpha": np.logspace(np.log10(1e-5), np.log10(2), num=30),
            "selection": ["cyclic", "random"],
            "random_state": [42]
        },
    )
    LinearSVR = (
        LinearSVR(random_state=42), {"C": [1, 2, 3, 5, 10], "epsilon": [0.01, 0.05, 0.1], "random_state": [42]})
    ELASTICNETCV = (
        ElasticNetCV(random_state=42),
        {"l1_ratio": np.linspace(0.3, 1, 10), "selection": ["cyclic", "random"], "random_state": [42]},
    )
    XGBRegressor = (
        XGBRegressor(random_state=42),
        {
            'n_estimators': [5, 10],
            'max_depth': [2, 7],
            "learning_rate": [0.1, 1],
            "reg_lambda": [0.8, 10],
            # "gamma": [0.9, 1.16467595, 2.248149123539492, 3.9963209507789], Removed for high number of combinations
            # 'colsample_bytree': [0.5, 1.0], Removed for high number of combinations
            "subsample": [0.1, 1],
            "random_state": [42]
        },
    )
    HUBERREGRESSOR = (
            HuberRegressor(),
            {
                'epsilon': [1.0, 1.35, 1.5],
                'alpha': np.logspace(np.log10(1e-3), np.log10(1e2), num=10)
            }
    )
    QUANTILEREGRESSOR = (
        QuantileRegressor(),
        {
            'alpha': np.logspace(np.log10(1e-3), np.log10(1e2), num=10),
            'quantile': [0.1, 0.25, 0.5, 0.75]
        },
    )

    @staticmethod
    def get_model_data(model_name: str):
        try:
            model_enum = ModelsEnum[model_name]
            return model_enum.value
        except KeyError:
            raise ValueError(
                f"Model '{model_name}' is not a valid model name. Choose from: {', '.join([model.name for model in ModelsEnum])}"
            )
