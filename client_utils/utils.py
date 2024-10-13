import pandas as pd
import numpy as np
from scipy.stats import linregress
import flwr as fl
from client_utils.file_controller import FileController
from client_utils.ModelEnum import ModelEnum
import xgboost as xgb
from io import BytesIO
from typing import cast
from typing import Any, List
import numpy as np
import numpy.typing as npt
from flwr.common.typing import Parameters

NDArray = npt.NDArray[Any]
NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float_]
NDArrays = List[NDArray]

import json
def detect_timeseries_type(data, feature):
    # Extracting time index
    data[feature] = pd.to_numeric(data[feature], errors='coerce')
    data = data[feature]
    time_index = np.arange(len(data))

    # Performing linear regression
    slope, intercept, _, _, _ = linregress(time_index, data)

    # Compute residuals
    residuals = data - (slope * time_index + intercept)

    # Calculate the variance of residuals
    residual_var = np.var(residuals)

    # Compute the mean of the time series data
    data_mean = np.mean(data)

    # Check if the residuals show a pattern (indicating multiplicative seasonality)
    if residual_var > data_mean:
        return "multiplicative"
    else:
        return "additive"


def log_transform(X, feature):
        """
        Apply natural logarithm transformation to the specified column of the input DataFrame.

        Parameters:
            X (pd.DataFrame): The input DataFrame.

            feature: feature used

        Returns:
            pd.DataFrame: The input DataFrame with logarithmically transformed column replacing the original one.
        """

        X_transformed = X.copy()
        min_val= min(X_transformed[feature])
        #shift the data to the positive side if column contains -ve values
        if min_val <= 0:
            X_transformed[feature] = X_transformed[feature]+(abs(min_val)+1)

        X_transformed[feature] = np.log(X_transformed[feature])
        X_transformed.dropna(inplace=True)
        return X_transformed




def get_model_weights(model):
    if hasattr(model, 'coef_'):
        return [model.coef_, model.intercept_]
    elif hasattr(model, 'coefs_'):
        return model.coefs_+ model.intercepts_
    # elif hasattr(model, 'feature_importances_'):
    #     return [model.get_booster().save_raw()[:4]]
    else:
        booster = model.get_booster()
        booster_json = booster.save_raw("json")
        return bytes(booster_json)
        # return json.loads(booster_json.decode('utf-8'))
# Define a function to set model weights
def set_model_weights(model, weights):
    if hasattr(model, 'coef_'):
        try:
            model.coef_ = weights[0]
            model.intercept_ = weights[1]
        except:
            print("skip setter")
    elif hasattr(model, "coefs_"):
        coefs = weights[:len(model.coefs_)]
        intercepts = weights[len(model.coefs_):]
        model.coefs_ = coefs
        model.intercepts_ = intercepts
    elif hasattr(model, 'feature_importances_'):
        booster = xgb.Booster()
        booster.load_model(weights)
        model._Booster = booster
    return model
def weights_to_parameters(weights):
    tensors = [ndarray_to_bytes(ndarray) for ndarray in weights]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")
# Convert Parameters object to model weights
def parameters_to_weights(parameters):
    r = [bytes_to_ndarray(tensor) for tensor in parameters]
    return r

def get_current_model(model_name, model_params ):
    model_class,_ = ModelEnum.get_model_data(model_name)
    model = model_class.__class__(**model_params)
    return model



def ndarray_to_bytes(ndarray: NDArray) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    bytes_io = BytesIO()
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
    np.save(bytes_io, ndarray, allow_pickle=False)
    return bytes_io.getvalue()


def bytes_to_ndarray(tensor: bytes) -> NDArray:
    """Deserialize NumPy ndarray from bytes."""
    bytes_io = BytesIO(tensor)
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.load.html
    ndarray_deserialized = np.load(bytes_io, allow_pickle=False)
    return cast(NDArray, ndarray_deserialized)
