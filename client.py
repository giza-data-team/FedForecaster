from log_exception import log_exception
import flwr as fl
from client_utils.prepare_data import PrepareData
from sklearn.metrics import root_mean_squared_error
from client_utils.utils import (parameters_to_ndarrays, ndarrays_to_parameters)
from kerasbeats import NBeatsModel
from client_utils.get_client_data import GetClientData
import numpy as np
import random
import argparse
import tensorflow as tf
import keras
import os
import signal
EXCEPTION_LOG_FILE = 'experiment_exceptions.csv'
tf.get_logger().setLevel('ERROR')
os.environ["GRPC_POLL_STRATEGY"] = "epoll1"

class FlowerClient(fl.client.Client):
    def __init__(self, cid, server_address, server_port, raw_data, number_clients):
        self.cid = cid
        self.server_address = server_address
        self.server_port = server_port
        self.raw_data = raw_data
        self.n_clients = number_clients
        self.X_train, self.X_test, self.y_train, self.y_test = PrepareData(data=self.raw_data,
                                                                           train_size=0.67).train_test_split()
        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
        keras.utils.set_random_seed(42)
        keras.mixed_precision.set_dtype_policy('mixed_float16')
        if number_clients > 5:
            self.nbeats_model = NBeatsModel(model_type='generic',
                                            lookback=7,
                                            horizon=1,
                                            num_generic_neurons=128,  # Reduced from 512
                                            num_generic_stacks=10,  # Reduced from 30
                                            num_generic_layers=2,  # Reduced from 4
                                            num_trend_neurons=64,  # Reduced from 256
                                            num_trend_stacks=2,  # Reduced from 3
                                            num_trend_layers=2,  # Reduced from 4
                                            num_seasonal_neurons=512,  # Reduced from 2048
                                            num_seasonal_stacks=2,  # Reduced from 3
                                            num_seasonal_layers=2,  # Reduced from 4
                                            num_harmonics=1,
                                            polynomial_term=2,  # Reduced from 3
                                            loss='mse',
                                            learning_rate=0.0005,
                                            batch_size=256)  # Reduced from 1024
        else:
            self.nbeats_model = NBeatsModel(model_type='generic',
                                            lookback=7,
                                            horizon=1,
                                            batch_size=256,  # Reduced from 1024
                                            num_seasonal_neurons=512,  # Reduced from 2048
                                            loss="mse",
                                            learning_rate=0.0005,
                                            num_generic_stacks=10)  # Reduced from 30
        self.nbeats_model.fit(self.X_train, self.y_train, epochs=0)
        super().__init__()  # Initialize the parent class

    def get_parameters(self, config):
        weights = self.nbeats_model.model.get_weights()
        parameters = ndarrays_to_parameters(weights)
        status = fl.common.Status(code=fl.common.Code.OK, message="done")
        return fl.common.typing.GetParametersRes(status=status, parameters=parameters)

    def fit(self, parameters):
        # Perform feature extraction
        metrics = {}
        weights = parameters_to_ndarrays(parameters)
        self.nbeats_model.model.set_weights(weights)
        self.nbeats_model.fit(self.X_train, self.y_train, epochs=1)
        weights = self.nbeats_model.model.get_weights()
        parameters = ndarrays_to_parameters(weights)
        status = fl.common.Status(code=fl.common.Code.OK, message="done")
        fit_res = fl.common.FitRes(
            parameters=parameters,
            num_examples=len(self.X_train),
            metrics=metrics,
            # Set metrics to an empty dictionary since you don't want to return any metrics
            status=status
        )
        return fit_res

    def evaluate(self, parameters):
        weights = parameters_to_ndarrays(parameters)
        self.nbeats_model.model.set_weights(weights)

        y_pred_train = self.nbeats_model.predict(self.X_train)
        train_rmse = root_mean_squared_error(self.y_train[:, 0], y_pred_train[:, 0])

        y_pred_test = self.nbeats_model.predict(self.X_test)
        test_rmse = root_mean_squared_error(self.y_test[:, 0], y_pred_test[:, 0])

        status = fl.common.Status(code=fl.common.Code.OK, message="done")
        return fl.common.EvaluateRes(loss=test_rmse, num_examples=len(self.X_test), metrics={"train_rmse": train_rmse},
                                     status=status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process named arguments.')

    # Define arguments
    parser.add_argument('--n_clients', type=str, required=True, help='Number of clients')
    parser.add_argument('--client_id', type=str, required=True, help='The ID of the client')
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset_name')

    # Parse the arguments
    args = parser.parse_args()

    # Access arguments by name
    client_id = int(args.client_id)
    n_clients = int(args.n_clients)
    dataset_name = args.dataset_name

    # --------- server configs --------------
    client_server_address = "localhost"  # Change to actual server address
    client_server_port = 5555  # Change to actual server port

    try:
        client_data = GetClientData(client_id=client_id, n_clients=n_clients, dataset_name=dataset_name)
        client_chunk = client_data.load_and_process_data()
        # # Create an instance of the client
        client = FlowerClient(cid=f"client_{client_id}", server_address=client_server_address,
                              server_port=client_server_port,
                              raw_data=client_chunk, number_clients=n_clients)
        # # # Connect the client to the server
        fl.client.start_client(server_address="localhost:5555", client=client)
    except Exception as e:
        error_message = f"Error in client script: {e}"
        experiment_info = {
            'dataset_name': dataset_name,
            'n_clients': int(n_clients),
            "type": f"client {client_id}"
        }
        if "Connection reset" not in error_message:
            print(error_message)
            log_exception(experiment_info, error_message)
        os.kill(os.getpid(), signal.SIGTERM)