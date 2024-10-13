import flwr as fl
import json
from client_utils.parameters_handler import ParametersHandler
from client_utils.read_preprocess_data import ReadPreprocessData
from client_utils.ModelEnum import ModelEnum
import pickle
from sklearn.metrics import  root_mean_squared_error
from client_utils.utils import (get_model_weights, set_model_weights, parameters_to_weights,
                                weights_to_parameters, get_best_model)
from client_utils.split_data import SplitData
from client_utils.get_client_data import GetClientData
from log_exception import log_exception

import argparse
import os
import signal

class FlowerClient(fl.client.Client):
    def __init__(self, cid, server_address, server_port, dataset_path, n_clients,train_freq = 0.8):
        self.cid = cid
        self.server_address = server_address
        self.server_port = server_port
        self.raw_data = GetClientData(client_id=int(cid), n_clients=n_clients,
                                       dataset_name=dataset_path).load_and_process_data()
        split_data = SplitData(data=self.raw_data, train_freq=train_freq)
        self.raw_train_data, self.raw_test_data = split_data.train_test_split()
        read_preprocess_data = ReadPreprocessData()
        self.preprocessed_train_data, self.columns_types, self.dataset_type = read_preprocess_data.fit_transform(self.raw_train_data)
        self.preprocessed_test_data = read_preprocess_data.transform(self.raw_test_data)
        self.parameters_handler = ParametersHandler(cid=self.cid,preprocessed_train_data=self.preprocessed_train_data,
                                                    preprocessed_test_data=self.preprocessed_test_data,
                                                    columns_types=self.columns_types, dataset_type=self.dataset_type)
        self.modelEnum = ModelEnum
        self.selected_features = []
        self.model = None
        super().__init__()  # Initialize the parent class

    def get_parameters(self, config):
        features_bytes = json.dumps({'server_round': 1}).encode("utf-8")
        parameters = fl.common.Parameters(tensors=[features_bytes], tensor_type="features_weights")
        status = fl.common.Status(code=fl.common.Code.OK, message="done")
        return fl.common.typing.GetParametersRes(status=status, parameters=parameters)

    def fit(self, parameters):
        print("inside fit >>>>>>>>>>>>>>>>>>>>>>>>>")
        # Perform feature extraction
        model = get_best_model()
        model_name = model.__class__.__name__
        # print(parameters.parameters)
        metrics = {}
        if parameters.parameters.tensor_type == "features_weights":
            data_list = [json.loads(tensor.decode("utf-8")) for tensor in parameters.parameters.tensors]
            if data_list[0]['server_round'] <= 4:
                output = self.parameters_handler.get_output(parameters, data_list)
                if isinstance(output, list):
                    model = get_best_model()
                    metrics["model"] = str(model.__class__.__name__)
                    parameters = weights_to_parameters(output)

                elif isinstance(output,bytes):
                    parameters = fl.common.Parameters(tensors=[output],
                                                      tensor_type="weights")
                else:
                    features_bytes = json.dumps(output).encode("utf-8")
                    # Create a Parameters object with features as bytes
                    parameters = fl.common.Parameters(tensors=[features_bytes], tensor_type="dict")

        elif parameters.parameters.tensor_type == "xgboost_weights":
            if model_name == "XGBRegressor":
                metrics["model"] = str(model_name)
                global_model = []
                for item in parameters.parameters.tensors:
                    global_model = bytearray(item)
                model.load_model(global_model)
                model = set_model_weights(model, global_model)
                with open(f'model_{self.cid}.pkl', 'wb') as model_file:
                    pickle.dump(self.model, model_file)
                output = get_model_weights(model)
                parameters = fl.common.Parameters(tensors=[output],
                                              tensor_type="weights")
            else:
                selected_features = self.parameters_handler.selected_features
                train_data = self.parameters_handler.train_data
                X, y = SplitData(data=train_data, selected_features=selected_features,
                                 target_column=self.columns_types['target']).x_y_split()
                model.fit(X, y)
                output = get_model_weights(model)
                metrics["model"] = str(model_name)
                parameters = weights_to_parameters(output)
            # print(output)
        else:
            if model_name == "XGBRegressor":
                selected_features = self.parameters_handler.selected_features
                train_data = self.parameters_handler.train_data
                X, y = SplitData(data=train_data, selected_features=selected_features,
                                 target_column=self.columns_types['target']).x_y_split()
                model.fit(X, y)
                metrics["model"] = str(model_name)
                output = get_model_weights(model)
                parameters = fl.common.Parameters(tensors=[output],
                                              tensor_type="weights")
            else:
                weights = parameters_to_weights(parameters)
                selected_features = self.parameters_handler.selected_features
                train_data = self.parameters_handler.train_data
                X, y = SplitData(data=train_data,
                                 selected_features=selected_features,
                                 target_column=self.columns_types['target']).x_y_split()
                # self.model = get_best_model()
                metrics["model"] = str(model_name)
                model = set_model_weights(model, weights)
                model.fit(X, y)
                with open(f'model_{self.cid}.pkl', 'wb') as model_file:
                    pickle.dump(model, model_file)
                parameters = get_model_weights(model)
                parameters = weights_to_parameters(parameters)
        status = fl.common.Status(code=fl.common.Code.OK, message="done")
        fit_res = fl.common.FitRes(
            parameters=parameters,
            num_examples=self.parameters_handler.data_length,
            metrics=metrics,
            # Set metrics to an empty dictionary since you don't want to return any metrics
            status=status
        )
        return fit_res

    def evaluate(self, parameters):
        self.model = get_best_model()
        tensor_type = parameters.parameters.tensor_type
        selected_features = self.parameters_handler.selected_features
        train_data = self.parameters_handler.train_data
        test_data = self.parameters_handler.test_data
        X, y = SplitData(data=train_data,
                        selected_features=selected_features,
                        target_column=self.columns_types['target']).x_y_split()
        X_test, y_test = SplitData(data=test_data,
                                           selected_features=selected_features,
                                           target_column=self.columns_types['target']).x_y_split()
        try:
            if tensor_type == "xgboost_weights":

                global_model = []
                for item in parameters.parameters.tensors:
                    global_model = bytearray(item)
                self.model.load_model(global_model)
                self.model = set_model_weights(self.model, global_model)
                # Make predictions
                y_pred = self.model.predict(X_test)
                loss = root_mean_squared_error(y_test, y_pred)
                train_pred = self.model.predict(X)
                train_loss = root_mean_squared_error(y,train_pred)
            else:

                weights = parameters_to_weights(parameters)
                model = get_best_model()
                model.fit(X, y)
                model = set_model_weights(model, weights)
                y_pred = model.predict(X_test)
                loss = root_mean_squared_error(y_test, y_pred)
                train_pred = model.predict(X)
                train_loss = root_mean_squared_error(y, train_pred)
        except:
            train_loss = 999999999
            loss = 999999999
        status = fl.common.Status(code=fl.common.Code.OK, message="done")
        return fl.common.EvaluateRes(loss=loss, num_examples=len(X_test), metrics={"train_loss":train_loss}, status=status)

if __name__ == "__main__":
    # Parse variables
    parser = argparse.ArgumentParser(description="Process named arguments.")

    # Define arguments
    parser.add_argument(
        "--n_clients", type=str, required=True, help="Number of clients"
    )
    parser.add_argument(
        "--client_id", type=str, required=True, help="The ID of the client"
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="dataset_name")
    parser.add_argument("--train_freq", type=str, required=True, help="train_freq")
    parser.add_argument("--port", type=str, required=True, help="port")
    # Parse the arguments
    args = parser.parse_args()

    # Access arguments by name
    client_id = int(args.client_id)
    n_clients = int(args.n_clients)
    dataset_name = str(args.dataset_name)
    train_freq = float(args.train_freq)
    port = int(args.port)
    # Parse variables
    # --------- server configs --------------
    client_server_address = "localhost"  # Change to actual server address
    client_server_port = port  # Change to actual server port
    try:
        # Create an instance of the client
        client = FlowerClient(
            cid=client_id,
            server_address=client_server_address,
            server_port=client_server_port,
            dataset_path=dataset_name,
            n_clients=n_clients,
            train_freq=train_freq
        )
        # Connect the client to the server
        fl.client.start_client(server_address=f"{client_server_address}:{client_server_port}", client=client)
    except Exception as e:
        error_message = f"Error in client script: {e}"
        experiment_info = {
            'dataset_name': dataset_name,
            'n_clients': int(n_clients),
            "type": f"client {client_id}"
        }
        if "Connection reset" not in error_message:
            print(error_message)
            log_exception(experiment_info, error_message, n_clients=n_clients)
        os.kill(os.getpid(), signal.SIGTERM)
