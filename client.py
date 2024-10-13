import os
import signal

import flwr as fl
import json


from client_utils.parameters_handler import ParametersHandler
from client_utils.read_preprocess_data import ReadPreprocessData
from client_utils.models_enum import ModelsEnum
from sklearn.metrics import (
    root_mean_squared_error,
)
from client_utils.utils import (
    get_model_weights,
    set_model_weights,
    parameters_to_weights,
    weights_to_parameters,
    get_current_model,
)
from client_utils.split_data import SplitData
from client_utils.get_client_data import GetClientData
import argparse
from log_exception import log_exception


class FlowerClient(fl.client.Client):
    def __init__(self, cid, server_address, server_port, dataset_path, n_clients, train_freq=0.8):
        self.cid = cid
        self.server_address = server_address
        self.server_port = server_port
        self.train_freq = float(train_freq)
        self.raw_data = GetClientData(
            client_id=cid, n_clients=n_clients, dataset_name=dataset_path
        ).load_and_process_data()
        split_data = SplitData(data=self.raw_data, train_freq=self.train_freq)
        self.raw_train_data, self.raw_test_data = split_data.train_test_split()
        read_preprocess_data = ReadPreprocessData()
        self.preprocessed_train_data, self.columns_types, self.dataset_type = (
            read_preprocess_data.fit_transform(self.raw_train_data)
        )
        self.preprocessed_test_data = read_preprocess_data.transform(self.raw_test_data)
        self.parameters_handler = ParametersHandler(
            raw_train_data = self.raw_train_data,
            preprocessed_train_data=self.preprocessed_train_data,
            preprocessed_test_data=self.preprocessed_test_data,
            columns_types=self.columns_types,
            dataset_type=self.dataset_type,
        )
        self.modelEnum = ModelsEnum
        self.selected_features = []
        super().__init__()  # Initialize the parent class

    def get_parameters(self, config):
        features_bytes = json.dumps({"server_round": 1}).encode("utf-8")
        parameters = fl.common.Parameters(
            tensors=[features_bytes], tensor_type="features_weights"
        )
        status = fl.common.Status(code=fl.common.Code.OK, message="done")
        return fl.common.typing.GetParametersRes(status=status, parameters=parameters)

    def fit(self, parameters):
        metrics = {}
        if parameters.parameters.tensor_type == "features_weights":
            data_list = [
                json.loads(tensor.decode("utf-8"))
                for tensor in parameters.parameters.tensors
            ]
            if data_list[0]["server_round"] <= 3:
                output = self.parameters_handler.get_output(parameters, data_list)
                if isinstance(output, list):
                    parameters = weights_to_parameters(output)

                elif isinstance(output, bytes):
                    parameters = fl.common.Parameters(
                        tensors=[output], tensor_type="weights"
                    )
                else:
                    features_bytes = json.dumps(output).encode("utf-8")
                    # Create a Parameters object with features as bytes
                    parameters = fl.common.Parameters(
                        tensors=[features_bytes], tensor_type="dict"
                    )
        else:
            tensor_type = parameters.parameters.tensor_type
            model_name = parameters.parameters.tensors[0].decode('utf-8')
            model_parameters = json.loads(parameters.parameters.tensors[1])
            model = get_current_model(model_name=model_name,
                                      model_params = model_parameters)
            if model_name == "XGBRegressor":
                metrics["model"] = str(model_name)
                global_model = []
                selected_features = self.parameters_handler.selected_features
                X, y = SplitData(
                    data=self.parameters_handler.train_data,
                    selected_features=selected_features,
                    target_column=self.columns_types["target"],
                ).x_y_split()
                if tensor_type != "xgboost_weights":
                    model.fit(X, y)
                else:
                    agg_weights = parameters.parameters.tensors[2:]
                    for item in agg_weights:
                        global_model = bytearray(item)
                    model.load_model(global_model)
                    model = set_model_weights(model, global_model)
                output = get_model_weights(model)
                parameters = fl.common.Parameters(tensors=[output], tensor_type="weights")
            else:
                selected_features = self.parameters_handler.selected_features
                X, y = SplitData(
                    data=self.parameters_handler.train_data,
                    selected_features=selected_features,
                    target_column=self.columns_types["target"],
                ).x_y_split()
                metrics["model"] = str(model_name)
                if tensor_type != "model_weights":
                    model.fit(X, y)
                else:
                    agg_weights = parameters.parameters.tensors[2:]
                    weights = parameters_to_weights(agg_weights)
                    model.fit(X, y)
                    model = set_model_weights(model, weights)
                parameters = get_model_weights(model)
                parameters = weights_to_parameters(parameters)

        status = fl.common.Status(code=fl.common.Code.OK, message="done")
        fit_res = fl.common.FitRes(
            parameters=parameters,
            num_examples=self.parameters_handler.data_length,
            metrics=metrics,
            # Set metrics to an empty dictionary since you don't want to return any metrics
            status=status,
        )
        print("inside fit client")
        return fit_res

    def evaluate(self, parameters):
        print("inside evaluate")
        model_name = parameters.parameters.tensors[0].decode('utf-8')
        model_parameters = json.loads(parameters.parameters.tensors[1])
        model = get_current_model(model_name=model_name,
                                    model_params = model_parameters)
        tensor_type = parameters.parameters.tensor_type
        selected_features = self.parameters_handler.selected_features
        X, y = SplitData(
            data=self.parameters_handler.train_data,
            selected_features=selected_features,
            target_column=self.columns_types["target"],
        ).x_y_split()
        X_test, y_test = SplitData(
            data=self.parameters_handler.test_data,
            selected_features=selected_features,
            target_column=self.columns_types["target"],
        ).x_y_split()
        parameters = parameters.parameters.tensors[2:]
        # print(model_name)
        # print("model_name")
        # print("model_parameters")
        # print(model_parameters)
        # print("weights")
        # print(parameters)
        # if tensor_type == "XGBRegressor":
        #         global_model = []
        #         for item in parameters:
        #             global_model = bytearray(item)
        #         model.load_model(global_model)
        #         model = set_model_weights(model, global_model)
        #         # Make predictions
        #         y_pred = model.predict(X_test)
        #         loss = root_mean_squared_error(y_test, y_pred)
        #         train_pred = model.predict(X)
        #         train_loss = root_mean_squared_error(y, train_pred)
        #
        # else:
        #         weights = parameters_to_weights(parameters)
        #         model.fit(X, y)
        #         model = set_model_weights(model, weights)
        #         y_pred = model.predict(X_test)
        #         loss = root_mean_squared_error(y_test, y_pred)
        #         train_pred = model.predict(X)
        #         train_loss = root_mean_squared_error(y, train_pred)
        try:
            if tensor_type == "XGBRegressor":
                global_model = []
                for item in parameters:
                    global_model = bytearray(item)
                model.load_model(global_model)
                model = set_model_weights(model, global_model)
                # Make predictions
                y_pred = model.predict(X_test)
                loss = root_mean_squared_error(y_test, y_pred)
                train_pred = model.predict(X)
                train_loss = root_mean_squared_error(y, train_pred)

            else:
                weights = parameters_to_weights(parameters)
                model.fit(X, y)
                model = set_model_weights(model, weights)
                y_pred = model.predict(X_test)
                loss = root_mean_squared_error(y_test, y_pred)
                train_pred = model.predict(X)
                train_loss = root_mean_squared_error(y, train_pred)
        except Exception as ex:
            train_loss = 9999999999
            loss = 9999999999
        status = fl.common.Status(code=fl.common.Code.OK, message="done")
        print("end evaluate")
        return fl.common.EvaluateRes(
            loss=loss,
            num_examples=len(X_test),
            metrics={"train_loss": train_loss},
            status=status,
        )


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
