import flwr as fl

from log_exception import log_exception
from server_utils.agg_strategy import CustomStrategy
import argparse
import os
import signal
import json


class FlowerServer(fl.server.Server):
    def __init__(self, strategy: fl.server.strategy.Strategy):
        # Create a SimpleClientManager to manage the clients
        client_manager = fl.server.SimpleClientManager()

        # Initialize the Server class with the SimpleClientManager and custom strategy
        super().__init__(client_manager=client_manager, strategy=strategy)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process named arguments.")
    # Create an instance of your custom strategy
    parser.add_argument(
        "--n_clients", type=str, required=True, help="Number of clients"
    )
    parser.add_argument("--dataset_name", type=str, required=True, help="dataset_name")
    parser.add_argument("--port", type=str, required=True, help="port")

    # Parse the arguments
    args = parser.parse_args()

    # Access arguments by name
    n_clients = int(args.n_clients)
    dataset_name = str(args.dataset_name)
    port = int(args.port)
    round_number = 9999999
    custom_strategy = CustomStrategy(
        round_number=round_number,
        dataset_name=dataset_name,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        n_clients=n_clients
    )

    try:
        # Create an instance of your server with the custom strategy
        server = FlowerServer(strategy=custom_strategy)
        print("start server")
        # Start the Flower server with the custom strategy
        fl.server.start_server(
            server_address=f"localhost:{port}",
            server=server,
            config=fl.server.ServerConfig(num_rounds=round_number),
        )
    except Exception as e:

        error_message = f"Error in server script: {e}"
        experiment_info = {
            'dataset_name': dataset_name,
            'n_clients': int(n_clients),
            "type": f"Server"
        }
        if "grid search done" not in error_message:
            print(error_message)
            log_exception(experiment_info, error_message, n_clients=n_clients)
        os.kill(os.getpid(), signal.SIGTERM)
