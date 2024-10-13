import flwr as fl
from server_utils.agg_strategy import CustomStrategy
import argparse
import tensorflow as tf
from log_exception import log_exception
import os
import signal

tf.get_logger().setLevel('ERROR')
os.environ["GRPC_POLL_STRATEGY"] = "epoll1"

class FlowerServer(fl.server.Server):
    def __init__(self, strategy: fl.server.strategy.Strategy):
        # Create a SimpleClientManager to manage the clients
        client_manager = fl.server.SimpleClientManager()

        # Initialize the Server class with the SimpleClientManager and custom strategy
        super().__init__(client_manager=client_manager, strategy=strategy)


if __name__ == "__main__":
    # Create an instance of your custom strategy
    parser = argparse.ArgumentParser(description='Process named arguments.')

    # Define arguments
    parser.add_argument('--round_number', type=str, required=True, help='Number of rounds')
    parser.add_argument('--n_clients', type=str, required=True, help='Number of clients')
    parser.add_argument('--max_time_minutes', type=str, required=True, help='max_time_minutes')
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset_name')
    # Parse the arguments
    args = parser.parse_args()

    # Access arguments by name
    round_number = int(args.round_number)
    n_clients = int(args.n_clients)
    max_time_minutes = int(args.max_time_minutes)
    dataset_name = args.dataset_name
    print(f"Next experiment: Dataset {dataset_name} with {n_clients} clients.")

    min_fit_clients = n_clients
    min_evaluate_clients = n_clients
    min_available_clients = n_clients
    try:
        custom_strategy = CustomStrategy(round_number=round_number, dataset_name=dataset_name,
                                         min_fit_clients=min_fit_clients,
                                         min_evaluate_clients=min_evaluate_clients,
                                         min_available_clients=min_available_clients,
                                         max_time_minutes=max_time_minutes,
                                         patience=3)

        # Create an instance of your server with the custom strategy
        server = FlowerServer(strategy=custom_strategy)
        print("start server")
        # Start the Flower server with the custom strategy

        fl.server.start_server(server_address="localhost:5555", server=server
                               , config=fl.server.ServerConfig(num_rounds=round_number))
    except Exception as e:
        error_message = f"Error in server script: {e}"
        experiment_info = {
            'dataset_name': dataset_name,
            'n_clients': int(n_clients),
            "type": f"Server"
        }
        if "Time done" not in error_message:
            print(error_message)
            log_exception(experiment_info, error_message)
        os.kill(os.getpid(), signal.SIGTERM)

