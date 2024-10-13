import os
import subprocess
import time
import signal
import json
import sys
from get_next_exp import GetNextExperiment
import argparse
import logging
from log_exception import log_exception

# Function to handle the signal for clean termination
def signal_handler(sig, frame):
    print("Script terminated.")
    sys.exit(0)

def terminate_processes():
    """Terminate all processes in the global list."""
    logging.info("Terminating all server and client processes...")
    for process in processes:
        try:
            process.terminate()
            process.wait()
        except Exception as e:
            logging.error(f"Error terminating process: {e}")


def signal_handler(sig, frame):
    """Handle termination signals to cleanly terminate processes."""
    terminate_processes()
    logging.info("Script terminated.")
    sys.exit(0)


def monitor_processes(server_process, client_processes):
    """Monitor server and client processes and terminate all if any process fails."""
    while server_process.poll() is None:
        for client_process in client_processes:
            if client_process.poll() is not None:
                if client_process.returncode != 0:
                    terminate_processes()
        time.sleep(1)  # Polling interval
    if server_process.returncode != 0:
        terminate_processes()



# Register the signal handler for clean termination
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Set the environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    if process.returncode != 0:
        print(f"Error: {err.decode('utf-8')}")
        sys.exit(1)
    return out.decode('utf-8')


# Change to the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
processes = []
train_size = .8
eval_size = 1 - train_size
n_clients = 10
port = "5432"

for n_clients in [5,15,10,9,20]:
    csv_file = f"results.csv"
    experiment_finder = GetNextExperiment(
        csv_file=csv_file, n_clients=n_clients, eval_size=eval_size
    )
    dataset_name = experiment_finder.get_next_experiment()
    while dataset_name:
        experiment_info = {
            'dataset_name': dataset_name,
            'n_clients': n_clients,
            'type': "Run file"
        }
        try:
            print("Starting server:")
            print(f"data set name: {dataset_name}")
            server_process = subprocess.Popen(
                [
                    "python",
                    "server.py",
                    "--n_clients",
                    str(n_clients),
                    "--dataset_name",
                    str(dataset_name),
                    "--port",
                    str(port)
                ]
            )
            processes.append(server_process)
            time.sleep(5)  # Sleep for 5 seconds to give the server enough time to start

            client_processes = []
            for j in range(n_clients):
                print(f"Starting client {j}")
                client_process = subprocess.Popen(
                    [
                        "python",
                        f"client.py",
                        "--client_id",
                        str(j),
                        "--n_clients",
                        str(n_clients),
                        "--dataset_name",
                        str(dataset_name),
                        "--train_freq",
                        str(train_size),
                        "--port",
                        str(port)
                    ]
                )
                client_processes.append(client_process)
                processes.append(client_process)

            # Wait for the server process to complete
            monitor_processes(server_process, client_processes)

            # Terminate all client processes
            for client_process in client_processes:
                client_process.terminate()
                client_process.wait()

            # Remove finished client processes from the global list
            processes[:] = [p for p in processes if p.poll() is None]

        except Exception as e:
            error_message = f"Error during run.py: {e}"
            logging.error(error_message)
            log_exception(experiment_info, error_message, n_clients=n_clients)

        finally:
            # Ensure processes are terminated after each experiment
            terminate_processes()
        experiment_finder = GetNextExperiment(
            csv_file=csv_file, n_clients=n_clients, eval_size=eval_size
        )
        dataset_name = experiment_finder.get_next_experiment()

