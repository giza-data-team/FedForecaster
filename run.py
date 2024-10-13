import os
import subprocess
import time
import signal
import sys
import logging
from get_next_exp import GetNextExperiment
from log_exception import log_exception

# Configurable parameters
CSV_FILE = 'results_5m.csv'

ROUND_NUMBER = 100
MAX_TIME_MINUTES = 5  # Minutes
SERVER_START_DELAY = 5  # seconds
SERVER_SCRIPT = 'server.py'
CLIENT_SCRIPT = 'client.py'
n_clients_in_experiments = [5, 9, 10, 15, 20]

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global list to keep track of processes
processes = []


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


# Register the signal handler for clean termination
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Set the environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def main():
    """Main function to manage experiments and processes."""
    # Change to the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Get next experiment
    experiment_manager = GetNextExperiment(csv_file=CSV_FILE, n_clients_in_experiments=n_clients_in_experiments)
    next_experiment = experiment_manager.get_next_experiment()

    while next_experiment:
        experiment_info = {
            'dataset_name': next_experiment[0],
            'n_clients': int(next_experiment[1]),
        }
        try:
            remaining_experiments_count, total_experiments_count = experiment_manager.count_remaining_experiments()
            logging.info(f'Total remaining experiments: {remaining_experiments_count} out of {total_experiments_count}')

            logging.info("Starting server:")
            server_process = subprocess.Popen(
                ['python', SERVER_SCRIPT, '--round_number', str(ROUND_NUMBER), '--n_clients',
                 str(experiment_info['n_clients']),
                 '--dataset_name', str(experiment_info['dataset_name']), '--max_time_minutes', str(MAX_TIME_MINUTES)]
            )
            processes.append(server_process)
            time.sleep(SERVER_START_DELAY)  # Sleep to give the server enough time to start

            client_processes = []
            for i in range(experiment_info['n_clients']):
                logging.info(f"Starting client {i}")
                client_process = subprocess.Popen(
                    ['python', CLIENT_SCRIPT, '--client_id', str(i), '--n_clients', str(experiment_info['n_clients']),
                     '--dataset_name',
                     str(experiment_info['dataset_name'])]
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
            log_exception(experiment_info, error_message)

        finally:
            # Ensure processes are terminated after each experiment
            terminate_processes()

        # Fetch the next experiment
        next_experiment = experiment_manager.get_next_experiment()


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


if __name__ == "__main__":
    main()
