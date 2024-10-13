import csv
import os


def log_exception(experiment_info, error_message, n_clients):
    EXCEPTION_LOG_FILE = f'experiment_exceptions_{n_clients}.csv'
    """Log exceptions to a CSV file."""
    # Prepare exception log CSV file with headers if it doesn't exist
    if not os.path.exists(EXCEPTION_LOG_FILE):
        with open(EXCEPTION_LOG_FILE, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['dataset_name', 'n_clients', 'type', 'error_message'])

    with open(EXCEPTION_LOG_FILE, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            [experiment_info['dataset_name'], experiment_info['n_clients'], experiment_info['type'], error_message])
