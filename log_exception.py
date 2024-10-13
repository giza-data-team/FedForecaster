import csv
import os

EXCEPTION_LOG_FILE = 'experiment_exceptions.csv'


def log_exception(experiment_info, error_message):
    """Log exceptions to a CSV file."""
    # Prepare exception log CSV file with headers if it doesn't exist
    if not os.path.exists(EXCEPTION_LOG_FILE):
        with open(EXCEPTION_LOG_FILE, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['dataset_name', 'n_clients', 'type', 'error_message'])

    with open(EXCEPTION_LOG_FILE, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            [experiment_info['dataset_name'], experiment_info['type'], experiment_info['n_clients'], error_message])
