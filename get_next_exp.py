import csv
import pandas as pd


def check_available_data(dataset_name, n_clients, eval_set_size=200):
    if isinstance(dataset_name, int):
        data = pd.read_csv(f"benchmarking_data/{str(dataset_name)}.csv").ffill()
        total_rows = len(data)
        rows_per_client = total_rows // n_clients

        if rows_per_client * 0.33 >= eval_set_size:
            return True
        else:
            return False
    else:
        return True


class GetNextExperiment:
    def __init__(self, csv_file, n_clients_in_experiments):
        self.csv_file = csv_file
        self.dataset_names = list(range(600, 613))
        self.n_clients = n_clients_in_experiments
        self.attempted_experiments = set()

    def read_results(self):
        """Read the results from the CSV file and return a set of completed experiments."""
        completed_experiments = set()
        try:
            with open(self.csv_file, mode='r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    dataset_name = str(row['dataset_name'])
                    num_clients = int(row['n_clients'])
                    completed_experiments.add((dataset_name, num_clients))
        except FileNotFoundError:
            # If file does not exist, return an empty set
            pass
        return completed_experiments

    def get_next_experiment(self):
        """Determine the next experiment to run based on the CSV file."""
        completed_experiments = self.read_results()
        for client_number in self.n_clients:
            if client_number == 10:
                dataset_names = self.dataset_names + ["XLE", "XLF", "XLI", "XLK"]
            elif client_number == 9:
                dataset_names = ["GDX", "XLU"]
            else:
                dataset_names = self.dataset_names
            for dataset_name in dataset_names:
                experiment = (str(dataset_name), client_number)
                if check_available_data(dataset_name=dataset_name, n_clients=client_number):
                    if experiment not in completed_experiments and experiment not in self.attempted_experiments:
                        self.attempted_experiments.add(experiment)
                        return experiment
        return None  # If all combinations are completed or attempted

    def count_remaining_experiments(self):
        """Count the total number of experiments remaining."""
        completed_experiments = self.read_results()
        total_experiments_count = 0
        remaining_experiments_count = 0

        for client_number in self.n_clients:
            if client_number == 10:
                dataset_names = self.dataset_names + ["XLE", "XLF", "XLI", "XLK"]
            elif client_number == 9:
                dataset_names = ["GDX", "XLU"]
            else:
                dataset_names = self.dataset_names
            for dataset_name in dataset_names:
                if check_available_data(dataset_name=dataset_name, n_clients=client_number):
                    total_experiments_count += 1
                    if (str(dataset_name), client_number) not in completed_experiments:
                        remaining_experiments_count += 1

        return remaining_experiments_count, total_experiments_count

    def reset_attempted_experiments(self):
        """Reset the list of attempted experiments."""
        self.attempted_experiments.clear()


if __name__ == "__main__":
    experiment_manager = GetNextExperiment(csv_file="archive/results.csv", n_clients_in_experiments=[5, 9, 10, 15, 20])
    count = 0
    while True:
        next_experiment = experiment_manager.get_next_experiment()
        if next_experiment:
            print(next_experiment)
            count += 1
        else:
            break
    print(count)
