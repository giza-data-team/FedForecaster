import csv
import pandas as pd
import os


class GetNextExperiment:
    def __init__(self, csv_file, n_clients, eval_size=0.33):
        self.csv_file = csv_file
        self.dataset_names = self._get_data_set_names()
        self.num_clients = int(n_clients)
        # self.grid_search = grid_search
        self.eval_size = eval_size
        self.attempted_datasets = set()
        self.completed_experiments = self.read_results()

    def _get_data_set_names(self):
        path = "benchmarking_datasets/"
        files = os.listdir(path)
        dataset_names = [file.split(".")[0] for file in files]
        names = []
        for name in dataset_names:
            names.append(name)
        return names

    def read_results(self):
        """Read the results from the CSV file and return a set of completed experiments."""
        completed_experiments = set()
        try:
            with open(
                    self.csv_file, mode="r", newline="", encoding="utf-8-sig"
            ) as file:
                reader = csv.DictReader(file)
                for row in reader:
                    dataset_name = str(row["dataset_name"])
                    num_clients = int(row["num_clients"])
                    completed_experiments.add(
                        (dataset_name, num_clients)
                    )
        except FileNotFoundError:
            # If file does not exist, return an empty set
            pass
        return completed_experiments

    def get_next_experiment(self):
        """Determine the next experiment to run based on the CSV file."""
        for dataset_name in self.dataset_names:
            if (str(dataset_name),int(self.num_clients)) not in self.completed_experiments:
                if self.check_available_data(
                        dataset_name=dataset_name, n_clients=self.num_clients
                ):
                    return dataset_name

        return None# If all combinations are completed

    def check_available_data(self, dataset_name, n_clients, eval_set_size=200):
        try:
            if n_clients == 9:
                print(dataset_name)
                raise
            data = pd.read_csv(f"benchmarking_datasets/{str(dataset_name)}.csv").ffill()

            total_rows = len(data)

            # Calculate the number of rows each client will have
            rows_per_client = total_rows // n_clients

            # Check if each client can have at least 200 rows in their evaluation set
            if (rows_per_client * self.eval_size) >= eval_set_size:
                return True
            else:
                return False
        except:
            if n_clients == 10 and dataset_name in ["XLE", "XLF", "XLI", "XLK"]:
                data = pd.read_csv(f"benchmarking_datasets/{str(dataset_name)}/1.csv")
                return True
            elif n_clients == 9 and dataset_name in ["GDX", "XLU"]:
                data = pd.read_csv(f"benchmarking_datasets/{str(dataset_name)}/1.csv")
                return True
            else:
                return False
