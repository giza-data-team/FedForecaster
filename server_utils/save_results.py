import os
import csv


class SaveResults:
    def __init__(self, n_clients):
        self.file_name = "FL_results.csv"
        self.columns = [
            "dataset_name",
            "num_clients",
            "train_loss",
            "test_loss",
            "time_taken",
            "model_name",
            "model_parameters",
            "first_model",
            "second_model",
            "third_model",
        ]

        # Check if the file exists, if not create it with header
        if not os.path.isfile(self.file_name):
            with open(self.file_name, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(self.columns)

    def save(
        self,
        dataset_name,
        num_clients,
        train_loss,
        test_loss,
        time_taken,
        model,
        parameters,
        models
    ):
        with open(self.file_name, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    dataset_name,
                    num_clients,
                    train_loss,
                    test_loss,
                    time_taken,
                    model,
                    parameters,
                    models[0],
                    models[1],
                    models[2],
                ]
            )
        print(f"results saved for dataset {dataset_name} and n_clients {num_clients}")
