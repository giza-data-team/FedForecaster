import os
import csv

class SaveResults:
    def __init__(self, file_name='results.csv'):
        self.file_name = file_name
        self.columns = ['dataset_name', 'num_clients', 'train_loss', 'test_loss', 'time_taken',
                        'model_name', "model_parameters"]

        # Check if the file exists, if not create it with header
        if not os.path.isfile(self.file_name):
            with open(self.file_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.columns)

    def save(self, dataset_name, num_clients, train_loss, test_loss, time_taken, model, parameters):
        with open(self.file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([dataset_name, num_clients, train_loss,
                             test_loss, time_taken, model, parameters])
        print(f"results saved for dataset {dataset_name} and n_clients {num_clients}")
