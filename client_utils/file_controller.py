import json
import os
import pandas as pd

class FileController:
    """
    This class is designed to handle client features.
    """
    def __init__(self, client_id = 0):
        self.client_id = str(client_id)

    def get_file(self, file_name, type="json"):
        """
        Return the last dictionary of client features from the JSON file.
        If the file doesn't exist, create a new file with an empty dictionary and return it.
        """
        if type == "json":
            if file_name == "best_model":
                file_path = f"D:/federatedLearning/GizaFederatedLearning/random-search/GizaFederatedML/{file_name}.json"
            else:
                file_path = os.path.join(f'./output{self.client_id}', f"{file_name}.json")
            self._check_file_availability(file_path)
            with open(file_path, 'r') as file:
                # Read the file line by line in reverse order
                lines = file.readlines()
                for line in reversed(lines):
                    try:
                        # Try to parse JSON from the current line
                        last_features = json.loads(line.strip())
                        break  # Stop parsing when the last valid JSON object is found
                    except Exception as ex:
                        # Ignore lines that are not valid JSON
                        print("error in file controller ")
                        print(ex)
            return last_features
        else:
            file_path = os.path.join(f'./output{self.client_id}', f"{file_name}.csv")
            df = pd.read_csv(file_path)
            return df

    def save_file(self, data, file_name, type="json"):
        """
        Take dict of features as an input and append it to the JSON file.
        """
        if type == "json":
            file_path = os.path.join(f'./output{self.client_id}', f"{file_name}.json")
            self._check_file_availability(file_path, create_file=True)
            with open(file_path, 'a') as file:
                file.write('\n')  # Ensure each dict is written on a new line
                json.dump(data, file)
        else:
            file_path = os.path.join(f'./output{self.client_id}', f"{file_name}.csv")
            self._check_file_availability(file_path, create_file=False)
            data.to_csv(file_path)

    def _check_file_availability(self, file_path, create_file=False):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        if create_file and not os.path.exists(file_path):
            with open(file_path, 'w') as file:
                json.dump({}, file)

