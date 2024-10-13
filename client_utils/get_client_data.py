import pandas as pd


class GetClientData:
    def __init__(self, client_id, n_clients, dataset_name):
        """
        Initializes the GetClientData object with client_id, n_clients, and dataset_name.

        Args:
            client_id (int): The ID of the client (0-indexed).
            n_clients (int): Total number of clients.
            dataset_name (str): Path to the dataset.
        """
        self.client_id = client_id
        self.n_clients = n_clients
        self.dataset_name = dataset_name
        self.data = None

    def load_and_process_data(self):
        """
        Loads the dataset, sorts it by Timestamp, and returns the chunk of data for the specified client.

        Returns:
            pd.DataFrame: A DataFrame containing the chunk of data for the client.
        """
        # Load the dataset
        data_flag = False
        try:
            try:
                int(self.dataset_name)
                data_flag = False
                self.data = pd.read_csv(f"./benchmarking_datasets/{self.dataset_name}.csv").ffill()
            except:
                data_flag = True
                self.data = pd.read_csv(f"./benchmarking_datasets/{self.dataset_name}/{self.client_id+1}.csv").ffill()
        except Exception as e:
            raise ValueError(f"Error reading the dataset: {e}")

        self.data.sort_index(inplace=True)
        if data_flag:
            self.data = self.data[['Timestamp','Close']]
            self.data = self.data.rename(columns={'Close':'Target'})
        else:
            self.data['Target'] = self.data['Target'].astype("float")

            # Calculate chunk indices
            total_rows = len(self.data)
            rows_per_chunk = total_rows // self.n_clients
            extra_rows = total_rows % self.n_clients
            self.client_id = self.client_id - 1
            # Calculate start and end indices for the client_id chunk
            start_idx = self.client_id * rows_per_chunk + min(self.client_id, extra_rows)
            end_idx = start_idx + rows_per_chunk + (1 if self.client_id < extra_rows else 0)

            # Return the chunk for the specified client_id
            self.data = self.data.iloc[start_idx:end_idx]
            # return self.data.iloc[start_idx:end_idx]
        return self.data
