import numpy as np
from server_utils.aggregators.base_aggregator import Aggregator
from server_utils.save_results import SaveResults

class FeatureImportanceAggregator(Aggregator):
    """
    Class to aggregate feature importance dictionaries.
    """

    def __init__(self):
        self.importance_threshold = 0.05
        self.save_result = SaveResults(file_name="features_importance.csv")

    def aggregate(self, parameters,data_sizes=[],dataset_name = ""):
        """
        Aggregate features importance and select top N features.

        Returns:
        - list: List of selected features.
        """
        features_importance = parameters["feature_importance"]

        aggregated_importance = {}
        n_clients = -1
        for feature, total_importance in features_importance.items():
            if n_clients == -1 :
                n_clients = len(total_importance)
            aggregated_importance[feature] = np.mean(total_importance)

        # Return only feature names
        features_name = [feature for feature, v in aggregated_importance.items() if
                                      v > self.importance_threshold]
        self.save_result.save(dataset_name=dataset_name,
                    num_clients=n_clients,
                    train_loss=-1,
                    test_loss=-1,
                    time_taken=-1,
                    model=-1,
                    parameters=features_name,
                    models=[-1,-1,-1])
        return {"selected_features": features_name}

# Example usage:
# if __name__ == "__main__":
#     # Sample feature importance dictionaries
#     features_importance_list = {"feature1": [0.01, .03, .01],
#                                 "feature2": [0.2, 3, .4],
#                                 "feature3": [0.22, .33, .4]}
#
#     # Create an instance of FeatureImportanceAggregator
#     aggregator = FeatureImportanceAggregator()
#
#     # Aggregate and print the top feature names
#     top_features = aggregator.aggregate(features_importance_list)
#     print("Top Features:")
#     for feature in top_features:
#         print(feature)
