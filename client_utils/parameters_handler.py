import json
import pandas as pd
import numpy as np
import os
from client_utils.features_extraction import features_extraction_pipeline
from client_utils.features_engineering import FeaturesEngineeringPipeline
from client_utils.extract_features_importance import FeatureImportanceExtraction
from client_utils.meta_features_before import meta_feature_extraction
from client_utils.meta_features_after import FEX_pipeline


class ParametersHandler:
    def __init__(self, raw_train_data, preprocessed_train_data, preprocessed_test_data, columns_types, dataset_type):
        self.preprocessed_train_data = preprocessed_train_data
        self.preprocessed_test_data = preprocessed_test_data
        self.raw_train_data = raw_train_data
        self.test_data = None
        self.train_data = None
        self.meta_features_before = {}
        self.meta_features_after = {}
        self.columns_types, self.dataset_type = columns_types, dataset_type
        self.data_length = None
        self.selected_features = []
        # self.cid = cid

    def get_output(self, parameters, data_list):
        server_round = data_list[0]['server_round']
        output = {}
        if server_round == 1:
            print(f"Round {server_round} started: Extract time series features")
            time_series_features, self.data_length = features_extraction_pipeline(self.preprocessed_train_data,
                                                                                  self.columns_types)
            output = time_series_features
            print(f"Round {server_round} Done: Extracted time series features and returned to the server")
        elif server_round == 2:
            print(
                f"Round {server_round} started: Feature engineering on selected time series features and Extract feature importance")
            del data_list[0]['server_round']
            pipeline = FeaturesEngineeringPipeline(features=data_list[0], columns_types=self.columns_types)
            # Fit and transform the train data
            self.train_data = pipeline.fit_transform(self.preprocessed_train_data)
            # Transform the test data
            self.test_data = pipeline.transform(self.preprocessed_test_data)
            feature_importance = FeatureImportanceExtraction(self.train_data,
                                                             target_column=self.columns_types['target'])
            output = feature_importance.extract_feature_importance()
            print(
                f"Round {server_round} Done: Applied feature engineering/Feature importance and returned to the server")
        elif server_round == 3:
            del data_list[0]['server_round']

            self.selected_features = data_list[0]['selected_features']
            self.meta_features_before = meta_feature_extraction(self.raw_train_data)
            # create a selected features dataframe
            # Reset the index and add it as a column to the DataFrame
            self.train_data = self.train_data.reset_index()
            columns_to_select = self.selected_features + ['Target', 'Timestamp']
            selected_features_df = self.train_data[columns_to_select].copy()
            self.meta_features_after = FEX_pipeline(selected_features_df)
            combined_meta_features = {
                "meta_features": {
                    **self.meta_features_before["meta_features"],
                    **self.meta_features_after["meta_features"]
                }
            }
            output = self._convert_to_serializable(combined_meta_features)
            print(
                f"Round {server_round} Done: ")
        return output

    def _convert_to_serializable(self, obj):
        """
        Recursively convert non-serializable types to serializable types.
        """
        if isinstance(obj, dict):
            return {self._convert_to_serializable(k): self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_serializable(i) for i in obj)
        elif isinstance(obj, set):
            return {self._convert_to_serializable(i) for i in obj}
        elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_to_serializable(obj.tolist())
        else:
            return obj
