import pandas as pd
import numpy as np
from math import log
import sys
import json


class MetaFeatures:
    def __init__(self, df):
        self.df = df
        self.categorical_columns = []
        self.numerical_columns = []

    def get_num_features(self):
        target_keywords = ['Close', 'close', 'value', 'Value']
        for col in self.df.columns:
            if any(keyword in col for keyword in target_keywords):
                self.df.rename(columns={col: 'Target'}, inplace=True)
                break
        return len(self.df.columns) - 2 if 'Timestamp' and 'Target' in self.df.columns else len(self.df.columns)

    def get_num_instances(self):
        return len(self.df)

    def get_num_missing_vals(self):
        value = self.df.isnull().sum().sum()

        return value if (value) else 0

    def get_target_missing_vals(self):
        return self.df['Target'].isnull().sum() if 'Target' in self.df.columns else 0


def meta_feature_extraction(df):
    mf = MetaFeatures(df)

    num_instances = mf.get_num_instances()
    num_features = mf.get_num_features()
    missing_vals = mf.get_num_missing_vals()
    target_missing_vals = mf.get_target_missing_vals()

    results = {
        "No. Of Instances": num_instances,
        "Dataset Missing Values %": ((missing_vals / (num_instances * num_features)) * 100 if num_features > 0 else 0),
        "Target Missing Values %": (target_missing_vals / num_instances) * 100
    }
    return {"meta_features": results}
