import pandas as pd
import os
import numpy as np
import logging
from statistics import mean, mode, stdev, StatisticsError
from scipy.stats import entropy
from server_utils.aggregators.base_aggregator import Aggregator

# Configure logging
logging.basicConfig(level=logging.DEBUG)


class MetaFeatureExtractionAggregator(Aggregator):

    def aggregate(self, parameters, data_sizes=[]):
        logging.debug(f"Parameters received for aggregation")
        out_parameters = {}

        for feature, feature_parameters in parameters.items():
            out_parameters[feature] = {}
            """Meta Feature Extraction Before preprocessing"""
            # for Instances in Clients
            out_parameters[feature]['Sum of Instances in Clients'] = self._aggregate_sum(
                feature_parameters.get('No. Of Instances', []))
            out_parameters[feature]['Max. Of Instances in Clients'] = self._aggregate_max(
                feature_parameters.get('No. Of Instances', []))
            out_parameters[feature]['Min. Of Instances in Clients'] = self._aggregate_min(
                feature_parameters.get('No. Of Instances', []))
            out_parameters[feature]['Stddev of Instances in Clients'] = self._aggregate_std(
                feature_parameters.get('No. Of Instances', []))
            # for dataset missing values
            out_parameters[feature]['Average Dataset Missing Values %'] = self._aggregate_average(
                feature_parameters.get('Dataset Missing Values %', []))
            out_parameters[feature]['Min Dataset Missing Values %'] = self._aggregate_min(
                feature_parameters.get('Dataset Missing Values %', []))
            out_parameters[feature]['Max Dataset Missing Values %'] = self._aggregate_max(
                feature_parameters.get('Dataset Missing Values %', []))
            out_parameters[feature]['Stddev Dataset Missing Values %'] = self._aggregate_std(
                feature_parameters.get('Dataset Missing Values %', []))
            # for target missing values
            out_parameters[feature]['Average Target Missing Values %'] = self._aggregate_average(
                feature_parameters.get('Target Missing Values %', []))
            out_parameters[feature]['Min Target Missing Values %'] = self._aggregate_min(
                feature_parameters.get('Target Missing Values %', []))
            out_parameters[feature]['Max Target Missing Values %'] = self._aggregate_max(
                feature_parameters.get('Target Missing Values %', []))
            out_parameters[feature]['Stddev Target Missing Values %'] = self._aggregate_std(
                feature_parameters.get('Target Missing Values %', []))
            # for No. Of Features
            out_parameters[feature]['No. Of Features'] = self._aggregate_mode(
                feature_parameters.get('No. Of Features', []))
            # for No. Of Numerical Features
            if 'No. Of Numerical Features' in feature_parameters:
                out_parameters[feature]['No. Of Numerical Features'] = self._aggregate_mode(
                    feature_parameters['No. Of Numerical Features'])
            else:
                # Handle the case where this feature is missing in meta_feature_before but should be in meta_feature_after
                logging.warning(
                    f"'No. Of Numerical Features' key missing in feature '{feature}', it should be in meta_feature_after.")
                continue  # Skip further processing for this feature if key is missing
            # for No. Of Categorical Features
            if 'No. Of Categorical Features' in feature_parameters:
                out_parameters[feature]['No. Of Categorical Features'] = self._aggregate_mode(
                    feature_parameters['No. Of Categorical Features'])
            else:
                # Handle the case where this feature is missing in meta_feature_before but should be in meta_feature_after
                logging.warning(
                    f"'No. Of Categorical Features' key missing in feature '{feature}', it should be in meta_feature_after.")
                continue  # Skip further processing for this feature if key is missing
            # for Sampling Rate
            out_parameters[feature]['Sampling Rate'] = self._aggregate_mode(feature_parameters.get('Sampling Rate', []))
            # for Skewness of Numerical Features
            out_parameters[feature]['Average Skewness of Numerical Features'] = self._aggregate_average(
                feature_parameters.get('Average Skewness of Numerical Features', []))
            out_parameters[feature]['Minimum Skewness of Numerical Features'] = self._aggregate_min(
                feature_parameters.get('Minimum Skewness of Numerical Features', []))
            out_parameters[feature]['Maximum Skewness of Numerical Features'] = self._aggregate_max(
                feature_parameters.get('Maximum Skewness of Numerical Features', []))
            out_parameters[feature]['Stddev Skewness of Numerical Features'] = self._aggregate_std(
                feature_parameters.get('Stddev Skewness of Numerical Features', []))
            # for Kurtosis of Numerical Features
            out_parameters[feature]['Average Kurtosis of Numerical Features'] = self._aggregate_average(
                feature_parameters.get('Average Kurtosis of Numerical Features', []))
            out_parameters[feature]['Minimum Kurtosis of Numerical Features'] = self._aggregate_min(
                feature_parameters.get('Minimum Kurtosis of Numerical Features', []))
            out_parameters[feature]['Maximum Kurtosis of Numerical Features'] = self._aggregate_max(
                feature_parameters.get('Maximum Kurtosis of Numerical Features', []))
            out_parameters[feature]['Stddev Kurtosis of Numerical Features'] = self._aggregate_std(
                feature_parameters.get('Stddev Kurtosis of Numerical Features', []))
            # for No. Of Symbols per Categorical Features
            out_parameters[feature]['Avg No. of Symbols per Categorical Features'] = self._aggregate_average(
                feature_parameters.get('Avg No. of Symbols per Categorical Features', []))
            out_parameters[feature]['Min. No. Of Symbols per Categorical Features'] = self._aggregate_min(
                feature_parameters.get('Min. No. Of Symbols per Categorical Features', []))
            out_parameters[feature]['Max. No. Of Symbols per Categorical Features'] = self._aggregate_max(
                feature_parameters.get('Max. No. Of Symbols per Categorical Features', []))
            out_parameters[feature]['Stddev No. Of Symbols per Categorical Features'] = self._aggregate_std(
                feature_parameters.get('Stddev No. Of Symbols per Categorical Features', []))

            """Meta Time Series Feature Extraction"""
            # No. Of Stationary Features
            out_parameters[feature]['Avg No. Of Stationary Features'] = self._aggregate_average(
                feature_parameters.get('No. Of Stationary Features', []))
            out_parameters[feature]['Min No. Of Stationary Features'] = self._aggregate_min(
                feature_parameters.get('No. Of Stationary Features', []))
            out_parameters[feature]['Max No. Of Stationary Features'] = self._aggregate_max(
                feature_parameters.get('No. Of Stationary Features', []))
            out_parameters[feature]['Stddev No. Of Stationary Features'] = self._aggregate_std(
                feature_parameters.get('No. Of Stationary Features', []))
            # No. Of Stationary Features after 1st order diff
            out_parameters[feature]['Avg No. Of Stationary Features after 1st order'] = self._aggregate_average(
                feature_parameters.get('No. Of Stationary Features after 1st order', []))
            out_parameters[feature]['Min No. Of Stationary Features after 1st order'] = self._aggregate_min(
                feature_parameters.get('No. Of Stationary Features after 1st order', []))
            out_parameters[feature]['Max No. Of Stationary Features after 1st order'] = self._aggregate_max(
                feature_parameters.get('No. Of Stationary Features after 1st order', []))
            out_parameters[feature]['Stddev No. Of Stationary Features after 1st order'] = self._aggregate_std(
                feature_parameters.get('No. Of Stationary Features after 1st order', []))
            # No. Of Stationary Features after 2nd order diff
            out_parameters[feature]['Avg No. Of Stationary Features after 2nd order'] = self._aggregate_average(
                feature_parameters.get('No. Of Stationary Features after 2nd order', []))
            out_parameters[feature]['Min No. Of Stationary Features after 2nd order'] = self._aggregate_min(
                feature_parameters.get('No. Of Stationary Features after 2nd order', []))
            out_parameters[feature]['Max No. Of Stationary Features after 2nd order'] = self._aggregate_max(
                feature_parameters.get('No. Of Stationary Features after 2nd order', []))
            out_parameters[feature]['Stddev No. Of Stationary Features after 2nd order'] = self._aggregate_std(
                feature_parameters.get('No. Of Stationary Features after 2nd order', []))
            # Significant Lags using pACF in Target
            out_parameters[feature]['Avg No. Of Significant Lags in Target'] = self._aggregate_average(
                feature_parameters.get('No. Of Significant Lags in Target', []))
            out_parameters[feature]['Min No. Of Significant Lags in Target'] = self._aggregate_min(
                feature_parameters.get('No. Of Significant Lags in Target', []))
            out_parameters[feature]['Max No. Of Significant Lags in Target'] = self._aggregate_max(
                feature_parameters.get('No. Of Significant Lags in Target', []))
            out_parameters[feature]['Stddev No. Of Significant Lags in Target'] = self._aggregate_std(
                feature_parameters.get('No. Of Significant Lags in Target', []))
            # No. Of Insignificant Lags between 1st and last significant ones in Target
            out_parameters[feature]['Avg No. Of Insignificant Lags in Target'] = self._aggregate_average(
                feature_parameters.get('No. Of Insignificant Lags in Target', []))
            out_parameters[feature]['Max No. Of Insignificant Lags in Target'] = self._aggregate_max(
                feature_parameters.get('No. Of Insignificant Lags in Target', []))
            out_parameters[feature]['Min No. Of Insignificant Lags in Target'] = self._aggregate_min(
                feature_parameters.get('No. Of Insignificant Lags in Target', []))
            out_parameters[feature]['Stddev No. Of Insignificant Lags in Target'] = self._aggregate_std(
                feature_parameters.get('No. Of Insignificant Lags in Target', []))
            # No. Of Seasonality Components in Target
            out_parameters[feature]['Avg. No. Of Seasonality Components in Target'] = self._aggregate_average(
                feature_parameters.get('No. Of Seasonality Components in Target', []))
            out_parameters[feature]['Max No. Of Seasonality Components in Target'] = self._aggregate_max(
                feature_parameters.get('No. Of Seasonality Components in Target', []))
            out_parameters[feature]['Min No. Of Seasonality Components in Target'] = self._aggregate_min(
                feature_parameters.get('No. Of Seasonality Components in Target', []))
            out_parameters[feature]['Stddev No. Of Seasonality Components in Target'] = self._aggregate_std(
                feature_parameters.get('No. Of Seasonality Components in Target', []))
            # Fractal Dimension Analysis of Target
            out_parameters[feature][
                'Average Fractal Dimensionality Across Clients of Target'] = self._aggregate_average(
                feature_parameters.get('Fractal Dimension Analysis of Target', []))
            # Period of Seasonality Components in Target
            out_parameters[feature][
                'Maximum Period of Seasonality Components in Target Across Clients'] = self._aggregate_max(
                feature_parameters.get('Maximum Period of Seasonality Components in Target', []))
            out_parameters[feature][
                'Minimum Period of Seasonality Components in Target Across Clients'] = self._aggregate_min(
                feature_parameters.get('Minimum Period of Seasonality Components in Target', []))
            # Target Stationarity Entropy
            out_parameters[feature]['Entropy of Target Stationarity'] = self._aggregate_entropy(
                feature_parameters.get('Target Stationarity', []))
        # self.metafeatuers_to_csv(out_parameters)
        return {"aggregated features": out_parameters}

    # Sum
    def _aggregate_sum(self, values):

        return sum(values)  # if (values)  else 9999 to check

    # max
    def _aggregate_max(self, values):
        return max(values)  # if (values)  else 9999 to check

    # min
    def _aggregate_min(self, values):
        return min(values)  # if (values)  else 9999 to check

    # mean
    def _aggregate_mean(self, values):
        return mean(values)  # if (values)  else 9999 to check

    # mode
    def _aggregate_mode(self, values):
        try:
            return mode(values)  # if (values)  else 9999 to check
        except StatisticsError:
            return "No unique mode"

    # std
    def _aggregate_std(self, values):
        if (values):
            return np.std(values)
        elif (np.isnan(values)):
            return 0
        else:
            # the standard devision measure the spread of the data so if all identical >> no variability >> return zero
            return 0

    # len
    def _aggregate_len(self, values):
        return len(values)  # if (values)  else 9999 to check

    # average
    def _aggregate_average(self, values):
        return (sum(values) / len(values))  # if (values)  else 9999 to check

    # entropy
    def _aggregate_entropy(self, values):
        if not values:
            return 0
        value_counts = [values.count(v) for v in set(values)]
        return entropy(value_counts, base=None)  # if (values)  else 9999 to check

    def metafeatuers_to_csv(self, results):

        meta_features = results.pop("meta_features", {})
        # Create a DataFrame from the updated data
        new_data_df = pd.DataFrame([meta_features])
        # Path to the CSV file
        file_path = 'Meta Features.csv'

        if os.path.exists(file_path):
            # If the CSV file exists, read the existing data
            existing_data_df = pd.read_csv(file_path)
            # Append the new data
            combined_df = pd.concat([existing_data_df, new_data_df], ignore_index=True)
        else:
            # If the CSV file doesn't exist, the new data is the combined data
            combined_df = new_data_df

        # Save the combined data to the CSV file
        combined_df.to_csv(file_path, index=False)
        print("combined_df")
