from collections import Counter
from typing import cast
from server_utils.aggregators.base_aggregator import Aggregator
import json
import flwr as fl
import numpy as np
class SVRWeightsAggregator(Aggregator):

    def __init__(self):
        self.global_model = None

    def parameters_to_weights(self,parameters):
        tensors = parameters.tensors
        return [np.frombuffer(t, dtype=np.float32) for t in tensors]

# Define a function to convert weights to Parameters
    def weights_to_parameters(self,weights):
        return fl.common.Parameters(tensors=[w.astype(np.float32).tobytes() for w in weights],
                                    tensor_type="model_weights")

    def aggregate(self, parameters, data_sizes=[]):
        print(parameters)
        weights_results = [self.parameters_to_weights(parameters_res.parameters) for _,parameters_res in parameters]
        print(weights_results[0])
        print(weights_results[1])
        print(len(weights_results[0][0]))
        print(len(weights_results[1][0]))
        dual_coefs = [param[0] for param in weights_results]
        supports = [param[1] for param in weights_results]
        intercepts = [param[2] for param in weights_results]
        support_vectors = [param[3] for param in weights_results]
        print(len(dual_coefs[0]))
        print(len(dual_coefs[1]))
        dual_coefs = np.stack(dual_coefs)
        avg_dual_coef = np.mean(dual_coefs, axis=0)
        print(avg_dual_coef)
        print(1)
        # Concatenate support indices and support vectors
        merged_supports = np.concatenate(supports)
        merged_support_vectors = np.concatenate(support_vectors, axis=0)
        print(1)
        # Ensure unique support vectors (and their indices)
        unique_support_vectors, unique_indices = np.unique(merged_support_vectors, axis=0, return_index=True)
        unique_supports = merged_supports[unique_indices]
        print(1)
        # Average intercepts
        avg_intercept = np.mean(intercepts, axis=0)
        weights = [avg_dual_coef, unique_supports, avg_intercept, unique_support_vectors]
        print(weights)
        weights = self.weights_to_parameters(weights)
        print(weights)
        return weights
