import joblib
import pandas as pd
import numpy as np
import os

class MetaModel:
    """
    A class to handle predictions using a pre-trained classification model.

    Attributes:
        prob_threshold (float): Minimum cumulative probability required to include a class in the output.
        top_n (int): The number of top classes to include based on their probabilities.
        model: The trained classification model.
        label_encoder: Optional LabelEncoder to map encoded labels back to original class names.

    Methods:
        predict_best_model(input_features: dict) -> list:
            Predicts class probabilities for a single record and returns the top classes based on the
            specified prob_threshold or top_n.
    """

    def __init__(self, prob_threshold: float = None, top_n: int = None, model_path: str = None,
                 encoder_path: str = None):
        """
        Initializes the MetaModel with a given probability threshold, number of top classes, and model paths.

        Parameters:
            prob_threshold (float): Minimum cumulative probability to include a class (between 0 and 1).
            top_n (int): Number of top classes to include based on their probabilities.
            model_path (str): Path to the saved model file.
            encoder_path (str): Optional path to the saved label encoder file.

        Raises:
            ValueError: If prob_threshold is not between 0 and 1, or if top_n is not a positive integer.
        """
        if prob_threshold is not None and not (0 <= prob_threshold <= 1):
            raise ValueError("prob_threshold must be between 0 and 1")
        if top_n is not None and top_n <= 0:
            raise ValueError("top_n must be a positive integer")
        self.prob_threshold = prob_threshold
        self.top_n = top_n
        current_directory = os.getcwd()
        # Construct the full path
        model_path = os.path.join(current_directory, model_path)
        self.model = joblib.load(model_path)

        if encoder_path:
            encoder_path = os.path.join(current_directory, encoder_path)
            self.label_encoder = joblib.load(encoder_path)
        else:
            self.label_encoder = None

    def recommendModels(self, input_features: dict) -> list:
        """
        Predicts class probabilities for a single record and returns the top classes based on either
        the probability threshold or the number of top classes specified.

        Parameters:
            input_features (dict): A dictionary of input features for the prediction.

        Returns:
            list: A list of tuples where each tuple contains a class name (or encoded label) and its
                  corresponding normalized probability.
        """
        # Convert input_features dict to DataFrame for prediction
        X_input = pd.DataFrame([input_features])

        # Predict probabilities
        y_pred_prob = self.model.predict_proba(X_input)

        # Get class labels
        class_labels = self.model.classes_

        # Get probabilities for the first record (since input_features is one record)
        prob_array = y_pred_prob[0]

        # Sort probabilities in descending order and get corresponding class labels
        sorted_indices = np.argsort(-prob_array)
        sorted_probs = prob_array[sorted_indices]
        sorted_classes = class_labels[sorted_indices]

        if self.top_n is not None:
            # Return the top_n classes with the highest probabilities
            top_n_probs = sorted_probs[:self.top_n]
            top_n_classes = sorted_classes[:self.top_n]
        else:
            # Accumulate probabilities until the threshold is exceeded
            accumulated_prob = 0
            selected_classes = []
            selected_probs = []

            for prob, cls in zip(sorted_probs, sorted_classes):
                accumulated_prob += prob
                selected_classes.append(cls)
                selected_probs.append(prob)
                if accumulated_prob >= self.prob_threshold:
                    break

            top_n_probs = selected_probs
            top_n_classes = selected_classes

        # Normalize probabilities so that their sum is 1
        total_prob = sum(top_n_probs)
        if total_prob > 0:
            normalized_probs = [prob / total_prob for prob in top_n_probs]
        else:
            normalized_probs = top_n_probs

        # Prepare the final list of class names with normalized probabilities
        # result = list(zip(top_n_classes, normalized_probs))
        # Map encoded labels back to original class names if the label encoder is provided
        if self.label_encoder:
            # Reverse the label encoding
            class_mapping = {i: label for i, label in enumerate(self.label_encoder.classes_)}
            top_n_classes = [class_mapping[cls] for cls in top_n_classes]

        print("recommended models from meta model")
        # top_n_classes = ['QUANTILEREGRESSOR', 'LinearSVR']
        print("-----------------------------")
        return top_n_classes, normalized_probs


if __name__ == "__main__":
    # Example usage:
    import json

    meta_model = MetaModel(prob_threshold=None, top_n=2, model_path='trained_best_model.pkl',
                           encoder_path='label_encoder.pkl')
    input_features = json.load(open('meta_features_example.json'))
    # print(input_features[0])
    models, probs = meta_model.recommendModels(input_features)
