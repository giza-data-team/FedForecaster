from pycaret.classification import *
import pandas as pd
import numpy as np
import joblib
import os

class MetaModelPyCaret:
    def __init__(self, prob_threshold: float = None, top_n: int = None, model_path: str = None,
                 encoder_path: str = None):
        if prob_threshold is not None and not (0 <= prob_threshold <= 1):
            raise ValueError("prob_threshold must be between 0 and 1")
        if top_n is not None and top_n <= 0:
            raise ValueError("top_n must be a positive integer")
        self.prob_threshold = prob_threshold
        self.top_n = top_n
        current_directory = os.getcwd()

        # Create the full path to the file
        model_path = os.path.join(current_directory, model_path)
        # model_path = f"D:/federatedLearning/GizaFederatedLearning/production-pipeline/GizaFederatedML/server_utils/{model_path}"
        self.model = joblib.load(model_path)

        if encoder_path:
            encoder_path = os.path.join(current_directory, encoder_path)
            # encoder_path = f"D:/federatedLearning/GizaFederatedLearning/production-pipeline/GizaFederatedML/server_utils/{encoder_path}"
            self.label_encoder = joblib.load(encoder_path)
        else:
            self.label_encoder = None

    def predict_best_model(self, input_features: dict) -> list:
        # Convert input_features dict to DataFrame for prediction
        X_input = pd.DataFrame([input_features])

        # Predict probabilities
        y_pred_prob = predict_model(self.model, data=X_input, raw_score=True)
        # Get class labels and probabilities
        class_labels = [int(col.split('_')[-1]) for col in y_pred_prob.columns if col.startswith('prediction_score_')]
        prob_array = y_pred_prob[[col for col in y_pred_prob.columns if col.startswith('prediction_score_')]].values[0]
        # Sort probabilities in descending order and get corresponding class labels
        sorted_indices = np.argsort(-prob_array)
        sorted_probs = prob_array[sorted_indices]
        sorted_classes = np.array(class_labels)[sorted_indices]

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
        return top_n_classes, normalized_probs


if __name__ == "__main__":
    # Example usage:
    import json

    # Create MetaModel instance
    meta_model = MetaModelPyCaret(prob_threshold=None, top_n=3, model_path='final_model.pkl',
                           encoder_path='label_encoder.pkl')
    # Make prediction
    input_features = json.load(open('D:/federatedLearning/GizaFederatedLearning/production-pipeline/GizaFederatedML/server_utils/meta_model/new_model/meta_feature_example.json'))
    predictions , nor= meta_model.predict_best_model(input_features)
    print(predictions)
    print(nor)
