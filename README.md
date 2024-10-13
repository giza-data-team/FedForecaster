## Overview

FedForecaster automates the complete pipeline of time-series forecasting, including feature engineering, algorithm selection, and hyperparameter tuning, all while ensuring that data remains decentralized. The engine employs a meta-model trained on a diverse collection of synthetic and real univariate time-series datasets to recommend the best forecasting algorithms based on aggregated statistical meta-features from multiple clients.


## Features

- **Model Conversion**: Any input data is automatically transformed into a regression problem, making it easier to handle different types of forecasting tasks.
- **Logging**: Detailed logging of model selection and search process.
- **Federated Learning Integration**:The FedForecaster model operates within a federated learning context, receiving input from multiple clients and aggregating their meta-features to recommend models.
- **Top 3 Model Recommendation**:Using aggregated meta-features, the system recommends the top 3 models for forecasting tasks based on previously trained models.
- **Time-Budgeted Optimization**:The available time budget is dynamically split across the top 3 recommended models. Each model undergoes optimization of hyperparameters after each round to improve performance.
- **Final Best Model**:At the end of the optimization process, the best model with the best hyperparameters is saved.

## Getting Started

To run the random search benchmarking for regression models, follow these steps:

### Prerequisites

- Python 3.12.3 or higher
- Required libraries such as XGBoost, scikit-learn, and others.

### Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/giza-data-team/FedForecaster.git
   cd FedForecaster

### Running the Experiments
 
2. Execute the main benchmarking script:
   ```bash
   python run.py
   ```
### WorksFlow
1. **Data Preprocessing**: Apply Feature Extraction and Engineering to the time-series data to convert it into a regression data.
2. **Meta-Feature Aggregation**: After preprocessing, the system aggregates meta-features from multiple clients in a federated learning setup. These meta-features are used to evaluate and recommend the top 3 models from the trained models.
3. **Model Recommendation**: The system recommends the top 3 models based on the aggregated meta-features, then distributes the time budget across these models for further optimization.
4. **Hyperparameter Optimization**: Each model’s hyperparameters are optimized after each round using the given time budget to find the best combination of model and hyperparameters.
5. **Final Selection**: After the time budget is exhausted, the model with the best performance and hyperparameters is selected and saved as the final result.
6. **Result Logging**: The results, including the best model and its hyperparameters, are logged for future reference.


4. For any questions or feedback, please feel free to open an issue or submit a pull request.
