# Random Search Benchmarking for Forecasted Regression Models

This branch contains the code and experimental setup for benchmarking various regression models using random search within the context of a forecasting problem. The models are selected from a predefined search space, and any input data is first transformed into a regression problem before being processed. The project aims to recommend an optimal model for forecasting based on regression metrics.

## Features

- **Search Space Algorithms**: A set of regression models (e.g., XGBoost, Linear SVR, Lasso, Huber Regression, ElasticNet, Quantile Regressor) is used to identify the best model for forecasting tasks.
- **Random Search Implementation**: The `random_search.py` script manages the model selection and hyperparameter tuning using random search.
- **Model Conversion**: Any input data is automatically transformed into a regression problem, making it easier to handle different types of forecasting tasks.
- **Logging**: Detailed logging of model selection and search process.
- **Customizable Parameters**: Easily adjustable search space and hyperparameters.

## Getting Started

To run the random search benchmarking for regression models, follow these steps:

### Prerequisites

- Python 3.12.3 or higher
- Required libraries such as XGBoost, scikit-learn, and others.

### Setup

1. Clone this repository:

   ```bash
   git clone https://your-repo-url.git
   cd your-project-directory

### Running the Experiments
 
1. Execute the main benchmarking script:
   ```bash
   python run.py
   ```
 
3. The results will be logged in `results.csv`.
 
### Customization

- **Search Space**: Modify the `search_space` in `random_search.py` to adjust the models included in the search process. By default, the following models are used:
  - **XGBoost**
  - **Linear SVR**
  - **Lasso**
  - **Huber Regressor**
  - **ElasticNet**
  - **Quantile Regressor**

- **Data Preprocessing**: The script automatically converts any input data into a regression problem, but you can customize the preprocessing logic by modifying the `FeatureExtraction` and `FeatureEngineerin` pipeline classes  in `client_utils`.

For any questions or feedback, please feel free to open an issue or submit a pull request.
