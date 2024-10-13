# N-BEATS Benchmarking for FedForecaster

This branch contains the code and experimental setup for benchmarking the N-BEATS algorithm within the context of the FedForecaster, a novel automated machine learning (AutoML) engine designed for univariate time-series forecasting in federated learning (FL) environments.

## Features

- **Experiment Management**: The `run.py` script manages the execution of experiments, including server and client processes.
- **Logging**: Comprehensive logging monitors experiment progress and diagnose issues.
- **Flexible Configuration**: Easily configurable parameters for customizing the benchmarking process.
- **Signal Handling**: Graceful termination of processes upon receiving termination signals.

## Getting Started

To run the benchmarking experiments, follow these steps:

### Prerequisites

- Python 3.12.3
- Required libraries (e.g., Flower framework, TensorFlow)

### Setup

1. Clone this repository

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Experiments

1. Execute the main benchmarking script:
   ```bash
   python run.py
   ```

3. The results will be logged in `results.csv`.

### Customization

- Modify the `n_clients_in_experiments` list in `run.py` to change the number of clients for experiments.
- Adjust other parameters such as `ROUND_NUMBER`, `MAX_TIME_MINUTES`, and `SERVER_START_DELAY` as needed.

For any questions or feedback, please feel free to open an issue or submit a pull request.