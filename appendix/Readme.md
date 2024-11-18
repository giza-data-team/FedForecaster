# Appendix

## Grid Search Combinations During Knowledge Base Construction
The grid search conducted in the knowledge base construction exists in [Grid Search Space](https://github.com/giza-data-team/FedForecaster/blob/FedForecaster/appendix/SearchSpace.csv).

| Model              | Library/Implementation                | Hyperparameters                                                                                                          | Runs |
|---------------------|---------------------------------------|-------------------------------------------------------------------------------------------------------------------------|------|
| Lasso              | `sklearn.linear_model.Lasso`          | `{'alpha': np.logspace(np.log10(1e-5), np.log10(2), num=30), 'selection': ['cyclic', 'random']}`                         | 60   |
| LinearSVR          | `sklearn.svm.LinearSVR`               | `{'C': [1, 2, 3, 5, 10], 'epsilon': [0.01, 0.05, 0.1]}`                                                                 | 15   |
| ElasticNetCV       | `sklearn.linear_model.ElasticNetCV`   | `{'l1_ratio': np.linspace(0.3, 1, 10), 'selection': ['cyclic', 'random']}`                                              | 20   |
| XGBRegressor       | `xgboost.XGBRegressor`                | `{'learning_rate': [0.1, 1], 'reg_lambda': [0.8, 10], 'gamma': [0.9, 1.16467595, 2.248149123539492, 3.9963209507789], 'subsample': [0.1, 1]}` | 32   |
| HuberRegressor     | `sklearn.linear_model.HuberRegressor` | `{'epsilon': [1.0, 1.35, 1.5], 'alpha': np.logspace(np.log10(1e-3), np.log10(1e2), num=10)}`                             | 30   |
| QuantileRegressor  | `sklearn.linear_model.QuantileRegressor` | `{'alpha': np.logspace(np.log10(1e-3), np.log10(1e2), num=10), 'quantile': [0.1, 0.25, 0.5, 0.75]}`                     | 40   |
| **Total**          |                                       |                                                                                                                         | **197** |

## Characteristics of the Benchmarking Datasets
The characteristics of the benchmark datasets are calculated in [Benchmark Datasets Characteristics](https://github.com/giza-data-team/FedForecaster/blob/FedForecaster/appendix/Benchmark%20Datasets%20Characteristics.csv).
The benchmark datasets have different numbers of instances count, features count, skewness & kurtosis of features, stationary features count, significant lags count, entropy / fractual dimensionality of the target.


## Comparison to Centralized Approach
Below is the performance Comparison of `FedForecaster`, `Random Search`, and `N-Beats` on 12 Datasets in terms of MSE. We have added a new column for `N-beats centralized` which shows that there exists a performance gain on N-beats when the dataset size increases showing the N-beats starts to work better on larger datasets to recognize patterns in the dataset.

| **Dataset**                           | **Len.** | **Clients** | **FedForecaster** | **Random Search** | **N-Beats** | **N-Beats Centralized** | **Best Model**       |
|---------------------------------------|----------|-------------|--------------------|-------------------|-------------|--------------------------|----------------------|
| BOE-XUDLERD                           | 15653    | 20          | <b>0.006</b>          | 0.011             | 0.071       | <b>0.00402</b>               | HuberRegressor       |
| SunSpotDaily                          | 73924    | 20          | <b>29.372</b>         | 32.069            | 63.384      | <b>16.51</b>              | Lasso                |
| USBirthsDaily                         | 7305     | 5           | <b>434.891</b>        | 533.366           | 983.357     | 820.02               | LinearSVR            |
| nasdaq_Brazil_Base_Financial_Rate     | 10091    | 10          | 0.058              | <b>0.048</b>         | 0.153       | <b>0.031</b>              | LinearSVR            |
| nasdaq_Brazil_Pr_Base_Financial_Rate  | 10091    | 15          | <b>0.008</b>          | 0.012             | 0.008       | <b>0.0014</b>              | HuberRegressor       |
| nasdaq_Brazil_Saving_Deposits1        | 812      | 5           | <b>0.028</b>          | 0.039             | 0.412       | <b>0.0252</b>              | Lasso                |
| nasdaq_Brazil_Saving_Deposits2        | 1182     | 10          | <b>0.020</b>          | 0.025             | 0.024       | <b>0.0057</b>              | XGBRegressor         |
| nasdaq_EIA_PET_RWTC                   | 9124     | 5           | <b>1.291</b>          | 1.404             | 8.663       | <b>1.108</b>               | LinearSVR            |
| nasdaq_WIKI_AAPL_Price                | 9124     | 15          | <b>3.761</b>          | 4.242             | 4.145       | <b>3.991</b>              | LinearSVR            |
| Energy Select Sector ETF              | 2517     | 10          | 3.441              | <b>2.867</b>         | 24.611      | 24.611                   | Lasso                |
| The Technology Sector ETF             | 2517     | 10          | <b>39.996</b>         | 101.702           | 75.980      | 75.980                   | QuantileRegressor    |
| Utilities Select Sector ETF           | 2517     | 10          | <b>1.297</b>          | 11.701            | 17.577      | 17.577                   | HuberRegressor       |


## Benchmarking Experiments with all clients' counts
The below table summarizes the MSE results for the benchmark datasets over all possible clients count keeping in mind that data sets with splits <500 instances are discarded.
# Performance Comparison of `FedForecaster`, Random Search, and `N-Beats` on Various Datasets

| **Dataset Name**                       | **Clients** | **FedForecaster** | **Random Search** | **N-Beats**         |
|---------------------------------------|-------------|--------------------|-------------------|---------------------|
| BOE-XUDLERD                           | 5           | <b>0.0261</b>         | 0.0282       | 0.2176            |
| BOE-XUDLERD                           | 10          | 0.0191         | <b>0.0142<b>       | 0.0120            |
| BOE-XUDLERD                           | 15          | <b>0.0116</b>        | 0.0140       | 0.0713            |
| BOE-XUDLERD                           | 20          | <b>0.006</b>          | 0.011             | 0.071               |
| SunSpotDaily                          | 5           | <b>24.554</b>             | 24.839955         | 34.34048            |
| SunSpotDaily                          | 10          | <b>29.301</b>             | 31.521126         | 54.949              |
| SunSpotDaily                          | 15          | <b>25.804</b>             | 26.30128          | 63.512              |
| SunSpotDaily                          | 20          | <b>29.372</b>         | 32.069            | 63.384              |
| USBirthsDaily                         | 5           | <b>434.891</b>        | 533.366           | 983.357             |
| USBirthsDaily                         | 10          | <b>535.652</b>            | 669.647           | 1147.762            |
| nasdaq_Brazil_Base_Financial_Rate     | 5           | 0.0459           | <b>0.0435</b>          | 0.559            |
| nasdaq_Brazil_Base_Financial_Rate     | 10          | 0.058              | <b>0.048</b>         | 0.153               |
| nasdaq_Brazil_Base_Financial_Rate     | 15          | <b>0.04574</b>           | 0.060963          | 0.189575            |
| nasdaq_Brazil_Pr_Base_Financial_Rate  | 5           | 0.00423           | 0.04199           | <b>0.003306</b>            |
| nasdaq_Brazil_Pr_Base_Financial_Rate  | 10          | <b>0.00697</b>            | 0.03215          | 0.00698             |
| nasdaq_Brazil_Pr_Base_Financial_Rate  | 15          | <b>0.008</b>          | 0.011775          | 0.008495            |
| nasdaq_Brazil_Saving_Deposits1        | 5           | <b>0.028</b>          | 0.039467          | 0.412               |
| nasdaq_Brazil_Saving_Deposits2        | 5           | <b>0.0102</b>           | 0.0211                 | 0.191782            |
| nasdaq_Brazil_Saving_Deposits2        | 10          | <b>0.020</b>          | 0.02536           | 0.024               |
| nasdaq_Brazil_Saving_Deposits2        | 15          | 0.0420           | <b>0.0363</b>          | 0.076547            |
| nasdaq_EIA_PET_RWTC                   | 5           | <b>1.291</b>          | 1.404             | 8.663               |
| nasdaq_EIA_PET_RWTC                   | 10          | <b>1.313</b>           | 2.100          | 1.529               |
| nasdaq_EIA_PET_RWTC                   | 15          | <b>1.3458</b>           | 2.093          | 1.544               |
| nasdaq_WIKI_AAPL_Price                | 5           | 18.609             | 19.595            | <b>17.856</b>            |
| nasdaq_WIKI_AAPL_Price                | 10          | <b>16.564</b>             | 54.534091         | 18.063154            |
| nasdaq_WIKI_AAPL_Price                | 15          | <b>3.761</b>          | 4.242377          | 4.145               |
| Energy Select Sector ETF              | 10          | 3.441              | <b>2.867</b>         | 24.611              |
| The Technology Sector ETF             | 10          | <b>39.996</b>         | 101.702           | 75.980              |
| Utilities Select Sector ETF           | 10           | <b>1.297</b>          | 11.701            | 17.577              |


## Feature Selection Threshold
Using accumulative feature importance of 80%, 90%, 95%, 98% results in feature losses of 38.7, 38.0, 36.5 and 36.1 respectively. Although increasing the features results in better losses value as the datset includes better information, there is a reduction in important features count and hence less iteration time 
| Accumulative Importance | Average of test_loss | Average of Count_Features | Avg iteration time |
|--------------------------|----------------------|----------------------------|-------------------------------------|
| 0.98                    | 36.11764665         | 7.3                        | 2.749494216                         |
| 0.95                    | 36.52490576         | 4.2                        | 1.197480258                         |
| 0.9                     | 37.97948914         | 3.7                        | 1.487262682                         |
| 0.8                     | 38.71268618         | 3.3                        | 1.51576263                          |
The detailed results of these experiments on the benchmark datasets exists here [Results with different feature selection threshold](https://github.com/giza-data-team/FedForecaster/blob/FedForecaster/appendix/Result_with_diff_threshold.csv).

## Offline Cost
### Knowledge base Construction
The total time consumed over all clients for running the grid search on the knowledge base records on average is summarized in the below table:
| Datasets with Clients Count # | Total Time Consumed (Grid Search) |
|-----------------------|-----------------------------------|
| 5             | 732.6015                         |
| 10            | 1073.338                         |
| 15            | 1597.956                         |
| 20            | 1959.802                         |
| **Average**   | **1340.924**                     |

### Meta-Features Extraction Per Client
The average meta-feature extraction time per client across all benchmarking datasets is: <b>2.7377</b> seconds.
