# ADRPairsTrading

## Data Pulling and Engineering
- `eric/jh_data` contains the stock and forex data used in backtesting.
- `eric_jh_data/pull_data.ipynb` is used to extract historical stock and forex data of interest.
- `eric_jh_data/filter_forex.ipynb` is used to filter the forex data pulled, for backtesting.

## Backtesting
- `strategies.py` contains the code to run all 4 variants of our pairs trading strategy, on a single pair.
- `helpers.py` contains code that estimates the ADR:ORD ratio, merges raw data into a single dataframe, and calculates useful statistics.
- `local_pairs_hp_tune.ipynb` conducts hyperparameter tuning to determine the best hyperparameters for each adr-stock pair, for each variant.
- `final_strategy.ipynb` selects the optimal variant, selects the pairs we wish to trade, defines the strategy that trades multiple pairs at once, incorporates global risk measures for the entire portfolio, conducts hyperparameter tuning for portfolio allocation, and reports final results. 

## Live Trading
- `Live Trading Implementation.ipynb` updates data, decides what trades to do and places trades.
- `Live-trading-monitoring.ipynb` contains useful code
