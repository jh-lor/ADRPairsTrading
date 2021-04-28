# ADRPairsTrading

## Data Pulling and Engineering
- `historical_data` contains the stock and forex data used in backtesting.
- `historical_data/pull_data.ipynb` is used to select our universe of stock-ADR pairs, and to extract historical stock and forex data of interest.
- `historical_data/filter_forex.ipynb` is used to filter the forex data pulled, for backtesting.
- `market_metadata` contains the interest rate and vix data used to select a suitable outsample period.
- `market_metadata/pull_vix.ipynb` and `market_metadata/generate_interest_rates.ipynb` are used to obtain interest rate and vix data over our backtesting period.

## Backtesting
- `strategies.py` contains the code to run all 4 variants of our pairs trading strategy, with local risk measures and realistic market assumptions, on a single pair.
- `helpers.py` contains code that estimates the ADR:ORD ratio, merges raw data into a single dataframe, and calculates useful statistics.
- `market_metadata/select_outsample.ipynb` is used to select our outsample period used in backtesting.
- `local_pairs_hp_tune.ipynb` conducts hyperparameter tuning to determine the best hyperparameters for each adr-stock pair, for each variant, on our in-sample period.
- `final_strategy.ipynb` selects the optimal variant, selects the pairs we wish to trade in our portfolio, defines the strategy that trades multiple pairs at once, incorporates global risk measures for the entire portfolio, conducts hyperparameter tuning for portfolio allocation, and reports final results. 

## Live Trading
- `live_trading_implementation.ipynb` updates data periodically, decides what trades to make and is used to place trades.

## Visualizations
- `Visualizations.ipynb` generates all the graphs that we use in our final report, which are stored in the `visualizations` folder.
