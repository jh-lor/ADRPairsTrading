import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

def get_ratio(country, adr, fx_dict):
    """
    Convert stock price to USD and estimate the number of stocks each ADR share represents.
    """
    adr_path = f'eric_jh_data/{country}/{adr}/adr.csv'
    stock_path =  f'eric_jh_data/{country}/{adr}/underlying.csv'
    fx_path = fx_dict[country][0]
    fx_type =  fx_dict[country][1]

    adr_df = pd.read_csv(adr_path, index_col = 0).rename(columns = {'close':'adr_close', 'open':'adr_open'})
    stock_df = pd.read_csv(stock_path, index_col = 0).rename(columns = {'close':'stock_close', 'open':'stock_open'})
    fx_df = pd.read_csv(fx_path, index_col = 0)

    merged_df = pd.merge(adr_df.loc[:,['date', 'adr_open','adr_close']], stock_df.loc[:,['date', 'stock_open','stock_close']])
    merged_df = pd.merge(merged_df, fx_df)

    if fx_type == 1:
        merged_df['stock_open_usd'] = merged_df['stock_open']/((merged_df['avg_bid_non_us_at'] + merged_df['avg_ask_non_us_at'])/2)
    else:
        merged_df['stock_open_usd'] = merged_df['stock_open']*((merged_df['avg_bid_non_us_at'] + merged_df['avg_ask_non_us_at'])/2)
    merged_df["ratio"] = merged_df['stock_open_usd']/merged_df['adr_close']
    
    ratio_geq_1 = True
    if np.mean(merged_df["ratio"] < 1):
        merged_df["ratio"] = 1/merged_df["ratio"]
        ratio_geq_1 = False
    
    return ratio_geq_1, np.round(np.mean(merged_df["ratio"]), 4)

def data_processing(country, adr, fx_dict, forex_bps = 10, adjust_forex_expense = True):
    """
    Return a consolidated dataframe for each adr-stock pair.
    """
    adr_path = f'eric_jh_data/{country}/{adr}/adr.csv'
    stock_path =  f'eric_jh_data/{country}/{adr}/underlying.csv'
    ratio_path = f'eric_jh_data/{country}/{adr}/ratio.csv'
    fx_path = fx_dict[country][0]
    fx_type =  fx_dict[country][1]

    adr_df = pd.read_csv(adr_path, index_col = 0).rename(columns = {'close':'adr_close', 'open':'adr_open', 'volume' : 'adr_volume'})
    stock_df = pd.read_csv(stock_path, index_col = 0).rename(columns = {'close':'stock_close', 'open':'stock_open', 'volume' : 'stock_volume'})
    fx_df = pd.read_csv(fx_path, index_col = 0)
    ratio_df = pd.read_csv(ratio_path, index_col = 0)

    # Invert fx data so that all prices are reflected in USD
    if fx_type == 0:
        inverted_fx_df = 1/fx_df.iloc[:,[2,1,3,5,4,6,8,7,9,11,10,12]].copy()
        inverted_fx_df.columns = fx_df.columns[1:-1]
        fx_df.iloc[:,1:-1] = inverted_fx_df
    merged_df = pd.merge(adr_df.loc[:,['date', 'adr_open','adr_close', 'adr_volume']], stock_df.loc[:,['date', 'stock_open','stock_close', 'stock_volume']])
    merged_df = pd.merge(merged_df, fx_df)
    ratio_geq_1, ratio = ratio_df["ratio_geq_1"].item(), ratio_df["ratio"].item()

#     ratio is (stock price in USD)/(ADR price)
#     If ratio >= 1, for one "unit" of trade, we shall buy 1 stock, and sell multiple adrs
#     If ratio < 1, for one "unit" of trade, we shall sell 1 adr, and buy multiple stocks
    if ratio_geq_1:
        merged_df["stock_num_per_unit"] = 1
        merged_df["adr_num_per_unit"] = ratio
        merged_df["stock_open_per_unit"] = merged_df["stock_open"]
        merged_df["stock_close_per_unit"] = merged_df["stock_close"]
        merged_df["adr_open_per_unit"] = merged_df["adr_open"]*ratio
        merged_df["adr_close_per_unit"] = merged_df["adr_close"]*ratio
    else:
        merged_df["stock_num_per_unit"] = ratio
        merged_df["adr_num_per_unit"] = 1
        merged_df["stock_open_per_unit"] = merged_df["stock_open"]*ratio
        merged_df["stock_close_per_unit"] = merged_df["stock_close"]*ratio
        merged_df["adr_open_per_unit"] = merged_df["adr_open"]
        merged_df["adr_close_per_unit"] = merged_df["adr_close"]    
    
    if adjust_forex_expense:
        # Added expense for trading small amounts in forex market
        forex_bid_multiplier = 1 - 0.0001*forex_bps
        forex_ask_multiplier = 1 + 0.0001*forex_bps
        merged_df.loc[:,merged_df.columns.str.contains("bid")] *= forex_bid_multiplier
        merged_df.loc[:,merged_df.columns.str.contains("ask")] *= forex_ask_multiplier
        
    merged_df["adr_volume"] *= 100
    
    return merged_df

def calc_max_drawdown(portfolio_values, method = "percentage"):
    """
    Returns Max Drawdown, the maximum observed loss from a peak to a trough of a portfolio, before a new peak is attained.
    """
    peak, trough = portfolio_values[0], portfolio_values[0]
    max_drawdown = 0
    for i in range(1, len(portfolio_values)):
        if portfolio_values[i] < trough:
            trough = portfolio_values[i]
            if method == "percentage":
                max_drawdown = max(max_drawdown, (peak - trough)/peak)
            else:
                max_drawdown = max(max_drawdown, peak - trough)
        elif portfolio_values[i] > peak:
            peak, trough = portfolio_values[i], portfolio_values[i]
    return max_drawdown

def get_risk_statistics(stock_values, adr_values, var_ci):
    """
    Given a set of values for foreign stocks (long position), and ADR shares (short position), calculate various risk statistics.
    """
    port_stock = stock_values - adr_values
    port = pd.DataFrame(data = port_stock)
    port_diff = port - port.shift(1)
    pnl = pd.DataFrame(port_diff).dropna()
    sigma = pnl.std()[0]
    pnl['pct_rank'] = pnl.rank(pct=True)
    pnl.columns =['daily_pl', 'pct_rank']
    var = abs(pnl[pnl.pct_rank< 1-var_ci].daily_pl.max())
    max_drawdown_abs = calc_max_drawdown(port_stock, "absolute")
    return sigma, var, max_drawdown_abs

def plot_returns(dates, portfolio_values, num_xticks = 5):
    """
    Plot returns given a set of dates and corresponding portfolio values.
    """
    plt.plot(portfolio_values)
    xticks_indices = np.arange(0, len(dates), (len(dates)-1)// num_xticks)
    plt.xticks(xticks_indices, itemgetter(*xticks_indices)(dates), rotation = 45)
    plt.show();
    
def calc_sharpe(portfolio_values):
    """
    Calculate the Sharpe Ratio given a set of portfolio values over a period of time.
    """
    portfolio_values = np.array(portfolio_values)
    returns = (portfolio_values[1:] - portfolio_values[:-1])/portfolio_values[:-1]
    return np.round(np.sqrt(252)*np.mean(returns)/np.std(returns), 2)
    
def report_and_store_statistics(pairs_trade_strategy, filename, list_pairs, fx_dict):
    for (country, adr) in list_pairs:
        merged_df = data_processing(country, adr, fx_dict)
        ret, trade_records, portfolio_values, hits, dates = pairs_trade_strategy(merged_df)
        ret = np.round(ret*100, 2)
        hit_ratio = None
        logs = [f'The return of ADR_underlying pairs trading for {adr} from {country} is {0.00}%, no trades were placed.\n']
        if hits:
            hit_ratio = np.round(np.mean(hits)*100,2)
            max_drawdown = np.round(calc_max_drawdown(portfolio_values)*100,2)
            logs = [f'The return of ADR_underlying pairs trading for {adr} from {country} is {ret}%\nThe hit ratio is {hit_ratio}%\nThe max drawdown is {max_drawdown}%\n']
            print("Country: {}, ADR_Stock: {}, Return: {}%, Hit Ratio: {}%, Max Drawdown: {}%".format(country, adr, ret, hit_ratio, max_drawdown))
        else:
            print("Country: {}, ADR_Stock: {}, Return: {}%, Hit Ratio: None, Max Drawdown: 0.00%".format(country, adr, ret))
        logs = logs + trade_records 
        fname = f'eric_jh_data/{country}/{adr}/logs/' + filename
        f = open(fname, 'w')
        f.writelines(logs)
        f.close()
    return dates, portfolio_values