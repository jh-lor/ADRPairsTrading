import numpy as np
import pandas as pd
from collections import deque
from helpers import *

def pairs_trade_v1(merged_df, lookback = 100, cash = 250000, entry = 1, exit = 0, stop_loss = 3, 
                   start_date = "2016-01-01", end_date = "2021-01-31", slippage_bps = 10, 
                   borrowing_bps = 50, risk_lookback = 100, var_ci = 0.95, var_limit = 0.1, max_drawdown_limit = 0.2, 
                   sigma_limit = 0.05, maximum_holding_period = 30, volume_lookback = 5):
    
    """
    Variant 1 - Begin each trade on Asian market open.


    For each row:
        The following events occur in order: stock_open, stock_close, adr_open, adr_close
        After these 4 events, assess entry/exit condition (right before the Asian market opens ~ 6.59PM EST)
        To open a position, we check the close price of adr, compared it to close px of stock of the same row. 
        To close a position,  we check the CLOSE price of adr, compared it to CLOSE px of stock of the same row. 
        Place trade on next row (First trade stock on Asian market open, then trade ADR on US market open)

    lookback -- length of lookback window (in trading days) used to calculate z-score 
    cash -- initial equity 
    entry, exit, stop_loss -- hyperparameters used to determine entry and exit conditions
    start_date - first date (EST) we may place a trade
    end_date - last date (EST) we may place a trade
    slippage_bps - BPS accounted for for each trade
    borrowing_bps - BPS accounted for for shorting ADR shares
    risk_lookback -- length of lookback window (in trading days) used for risk metrics
    var_ci -- percentile used in VaR calculation
    var_limit -- percentage of current equity that is the upper limit for historical VaR
    max_drawdown_limit -- percentage of initial equity that is the upper limit for historical max drawdown
    sigma_limit -- percentage of current equity that is the upper limit for historical PnL volatility
    maximum_holding_period -- Maximum holding period (in trading days) of a adr-stock pair, before liquidation
    volume_lookback -- length of lookback window (in trading days) used to estimate Average Daily Trading Volume
    """
    
    # Accounts for slippage and transaction costs
    short_multiplier = 1 - 0.0001*slippage_bps
    long_multiplier = 1 + 0.0001*slippage_bps
    starting_cash = cash
    stock_pos, adr_pos = 0, 0
    # For book-keeping, since we shall store the portfolio value of the day before
    prev_cash, prev_adr_pos = cash, adr_pos
    forex_cash = 0
    holding_period = None
    diff_record = deque(maxlen = lookback)
    trade_records = []
    # Stores value of portfolio at each date from one day before the start_date, to the end_date (inclusive), when Asian market opens (EST ~ 7:00 PM)
    portfolio_values = []
    dates = []
    hits = []
    
    # Make sure that merged_df before end date is not empty
    if merged_df[merged_df['date'] < end_date].empty:
        return 0, trade_records, portfolio_values, hits, dates

    for index, row in merged_df.iterrows():

        if index+1 < len(merged_df) and index > 0:
            
            # Log portfolio value of the day before, when the Asian market opens
            prev_date = merged_df.loc[index - 1, "date"]
            if row["date"] >= start_date and prev_date <= end_date:
                dates.append(prev_date)
                # Convert equity in foreign stocks and currency to USD
                prev_forex_value = forex_cash + stock_pos*row["stock_open"]
                if prev_forex_value > 0:
                    prev_forex_value /= row['avg_ask_non_us_at']
                else:
                    prev_forex_value /= row['avg_bid_non_us_at']
                
                portfolio_values.append(prev_cash + prev_adr_pos*merged_df.loc[index - 1, 'adr_close'] 
                                        + prev_forex_value)

            # Before the Asian Market open, append price difference between ADR and stock
            diff_record.append(row['adr_close_per_unit'] 
                                   - row['stock_close_per_unit']/merged_df.loc[index+1,'avg_non_us_before'])

            # We place one trade the day itself (Asian), one trade the day after (US)
            if len(diff_record) < lookback or row["date"] < start_date or merged_df.loc[index+1, "date"] > end_date:
                continue
            
            # Account for borrowing cost
            if stock_pos > 0:
                holding_period += 1
                cash -= 0.0001*borrowing_bps*(1/252)*abs(adr_pos)*merged_df.loc[index - 1, 'adr_close']
                multiplier = (1 + 0.01*(2 + row["ir"])*(1/252))
                forex_cash *= multiplier
            prev_cash, prev_adr_pos = cash, adr_pos

            mean = np.array(diff_record).mean()
            std = np.array(diff_record).std()
            
            # Entry condition
            if diff_record[-1] > mean + entry*std and diff_record[-1] <= mean + stop_loss*std:
                if stock_pos == 0 and adr_pos == 0:
                    portfolio_value_before_entering = cash
                    # Allow ourselves to trade 20% of ADT volume over the past 5 trading days
                    # We take the median to make this estimate more robust to extreme values
                    adr_volume = 0.2*(merged_df.loc[index-volume_lookback+1:index,:]["adr_volume"].median()/row["adr_num_per_unit"])
                    stock_volume = 0.2*(merged_df.loc[index-volume_lookback+1:index,:]["stock_volume"].median()/row["stock_num_per_unit"])
                    units = int(min(cash/row['adr_close_per_unit'],
                                    cash/(row['stock_close_per_unit']/merged_df.loc[index+1,'avg_non_us_before']), 
                                    adr_volume, 
                                    stock_volume))
                    adr_quantity = int(units*row["adr_num_per_unit"])
                    stock_quantity = int(units*row["stock_num_per_unit"])
                    
                    # Take portfolio value for each previous day (till today) right before the Asian market opens
                    # Further adjust volume based on historical max drawdown, VaR and PnL volatility
                    temp_risk_lookback = min(risk_lookback, index)
                    current = merged_df.loc[(index - temp_risk_lookback + 1):index].copy()
                    next_day = merged_df.loc[(index - temp_risk_lookback + 2):(index + 1)].copy()
                    stock_values = (np.array((current["stock_close"])/np.array(next_day["avg_non_us_before"]))*stock_quantity) 
                    adr_values = np.array(current["adr_close"]*adr_quantity)
                    sigma, var, max_drawdown_abs = get_risk_statistics(stock_values, adr_values, var_ci)
                    if (var > portfolio_value_before_entering*var_limit or 
                        max_drawdown_abs > max_drawdown_limit*starting_cash or 
                        sigma > portfolio_value_before_entering*sigma_limit):
                        frac = min((portfolio_value_before_entering*var_limit)/var, 
                                   (max_drawdown_limit*starting_cash)/max_drawdown_abs,
                                  (portfolio_value_before_entering*sigma_limit)/sigma)
                        units = int(frac*units)
                        if units == 0:
                            continue
                        adr_quantity = int(units*row["adr_num_per_unit"])
                        stock_quantity = int(units*row["stock_num_per_unit"])                        
                    
                    stock_pos += stock_quantity
                    stock_px_fx = merged_df.loc[index+1,'stock_open']*long_multiplier
                    forex_cash -= stock_px_fx*stock_quantity
                    # We store the current cash/adr position, because the trade below will occur on the next day (EST)
                    prev_cash, prev_adr_pos = cash, adr_pos
                    
                    adr_pos -= adr_quantity
                    adr_px = merged_df.loc[index+1,'adr_open']*short_multiplier
                    cash += adr_quantity*adr_px
                    
                    holding_period = 0
                    trade_records.append("Opening positions:\n")
                    # Times in EST
                    trade_records.append(f"We bought {stock_quantity} shares of underlying stock at the price of {stock_px_fx} foreign dollars on {row['date']}\n")
                    trade_records.append(f"We sold {adr_quantity} shares of ADR at the price of {adr_px} on {merged_df.loc[index+1,'date']}\n")

            # Liquidation condition
            elif (diff_record[-1] < mean + exit*std or 
                  diff_record[-1] > mean + stop_loss*std or 
                  holding_period == maximum_holding_period):
                if stock_pos > 0 and adr_pos < 0 : 
                    stock_px_fx = merged_df.loc[index+1,'stock_open']*short_multiplier
                    forex_cash += stock_px_fx*stock_pos
                    if forex_cash > 0:
                        forex_cash /= merged_df.loc[index+1,'avg_ask_non_us_at']
                    else:
                        forex_cash /= merged_df.loc[index+1,'avg_bid_non_us_at']
                    cash += forex_cash
                    forex_cash = 0
                    
                    # We store the current cash/adr position, because the trade below will occur on the next day (EST)
                    prev_cash, prev_adr_pos = cash, adr_pos
                    
                    adr_px = merged_df.loc[index+1,'adr_open']*long_multiplier
                    cash -= abs(adr_pos)*adr_px
                    trade_records.append("Closing positions:\n")
                    # Times in EST
                    trade_records.append(f"We sold {stock_pos} shares of underlying stock at the price of {stock_px_fx} foreign dollars on {row['date']}\n")
                    trade_records.append(f"We bought {-adr_pos} shares of ADR at the price of {adr_px} on {merged_df.loc[index+1,'date']}\n")
                    stock_pos, adr_pos = 0, 0
                    holding_period = None
                    if cash > portfolio_value_before_entering:
                        hits.append(1)
                    else:
                        hits.append(0)

    ret = (portfolio_values[-1] - starting_cash)/starting_cash
    
    return ret, trade_records, portfolio_values, hits, dates

"""
Variant 2 - Begin each trade on US market open (Evaluate after Asian market closes)

To open a position, we check the CLOSE price of adr of the previous row, compared to CLOSE px of 
stock of the current row. We buy the stock on the next trading OPEN for Asian/US market

To close a position, we check the CLOSE price of adr of the previous row, compared to CLOSE px of 
stock of the current row. We sell the stock on the next trading next OPEN for Asian/US market

For each row:
    stock_open, stock_close, (assess), adr_open, adr_close
    After first 2 events events, assess condition (right before the US market opens ~ 9.29AM EST)
    Place trade on current and next row (First trade ADR on US market open, then trade stock on Asian market open)
    
start_date: First date (EST) we may place a trade
end_date: Last date (EST) we may place a trade
portfolio_values: Stores value of portfolio at each date from one day before the start_date, to the end_date, when the Asian market opens
"""
def pairs_trade_v2(merged_df, lookback = 100, cash = 250000, entry = 1, exit = 0, stop_loss = 3, 
                   start_date = "2016-01-01", end_date = "2021-01-31", slippage_bps = 10, 
                   borrowing_bps = 50, risk_lookback = 100, var_ci = 0.95, var_limit = 0.1, max_drawdown_limit = 0.2, 
                   sigma_limit = 0.05, maximum_holding_period = 30, volume_lookback = 5):
    
    # Accounts for slippage and transaction costs
    short_multiplier = 1 - 0.0001*slippage_bps
    long_multiplier = 1 + 0.0001*slippage_bps
    starting_cash = cash
    stock_pos, adr_pos = 0, 0
    holding_period = None
    forex_cash = 0
    diff_record = deque(maxlen = lookback)
    trade_records = []
    portfolio_values = []
    dates = []
    hits = []
    
    # Make sure that merged_df before end date is not empty
    if merged_df[merged_df['date'] < end_date].empty:
        return 0, trade_records, portfolio_values, hits, dates

    for index, row in merged_df.iterrows():
        
        if index+1 < len(merged_df) and index > 0:
            
            # Add portfolio value for the day before
            prev_date = merged_df.loc[index - 1, "date"]
            if row["date"] >= start_date and prev_date <= end_date:
                dates.append(prev_date)
                prev_forex_value = forex_cash + stock_pos*row["stock_open"]
                if prev_forex_value > 0:
                    prev_forex_value /= row['avg_ask_non_us_at']
                else:
                    prev_forex_value /= row['avg_bid_non_us_at']
                
                portfolio_values.append(cash + adr_pos*merged_df.loc[index - 1, 'adr_close'] 
                                        + prev_forex_value)

            diff_record.append(merged_df.loc[index-1,'adr_close_per_unit'] 
                                   - row['stock_close_per_unit']/row['avg_us_before'])
            
            # We place both trades the day itself
            if len(diff_record) < lookback or row["date"] < start_date or row["date"] > end_date:
                continue

            if stock_pos > 0:
                holding_period += 1
                cash -= 0.0001*borrowing_bps*(1/252)*abs(adr_pos)*merged_df.loc[index - 1, 'adr_close']
                multiplier = (1 + 0.01*(2 + row["ir"])*(1/252))
                forex_cash *= multiplier
                
            mean = np.array(diff_record).mean()
            std = np.array(diff_record).std()
            
            # If we have passed the initial lookback window and are in the specified dates
            # enter the position if diff is significant
            if diff_record[-1] > mean + entry*std and diff_record[-1] <= mean + stop_loss*std:
                if stock_pos == 0 and adr_pos == 0:
                    portfolio_value_before_entering = cash
                    adr_volume = 0.2*(merged_df.loc[index-volume_lookback:index - 1,:]["adr_volume"].median()/row["adr_num_per_unit"])
                    stock_volume = 0.2*(merged_df.loc[index-volume_lookback+1:index,:]["stock_volume"].median()/row["stock_num_per_unit"])
                    units = int(min(cash/merged_df.loc[index-1,'adr_close_per_unit'],
                                    cash/(row['stock_close_per_unit']/row['avg_us_before']), 
                                    adr_volume, 
                                    stock_volume))
                    adr_quantity = int(units*row["adr_num_per_unit"])
                    stock_quantity = int(units*row["stock_num_per_unit"])
                    # Take portfolio value for each previous day (till today) right before the US market opens
                    temp_risk_lookback = min(risk_lookback, index)
                    current = merged_df.loc[(index - temp_risk_lookback + 1):index].copy()
                    stock_values = np.array((current["stock_close"]/current["avg_us_before"])*stock_quantity) 
                    adr_values = np.array(merged_df.loc[(index - temp_risk_lookback):(index-1)]["adr_close"]*adr_quantity)
                    sigma, var, max_drawdown_abs = get_risk_statistics(stock_values, adr_values, var_ci)
                    if (var > portfolio_value_before_entering*var_limit or 
                        max_drawdown_abs > max_drawdown_limit*starting_cash or 
                        sigma > portfolio_value_before_entering*sigma_limit):
                        frac = min((portfolio_value_before_entering*var_limit)/var, 
                                   (max_drawdown_limit*starting_cash)/max_drawdown_abs,
                                  (portfolio_value_before_entering*sigma_limit)/sigma)
                        units = int(frac*units)
                        if units == 0:
                            continue
                        adr_quantity = int(units*row["adr_num_per_unit"])
                        stock_quantity = int(units*row["stock_num_per_unit"])  
                    
                    adr_pos -= adr_quantity
                    adr_px = row['adr_open']*short_multiplier
                    cash += adr_quantity*adr_px
                    
                    stock_pos += stock_quantity
                    stock_px_fx = merged_df.loc[index+1,'stock_open']*long_multiplier
                    forex_cash -= stock_px_fx*stock_quantity
#                     stock_px = stock_px_fx/merged_df.loc[index+1,'avg_bid_non_us_at']
#                     cash -= stock_px*stock_quantity
                    
                    holding_period = 0
                    trade_records.append("Opening positions:\n")
                    # Times in EST
                    trade_records.append(f"We sold {adr_quantity} shares of ADR at the price of {adr_px} on {row['date']}\n")
                    trade_records.append(f"We bought {stock_quantity} shares of underlying stock at the price of {stock_px_fx} foreign dollars on {row['date']}\n")

            # Liquidation condition
            elif (diff_record[-1] < mean + exit*std or 
                  diff_record[-1] > mean + stop_loss*std or 
                  holding_period == maximum_holding_period):
                if stock_pos > 0 and adr_pos < 0 : 
                    adr_px = row['adr_open']*long_multiplier
                    cash -= abs(adr_pos)*adr_px
                    stock_px_fx = merged_df.loc[index+1,'stock_open']*short_multiplier
                    forex_cash += stock_px_fx*stock_pos
                    if forex_cash > 0:
                        forex_cash /= merged_df.loc[index+1,'avg_ask_non_us_at']
                    else:
                        forex_cash /= merged_df.loc[index+1,'avg_bid_non_us_at']
                    cash += forex_cash
                    forex_cash = 0
                    
                    trade_records.append("Closing positions:\n")
                    # Times in EST
                    trade_records.append(f"We bought {-adr_pos} shares of ADR at the price of {adr_px} on {row['date']}\n")
                    trade_records.append(f"We sold {stock_pos} shares of underlying stock at the price of {stock_px_fx} foreign dollars on {row['date']}\n")
                    stock_pos, adr_pos = 0, 0
                    holding_period = None
                    if cash > portfolio_value_before_entering:
                        hits.append(1)
                    else:
                        hits.append(0)

    ret = (portfolio_values[-1] - starting_cash)/starting_cash
    
    return ret, trade_records, portfolio_values, hits, dates

"""
Variant 3a 
- Begin each trade on either US market open or Asian market open
- Regressions are done for the similar "type" of trade
    i.e. if we are entering at a certain time, we do a regression based on the values obtained at the same time each day

For each row:
    stock_open, stock_close, (assess condition 1), adr_open, adr_close, (assess condition 2)
    If not condition 2 - No action taken: 
        After first 2 events, Assess condition 1 (right before the US market opens ~ 9.29AM EST)
        If condition 1:
            Place trade on current and next row (First trade ADR on US market open, then trade stock on Asian market open)
    If not condition 1 - No action taken:
        After next 2 events occur, assess condition 2
        If condition 2:
            Place trade on next row (First trade ADR on Asian market open, then trade stock on US market open)
    
start_date: First date (EST) we may place a trade
end_date: Last date (EST) we may place a trade
portfolio_values: Stores value of portfolio at each date from one day before the start_date, to the end_date, when the Asian market opens
"""
def pairs_trade_v3a(merged_df, lookback = 100, cash = 250000, entry_cond1_val = 1, entry_cond2_val = 1, 
                    exit_cond1_val = 0, exit_cond2_val = 0, stop_loss_cond1 = 3, stop_loss_cond2 = 3, 
                    start_date = "2016-01-01", end_date = "2021-01-31", slippage_bps = 10, 
                    borrowing_bps = 50, risk_lookback = 100, var_ci = 0.95, var_limit = 0.1, max_drawdown_limit = 0.2, 
                    sigma_limit = 0.05, maximum_holding_period = 30, volume_lookback = 5):

    # Accounts for slippage and transaction costs
    short_multiplier = 1 - 0.0001*slippage_bps
    long_multiplier = 1 + 0.0001*slippage_bps
    starting_cash = cash
    stock_pos, adr_pos = 0, 0
    holding_period = None
    trade_type = None
    forex_cash = 0
    # For book-keeping, since we shall store the portfolio value of the day before
    prev_cash, prev_adr_pos = cash, adr_pos
    diff_record_cond1 = deque(maxlen = lookback)
    diff_record_cond2 = deque(maxlen = lookback)
    trade_records = []
    portfolio_values = []
    dates = []
    hits = []
    enter_cond1, exit_cond1, enter_cond2, exit_cond2 = False, False, False, False
    
    # Make sure that merged_df before end date is not empty
    if merged_df[merged_df['date'] < end_date].empty:
        return 0, trade_records, portfolio_values, hits, dates
    
    for index, row in merged_df.iterrows():
                    
        if index+1 < len(merged_df) and index > 0:
            
            # Add portfolio value for the day before
            prev_date = merged_df.loc[index - 1, "date"]
            if row["date"] >= start_date and prev_date <= end_date:
                dates.append(prev_date)
                prev_forex_value = forex_cash + stock_pos*row["stock_open"]
                if prev_forex_value > 0:
                    prev_forex_value /= row['avg_ask_non_us_at']
                else:
                    prev_forex_value /= row['avg_bid_non_us_at']
                portfolio_values.append(prev_cash + prev_adr_pos*merged_df.loc[index - 1, 'adr_close'] 
                                        + prev_forex_value)
            
            # Before US Market Opens
            diff_record_cond1.append(merged_df.loc[index-1,'adr_close_per_unit'] - row['stock_close_per_unit']/row['avg_us_before'])
            # Before Asian Market Opens
            diff_record_cond2.append(row['adr_close_per_unit'] 
                                   - row['stock_close_per_unit']/merged_df.loc[index+1,'avg_non_us_before'])


            # row["date"] is between start_date (inclusive) and end_date (inclusive)
            if len(diff_record_cond1) < lookback or row["date"] < start_date or row["date"] > end_date:
                continue

            # Update cash/adr position after portfolio values has been updated
            if stock_pos > 0:
                holding_period += 1
                cash -= 0.0001*borrowing_bps*(1/252)*abs(adr_pos)*merged_df.loc[index - 1, 'adr_close']
                multiplier = (1 + 0.01*(2 + row["ir"])*(1/252))
                forex_cash *= multiplier
            prev_cash, prev_adr_pos = cash, adr_pos
                
            mean_cond1 = np.array(diff_record_cond1).mean()
            std_cond1 = np.array(diff_record_cond1).std()
            mean_cond2 = np.array(diff_record_cond2).mean()
            std_cond2 = np.array(diff_record_cond2).std()
            
            # If a concurrent trade is not already being placed
            if not (enter_cond2 or exit_cond2):
                enter_cond1 = (diff_record_cond1[-1] > mean_cond1 + entry_cond1_val*std_cond1 
                               and diff_record_cond1[-1] <= mean_cond1 + stop_loss_cond1*std_cond1
                               and stock_pos == 0 and adr_pos == 0)
                exit_cond1 = ((diff_record_cond1[-1] < mean_cond1 + exit_cond1_val*std_cond1 
                              or diff_record_cond1[-1] > mean_cond1 + stop_loss_cond1*std_cond1
                              or (holding_period == maximum_holding_period and trade_type == 1))
                              and stock_pos > 0 and adr_pos < 0)
                    
                if enter_cond1:
                    portfolio_value_before_entering = cash
                    adr_volume = 0.2*(merged_df.loc[index-volume_lookback:index - 1,:]["adr_volume"].median()/row["adr_num_per_unit"])
                    stock_volume = 0.2*(merged_df.loc[index-volume_lookback+1:index,:]["stock_volume"].median()/row["stock_num_per_unit"])
                    units = int(min(cash/row['adr_close_per_unit'],
                                    cash/(row['stock_close_per_unit']/merged_df.loc[index+1,'avg_non_us_before']), 
                                    adr_volume, 
                                    stock_volume))
                    adr_quantity = int(units*row["adr_num_per_unit"])
                    stock_quantity = int(units*row["stock_num_per_unit"])
                    
                    # Take portfolio value for each previous day when the Asian market opens
                    temp_risk_lookback = min(risk_lookback, index)
                    current = merged_df.loc[(index - temp_risk_lookback + 1):index].copy()
                    stock_values = np.array((current["stock_close"]/current["avg_us_before"])*stock_quantity) 
                    adr_values = np.array(merged_df.loc[(index - temp_risk_lookback):(index-1)]["adr_close"]*adr_quantity)
                    sigma, var, max_drawdown_abs = get_risk_statistics(stock_values, adr_values, var_ci)
                    if (var > portfolio_value_before_entering*var_limit or 
                        max_drawdown_abs > max_drawdown_limit*starting_cash or 
                        sigma > portfolio_value_before_entering*sigma_limit):
                        frac = min((portfolio_value_before_entering*var_limit)/var, 
                                   (max_drawdown_limit*starting_cash)/max_drawdown_abs,
                                  (portfolio_value_before_entering*sigma_limit)/sigma)
                        units = int(frac*units)
                        if units == 0:
                            enter_cond1 = False
                        adr_quantity = int(units*row["adr_num_per_unit"])
                        stock_quantity = int(units*row["stock_num_per_unit"])
                    if units != 0:
                                
                        adr_pos -= adr_quantity
                        adr_px = row['adr_open']*short_multiplier
                        cash += adr_quantity*adr_px

                        stock_pos += stock_quantity
                        stock_px_fx = merged_df.loc[index+1,'stock_open']*long_multiplier
                        forex_cash -= stock_px_fx*stock_quantity
                        prev_cash, prev_adr_pos = cash, adr_pos
                        holding_period = 0
                        trade_type = 1
                        
                        trade_records.append("Opening positions:\n")
                        # Times in EST
                        trade_records.append(f"We sold {adr_quantity} shares of ADR at the price of {adr_px} on {row['date']}\n")
                        trade_records.append(f"We bought {stock_quantity} shares of underlying stock at the price of {stock_px_fx} foreign dollars on {row['date']}\n")

                elif exit_cond1:
                    
                    adr_px = row['adr_open']*long_multiplier
                    cash -= abs(adr_pos)*adr_px
                    stock_px_fx = merged_df.loc[index+1,'stock_open']*short_multiplier
                    forex_cash += stock_px_fx*stock_pos
                    if forex_cash > 0:
                        forex_cash /= merged_df.loc[index+1,'avg_ask_non_us_at']
                    else:
                        forex_cash /= merged_df.loc[index+1,'avg_bid_non_us_at']
                    cash += forex_cash
                    forex_cash = 0
                    trade_records.append("Closing positions:\n")
                    # Times in EST
                    trade_records.append(f"We bought {-adr_pos} shares of ADR at the price of {adr_px} on {row['date']}\n")
                    trade_records.append(f"We sold {stock_pos} shares of underlying stock at the price of {stock_px_fx} foreign dollars on {row['date']}\n")
                    stock_pos, adr_pos = 0, 0
                    holding_period = None
                    trade_type = None
                    prev_cash, prev_adr_pos = cash, adr_pos
                    if cash > portfolio_value_before_entering:
                        hits.append(1)
                    else:
                        hits.append(0)
                    
            # If a concurrent trade is not already being placed
            # The 2nd trade of condition 2 falls on the next day
            if not (enter_cond1 or exit_cond1) and merged_df.loc[index+1, "date"] <= end_date:
                # Check and possibly trade condition 2
                enter_cond2 = (diff_record_cond2[-1] > mean_cond2 + entry_cond2_val*std_cond2 
                               and diff_record_cond2[-1] <= mean_cond2 + stop_loss_cond2*std_cond2
                               and stock_pos == 0 and adr_pos == 0)
                exit_cond2 = ((diff_record_cond2[-1] < mean_cond2 + exit_cond2_val*std_cond2 
                              or diff_record_cond2[-1] > mean_cond2 + stop_loss_cond2*std_cond2
                              or (holding_period == maximum_holding_period and trade_type == 2))
                              and stock_pos > 0 and adr_pos < 0)
                    
                if enter_cond2:
                    portfolio_value_before_entering = cash
                    adr_volume = 0.2*(merged_df.loc[index-volume_lookback+1:index,:]["adr_volume"].median()/row["adr_num_per_unit"])
                    stock_volume = 0.2*(merged_df.loc[index-volume_lookback+1:index,:]["stock_volume"].median()/row["stock_num_per_unit"])
                    units = int(min(cash/merged_df.loc[index-1,'adr_close_per_unit'],
                                    cash/(row['stock_close_per_unit']/row['avg_us_before']), 
                                    adr_volume, 
                                    stock_volume))
                    adr_quantity = int(units*row["adr_num_per_unit"])
                    stock_quantity = int(units*row["stock_num_per_unit"])
                    
                    # Take portfolio value for each previous day when the Asian market opens
                    temp_risk_lookback = min(risk_lookback, index)
                    current = merged_df.loc[(index - temp_risk_lookback + 1):index].copy()
                    next_day = merged_df.loc[(index - temp_risk_lookback + 2):(index + 1)].copy()
                    stock_values = (np.array((current["stock_close"])/np.array(next_day["avg_non_us_before"]))*stock_quantity) 
                    adr_values = np.array(current["adr_close"]*adr_quantity)
                    sigma, var, max_drawdown_abs = get_risk_statistics(stock_values, adr_values, var_ci)
                    if (var > portfolio_value_before_entering*var_limit or 
                        max_drawdown_abs > max_drawdown_limit*starting_cash or 
                        sigma > portfolio_value_before_entering*sigma_limit):
                        frac = min((portfolio_value_before_entering*var_limit)/var, 
                                   (max_drawdown_limit*starting_cash)/max_drawdown_abs,
                                  (portfolio_value_before_entering*sigma_limit)/sigma)
                        units = int(frac*units)
                        if units == 0:
                            enter_cond2 = False
                        adr_quantity = int(units*row["adr_num_per_unit"])
                        stock_quantity = int(units*row["stock_num_per_unit"]) 
                    if units != 0:
                        stock_pos += stock_quantity
                        stock_px_fx = merged_df.loc[index+1,'stock_open']*long_multiplier
                        forex_cash -= stock_px_fx*stock_quantity
                        # We store the current cash/adr position, because the trade below will occur on the next day (EST)
                        prev_cash, prev_adr_pos = cash, adr_pos
                        
                        adr_pos -= adr_quantity
                        adr_px = merged_df.loc[index+1,'adr_open']*short_multiplier
                        cash += adr_quantity*adr_px
                        
                        holding_period = 0
                        trade_type = 2
                        trade_records.append("Opening positions:\n")
                        # Times in EST
                        trade_records.append(f"We bought {stock_quantity} shares of underlying stock at the price of {stock_px_fx} foreign dollars on {row['date']}\n")
                        trade_records.append(f"We sold {adr_quantity} shares of ADR at the price of {adr_px} on {merged_df.loc[index+1,'date']}\n")

                elif exit_cond2:
                    stock_px_fx = merged_df.loc[index+1,'stock_open']*short_multiplier
                    forex_cash += stock_px_fx*stock_pos
                    if forex_cash > 0:
                        forex_cash /= merged_df.loc[index+1,'avg_ask_non_us_at']
                    else:
                        forex_cash /= merged_df.loc[index+1,'avg_bid_non_us_at']
                    cash += forex_cash
                    forex_cash = 0
                    # We store the current cash/adr position, because the trade below will occur on the next day (EST)
                    prev_cash, prev_adr_pos = cash, adr_pos
                    
                    adr_px = merged_df.loc[index+1,'adr_open']*long_multiplier
                    cash -= abs(adr_pos)*adr_px
                    trade_records.append("Closing positions:\n")
                    # Times in EST
                    trade_records.append(f"We sold {stock_pos} shares of underlying stock at the price of {stock_px_fx} foreign dollars on {row['date']}\n")
                    trade_records.append(f"We bought {-adr_pos} shares of ADR at the price of {adr_px} on {merged_df.loc[index+1,'date']}\n")
                    stock_pos, adr_pos = 0, 0
                    holding_period = None
                    trade_type = None
                    if cash > portfolio_value_before_entering:
                        hits.append(1)
                    else:
                        hits.append(0)

    ret = (portfolio_values[-1] - starting_cash)/starting_cash

    return ret, trade_records, portfolio_values, hits, dates

"""
Variant 3b
- Begin each trade on either US market open or Asian market open
- Regressions are done for all data collected in lookback window

For each row:
    stock_open, stock_close, (assess condition 1), adr_open, adr_close, (assess condition 2)
    If not condition 2 - No action taken: 
        After first 2 events, Assess condition 1 (right before the US market opens ~ 9.29AM EST)
        If condition 1:
            Place trade on current and next row (First trade ADR on US market open, then trade stock on Asian market open)
    If not condition 1 - No action taken:
        After next 2 events occur, assess condition 2
        If condition 2:
            Place trade on next row (First trade ADR on Asian market open, then trade stock on US market open)
    
start_date: First date (EST) we may place a trade
end_date: Last date (EST) we may place a trade
portfolio_values: Stores value of portfolio at each date from one day before the start_date, to the end_date, when the Asian market opens
"""
def pairs_trade_v3b(merged_df, lookback = 100, cash = 250000, entry_cond1_val = 1, entry_cond2_val = 1, 
                    exit_cond1_val = 0, exit_cond2_val = 0, stop_loss_cond1 = 3, stop_loss_cond2 = 3, 
                    start_date = "2016-01-01", end_date = "2021-01-31", slippage_bps = 10, 
                    borrowing_bps = 50, risk_lookback = 100, var_ci = 0.95, var_limit = 0.1, max_drawdown_limit = 0.2,
                    sigma_limit = 0.05, maximum_holding_period = 30, volume_lookback = 5):
    
    # Accounts for slippage and transaction costs
    short_multiplier = 1 - 0.0001*slippage_bps
    long_multiplier = 1 + 0.0001*slippage_bps
    # We assume lookback is given in terms of days
    lookback *= 2
    starting_cash = cash
    stock_pos, adr_pos = 0, 0
    holding_period = None
    trade_type = None
    forex_cash = 0
    # For book-keeping, since we shall store the portfolio value of the day before
    prev_cash, prev_adr_pos = cash, adr_pos
    diff_record = deque(maxlen = lookback)
    trade_records = []
    portfolio_values = []
    dates = []
    hits = []
    enter_cond1, exit_cond1, enter_cond2, exit_cond2 = False, False, False, False
    
    # Make sure that merged_df before end date is not empty
    if merged_df[merged_df['date'] < end_date].empty:
        return 0, trade_records, portfolio_values, hits, dates

    for index, row in merged_df.iterrows():

        if index+1 < len(merged_df) and index > 0:
            
            # Add portfolio value for the day before
            prev_date = merged_df.loc[index - 1, "date"]
            if row["date"] >= start_date and prev_date <= end_date:
                dates.append(prev_date)
                prev_forex_value = forex_cash + stock_pos*row["stock_open"]
                if prev_forex_value > 0:
                    prev_forex_value /= row['avg_ask_non_us_at']
                else:
                    prev_forex_value /= row['avg_bid_non_us_at']
                portfolio_values.append(prev_cash + prev_adr_pos*merged_df.loc[index - 1, 'adr_close'] 
                                        + prev_forex_value)
                
            # Before US Market Opens
            diff_record.append(merged_df.loc[index-1,'adr_close_per_unit'] - row['stock_close_per_unit']/row['avg_us_before'])
            if len(diff_record) == lookback and row["date"] >= start_date and row["date"] <= end_date:
                if stock_pos > 0:
                    holding_period += 1
                    cash -= 0.0001*borrowing_bps*(1/252)*abs(adr_pos)*merged_df.loc[index - 1, 'adr_close']
                    multiplier = (1 + 0.01*(2 + row["ir"])*(1/252))
                    forex_cash *= multiplier
                prev_cash, prev_adr_pos = cash, adr_pos
                    
                mean = np.array(diff_record).mean()
                std = np.array(diff_record).std()
                
                # If a concurrent trade is not already being placed
                if not (enter_cond2 or exit_cond2):
                    enter_cond1 = (diff_record[-1] > mean + entry_cond1_val*std
                                   and diff_record[-1] <= mean + stop_loss_cond1*std
                                   and stock_pos == 0 and adr_pos == 0)
                    exit_cond1 = ((diff_record[-1] < mean + exit_cond1_val*std
                                  or diff_record[-1] > mean + stop_loss_cond1*std
                                  or (holding_period == maximum_holding_period and trade_type == 1))
                                  and stock_pos > 0 and adr_pos < 0)
                    
                    if enter_cond1:
                        portfolio_value_before_entering = cash
                        adr_volume = 0.2*(merged_df.loc[index-volume_lookback:index - 1,:]["adr_volume"].median()/row["adr_num_per_unit"])
                        stock_volume = 0.2*(merged_df.loc[index-volume_lookback+1:index,:]["stock_volume"].median()/row["stock_num_per_unit"])
                        units = int(min(cash/row['adr_close_per_unit'],
                                        cash/(row['stock_close_per_unit']/merged_df.loc[index+1,'avg_non_us_before']), 
                                        adr_volume, 
                                        stock_volume))
                        adr_quantity = int(units*row["adr_num_per_unit"])
                        stock_quantity = int(units*row["stock_num_per_unit"])
                        
                        temp_risk_lookback = min(risk_lookback, index)
                        current = merged_df.loc[(index - temp_risk_lookback + 1):index].copy()
                        stock_values = np.array((current["stock_close"]/current["avg_us_before"])*stock_quantity) 
                        adr_values = np.array(merged_df.loc[(index - temp_risk_lookback):(index-1)]["adr_close"]*adr_quantity)
                        sigma, var, max_drawdown_abs = get_risk_statistics(stock_values, adr_values, var_ci)
                        if (var > portfolio_value_before_entering*var_limit or 
                            max_drawdown_abs > max_drawdown_limit*starting_cash or 
                            sigma > portfolio_value_before_entering*sigma_limit):
                            frac = min((portfolio_value_before_entering*var_limit)/var, 
                                       (max_drawdown_limit*starting_cash)/max_drawdown_abs,
                                      (portfolio_value_before_entering*sigma_limit)/sigma)
                            units = int(frac*units)
                            if units == 0:
                                enter_cond1 = False
                            adr_quantity = int(units*row["adr_num_per_unit"])
                            stock_quantity = int(units*row["stock_num_per_unit"]) 
                        if units != 0:
                            adr_pos -= adr_quantity
                            adr_px = row['adr_open']*short_multiplier
                            cash += adr_quantity*adr_px

                            stock_pos += stock_quantity
                            stock_px_fx = merged_df.loc[index+1,'stock_open']*long_multiplier
                            forex_cash -= stock_px_fx*stock_quantity
                            prev_cash, prev_adr_pos = cash, adr_pos
                            holding_period = 0
                            trade_type = 1
                            trade_records.append("Opening positions:\n")
                            # Times in EST
                            trade_records.append(f"We sold {adr_quantity} shares of ADR at the price of {adr_px} on {row['date']}\n")
                            trade_records.append(f"We bought {stock_quantity} shares of underlying stock at the price of {stock_px_fx} foreign dollars on {row['date']}\n")

                    elif exit_cond1:

                        adr_px = row['adr_open']*long_multiplier
                        cash -= abs(adr_pos)*adr_px
                        stock_px_fx = merged_df.loc[index+1,'stock_open']*short_multiplier
                        forex_cash += stock_px_fx*stock_pos
                        if forex_cash > 0:
                            forex_cash /= merged_df.loc[index+1,'avg_ask_non_us_at']
                        else:
                            forex_cash /= merged_df.loc[index+1,'avg_bid_non_us_at']
                        cash += forex_cash
                        forex_cash = 0
                        trade_records.append("Closing positions:\n")
                        # Times in EST
                        trade_records.append(f"We bought {-adr_pos} shares of ADR at the price of {adr_px} on {row['date']}\n")
                        trade_records.append(f"We sold {stock_pos} shares of underlying stock at the price of {stock_px_fx} foreign dollars on {row['date']}\n")
                        stock_pos, adr_pos = 0, 0
                        holding_period = None
                        trade_type = None
                        prev_cash, prev_adr_pos = cash, adr_pos
                        if cash > portfolio_value_before_entering:
                            hits.append(1)
                        else:
                            hits.append(0)
                        
            # Before Asian Market Opens
            diff_record.append(row['adr_close_per_unit'] - row['stock_close_per_unit']/merged_df.loc[index+1,'avg_non_us_before'])
            # The 2nd trade of condition 2 falls on the next day
            if len(diff_record) == lookback and row["date"] >= start_date and merged_df.loc[index+1,"date"] <= end_date:
                mean = np.array(diff_record).mean()
                std = np.array(diff_record).std()
                # If a concurrent trade is not already being placed
                if not (enter_cond1 or exit_cond1):
                    enter_cond2 = (diff_record[-1] > mean + entry_cond2_val*std
                                   and diff_record[-1] <= mean + stop_loss_cond2*std
                                   and stock_pos == 0 and adr_pos == 0)
                    exit_cond2 = ((diff_record[-1] < mean + exit_cond2_val*std
                                  or diff_record[-1] > mean + stop_loss_cond2*std
                                  or (holding_period == maximum_holding_period and trade_type == 2))
                                  and stock_pos > 0 and adr_pos < 0)

                    if enter_cond2:
                        portfolio_value_before_entering = cash
                        adr_volume = 0.2*(merged_df.loc[index-volume_lookback+1:index,:]["adr_volume"].median()/row["adr_num_per_unit"])
                        stock_volume = 0.2*(merged_df.loc[index-volume_lookback+1:index,:]["stock_volume"].median()/row["stock_num_per_unit"])
                        units = int(min(cash/merged_df.loc[index-1,'adr_close_per_unit'],
                                        cash/(row['stock_close_per_unit']/row['avg_us_before']), 
                                        adr_volume, 
                                        stock_volume))
                        adr_quantity = int(units*row["adr_num_per_unit"])
                        stock_quantity = int(units*row["stock_num_per_unit"])
                        temp_risk_lookback = min(risk_lookback, index)
                        current = merged_df.loc[(index - temp_risk_lookback + 1):index].copy()
                        next_day = merged_df.loc[(index - temp_risk_lookback + 2):(index + 1)].copy()
                        stock_values = (np.array((current["stock_close"])/np.array(next_day["avg_non_us_before"]))*stock_quantity) 
                        adr_values = np.array(current["adr_close"]*adr_quantity)
                        sigma, var, max_drawdown_abs = get_risk_statistics(stock_values, adr_values, var_ci)
                        if (var > portfolio_value_before_entering*var_limit or 
                            max_drawdown_abs > max_drawdown_limit*starting_cash or 
                            sigma > portfolio_value_before_entering*sigma_limit):
                            frac = min((portfolio_value_before_entering*var_limit)/var, 
                                       (max_drawdown_limit*starting_cash)/max_drawdown_abs,
                                      (portfolio_value_before_entering*sigma_limit)/sigma)
                            units = int(frac*units)
                            if units == 0:
                                enter_cond2 = False
                            adr_quantity = int(units*row["adr_num_per_unit"])
                            stock_quantity = int(units*row["stock_num_per_unit"])  
                        if units != 0:
                            stock_pos += stock_quantity
                            stock_px_fx = merged_df.loc[index+1,'stock_open']*long_multiplier
                            forex_cash -= stock_px_fx*stock_quantity
                            # We store the current cash/adr position, because the trade below will occur on the next day (EST)
                            prev_cash, prev_adr_pos = cash, adr_pos

                            adr_pos -= adr_quantity
                            adr_px = merged_df.loc[index+1,'adr_open']*short_multiplier
                            cash += adr_quantity*adr_px
                            holding_period = 0
                            trade_type = 2
                            trade_records.append("Opening positions:\n")
                            # Times in EST
                            trade_records.append(f"We bought {stock_quantity} shares of underlying stock at the price of {stock_px_fx} foreign dollars on {row['date']}\n")
                            trade_records.append(f"We sold {adr_quantity} shares of ADR at the price of {adr_px} on {merged_df.loc[index+1,'date']}\n")

                    elif exit_cond2:
                        stock_px_fx = merged_df.loc[index+1,'stock_open']*short_multiplier
                        forex_cash += stock_px_fx*stock_pos
                        if forex_cash > 0:
                            forex_cash /= merged_df.loc[index+1,'avg_ask_non_us_at']
                        else:
                            forex_cash /= merged_df.loc[index+1,'avg_bid_non_us_at']
                        cash += forex_cash
                        forex_cash = 0
                        # We store the current cash/adr position, because the trade below will occur on the next day (EST)
                        prev_cash, prev_adr_pos = cash, adr_pos
                        
                        adr_px = merged_df.loc[index+1,'adr_open']*long_multiplier
                        cash -= abs(adr_pos)*adr_px
                        trade_records.append("Closing positions:\n")
                        # Times in EST
                        trade_records.append(f"We sold {stock_pos} shares of underlying stock at the price of {stock_px_fx} foreign dollars on {row['date']}\n")
                        trade_records.append(f"We bought {-adr_pos} shares of ADR at the price of {adr_px} on {merged_df.loc[index+1,'date']}\n")
                        stock_pos, adr_pos = 0, 0
                        holding_period = None
                        trade_type = None
                        if cash > portfolio_value_before_entering:
                            hits.append(1)
                        else:
                            hits.append(0)

    ret = (portfolio_values[-1] - starting_cash)/starting_cash

    return ret, trade_records, portfolio_values, hits, dates

