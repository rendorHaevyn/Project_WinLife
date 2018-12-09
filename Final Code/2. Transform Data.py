import json
import time
import requests
import Constants

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import random
import imp
import gc
import os
import sys
import re
import math
import traceback
import pandas as pd
import Constants
import gc
import threading

N_LAGS = 10 # Choose number of lags to look back. The size of the data scales linearly with this single number
TIME_FRAME = Constants.TF_5M
TF_IN_MINS = Constants.TF_TO_MIN[TIME_FRAME]

IS_FOREX = False

# Load in the un-transformed data
EXPORT_PATH = "Data/Crypto"
main_data    = pd.read_csv("{}/{}/ALL.csv".format(EXPORT_PATH, TIME_FRAME))

# Use this as an easy way to truncate the data from a certain date. Less data means faster processing!
cut_off_date = int(time.mktime(time.strptime('01/4/2018', "%d/%m/%Y"))) * 1000
main_data    = main_data[main_data['date'] > cut_off_date].reset_index(drop=True)

# Any column with a "close_" will be followed by the name of the coin. 
# Find all the coins that are in the data and store them as a list
COINS = [x[len("close_"):] for x in main_data.columns if "close_" in x]
# Manual override to only load selected coins that you want
COINS = ["BMXBTCUSD", "BFXBTCUSDT", "BINBTCUSDT", "GDXBTCUSD", "BFXXRPUSDT", "BINETHUSDT"]


if IS_FOREX:
    denoms = {}
    denoms["AUD"] = np.repeat(1, len(main_data))
    for C in COINS:
        if C.endswith("AUD"):
            denoms[C[:3]] = main_data["close_"+C]
        elif C[:3] == "AUD":
            scaler = 100 if C.endswith("JPY") else 1
            denoms[C[-3:]] = scaler / main_data["close_"+C]
    
    for C in COINS:
        main_data['pipcost_'+C] = denoms[C[-3:]]


COINS = sorted(COINS)

L_SCALES        = [3, 6]    # Mulitpliers for the far reaching lag features
MA_PERIODS      = [5, 10, 20, 30, 50] # Periods for the Moving Average Features
RSI_PERIODS     = [5, 10, 20, 30, 50] # Periods for the RSI Features
MIN_MAX_PERIODS = [15] # Periods for the Support / Resistance Features
REG_LOOKS       = [6, 12, 18]  # Periods for the Linear Regression Features

# Dictionary that will store the 
finished_data = {}

# Function to create feature for a selection Coin (C). This will be run across multiple threads for speed
def TransformData(C):
    
    # I played around with looking at different market times of day as an indicator but determined
    # through modelling that there is no informational value held in this data. Code is archived below
    # for fun.
    '''
    df["DAY_OF_WEEK"] = df['date'].apply( lambda x : pd.to_datetime(str(x), unit="ms").dayofweek )
    df["HOUR_OF_DAY"] = df['date'].apply( lambda x : pd.to_datetime(str(x), unit="ms").hour )
    
    for lag in range(N_LAGS-1, -1, -1):
        df['DAY_OF_WEEK_{}_{}'.format(lag+1, C)] = df["DAY_OF_WEEK"].shift(lag)
    for lag in range(N_LAGS-1, -1, -1):
        df['HOUR_OF_DAY_{}_{}'.format(lag+1, C)] = df["HOUR_OF_DAY"].shift(lag)
        
    df.drop(['DAY_OF_WEEK','HOUR_OF_DAY'], axis=1, inplace=True)
    '''
     
    # Ensure that the requested coin name exists in the data, if not then just return nothing
    if "close_{}".format(C) not in main_data.columns:
        return None
    
    # Clean up memory
    gc.collect()
    
    # Create a local copy of the data, but only bring in the columns that are relevent to this coin
    df = main_data[['date'] + [x for x in main_data.columns if x.endswith(C)]].reset_index(drop=True)
    
    print("Feature Engineering {}".format(C))
    
    if IS_FOREX:
        scaler = 100 if "JPY" in C else 10000
        df['reward_'+C] = scaler * (df['close_'+C].shift(-1) - df['close_'+C]) * df['pipcost_'+C]
    else:
        df['reward_'+C] = df['close_'+C].shift(-1) / df['close_'+C]
        df['reward_'+C] = df["reward_"+C].apply(lambda x : math.log10(x)) # Returns are log base 10
    
    # Alternative reward using stop & limits. This may come in handy as a different trading strategy
    #------------------------------------------------------
    
    '''cut_low = 0.95
    cut_high = 1.05
    limits = []
    for i in range(len(df)):
        close_price = df.at[i, 'close_'+C]
        appended = False
        for j in range(i+1,len(df)):
            if df.at[j, 'low_'+C] / close_price <= cut_low:
                limits.append(math.log10(cut_low))
                appended = True
                break
            if df.at[j, 'high_'+C] / close_price >= cut_high:
                limits.append(math.log10(cut_high))
                appended = True
                break
        if not appended:
            limits.append(0)
    
    df['limit_'+C] = limits'''
    
    #------------------------------------------------------
    
    print("{}: LAGS".format(C))
    for col_type in ['L_CLOSE', 'L_LOW', 'L_HIGH', 'L_VOLUME', 'L_RET', 'L_VOLPRICE']:
        
        for lag in range(N_LAGS-1, -1, -1):
            
            lookup = {"L_CLOSE"  : "close", 
                      "L_LOW"    : "low",
                      "L_HIGH"   : "high",
                      "L_VOLUME" : "volume"}
            
            col = 'close' if col_type not in lookup else lookup[col_type]

            # Store the Z-score of the volume over the last 50 periods
            if col_type == "L_VOLUME":
                df['vol_mean'] = df["volume_"+C].shift(lag).rolling(center=False,window=50).mean()
                df['vol_std']  = df["volume_"+C].shift(lag).rolling(center=False,window=50).std()
                df["{}_{}_{}".format(col_type, lag+1, C)] = (df["volume_"+C].shift(lag) - df["vol_mean"])/df["vol_std"].apply(lambda x : 0 if np.isnan(x) else x)
            
            # Store the lagged return
            elif col_type == "L_CLOSE":
                df["{}_{}_{}".format(col_type, lag+1, C)] = df["{}_{}_{}".format(col_type, lag+1, C)] = df["close_"+C].shift(lag) / df["close_"+C].shift(lag+1)
            
            # Store the cumnulative lagged returns
            elif col_type == "L_RET":
                df["{}_{}_{}".format(col_type, lag+1, C)] = df["close_"+C].shift(lag) / df["close_"+C].shift(N_LAGS)
            
            # Store the price movement multiplied by the volume
            elif col_type == "L_VOLPRICE":
                df['vol_mean'] = df["volume_"+C].shift(lag).rolling(center=False,window=50).mean()
                df['vol_std']  = df["volume_"+C].shift(lag).rolling(center=False,window=50).std()
                df["{}_{}_{}".format(col_type, lag+1, C)] = (df["close_"+C].shift(lag) / df["close_"+C].shift(lag+1)) - 1
                df["{}_{}_{}".format(col_type, lag+1, C)] = df["{}_{}_{}".format(col_type, lag+1, C)] * ( df["volume_"+C].shift(lag) / (df['vol_mean']+df['vol_std']) )
            
            # Store High/Low divided by the closing price of the same lag
            else:
                df["{}_{}_{}".format(col_type, lag+1, C)] = df[col+"_"+C].shift(lag) / df["close_"+C].shift(lag)
                
    # Drop the vol mean & stdev - we dont need it anymore
    df.drop(['vol_mean','vol_std'], axis=1, inplace=True)
    
    #------------------------------------------------------
    print("{}: LAGS Scaled".format(C))
    # Iterate through each far reaching lag scale
    for i, scaler in enumerate(L_SCALES):
        for col_type in ['L_CLOSE', 'L_HIGH', 'L_LOW', 'L_VOLUME']:
            
            for lag in range(N_LAGS-1, -1, -1):
                
                lookup = {"L_CLOSE"  : "close", 
                          "L_LOW"    : "low",
                          "L_HIGH"   : "high",
                          "L_VOLUME" : "volume"}
                
                col = 'close' if col_type not in lookup else lookup[col_type]
                    
                col_type = col_type.replace("L_", "L{}_".format(i+2))
    
                
                if "VOLUME" in col_type:
                    pass # I dont think this feature is useful, so I am not using it
                    #stmt = "df['{}_{}_{}'.format(col_type, lag+1, C)] = "
                    #for x in range(scaler):
                    #    stmt += "df['volume_'+C].shift((scaler*lag)+{}){}".format(x, "" if x == scaler-1 else "+")
                    #exec(stmt)
                else:
                    # Build a statement that sums up the price, we use the average of this price
                    stmt = "df['{}_{}_{}'.format(col_type, lag+1, C)] = "
                    for x in range(scaler):
                        stmt += "df['{}_'+C].shift((scaler*lag)+{}){}".format(col, x, "" if x == scaler-1 else "+")
                    exec(stmt)
                    # Calc average and divide it by the current, non-lagged price
                    df['{}_{}_{}'.format(col_type, lag+1, C)] = (df['{}_{}_{}'.format(col_type, lag+1, C)] / scaler) / df['close_'+C] 
    
    #------------------------------------------------------
    print("{}: SMA_CLOSE".format(C))
    # Calculate each Moving Average feature
    for i, MA in enumerate(MA_PERIODS):
        for lag in range(N_LAGS-1, -1, -1):
                df["{}_{}_{}".format("SMACLOSE{}".format(i+1), lag+1, C)] = df["close_"+C].shift(lag) / df["close_"+C].shift(lag).rolling(center=False,window=MA).mean()
    print("{}: SMA_LOW".format(C))
    # Calculate each Moving Average feature
    for i, MA in enumerate(MA_PERIODS):
        for lag in range(N_LAGS-1, -1, -1):
                df["{}_{}_{}".format("SMALOW{}".format(i+1), lag+1, C)] = df["close_"+C].shift(lag) / df["low_"+C].shift(lag).rolling(center=False,window=MA).mean()
    print("{}: SMA_HIGH".format(C))
    # Calculate each Moving Average feature
    for i, MA in enumerate(MA_PERIODS):
        for lag in range(N_LAGS-1, -1, -1):
                df["{}_{}_{}".format("SMAHIGH{}".format(i+1), lag+1, C)] = df["close_"+C].shift(lag) / df["high_"+C].shift(lag).rolling(center=False,window=MA).mean()
    #------------------------------------------------------
    print("{}: RSI".format(C))
    # Calculate each Relative Strength Index (RSI) feature
    for i, RSI in enumerate(RSI_PERIODS):
        for lag in range(N_LAGS-1, -1, -1):
                df['rsi_chng'] = (df["close_"+C].shift(lag) - df["close_"+C].shift(lag+1))
                df['rsi_gain'] = df['rsi_chng'].apply(lambda x : 0 if x < 0 else x).rolling(center=False,window=RSI).mean()
                df['rsi_loss'] = df['rsi_chng'].apply(lambda x : 0 if x > 0 else abs(x)).rolling(center=False,window=RSI).mean()
                df["{}_{}_{}".format("RSI{}".format(i+1), lag+1, C)] = 100 - (100 / (1 + df['rsi_gain']/df['rsi_loss']) )
            
    # Drop the rsi data - we dont need it anymore
    df.drop(['rsi_chng','rsi_gain','rsi_loss'], axis=1, inplace=True)
    #------------------------------------------------------
    print("{}: SUPPORT & RESISTANCE".format(C))
    # Calculate Support & Resistance. These are simply defined by the Min/Max over a specific period. We normalize the numbers
    # by dividing the current price by the support/resistance price
    for i, p in enumerate(MIN_MAX_PERIODS):
        for lag in range(N_LAGS-1, -1, -1):  
            df["{}_{}_{}".format("SUPPORT{}".format(i+1), lag+1, C)]  = df['close_'+C].shift(lag) / df['low_'+C].rolling(center=False,window=p).min().shift(lag)
    
    for i, p in enumerate(MIN_MAX_PERIODS):
        for lag in range(N_LAGS-1, -1, -1):
            df["{}_{}_{}".format("RESIST{}".format(i+1), lag+1, C)]   = df['close_'+C].shift(lag) / df['high_'+C].rolling(center=False,window=p).max().shift(lag)
    #------------------------------------------------------
    
    # Calculate Linear Regressions. This takes a while, so we split it up into multiple threads for speed. The regression
    # captures the "m" coefficient of the form y = mx + c, but we force the c to be zero, and shift the relative point
    # as being the first price from the lookback period. This is to fix the removal of the bias term.
    print("{}: LINEAR".format(C))
    coef_lists  = [ [] for _ in range(len(REG_LOOKS)) ]
    threads     = []
    
    def calcReg(index, LB):
        
         for row_n in range(len(df)):
            
            idx1 = row_n - LB
            if idx1 < 0:
                coef_lists[index].append(np.nan)
                continue
            else:
                reg_data = df.ix[(idx1+1):(row_n),'close_'+C] / df.ix[idx1,'close_'+C]
                reg_data = reg_data.apply(lambda x : math.log10(x))
                coeff = np.linalg.lstsq(np.reshape(range(len(reg_data)), (-1, 1)), reg_data)[0]
                coef_lists[index].append(coeff[0]) # Keep the coefficent in front of 'time' only
    
    for i, lookback in enumerate(REG_LOOKS):
        new_thread = threading.Thread(target = calcReg, args=(i, lookback))
        threads.append(new_thread)
        threads[-1].start()
        
    for thr in threads:
        thr.join()
            
    for i, coef_list in enumerate(coef_lists):
        for lag in range(N_LAGS-1, -1, -1):
            df["{}_{}_{}".format("LINEAR{}".format(i+1), lag+1, C)] = pd.Series(coef_list).shift(lag)
    
    # Clean up memory
    gc.collect()
    # Done Transforming this coin!
    print("{} finished processing!".format(C))
    
    finished_data[C] = df
    #------------------------------------------------------     
    
# Run a thread per coin - Much faster!
#----------------------------------------------
threads = []
for C in COINS:
    threads.append(threading.Thread(target=TransformData, args=(C,)))
    threads[-1].start()
for thr in threads:
    thr.join()
#----------------------------------------------
    
# All data is garuanteed to be aligned due to the extract process being an inner join, because of this
# we can simply append each column and we can be sure that they align properly
finalData = None

#----------------------------------------------
for c, v in sorted(finished_data.items()):
    
    if finalData is None:
        finalData = v
    else:
        print("Merging on {}".format(c))
        for col in v.columns:
            if col == 'date':
                continue
            finalData[col] = v[col]
#----------------------------------------------
        
# This is just some Forex renaming - we cant have any number in the column name, as we use that to determine
# what lag # we are on in the modelling phase
rename_maps = {"GER30"  : "GER",
               "NAS100" : "NAS",
               "AUS200" : "ASX",
               "US30"   : "US",
               "SPX500" : "SPX",
               "UK100"  : "UK",
               "FRA40"  : "FRA"}

cols = []
for c in finalData.columns:
    for k in rename_maps:
        if k in c:
            c = c.replace(k, rename_maps[k])
            print(c)
            break    
    cols.append(c)
    
finalData.columns = cols

# Start exporting the data!
print("Exporting {} rows, {} columns...".format(len(finalData), len(cols)))
# Done!

try:
    os.mkdir("{}/{}".format(EXPORT_PATH, TIME_FRAME))
except FileExistsError as e1:
    pass
except OSError as e2:
    print('Failed to create directory {} - Incorrect syntax?'.format(TIME_FRAME))
except:
    print('Error occurred - {}.'.format(sys.exc_info()[0]))
    
finalData.to_csv("{}/{}/ALL_MOD.csv".format(EXPORT_PATH, TIME_FRAME), index=False)
