from __future__ import print_function
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
import sys
import re
import math
import traceback
import pandas as pd
import Constants
import gc

N_LAGS = 30
TIME_FRAME = Constants.TF_15M
TF_IN_MINS = Constants.TF_TO_MIN[TIME_FRAME]

df = pd.read_csv("Forex/{}/ALL.csv".format(TIME_FRAME))

cut_off_date = int(time.mktime(time.strptime('01/12/2017', "%d/%m/%Y"))) * 1000
#df = df[df['date'] > cut_off_date].reset_index(drop=True)

COINS = [x[len("close_"):] for x in df.columns if "close_" in x]
#COINS = sorted(['BCH', 'BTC', 'ETH', 'IOTA', 'EOS' ,'XRP', 'NEO'])
#COINS = ['IOTA']

#PIECE_HOURS = [24 * 14, 24 * 7, 24 * 3, 24, 12, 4, 2, 1]
#REG_HOURS   = [24 * 14, 24 * 7, 24 * 3, 24, 12, 4, 2, 1]

PIECE_HOURS = [24*7, 24*3, 24, 16, 10, 6, 2]
REG_HOURS   = [24*7, 24*3, 24, 16, 10, 6, 2]
#PIECE_HOURS = []
#REG_HOURS   = []
MA_PERIODS  = [10, 20, 30, 50]
RSI_PERIODS = [10, 20, 30, 50]

#PIECE_HOURS = []
#REG_HOURS   = []

L2_SCALE = 5

for C in COINS:
    
    gc.collect()
    
    print("Feature Engineering {}".format(C))
    df['volume_'+C] = df['volume_'+C] * df['close_'+C]
    df['reward_'+C] = df['close_'+C].shift(-1) / df['close_'+C]
    df['reward_'+C] = df["reward_"+C].apply(lambda x : math.log10(x))
    
    cut_low = 0.95
    cut_high = 1.05

    '''limits = []
    for i in range(len(df)):
        cls = df.at[i, 'close_'+C]
        appended = False
        for j in range(i+1,len(df)):
            if df.at[j, 'low_'+C] / cls <= cut_low:
                limits.append(math.log10(cut_low))
                appended = True
                break
            if df.at[j, 'high_'+C] / cls >= cut_high:
                limits.append(math.log10(cut_high))
                appended = True
                break
        if not appended:
            limits.append(0)
    
    df['limit_'+C] = limits'''
    
    print("{}: MOVS".format(C))
    for MA in MA_PERIODS:
        for lag in range(N_LAGS-1, -1, -1):
                df["{}_{}_{}".format("MOV{}".format(MA), lag+1, C)] = df["close_"+C].shift(lag) / df["close_"+C].shift(lag).rolling(center=False,window=MA).mean()
                df["{}_{}_{}".format("MOV{}".format(MA), lag+1, C)] = df["{}_{}_{}".format("MOV{}".format(MA), lag+1, C)].apply(lambda x : math.log10(x))
    
    print("{}: RSI".format(C))
    for RSI in RSI_PERIODS:
        for lag in range(N_LAGS-1, -1, -1):
                df['rsi_chng'] = (df["close_"+C].shift(lag) - df["close_"+C].shift(lag+1))
                df['rsi_gain'] = df['rsi_chng'].apply(lambda x : 0 if x < 0 else x).rolling(center=False,window=RSI).mean()
                df['rsi_loss'] = df['rsi_chng'].apply(lambda x : 0 if x > 0 else abs(x)).rolling(center=False,window=RSI).mean()
                df["{}_{}_{}".format("RSI{}".format(RSI), lag+1, C)] = 100 - (100 / (1 + df['rsi_gain']/df['rsi_loss']) )
    df.drop(['rsi_chng','rsi_gain','rsi_loss'], axis=1, inplace=True)
    
    
    print("{}: CLOUD".format(C))
    '''Tenkan, Kijun, SenkouB = [], [], []
    for i in range(len(df)):
        df['high_9'] = df['high_c'+C].rolling(center=False,window=9).max()
        high_9  = df['high_'+C][max(0,i-9):i].describe()['max']
        low_9   = df['low_'+C][max(0,i-9):i].describe()['min']
        high_26 = df['high_'+C][max(0,i-26):i].describe()['max']
        low_26  = df['low_'+C][max(0,i-26):i].describe()['min']
        high_52 = df['high_'+C][max(0,i-52):i].describe()['max']
        low_52  = df['low_'+C][max(0,i-52):i].describe()['min']
        Tenkan.append( (high_9 + low_9) / 2 )
        Kijun.append( (high_26 + low_26) / 2 )
        SenkouB.append( (high_52 + low_52) / 2 )
    
    Tenkan  = pd.Series(Tenkan)
    Kijun   = pd.Series(Kijun)
    SenkouB = pd.Series(SenkouB)'''
        
    for lag in range(N_LAGS-1, -1, -1):  
        
        df['high_9']  = df['high_'+C].rolling(center=False,window=9).max().shift(lag)
        df['low_9']   = df['low_'+C].rolling(center=False,window=9).min().shift(lag)
        df['high_26'] = df['high_'+C].rolling(center=False,window=26).max().shift(lag)
        df['low_26']  = df['low_'+C].rolling(center=False,window=26).min().shift(lag)
        df['high_52'] = df['high_'+C].rolling(center=False,window=52).max().shift(lag)
        df['low_52']  = df['low_'+C].rolling(center=False,window=52).min().shift(lag)
        
        df['Tenkan'] = (df['high_9'] + df['low_9']) / 2
        df['Kijun'] = (df['high_26'] + df['low_26']) / 2
        df['SenkouA'] = (df['Tenkan'] + df['Kijun']) / 2
        df['SenkouB'] = (df['high_52'] + df['low_52']) / 2
        
        df["{}_{}_{}".format("CLOUD_Tenkan", lag+1, C)] = df["close_"+C].shift(lag) / df['Tenkan']
        df["{}_{}_{}".format("CLOUD_Kijun", lag+1, C)] = df["close_"+C].shift(lag) / df['Kijun']
        df["{}_{}_{}".format("CLOUD_SenkouA", lag+1, C)] = df["close_"+C].shift(lag) / df['SenkouA']
        df["{}_{}_{}".format("CLOUD_SenkouB", lag+1, C)] = df["close_"+C].shift(lag) / df['SenkouB']
    
    
    df.drop(['high_9','low_9','high_26','low_26','high_52','low_52'], axis=1, inplace=True)
    df.drop(['Tenkan','Kijun','SenkouA','SenkouB'], axis=1, inplace=True)

    print("{}: LAGS".format(C))
    for col_type in ['L_CLOSE', 'L_HIGH', 'L_LOW', 'L_VOLUME']:
        
        for lag in range(N_LAGS, 0, -1):
            col = 'close'
            if col_type == 'L_HIGH':
                col = 'high'
            if col_type == 'L_LOW':
                col = 'low'
            if col_type == 'L_VOLUME':
                col = 'volume'

            if col_type == "L_VOLUME":
                df['vol_mean'] = df["volume_"+C].shift(lag-1).rolling(center=False,window=300).mean()
                df['vol_std'] = df["volume_"+C].shift(lag-1).rolling(center=False,window=300).std()
                df["{}_{}_{}".format(col_type, lag, C)] = (df["volume_"+C].shift(lag-1) - df["vol_mean"])/df["vol_std"].apply(lambda x : 1 if np.isnan(x) else x)
                #df['vol_mean'] = df["volume_"+C].shift(lag-1).rolling(center=False,window=50).mean()
                #df['vol_std'] = df["volume_"+C].shift(lag-1).rolling(center=False,window=50).std()
                #df["{}_{}_{}".format("VOL_STD_50", lag, C)] = (df["volume_"+C].shift(lag-1) - df["vol_mean"])/df["vol_std"].apply(lambda x : 1 if np.isnan(x) else x)
                #df["{}_{}_{}".format(col_type, lag, C)] = df[col+"_"+C].shift((lag)) / df['volume_'+C]
            elif col_type == "L_CLOSE":
                df["{}_{}_{}".format(col_type, lag, C)] = df["reward_"+C].shift(lag)
            else:
                df["{}_{}_{}".format(col_type, lag, C)] = df[col+"_"+C].shift(lag) / df["close_"+C].shift(lag)
                df["{}_{}_{}".format(col_type, lag, C)] = df["{}_{}_{}".format(col_type, lag, C)].apply(lambda x : math.log10(x))
    df.drop(['vol_mean','vol_std'], axis=1, inplace=True)
    print("{}: LAGS2".format(C))
    for col_type in ['L2_CLOSE', 'L2_HIGH', 'L2_LOW', 'L2_VOLUME']:
        
        for lag in range(N_LAGS, 0, -1):
            col = 'close'
            if col_type == 'L2_HIGH':
                col = 'high'
            if col_type == 'L2_LOW':
                col = 'low'
            if col_type == 'L2_VOLUME':
                col = 'volume'

            if col_type == "L2_VOLUME":
                stmt = "df['{}_{}_{}'.format(col_type, lag, C)] = "
                for x in range(L2_SCALE):
                    stmt += "df['volume_'+C].shift((L2_SCALE*lag)-{}){}".format(x, "" if x == L2_SCALE-1 else "+")
                exec(stmt)
                #df["{}_{}_{}".format(col_type, lag, C)] = df["volume_"+C].shift((L2_SCALE*lag))
                #df["{}_{}_{}".format(col_type, lag, C)] = df["volume_"+C].shift((L2_SCALE*lag)) / df['volume_'+C]
            else:
                stmt = "df['{}_{}_{}'.format(col_type, lag, C)] = "
                for x in range(L2_SCALE):
                    stmt += "df['{}_'+C].shift((L2_SCALE*lag)-{}){}".format(col, x, "" if x == L2_SCALE-1 else "+")
                exec(stmt)
                df['{}_{}_{}'.format(col_type, lag, C)] = df['{}_{}_{}'.format(col_type, lag, C)] / L2_SCALE
                df['{}_{}_{}'.format(col_type, lag, C)] = df['{}_{}_{}'.format(col_type, lag, C)] / df['close_'+C]
            
            df["{}_{}_{}".format(col_type, lag, C)] = df["{}_{}_{}".format(col_type, lag, C)].apply(lambda x : math.log10(x))
    
    #print("{}: PRICE CHANGE".format(C))
    #-------------------------------------------------------------------
    #for i in range(len(PIECE_HOURS)-1):
    #    look_back1 = (PIECE_HOURS[i+0] * 60) // TF_IN_MINS
    #    look_back2 = (PIECE_HOURS[i+1] * 60) // TF_IN_MINS
    #    df["L_CHNG_{}_{}".format(i+1,C)] = df['close_'+C].shift(look_back2) / df['close_'+C].shift(look_back1)
    print("{}: PIECEWISE".format(C))
    #-------------------------------------------------------------------
    for i in range(len(PIECE_HOURS)):
        L = len(df)
        look_back1 = (PIECE_HOURS[i+0] * 60) // TF_IN_MINS
        if i == len(PIECE_HOURS) - 1:
            look_back2 = 0
        else:
            look_back2 = (PIECE_HOURS[i+1] * 60) // TF_IN_MINS
        coefs = []
        for row_n in range(L):
            try:
                idx1 = row_n - look_back1
                if idx1 < 0:
                    coefs.append(np.nan)
                    continue
                idx2 = row_n - look_back2
                data = df.ix[idx1:idx2,'close_'+C] / df.ix[idx1,'close_'+C]
                data = data.apply(lambda x : math.log10(x))
                coeff = np.linalg.lstsq(np.reshape(range(len(data)), (-1, 1)), data)[0]
                #print("Coefficients", coeff)
                coefs.append(coeff[0])
                #print(coefs)
            except Exception as err:
                print("error is ", err, idx1, idx2, df.ix[idx1,'close_'+C])
                coefs.append(np.nan)
        print("PIECE_CLOSE_{}/{}_{}".format(i+1, len(PIECE_HOURS), C))
        df["PIECE_CLOSE_{}_{}".format(i+1, C)] = coefs

    all_close_coefs = []
    all_vol_coefs = []
    print("{}: REGRESSIONS".format(C))
    for i, hours in enumerate(REG_HOURS):
        L = len(df)
        look_back1 = (hours * 60) // TF_IN_MINS
        coefs_close, coefs_vol = [], []
        for row_n in range(L):
            
            # close
            idx1 = row_n - look_back1
            if idx1 < 0:
                coefs_close.append(np.nan)
                coefs_vol.append(np.nan)
                continue
            else:
                data = df.ix[idx1:(row_n),'close_'+C] / df.ix[idx1,'close_'+C]
                data = data.apply(lambda x : math.log10(x))
                coeff = np.linalg.lstsq(np.reshape(range(len(data)), (-1, 1)), data)[0]
                coefs_close.append(coeff[0])
            # volume
            idx1 = row_n - look_back1
            denom = df.ix[idx1,'volume_'+C]
            loops = 1
            broke = False
            while denom == 0:
                new_idx = idx1 - loops
                if new_idx < 0:
                    coefs_vol.append(0)
                    broke = True
                    break
                denom = df.ix[new_idx,'volume_'+C]
                loops += 1
            if broke:
                continue
            data = df.ix[idx1:(row_n),'volume_'+C] / denom - 1
            coeff = np.linalg.lstsq(np.reshape(range(len(data)), (-1, 1)), data)[0]
            coefs_vol.append(coeff[0])
        all_close_coefs.append(coefs_close)
        all_vol_coefs.append(coefs_vol)
        
        print("REG_CLOSE_{}/{}_{}".format(i+1, len(REG_HOURS), C))
    
    for i, coefs in enumerate(all_close_coefs):
        df["REG_CLOSE_{}_{}".format(i+1,C)] = coefs
    for i, coefs in enumerate(all_vol_coefs):
        df["REG_VOLUME_{}_{}".format(i+1,C)] = coefs
        
        
rename_maps = {"GER30" : "GER",
               "NAS100" : "NAS",
               "AUS200" : "ASX",
               "US30" : "US",
               "SPX500" : "SPX",
               "UK100" : "UK",
               "FRA40" : "FRA"}

cols = []
for c in df.columns:
    for k in rename_maps:
        if k in c:
            c = c.replace(k, rename_maps[k])
            print(c)
            break    
    cols.append(c)
    
df.columns = cols
    
df.to_csv("Forex/{}/{}.csv".format(TIME_FRAME, "ALL_MOD"), index=False)

plt.ion()

#plt.plot(df.close_IOTA)
#plt.show()

for i in range(14000,len(df),1):
    #plt.plot(df.close_IOTA)
    row = df.iloc[i,:]
    price = row.close_IOTA
    lines = []
    for j, hours in enumerate(REG_HOURS):
        look_back1 = (hours * 60) // TF_IN_MINS
        plt.plot( (i-look_back1, i), (math.log10(price)-look_back1*row["REG_CLOSE_{}_IOTA".format(j+1)], math.log10(price)) )
        if j == 0:
            break
        
    print(i/len(df))
    
    #plt.pause(0.5)
    #plt.show()

plt.show()

hold = 0
profits = [0]
for i in range(10000, len(df)):
    row = df.iloc[i,:]
    profit = 0
    if row.REG_CLOSE_4_BTC < -0.0002 and hold == 0:
        hold = 1
        price_buy = row.close_BTC
    elif row.REG_CLOSE_4_BTC > 0.000 and hold == 1:
        hold = 0
        profit = math.log10(row.close_BTC / price_buy)
    
    profits.append(profits[-1]+profit)

plt.plot(profits)


