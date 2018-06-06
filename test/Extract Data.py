from __future__ import print_function
import json
import time
import hmac,hashlib
import Client_Poloniex
import requests
import os
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

def ExtractData(N_LAGS=5):

    POL = Client_Poloniex.Poloniex("FKF20K52-896REY9Z-RUI9HW0F-X5FH3SI8",
                       "7096da20ec5e1bfa5f2399047c98e242bb35ad680a424b0c9fef45d4ee5c0220e603ce912ef737ef6ea98af51fb150ebf433fc4f437aa35f3eed4656a6704925")

    ALL_TICKERS = sorted(POL.returnTicker().keys())

    JUNE_1_2017 = 1496275200
    NOV_1_2017 = 1509454800
    START_TIME = JUNE_1_2017

    tickers = []
    for T in ALL_TICKERS:
        tickers.append((T, T[:T.index("_")], T[T.index("_")+1:], 1 if "USDT" in T else 0))
        
    new_tickers = []
    for t in tickers:
        keep = False
        for coin in sorted(['BCH', 'BTC', 'DASH', 'ETC', 'ETH', 'LTC', 'XMR', 'XRP', 'ZEC']):
            if coin in t:
                keep = True
                break
        if keep:
            new_tickers.append(t)
    tickers = new_tickers

    coin_list = pd.DataFrame(tickers)
    coin_list.columns = ['Pair', 'Base','Coin', 'Extract']
    coin_list.to_csv("coins.csv", index=False)

    tm = time.time()

    tmap    = {86400 : "D1", 14400 : "H4", 7200 : "H2", 1800 : "M30", 900 : "M15", 300 : "M5"}
    highmap = {86400 : 900,  14400 : 300,  7200 : 300,  1800 : 300,   900 : 300,   300 : 300 }

    SHIFT_POINTS_HOURS = [24 * 14, 24 * 7, 24 * 3, 24, 0]
    REG_HOURS          = [24 * 14, 24 * 7, 24 * 3, 24, 12, 4, 2, 1]

    CALC_R2 = False

    for tf in [86400, 14400, 7200, 1800, 900, 300]:
        merged = None
        for idx in range(len(coin_list)):
            coin = coin_list.iloc[idx,:]
            if not coin.Extract:
                continue
            TICK = coin.Pair
            while True:
                try:
                    data = POL.returnChartData(TICK,tf,START_TIME,tm)
                    break
                except KeyboardInterrupt:
                    sys.exit()
                except Exception as error:
                    print(error)
            df = pd.DataFrame(data)
            df['reward'] = df['close'].shift(-1) / df['close']
            df['reward'] = df.reward.apply(lambda x : math.log10(x))

            if CALC_R2:
                m5 =   pd.DataFrame(POL.returnChartData(TICK,highmap[tf],0,tm))
                m5['reward'] = m5['close'].shift(0) / m5['close'].shift(1)
                m5['reward'] = m5.reward.apply(lambda x : math.log10(x))
                
                r2 = list(df.reward)
                if tf > 300:
                    for i in range(len(df)):
                        m5 = m5[m5.date >= df.ix[i, 'date']]
                        sub_time = m5[m5.date < df.ix[i, 'date']+tf].reset_index(drop=True)
                        r2[i] = sub_time.reward.describe()[1] / sub_time.reward.describe()[2]
            
                df['reward2'] = pd.Series(r2).shift(-1)

            df.drop(['weightedAverage', 'quoteVolume', 'open'], axis=1, inplace=True)
            for col_type in ['L_CLOSE', 'L_HIGH', 'L_LOW', 'L_VOLUME']:
                
                for lag in range(N_LAGS):
                    col = 'close'
                    if col_type == 'L_HIGH':
                        col = 'high'
                    if col_type == 'L_LOW':
                        col = 'low'
                    if col_type == 'L_VOLUME':
                        col = 'volume'
                    df["{}_{}".format(col_type, lag+1)] = df[col].shift((lag+1)) / df['close' if col != 'volume' else 'volume'].shift(0)
            
            #-------------------------------------------------------------------
            for i in range(len(SHIFT_POINTS_HOURS)-1):
                look_back1 = (SHIFT_POINTS_HOURS[i+0] * 3600) // tf
                look_back2 = (SHIFT_POINTS_HOURS[i+1] * 3600) // tf
                df["L_CHNG_{}".format(i+1)] = df['close'].shift(look_back2) / df['close'].shift(look_back1)
            #-------------------------------------------------------------------
            #-------------------------------------------------------------------
            for i in range(len(SHIFT_POINTS_HOURS)-1):
                L = len(df)
                look_back1 = (SHIFT_POINTS_HOURS[i+0] * 3600) // tf
                look_back2 = (SHIFT_POINTS_HOURS[i+1] * 3600) // tf
                coefs = []
                for row_n in range(L):
                    try:
                        idx1 = row_n - look_back1
                        if idx1 < 0:
                            coefs.append(np.nan)
                            continue
                        idx2 = row_n - look_back2
                        data = df.ix[idx1:idx2,'close'] / df.ix[idx1,'close']
                        data = data.apply(lambda x : math.log10(x))
                        coeff = np.linalg.lstsq(np.reshape(range(len(data)), (-1, 1)), data)[0]
                        #print("Coefficients", coeff)
                        coefs.append(coeff[0])
                        #print(coefs)
                    except Exception as err:
                        print("error is ", err, idx1, idx2, df.ix[idx1,'close'])
                        coefs.append(np.nan)
                df["L_REG_CLOSE_{}".format(i+1)] = coefs

            all_close_coefs = []
            all_vol_coefs = []
            for i, hours in enumerate(REG_HOURS):
                L = len(df)
                look_back1 = (hours * 3600) // tf
                coefs_close, coefs_vol = [], []
                for row_n in range(L):
                    
                    # close
                    idx1 = row_n - look_back1
                    if idx1 < 0:
                        coefs_close.append(np.nan)
                        coefs_vol.append(np.nan)
                        continue
                    else:
                        data = df.ix[idx1:(row_n),'close'] / df.ix[idx1,'close']
                        data = data.apply(lambda x : math.log10(x))
                        coeff = np.linalg.lstsq(np.reshape(range(len(data)), (-1, 1)), data)[0]
                        coefs_close.append(coeff[0])
                    # volume
                    idx1 = row_n - look_back1
                    denom = df.ix[idx1,'volume']
                    loops = 1
                    broke = False
                    while denom == 0:
                        new_idx = idx1 - loops
                        if new_idx < 0:
                            coefs_vol.append(0)
                            broke = True
                            break
                        denom = df.ix[new_idx,'volume']
                        loops += 1
                    if broke:
                        continue
                    data = df.ix[idx1:(row_n),'volume'] / denom - 1
                    coeff = np.linalg.lstsq(np.reshape(range(len(data)), (-1, 1)), data)[0]
                    coefs_vol.append(coeff[0])
                all_close_coefs.append(coefs_close)
                all_vol_coefs.append(coefs_vol)
            
            for i, coefs in enumerate(all_close_coefs):
                df["REG_CLOSE_{}".format(i+1)] = coefs
            for i, coefs in enumerate(all_vol_coefs):
                df["REG_VOLUME_{}".format(i+1)] = coefs
                
            #-------------------------------------------------------------------
            df=df[['date']+[c for c in df.columns if c != 'date']]
            newnames = [c if c == "date" else c+TICK[4:] for c in df.columns]
            df.columns = newnames
            #-------------------------------------------------------------------
            try:
                os.mkdir(tmap[tf])
            except FileExistsError as e1:
                pass
            except OSError as e2:
                print('Failed to create directory {} - Incorrect syntax?'.format(tmap[tf]))
            except:
                print('Error occurred - {}.'.format(sys.exc_info()[0]))                        
            df.to_csv("{}/{}.csv".format(tmap[tf], TICK), index=False)
            print("Wrote: {} {}".format(TICK, tmap[tf]))
            if merged is None:
                merged = df
            else:
                merged = merged.merge(df, how='inner', on='date')        
        merged.to_csv("{}/{}.csv".format(tmap[tf], "ALL"), index=False)

ExtractData(30)
