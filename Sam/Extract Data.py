import json
import time
import hmac,hashlib
import Client_Poloniex
import requests

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


    tickers = []
    for T in ALL_TICKERS:
        tickers.append((T, T[:T.index("_")], T[T.index("_")+1:], 1 if "USDT" in T else 0))

    #coin_list = pd.DataFrame(tickers)
    #coin_list.columns = ['Pair', 'Base','Coin', 'Extract']
    #coin_list.to_csv("coins.csv", index=False)

    tm = time.time()

    tmap    = {86400 : "D1", 14400 : "H4", 7200 : "H2", 1800 : "M30", 900 : "M15", 300 : "M5"}
    highmap = {86400 : 900,  14400 : 300,  7200 : 300,  1800 : 300,   900 : 300,   300 : 300 }

    CALC_R2 = False

    for tf in [86400, 14400, 7200, 1800, 900, 300]:
        merged = None
        for idx in range(len(coin_list)):
            coin = coin_list.iloc[idx,:]
            if not coin.Extract:
                continue
            TICK = coin.Pair
            data = POL.returnChartData(TICK,tf,JUNE_1_2017,tm)
            df = pd.DataFrame(data)
            df['reward'] = df['close'].shift(-1) / df['close'] - 1
            df['reward'] = df.reward.apply(lambda x : math.log(x+1, 10))

            if CALC_R2:
                m5 =   pd.DataFrame(POL.returnChartData(TICK,highmap[tf],0,tm))
                m5['reward'] = m5['close'].shift(0) / m5['close'].shift(1) - 1
                m5['reward'] = m5.reward.apply(lambda x : math.log(x+1, 10))
                
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
            df=df[['date']+[c for c in df.columns if c != 'date']]
            newnames = [c if c == "date" else c+TICK[4:] for c in df.columns]
            df.columns = newnames
            
            df.to_csv("{}/{}.csv".format(tmap[tf], TICK), index=False)
            print("Wrote: {} {}".format(TICK, tmap[tf]))
            if merged is None:
                merged = df
            else:
                merged = merged.merge(df, how='inner', on='date')
        merged.to_csv("{}/{}.csv".format(tmap[tf], "ALL"), index=False)

ExtractData(10)
