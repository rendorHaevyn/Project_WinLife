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

N_LAGS = 30
TIME_FRAME = Constants.TF_15M
TF_IN_MINS = Constants.TF_TO_MIN[TIME_FRAME]

df = pd.read_csv("{}/ALL.csv".format(TIME_FRAME))

cut_off_date = int(time.mktime(time.strptime('01/12/2017', "%d/%m/%Y"))) * 1000
#df = df[df['date'] > cut_off_date].reset_index(drop=True)

COINS = [x[len("close_"):] for x in df.columns if "close_" in x]

#SHIFT_POINTS_HOURS = [24 * 14, 24 * 7, 24 * 3, 24, 12, 4, 2, 1]
#REG_HOURS          = [24 * 14, 24 * 7, 24 * 3, 24, 12, 4, 2, 1]

SHIFT_POINTS_HOURS = [24, 12, 4, 2, 1]
REG_HOURS          = [24, 12, 4, 2, 1]

#SHIFT_POINTS_HOURS = []
#REG_HOURS          = []

L2_SCALE = 5

for C in COINS:
    
    print("Feature Engineering {}".format(C))
    df['volume_'+C] = df['volume_'+C] * df['close_'+C]
    df['reward_'+C] = df['close_'+C].shift(-1) / df['close_'+C]
    df['reward_'+C] = df["reward_"+C].apply(lambda x : math.log10(x))
    
    cut_low = 0.95
    cut_high = 1.05

    limits = []
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
    
    df['limit_'+C] = limits
        
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
                df["{}_{}_{}".format(col_type, lag, C)] = df["volume_"+C].shift((lag))
                #df["{}_{}_{}".format(col_type, lag, C)] = df[col+"_"+C].shift((lag)) / df['volume_'+C]
            else:
                df["{}_{}_{}".format(col_type, lag, C)] = df[col+"_"+C].shift((lag)) / df['close_'+C]
            df["{}_{}_{}".format(col_type, lag, C)] = df["{}_{}_{}".format(col_type, lag, C)].apply(lambda x : math.log10(x))
            
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
    
    print("{}: PRICE CHANGE".format(C))
    #-------------------------------------------------------------------
    for i in range(len(SHIFT_POINTS_HOURS)-1):
        look_back1 = (SHIFT_POINTS_HOURS[i+0] * 60) // TF_IN_MINS
        look_back2 = (SHIFT_POINTS_HOURS[i+1] * 60) // TF_IN_MINS
        df["L_CHNG_{}_{}".format(i+1,C)] = df['close_'+C].shift(look_back2) / df['close_'+C].shift(look_back1)
    print("{}: PIECEWISE".format(C))
    #-------------------------------------------------------------------
    for i in range(len(SHIFT_POINTS_HOURS)-1):
        L = len(df)
        look_back1 = (SHIFT_POINTS_HOURS[i+0] * 60) // TF_IN_MINS
        look_back2 = (SHIFT_POINTS_HOURS[i+1] * 60) // TF_IN_MINS
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
        df["L_REG_CLOSE_{}_{}".format(i+1, C)] = coefs

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
    
    for i, coefs in enumerate(all_close_coefs):
        df["REG_CLOSE_{}_{}".format(i+1,C)] = coefs
    for i, coefs in enumerate(all_vol_coefs):
        df["REG_VOLUME_{}_{}".format(i+1,C)] = coefs
        
df.to_csv("{}/{}.csv".format(TIME_FRAME, "ALL_MOD"), index=False)
