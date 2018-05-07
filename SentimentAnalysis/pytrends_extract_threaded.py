# -*- coding: utf-8 -*-
"""
Created on Fri May 04 21:41:47 2018

@author: rendorHaevyn
URL: https://github.com/GeneralMills/pytrends
"""

## TODO re-cut interest by 8 minute intervals into interest by 15 minute intervals 


## IMPORTS
from __future__ import print_function
from pytrends.request import TrendReq
import pandas as pd
import os
import sys
from dateutil import rrule
from datetime import datetime, timedelta
from threading import Thread
import time

sys.path.insert(0, "C:\\Users\\Admin\\Documents\\GitHub\\Project_WinLife\\SentimentAnalysis\\")
import LogThread

## LOAD DATA
OUTDIR           = 'C:\\Users\\Admin\\Documents\\GitHub\\Project_WinLife\\SentimentAnalysis\\pt_data'
INDIR           = 'C:\\Users\\Admin\\Documents\\GitHub\\Project_WinLife\\SentimentAnalysis\\pt_inputs'
os.chdir(OUTDIR)
kw_df = pd.read_csv(INDIR + os.sep + 'kw.csv',delimiter='|')

## HACK FOR GOOGLE 429 ERRORS
GGL_HACK        = 1


## CONSTANTS
CATEGORY        = 107 # INVESTING
PROPERTY        = 'news' # SET TO EMPTY, ELSE images, news, youtube or froogle
GEOLOC          = '' # SET TO EMPTY, ELSE 2 LETTER COUNTRY ABBREVIATION
NOW             = datetime.utcnow() #datetime(2018, 5, 6, 10, 58, 14, 198000) - locking this to current to baseline all coins
WKS_BACK        = 12
YR_BACK         = NOW - timedelta(weeks=WKS_BACK)
DAY_CNT         = WKS_BACK * 7
COIN_CNT        = len(kw_df)

# Login to Google. Only need to run this once, the rest of requests will use the same session.
pytrend = TrendReq()

# Populate list of days in prior year    
day_lst = list(rrule.rrule(rrule.DAILY, dtstart=YR_BACK, until=NOW))


# Func to use in threading
def get_trend(i,results,coin):
    s_tf = day_lst[i].strftime("%Y-%m-%dT00") + ' ' + day_lst[i+1].strftime("%Y-%m-%dT00")
    #print('Fetching: coin - {}, day - {}'.format(coin,s_tf))
    sys.stdout.write('Fetching: coin - {}, day - {}\r'.format(coin,s_tf))
    sys.stdout.flush()    
    pytrend.build_payload(kw_list   = kw
                        ,cat        = CATEGORY
                        ,geo        = GEOLOC
                        ,gprop      = PROPERTY
                        ,timeframe  = s_tf
                        )
    iot_df = pytrend.interest_over_time()
    iot_df = iot_df.drop(['isPartial'],axis=1)
    results[i] = iot_df

# Create empty lists for threads and data frame results
threads     = [None] * DAY_CNT
results     = [None] * DAY_CNT
coin_trends = [None] * COIN_CNT
df_consolidated = pd.DataFrame()

# Iterate keyword list by coin of interest
for indx,vals in kw_df.iterrows():
#    if indx >= 2:
#        break
    if indx > 0 and GGL_HACK == 1: # Restriction to usurp getting google 429 error - too-many-requests
        for i in range(10): 
            sys.stdout.write('Sleeping for next coin - {} of 30\r'.format(i))
            time.sleep(1)
            sys.stdout.flush()
    if indx >= 0:
        kw = vals['kw_lst'].split(',')
    # Iterate days in period
        coin_trends[indx] = pd.DataFrame()
        for i in range(DAY_CNT):
            if  GGL_HACK == 1:  # Restriction to usurp getting google 429 error - too-many-requests
                time.sleep(2)
            threads[i] = Thread(target=get_trend, args=(i,results,vals['coin']))
            threads[i].start()        
        for i in range(DAY_CNT):
            threads[i].join()
        coin_trends[indx] = coin_trends[indx].append(results)
        coin_trends[indx] = coin_trends[indx].sort_index()
        # Export file
        coin_trends[indx].to_csv(OUTDIR 
                            + os.sep 
                            + 'coin_kw_trends_{}_{}_{}_at-{}.csv'.format(vals['coin']
                            ,NOW.strftime("%Y-%m-%d-%H%M")
                            ,YR_BACK.strftime("%Y-%m-%d")
                            ,NOW.strftime("%H%M")))


# Mergetime series pytrend data frames
df_consolidated = pd.concat(coin_trends, axis=1)

# Export file
df_consolidated.to_csv(OUTDIR 
                    + os.sep 
                    + 'coin_kw_trends_complete_{}_{}_at-{}.csv'
                    .format(
                    NOW.strftime("%Y-%m-%d-%H%M")
                    ,YR_BACK.strftime("%Y-%m-%d")
                    ,NOW.strftime("%H%M")))



# Interest by Region
"""
ibr_df = pytrend.interest_by_region(resolution='CITY') #COUNTRY/CITY/DMA/REGION - 'REGION' seems to fail
print(ibr_df.head())
ibr_gt0 = ibr_df[(ibr_df['IOTA'] > 0) | 
                (ibr_df['iOTA'] > 0)  |
                (ibr_df['iota'] > 0)  |
                (ibr_df['Iota'] > 0)
                ]
"""

import string
import random
def random_word(length):
    """Return a random word of 'length' letters."""
    return ''.join(random.choice(string.ascii_letters) for i in range(length))
