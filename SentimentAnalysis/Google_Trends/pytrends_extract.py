# -*- coding: utf-8 -*-
"""
Created on Fri May 04 21:41:47 2018

@author: rendorHaevyn
URL: https://github.com/GeneralMills/pytrends
"""

## IMPORTS

from __future__ import print_function
from pytrends.request import TrendReq
import pandas as pd
import os
from dateutil import rrule
from datetime import datetime, timedelta

## CONSTANTS
INDIR           = 'C:\\Users\\Admin\\Documents\\GitHub\\Project_WinLife\\SentimentAnalysis\\Google_Trends\\pt_inputs'
OUTDIR          = 'C:\\Users\\Admin\\Documents\\GitHub\\Project_WinLife\\SentimentAnalysis\\Google_Trends\\pt_data'
CATEGORY        = 0 # 107 = INVESTING, 0 = everything
PROPERTY        = '' # SET TO EMPTY, ELSE images, news, youtube or froogle
GEOLOC          = '' # SET TO EMPTY, ELSE 2 LETTER COUNTRY ABBREVIATION
NOW             = datetime.utcnow()
YR_BACK         = NOW - timedelta(weeks=12)
S_TF            = 'today 3-m' # 3 months back
IOT_TYPE        = 'daily' # 'minute' for 8-minute daily extract or 'daily' for daily monthly extract
#YR_BACK         = NOW - timedelta(weeks=52)


## LOAD DATA
os.chdir(OUTDIR)
kw_df = pd.read_csv(INDIR + os.sep + 'kw.csv',delimiter='|')


# Login to Google. Only need to run this once, the rest of requests will use the same session.
pytrend = TrendReq()

# Populate list of days in prior year    
day_lst = list(rrule.rrule(rrule.DAILY, dtstart=YR_BACK, until=NOW))

# Iterate keyword list by coin of interest
df_consolidated = pd.DataFrame()
for indx,vals in kw_df.iterrows():
    #print('PyTrend for {}, using keys {}.'.format(vals['coin'],vals['kw_lst']))
    kw = vals['kw_lst'].split(',')
    if IOT_TYPE == 'minute': # iterate days for minute data
        df_trend = pd.DataFrame()
        for i in range(len(day_lst)-1):
            s_tf = day_lst[i].strftime("%Y-%m-%dT00") + ' ' + day_lst[i+1].strftime("%Y-%m-%dT00")
            print('Fetching: coin - {}, day - {}'.format(vals['coin'],s_tf))
            pytrend.build_payload(kw_list   = kw
                                ,cat        = CATEGORY
                                ,geo        = GEOLOC
                                ,gprop      = PROPERTY
                                ,timeframe  = s_tf
                                )
            iot_df = pytrend.interest_over_time()
            iot_df = iot_df.drop(['isPartial'],axis=1)
            df_trend = df_trend.append(iot_df)
    else: # iterate months for daily data
        df_trend = pd.DataFrame()
        print('Fetching: coin - {}, day - {}'.format(vals['coin'],S_TF))
        pytrend.build_payload(kw_list   = kw
                            ,cat        = CATEGORY
                            ,geo        = GEOLOC
                            ,gprop      = PROPERTY
                            ,timeframe  = S_TF
                            )
        iot_df = pytrend.interest_over_time()
        iot_df = iot_df.drop(['isPartial'],axis=1)
        df_trend = df_trend.append(iot_df)        
    # Mergetime series pytrend data frames
    df_consolidated = pd.concat([df_consolidated,df_trend], axis=1)

# Export file
df_consolidated.to_csv(OUTDIR + os.sep + 'coin_kw_trends_monthly_{}.csv'.format(datetime.utcnow().strftime('%d%Y')))



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