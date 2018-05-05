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
WKDIR           = 'C:\\Users\\Admin\\Documents\\GitHub\\Project_WinLife\\SentimentAnalysis\\pt_data'
CATEGORY        = 107 # INVESTING
PROPERTY        = 'news' # SET TO EMPTY, ELSE images, news, youtube or froogle
GEOLOC          = '' # SET TO EMPTY, ELSE 2 LETTER COUNTRY ABBREVIATION
NOW             = datetime.utcnow()
#YR_BACK         = now - timedelta(weeks=1)
YR_BACK         = NOW - timedelta(weeks=52)


## LOAD DATA
os.chdir(WKDIR)
kw_df = pd.read_csv(WKDIR + os.sep + 'kw.csv',delimiter='|')


## TODO Parallel process pytrends calls
## TODO re-cut interest by 8 minute intervals into interest by 15 minute intervals


# Login to Google. Only need to run this once, the rest of requests will use the same session.
pytrend = TrendReq()

# Populate list of days in prior year    
day_lst = list(rrule.rrule(rrule.DAILY, dtstart=YR_BACK, until=NOW))

# Iterate keyword list by coin of interest
df_consolidated = pd.DataFrame()
for indx,vals in kw_df.iterrows():
    #print('PyTrend for {}, using keys {}.'.format(vals['coin'],vals['kw_lst']))
    kw = vals['kw_lst'].split(',')
# Iterate days in period
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
# Mergetime series pytrend data frames
    df_consolidated = pd.concat([df_consolidated,df_trend], axis=1)

# Export file
df_consolidated.to_csv(WKDIR + os.sep + 'coin_kw_trends.csv')



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