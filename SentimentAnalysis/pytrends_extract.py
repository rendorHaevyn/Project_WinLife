# -*- coding: utf-8 -*-
"""
Created on Fri May 04 21:41:47 2018

@author: rendorHaevyn
URL: https://github.com/GeneralMills/pytrends
"""

## IMPORTS

from pytrends.request import TrendReq
import pandas as pd
import os

## CONSTANTS
wkdir = 'C:\\Users\\Admin\\Documents\\GitHub\\Project_WinLife\\SentimentAnalysis\\pt_data'

## LOAD DATA
os.chdir(wkdir)
kw_df = pd.read_csv(wkdir + os.sep + 'kw.csv',delimiter='|')


## TODO 1. iterate years, months, days, hours for timeframe
## TODO 2. iterate keyword lists 
## TODO 3. generate interest-over-time overall table (8 minute intervals)
## TODO 4. generate interest by region / city hourly table

# Login to Google. Only need to run this once, the rest of requests will use the same session.
pytrend = TrendReq()

# Iterate keyword list by coin of interest
for indx,vals in kw_df.iterrows():
    print('Generating trends for {}.'.format(vals['coin']))
    kw = vals['kw_lst']

## TODO - here, iterate hour/day/year and make call to def function to call / create tables.

# Create payload and capture API tokens. Only needed for interest_over_time(), interest_by_region() & related_queries()
pytrend.build_payload(kw_list=kw
                    ,cat=107
                    ,geo=''
                    ,timeframe='2017-12-01T00 2017-12-02T00'
                    ,gprop='news'
                    ) # cat = Investing

# Interest Over Time
iot_df = pytrend.interest_over_time() 
print(iot_df.head())

# Interest by Region
ibr_df = pytrend.interest_by_region(resolution='CITY') #COUNTRY/CITY/DMA/REGION - 'REGION' seems to fail
print(ibr_df.head())
ibr_gt0 = ibr_df[(ibr_df['IOTA'] > 0) | 
                (ibr_df['iOTA'] > 0)  |
                (ibr_df['iota'] > 0)  |
                (ibr_df['Iota'] > 0)
                ]


# Related Queries, returns a dictionary of dataframes
related_queries_dict = pytrend.related_queries()
print(related_queries_dict)

# Get Google Hot Trends data
trending_searches_df = pytrend.trending_searches()
print(trending_searches_df.head())

# Get Google Top Charts
top_charts_df = pytrend.top_charts(cid='actors', date=201611)
print(top_charts_df.head())

# Get Google Keyword Suggestions
suggestions_dict = pytrend.suggestions(keyword='pizza')
print(suggestions_dict)
