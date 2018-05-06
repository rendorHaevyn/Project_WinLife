
"""
https://github.com/jblemoine/Big_data_trading_algorithm/blob/master/Pytrends.ipynb
"""

## BATCH QUERY

#import modules to query data
import socket
import socks

import requests
from bs4 import BeautifulSoup

#connect to Tor Browser
from stem import Signal
from stem.control import Controller
controller = Controller.from_port(port=9151)
controller.authenticate()

def connectTor():
    socks.setdefaultproxy(socks.PROXY_TYPE_SOCKS5 , "127.0.0.1", 9150, True)
    socket.socket = socks.socksocket
        
def disconnectTor():
    controller.close()
        
def renew_tor():
    controller.signal(Signal.NEWNYM)

def showmyip():
    url = "http://www.showmyip.gr/"
    r = requests.Session()
    page = r.get(url)
    soup = BeautifulSoup(page.content, "lxml")
    ip_address = soup.find("span",{"class":"ip_address"}).text.strip()
    print(ip_address)
    
    
## GENERAL BATCH QUERY
import pandas as pd
import numpy as np
from io import StringIO
import requests
import os
import re
import time
import datetime
from random import randint
#payload  = {'q': 'apple', 'geo':'US','start_date': '2012-01-25','end_date': '2017-01-25'} payload style
# This script is based on unofficial Pytrends API  https://github.com/GeneralMills/pytrends 
#As Google has changed recently their API, I manually changed the script to fit my requirements  

def PyTrends(payload):
    #first we need to get the token from Google Trends
    #get the url with the token 
    r = requests.get(
                "https://www.google.com/trends/api/explore?hl=en-US&tz=360&req=%7B%22comparisonItem%22:%5B%7B%22keyword%22:%22{0}%22,%22geo%22:%22{1}%22,%22time%22:%22{2}%20{3}%22%7D%5D,%22category%22:0,%22property%22:%22%22%7D&tz=360" .format(payload['q'],payload['geo'],payload['start_date'],payload['end_date']))
    match = re.search('token', r.text)
    
    start = match.start()
    token = r.text[start + 8:start + 52] #get the token
    
    #download results
    req_url ="https://www.google.com/trends/api/widgetdata/multiline/csv?req=%7B%22time%22%3A%22{0}%20{1}%22%2C%22resolution%22%3A%22DAY%22%2C%22locale%22%3A%22en-US%22%2C%22comparisonItem%22%3A%5B%7B%22geo%22%3A%7B%22country%22%3A%22US%22%7D%2C%22complexKeywordsRestriction%22%3A%7B%22keyword%22%3A%5B%7B%22type%22%3A%22BROAD%22%2C%22value%22%3A%22{2}%22%7D%5D%7D%7D%5D%2C%22requestOptions%22%3A%7B%22property%22%3A%22%22%2C%22backend%22%3A%22IZG%22%2C%22category%22%3A0%7D%7D&token={3}&tz=360" .format(payload['start_date'],payload['end_date'],payload['q'],token)

    req = requests.get(req_url)
    text=req.text
    data=StringIO(text)
    df=pd.read_csv(data,sep='\n',delimiter=',',skiprows=1,index_col=0)[::-1]
    return df

### Return all daily data From google trends

def Pytrends_daily(payload):
    start_date=str(payload['start_date'])
    end_date=str(payload['end_date'])
    keyword=str(payload['q'])
    
    """First Period"""
    period_end_date=end_date
    print(period_end_date)
    
    #compute the beginning date of the period
#    d=datetime.datetime.strptime(end_date, '%Y-%m-%d')+pd.DateOffset(months=-8)  #offset 8 months ealier max if daily 
#    period_start_date=d.strftime('%Y-%m-%d')
    d=datetime.datetime.strptime(start_date, '%Y-%m-%d')
    period_start_date=d.strftime('%Y-%m-%d')
    print(period_start_date)    
    
    payload= {'q': keyword, 'geo':'US','start_date': period_start_date,'end_date': period_end_date}

    #request data from Google Trends for the first period
    dailyData=PyTrends(payload)
    
    #for the period the end_date is the start_date of the previous period
    period_end_date=period_start_date
  
    RemainingTime=d-datetime.datetime.strptime(start_date, '%Y-%m-%d')
    
    """ Loop"""
    while   RemainingTime.days >0:
        
        d=datetime.datetime.strptime(period_end_date, '%Y-%m-%d')+pd.DateOffset(months=-8)  #offset 8 months ealier max if daily 
        period_start_date=d.strftime('%Y-%m-%d')
        payload= {'q': keyword, 'geo':'US','start_date': period_start_date,'end_date': period_end_date}
        
        #request data from Google Trends for the new period 
        NewdailyData=PyTrends(payload)
        
        #for the period the end_date is the start_date of the previous period
        period_end_date=period_start_date
        
        NewdailyData=NewdailyData[NewdailyData.index<=dailyData.index[-1]] 
        
        #compute the coef in order to rescale the new data
        MatrixFullCoef= pd.DataFrame(index=NewdailyData.index, columns=NewdailyData.columns)
        #print(dailyData[-1:].index==NewdailyData[:1].index)
        MatrixFullCoef[:1]=[np.mean(dailyData[-15:])[0]/np.mean(NewdailyData[:15])[0]]
        
        for i in range(len(MatrixFullCoef)):
            MatrixFullCoef.iloc[i]=MatrixFullCoef.iloc[0]
        
        #Normalize the NewDailyData with the coef calculated before
        AdjustedNewdailyData=(NewdailyData*MatrixFullCoef)[1:]
        
        #add the new data Frame of daily datas to the existing one
        dailyData= dailyData.append(AdjustedNewdailyData,ignore_index=False)
        
        RemainingTime=d-datetime.datetime.strptime(start_date, '%Y-%m-%d')
        
    return dailyData

#Load the inputs 
WKDIR           = 'C:\\Users\\Admin\\Documents\\GitHub\\Project_WinLife\\SentimentAnalysis\\pt_data'
os.chdir(WKDIR)

#Input=pd.read_excel('C:/Users/Jean-Baptiste/Google Drive/M2/GoogleTrendsData/google/Inputs.xlsx', na_values='NA')
Output=pd.DataFrame()

ST_DATE         = '2016-05-29'
END_DATE        = '2017-01-29'

#for name in Input['Name']:
for name in ['miota']:
    count=0
    try:
        payload = {'q': str(name), 'geo':'US','start_date': ST_DATE,'end_date': END_DATE}
        Data=Pytrends_daily(payload)
        Output=pd.concat([Output,Data],axis=1,join='outer')
        Output.to_excel('Output.xlsx') 
        print('request ok')
#        print('{} restants'.format(len(Input)-len(Output.T)))
        
    except:
        print('request failed')
        time.sleep(randint(10,25))
        renew_tor()
        connectTor()
        showmyip()
        count+=1
        if count>5:
            pass
