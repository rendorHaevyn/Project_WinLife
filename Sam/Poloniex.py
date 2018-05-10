import urllib
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2
import json
import time
import hmac,hashlib
import poloniex as pol
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
import sklearn as sk
import math
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import sklearn.ensemble
import sklearn.neighbors
import sklearn.neural_network
import sklearn.naive_bayes
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn import linear_model
from scipy import stats
import scipy
import utils
import traceback

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import _tree, export_graphviz

import matplotlib.pyplot as plt

from sklearn.utils import check_random_state

def createTimeStamp(datestr, format="%Y-%m-%d %H:%M:%S"):
    return time.mktime(time.strptime(datestr, format))

class poloniex:
    def __init__(self, APIKey, Secret):
        self.APIKey = APIKey
        self.Secret = Secret

    def post_process(self, before):
        after = before

        # Add timestamps if there isnt one but is a datetime
        if('return' in after):
            if(isinstance(after['return'], list)):
                for x in range(0, len(after['return'])):
                    if(isinstance(after['return'][x], dict)):
                        if('datetime' in after['return'][x] and 'timestamp' not in after['return'][x]):
                            after['return'][x]['timestamp'] = float(createTimeStamp(after['return'][x]['datetime']))
                            
        return after

    def api_query(self, command, req={}):

        if(command == "returnTicker" or command == "return24Volume"):
            ret = urllib2.urlopen(urllib2.Request('https://poloniex.com/public?command=' + command))
            return json.loads(ret.read().decode('utf8'))
        elif(command == "returnOrderBook"):
            ret = urllib2.urlopen(urllib2.Request('https://poloniex.com/public?command=' + command + '&currencyPair=' + str(req['currencyPair'])))
            return json.loads(ret.read().decode('utf8'))
        elif(command == "returnMarketTradeHistory"):
            ret = urllib2.urlopen(urllib2.Request('https://poloniex.com/public?command=' + "returnTradeHistory" + '&currencyPair=' + str(req['currencyPair'])))
            return json.loads(ret.read().decode('utf8'))
        else:
            req['command'] = command
            req['nonce'] = int(time.time()*1000)
            post_data = urllib.parse.urlencode(req).encode('utf8')

            sign = hmac.new(self.Secret, post_data, hashlib.sha512).hexdigest()
            headers = {
                'Sign': sign,
                'Key': self.APIKey
            }

            ret = requests.post('https://poloniex.com/tradingApi', data=req, headers=headers)
            jsonRet = json.loads(ret.text)
            #ret = urllib2.urlopen(urllib2.Request('https://poloniex.com/tradingApi', post_data, headers))
            #jsonRet = json.loads(ret.read().decode('utf8'))
            return self.post_process(jsonRet)


    def returnTicker(self):
        return self.api_query("returnTicker")

    def return24Volume(self):
        return self.api_query("return24Volume")

    def returnOrderBook (self, currencyPair):
        return self.api_query("returnOrderBook", {'currencyPair': currencyPair})

    def returnMarketTradeHistory (self, currencyPair):
        return self.api_query("returnMarketTradeHistory", {'currencyPair': currencyPair})


    # Returns all of your balances.
    # Outputs: 
    # {"BTC":"0.59098578","LTC":"3.31117268", ... }
    def returnBalances(self):
        return self.api_query('returnBalances')

    # Returns your open orders for a given market, specified by the "currencyPair" POST parameter, e.g. "BTC_XCP"
    # Inputs:
    # currencyPair  The currency pair e.g. "BTC_XCP"
    # Outputs: 
    # orderNumber   The order number
    # type          sell or buy
    # rate          Price the order is selling or buying at
    # Amount        Quantity of order
    # total         Total value of order (price * quantity)
    def returnOpenOrders(self,currencyPair):
        return self.api_query('returnOpenOrders',{"currencyPair":currencyPair})


    # Returns your trade history for a given market, specified by the "currencyPair" POST parameter
    # Inputs:
    # currencyPair  The currency pair e.g. "BTC_XCP"
    # Outputs: 
    # date          Date in the form: "2014-02-19 03:44:59"
    # rate          Price the order is selling or buying at
    # amount        Quantity of order
    # total         Total value of order (price * quantity)
    # type          sell or buy
    def returnTradeHistory(self,currencyPair):
        return self.api_query('returnTradeHistory',{"currencyPair":currencyPair})

    # Places a buy order in a given market. Required POST parameters are "currencyPair", "rate", and "amount". If successful, the method will return the order number.
    # Inputs:
    # currencyPair  The curreny pair
    # rate          price the order is buying at
    # amount        Amount of coins to buy
    # Outputs: 
    # orderNumber   The order number
    def buy(self,currencyPair,rate,amount):
        return self.api_query('buy',{"currencyPair":currencyPair,"rate":rate,"amount":amount})

    # Places a sell order in a given market. Required POST parameters are "currencyPair", "rate", and "amount". If successful, the method will return the order number.
    # Inputs:
    # currencyPair  The curreny pair
    # rate          price the order is selling at
    # amount        Amount of coins to sell
    # Outputs: 
    # orderNumber   The order number
    def sell(self,currencyPair,rate,amount):
        return self.api_query('sell',{"currencyPair":currencyPair,"rate":rate,"amount":amount})

    # Cancels an order you have placed in a given market. Required POST parameters are "currencyPair" and "orderNumber".
    # Inputs:
    # currencyPair  The curreny pair
    # orderNumber   The order number to cancel
    # Outputs: 
    # succes        1 or 0
    def cancel(self,currencyPair,orderNumber):
        return self.api_query('cancelOrder',{"currencyPair":currencyPair,"orderNumber":orderNumber})

    # Immediately places a withdrawal for a given currency, with no email confirmation. In order to use this method, the withdrawal privilege must be enabled for your API key. Required POST parameters are "currency", "amount", and "address". Sample output: {"response":"Withdrew 2398 NXT."} 
    # Inputs:
    # currency      The currency to withdraw
    # amount        The amount of this coin to withdraw
    # address       The withdrawal address
    # Outputs: 
    # response      Text containing message about the withdrawal
    def withdraw(self, currency, amount, address):
        return self.api_query('withdraw',{"currency":currency, "amount":amount, "address":address})
    
        
POL = poloniex("2FZIK8M1-SB3G51C8-8FOHONNK-WSDK2EXU",
               "39630d5e58e75ba33e0aa8dc82f585f6d06d5aaec5c59479ba8554dc2782c49350e0badf2f1bce4e1a1c58db982a4677cee65bed25b9cb33a2801ad956cd256b")
POL.Secret = POL.Secret.encode('utf8')


class Trade:
    
    PAIR_PRICES = {}
    
    def __init__(self, pair):
        self.openOrder = []
        self.closeOrder = []
        self.pair = pair
        self.coins = 0
        self.openBTC = 0
        self.openPrice = 0
        self.openFee = 0
        self.closeFee = 0
        self.closePrice = 0
        self.profit = 0
        self.openOrderID = '0'
        self.closeOrderID = '0'
        self.matchedBuy = False
        self.matchedSell = False
        
    def setOrder(self, dic):
        self.openOrder = dic
        self.coins = float(dic['amount'])
        self.openBTC = float(dic['total'])
        self.openPrice = float(dic['rate'])
        self.openFee = float(dic['fee'])
        
class TradingPair:
    
    def __init__(self, pair):
        self.openTrades = []
        self.closedTrades = []
        self.profit = 0
        self.profEq = 0
        self.profitBTC = 0
        self.profitBTC_EQ = 0
        self.pair = pair
        self.nxt_open = 99999 
        self.nxt_close = 0

ALL_TICKERS = sorted(POL.returnTicker().keys())

TRADING_PAIRS = [TradingPair(t) for t in ALL_TICKERS if "BTC_" in t and not (t == "BTC_DGB" or t == "BTC_SC")]
'''TRADING_PAIRS = [TradingPair("BTC_SC"), 
                 TradingPair("BTC_NXC"), 
                 TradingPair("BTC_DOGE"), 
                 TradingPair("BTC_OMNI"),
                 TradingPair("BTC_GNT"),
                 TradingPair("BTC_ETH"),
                 TradingPair("BTC_LTC"),
                 TradingPair("BTC_BCN"),
                 TradingPair("BTC_STEEM"),
                 TradingPair("BTC_PINK"),
                 TradingPair("BTC_STRAT")]'''

#TRADING_PAIRS = [TradingPair("BTC_OMNI")]

pendingOpenIDs = []
pendingCloseIDs = []
NET_GAIN = []
FLOATING_GAIN = []
RR = 1.04
GAP_OPEN = 0.97
PERCENT_OPEN = 0.01
loop = True
while loop:
    USD_PRICE = float(POL.returnTicker()['USDT_BTC']['last'])
    for TP in TRADING_PAIRS:
        try:
            PAIR = TP.pair
            '''COIN = POL.returnOpenOrders(PAIR)
            if 'error' in COIN:
                continue
            COIN_ORDER_BUY = [c for c in COIN if c['type'] == 'buy']
            COIN_ORDER_SELL = [c for c in COIN if c['type'] == 'sell']'''
            COIN_LAST = POL.returnTicker()[PAIR]
            if 'error' in COIN_LAST:
                continue
            
            COIN_PRICE = round(float(COIN_LAST['last']),8)
            Trade.PAIR_PRICES[PAIR] = COIN_PRICE
            COIN_HIGH = round(float(COIN_LAST['high24hr']),8)
            COIN_LOW = round(float(COIN_LAST['low24hr']),8)
            COIN_MID = round((0.5*COIN_HIGH+0.5*COIN_LOW),8)
            
            #COIN_MID = 0.00000110
            RR_act = (1+(RR-1))**(1+0.05*(len(TP.openTrades)-1))
            shouldClose = [t for t in TP.openTrades if t.openPrice > 0 and 
                                                       COIN_PRICE / t.openPrice > RR_act and 
                                                       t.closeOrderID == '0']
            #print(RR_act)
                                                       
            if len(TP.openTrades) == 1 and COIN_PRICE < COIN_MID:
                shouldClose = []

            for t in shouldClose:
                amtHolding = float(POL.returnBalances()[PAIR[4:]])
                print(PAIR + " CLOSING: {:.8f} {:.8f}, {} {}".
                format(t.openPrice, COIN_PRICE, t.coins, amtHolding))
                ret = POL.sell(PAIR, COIN_PRICE*0.995, min(amtHolding, t.coins))
                if 'error' in ret:
                    print(shouldClose)
                    print(TP.openTrades)
                    print(len(shouldClose))
                else:
                    t.closeOrderID = ret['orderNumber']
                    pendingCloseIDs.append((PAIR, ret['orderNumber']))
            TP.profit = 0
            for closed in TP.closedTrades:
                TP.profit += USD_PRICE * closed.profit
            TP.profEq = TP.profit
            for openT in TP.openTrades:
                gain = (COIN_PRICE - openT.openPrice)*openT.coins
                TP.profEq += USD_PRICE * gain
            if len(TP.openTrades) == 0:
                TP.nxt_open = 99999
            else: 
                TP.nxt_open = (1-(1-GAP_OPEN)*len(TP.openTrades))*max([t.openPrice for t in TP.openTrades])
            if COIN_PRICE < COIN_MID and COIN_PRICE < TP.nxt_open:
                print(PAIR, TP.nxt_open)
                btc_held = float(POL.returnBalances()['BTC'])
                if btc_held > 0.0001:
                    book = POL.returnOrderBook(PAIR)
                    asks = [[float(x[0]), x[1]] for x in book['asks']]
                    bids = [[float(x[0]), x[1]] for x in book['bids']]
                    p = COIN_PRICE
                    rng = 0.10 * p
                    pmax = p+rng
                    pmin = p-rng
                    asks_r = [a for a in asks if a[0] < p + rng]
                    bids_r = [b for b in bids if b[0] > p - rng]
                    asks_t = sum([a[0]*a[1] for a in asks_r])
                    bids_t = sum([b[0]*b[1] for b in bids_r])
                    asks_ts = sum([a[0]*a[1] * (1-(a[0]-p)/rng)**1 for a in asks_r])
                    bids_ts = sum([b[0]*b[1] * (1-(p-b[0])/rng)**1 for b in bids_r])
                    openSize_BTC = max(0.00012, btc_held*PERCENT_OPEN)
                    rat = 1
                    try:
                        rat = (bids_ts/asks_ts)**0.5
                    except:
                        pass
                    openSize_BTC *= max(min(rat**0.5, 4),1)
                    openSize_BTC = max(0.00012, openSize_BTC)
                    openSize_BTC = 0.001
                    openSize_COINS = openSize_BTC / COIN_PRICE
                    val_b = POL.buy(PAIR, 1.1*COIN_PRICE, openSize_COINS)
                    if 'orderNumber' in val_b:
                        nt = Trade(PAIR)
                        nt.openOrderID = val_b['orderNumber']
                        nt.openPrice = COIN_PRICE
                        TP.openTrades.append(nt)
                        pendingOpenIDs.append((PAIR, val_b['orderNumber'], btc_held))
                    elif 'error' not in val_b:
                        pass
                    else:
                        print(val_b)
                        print("{} {:.8f} {:.8f}".format(PAIR, openSize_BTC, openSize_COINS))
            
            for val in pendingOpenIDs:
                pair = val[0]
                ID = val[1]
                for T in [t for t in TP.openTrades if not t.matchedBuy]:
                    if ID == T.openOrderID:
                        hist = POL.returnTradeHistory(PAIR)
                        for H in hist:
                            if H['orderNumber'] == T.openOrderID:
                                T.openOrder = H
                                T.coins += float(H['amount'])
                                T.openBTC += float(H['total'])
                                T.openPrice = float(H['rate'])
                                T.openFee = float(H['fee'])
                                print("Bought {:.8f} {} Coins at {:.8f} each ({:.2f}%)".format(T.coins,
                                      PAIR, T.openPrice, 100*T.openBTC/val[2]))
                                T.matchedBuy = True
                                if val in pendingOpenIDs:
                                    pendingOpenIDs.remove(val)
               
            for val in pendingCloseIDs:
                pair = val[0]
                ID = val[1]
                for t in TP.openTrades:
                    if ID == t.closeOrderID:
                        try:
                            theTrade = [p for p in POL.returnTradeHistory(PAIR) if p['orderNumber']==ID]
                            print(theTrade[0])
                        except:
                            continue
                        print("Found Sale: ", theTrade)
                        TP.openTrades.remove(t)
                        t.closeOrder = theTrade
                        t.closePrice = sum([float(x['rate']) for x in theTrade])/len(theTrade)
                        t.closeBTC = sum([float(x['total']) for x in theTrade])
                        t.coinProfit = t.closePrice - t.openPrice
                        t.profit = t.closeBTC - t.openBTC
                        prof = t.profit * USD_PRICE
                        print("Closed {} Trade for ${:.2f} Profit".format(PAIR, prof))
                        TP.closedTrades.append(t)
                        TP.profit = 0
                        for closed in TP.closedTrades:
                            TP.profit += USD_PRICE * closed.profit
                            TP.profitBTC += closed.profit
                        TP.profEq = TP.profit
                        for openT in TP.openTrades:
                            gain = (COIN_PRICE - openT.openPrice)*openT.coins
                            TP.profitBTC_EQ += gain
                            TP.profEq += USD_PRICE * gain
                        netGain = 0
                        netGainEq = 0
                        for TP2 in TRADING_PAIRS:
                            netGain += TP2.profit
                            netGainEq += TP2.profEq
                        NET_GAIN.append(netGain)
                        FLOATING_GAIN.append(netGainEq)
                        print("Total Profit: ${:.2f}".format(TP.profit))
                        pendingCloseIDs.remove(val)
                        t.matchedSell = True
                        
            TP.nxt_close = 0 if not TP.openTrades else RR_act * min([t.openPrice for t in TP.openTrades])
            print("PAIR: {:>10} (${:.4f}) / (${:.4f})".format(PAIR, TP.profit, TP.profEq))
            print("LEVEL: {:.8f}".format(COIN_MID))
            print("NOW:   {:.8f}".format(COIN_PRICE))
            print("Open:   ")
            for x in [t for t in TP.openTrades if t.matchedBuy]:
                print("        {}".format((x.openPrice,x.openBTC,"{:.1f}%".format(100*(COIN_PRICE/x.openPrice-1)))))
            print("NxtOp:  {:.8f} ({:.8f})".format(TP.nxt_open, TP.nxt_open-COIN_PRICE) )
            print("NxtCls: {:.8f} ({:.8f})".format(TP.nxt_close, COIN_PRICE-TP.nxt_close))
            print("")
            
            netGain = 0
            netGainEq = 0
            n_closed = 0
            n_open = 0
            for TP2 in TRADING_PAIRS:
                netGain += TP2.profit
                netGainEq += TP2.profEq
                n_open += len(TP2.openTrades)
                n_closed += len(TP2.closedTrades)
            print("Profit: ${:.4f} ({})".format(netGain, n_closed))
            print("Float:  ${:.4f} ({})".format(netGainEq, n_open))
            print()
            
        except (KeyboardInterrupt, SystemExit):
            loop = False
            plt.plot(NET_GAIN,'b')
            plt.plot(FLOATING_GAIN,'r')
            break
        except:
            print("Exception Thrown", PAIR)
            traceback.print_exc()
            