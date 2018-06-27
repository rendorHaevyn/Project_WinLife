# -*- coding: utf-8 -*-

import os
import sys
import time
import pandas as pd
import numpy as np
import ccxt
import Constants

# -----------------------------------------------------------------------------
# common constants

msec = 1000
minute = 60 * msec
hold = 30

# -----------------------------------------------------------------------------

exchange = ccxt.bitfinex({
    'rateLimit': 8000,
    'enableRateLimit': True,
    # 'verbose': True,
})

for timeframe in Constants.TIME_FRAMES:
    
    merged = None
    for p in Constants.ALL_PAIRS:
        # -----------------------------------------------------------------------------
        
        from_datetime = '2017-01-01 00:00:00'
        from_timestamp = exchange.parse8601(from_datetime)
        
        saved_data = None
        coin = Constants.PairToCoin(p)
        
        try:
            saved_data = pd.read_csv("{}/{}.csv".format(timeframe, p.replace("/", "_")))
            from_timestamp = saved_data['date'].describe()[7]+1
        except:
            pass
        
        # -----------------------------------------------------------------------------
        
        now = exchange.milliseconds()
        
        # -----------------------------------------------------------------------------
        
        data = []
        
        while from_timestamp < now:
        
            try:
        
                print(exchange.milliseconds(), 'Fetching candles starting from', exchange.iso8601(from_timestamp))
                ohlcvs = exchange.fetch_ohlcv(p, timeframe, from_timestamp, limit=1000)
                print(exchange.milliseconds(), 'Fetched', len(ohlcvs), 'candles')
                if len(ohlcvs) < 1:
                    break
                first = ohlcvs[0][0]
                last = ohlcvs[-1][0]
                print("{:<6} : {:<4} -- First = {}, Last = {}".format( \
                      coin, timeframe, exchange.iso8601(first),  exchange.iso8601(last)))
                from_timestamp = last + minute * Constants.TF_TO_MIN[timeframe]
                data += ohlcvs
        
            except (ccxt.ExchangeError, ccxt.AuthenticationError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as error:
        
                print('Got an error', type(error).__name__, error.args, ', retrying in', hold, 'seconds...')
                time.sleep(hold)
        
        col_names = ['date'] + ["{}_{}".format(x, coin) for x in ['open', 'high', 'low', 'close', 'volume']]
        if saved_data is None:
            out_data = pd.DataFrame(data, columns=col_names)
        else:
            df = pd.DataFrame(data, columns=col_names)
            out_data = saved_data.append(df, ignore_index=True)
        
        out_data.drop('open_'+coin, axis=1, inplace=True)
        out_data.drop_duplicates(inplace=True)
        out_data = out_data.sort_values('date')[:-1]

        try:
            os.mkdir("{}".format(timeframe))
        except FileExistsError as e1:
            pass
        except OSError as e2:
            print('Failed to create directory {} - Incorrect syntax?'.format(timeframe))
        except:
            print('Error occurred - {}.'.format(sys.exc_info()[0]))
        
        out_data.to_csv("{}\{}.csv".format(timeframe, p.replace("/", "_")), index=False)
        
        if merged is None:
            merged = out_data
        else:
            merged = merged.merge(out_data, how='inner', on='date')
    
    merged.to_csv("{}\ALL.csv".format(timeframe), index=False)
        
        
        
        