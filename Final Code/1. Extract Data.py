import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import threading
import copy

import ccxt # pip install ccxt
import Constants

EXPORT_PATH = "Data/Crypto"

# Choose time-frame to extract data. Available TimeFrames are stored in "Constants.py"
timeframe = Constants.TF_5M
TimeStep  = Constants.TF_TO_MS[timeframe]

# List of Exchanges to extract data from
bitfinex    = ccxt.bitfinex({'rateLimit' : 2500,'enableRateLimit' : True,'verbose': False})
binance     = ccxt.binance( {'rateLimit' : 2500,'enableRateLimit' : True,'verbose': False})
bitmex      = ccxt.bitmex(  {'rateLimit' : 2500,'enableRateLimit' : True,'verbose': False})
gdax        = ccxt.gdax(    {'rateLimit' : 2500,'enableRateLimit' : True,'verbose': False})
bittrex     = ccxt.bittrex( {'rateLimit' : 2500,'enableRateLimit' : True,'verbose': False})
poloniex    = ccxt.poloniex({'rateLimit' : 2500,'enableRateLimit' : True,'verbose': False})
huobi       = ccxt.huobipro({'rateLimit' : 2500,'enableRateLimit' : True,'verbose': False})
hit         = ccxt.hitbtc2( {'rateLimit' : 2500,'enableRateLimit' : True,'verbose': False})
okex        = ccxt.okex(    {'rateLimit' : 2500,'enableRateLimit' : True,'verbose': False})

all_data = {} # Stores all combined data from a specific exchange
datasets = {} # Stores data for each exchange x coin combination. Eg datasets['BMX']['BMXBTCUSD']

# Function to get request data from an exchange, timeframe can be whatever you want
def getData(exchange,           # ccxt exchange as input 
            exch_name,          # String input, an alias for the exchange name
            LIMIT,              # Maximum number of records to get per query to the api
            timeframe,          # the timeframe of the request data 
            coins=['BTC/USDT'], # List of coins that you want to extract data for 
            params={}           # any special parameters that are needed for a specific exchange in regards to fetchOHLCV
            ):
    
    # Set all entries to blank for every run of this function
    datasets[exch_name] = {}
    
    start_time      = exchange.milliseconds()
    TimeStep        = Constants.TF_TO_MS[timeframe]           # Number of Milliseconds for one time step
    from_datetime   = '2018-01-01 00:00:00'                   # YYYY-MM-DD Extract data as far back as this date
    EARLIEST        = int(exchange.parse8601(from_datetime))  # Convert the string-readable date time to a unix time-stamp
    n_data          = (start_time - EARLIEST) // TimeStep     # Stores the expected number of rows of the final extract
    
    buffer = 5 # Move our next starting point slightly forward, to ensure we capture all data
    
    #-----------------------------------------------------
    # Iterate through all coins needed to extract
    for i, coin in enumerate(sorted(coins)):
        
        coin_name   = coin.replace("/", "") # ccxt format guarantees that "/" is in the coin name so this is fine for all exchanges
        since_time  = max(exchange.milliseconds() - TimeStep * (LIMIT - buffer), EARLIEST)
        since_time  = int(since_time)
        
        # OKEX exchange does not allow the "limit" parameter, so we need to explicitly handle it differently
        if type(exchange) is ccxt.okex:
            data = exchange.fetchOHLCV(coin, timeframe, since = since_time, params=params )
        else:
            data = exchange.fetchOHLCV(coin, timeframe, since = since_time, limit=LIMIT, params=params )
        
        since_time  = min([x[0] for x in data]) # Set the starting time to be the earliest date we found in the extract
        data        = pd.DataFrame(data)        # data is now a dataframe - columns are renamed later to [date, open, high, low, close, volume]
        
        #--------------------------------------------------------------------
        # Keep going until we have hit the earliest date we wanted
        while since_time > EARLIEST:
            
            # Need this loop in-case we get an exception, we only break this loop if the query goes through successfully
            while True:
                
                starting_time = int(since_time - TimeStep * (LIMIT-buffer))
                
                try:
                    # since_time is moved back by the Limit - buffer, again to guarantee that all data is picked up without gaps
                    if type(exchange) is ccxt.okex:
                        d2 = exchange.fetchOHLCV(coin, timeframe, since = starting_time, params=params)
                    else:
                        d2 = exchange.fetchOHLCV(coin, timeframe, since = starting_time, limit=LIMIT, params=params)
                    break
                
                except KeyboardInterrupt:
                    return None
                
                except Exception as ex:
                    print(ex)
                    if "ratelimit" in str(ex):
                        time.sleep(20)
            
            # Because of the buffer, we want to make sure we didn't double count a row. 
            # Remove duplicates is fine for this purpose
            data = data.append(d2).drop_duplicates().reset_index(drop=True)
            
            # should_break is a flag to see if we should stop extracting data. There are a few reasons we might want to do this:
            # 1. The API stopped returning new data (ie, d2 = [] or None)
            # 2. The next extract period would be before our "EARLIEST" date
            # 3. Another coin on the same exchange only has data above the current date,
            #    so the final inner join later will never pick up these rows, so theres no use extracting the data any further
            should_break = False
            
            # check all datasets that are being extracted, and ensure none
            # of them terminated before the date we are currently up to
            for k, v in datasets.items():
                for v2 in v:
                    if min(v[v2].date) > since_time:
                        should_break = True
                        
            # Exit criteria has been met
            if not d2 or (since_time - TimeStep * LIMIT) <= EARLIEST or len(d2) == 0 or should_break:
                print("{:<6} - {}: {} / {}".format(exch_name, coin_name, len(data), n_data))
                break
            else:
                new_since  = int(max( min([x[0] for x in d2]), EARLIEST))
                # starting time didnt move, so no reason to keep extracting
                if since_time == new_since:
                    print("Same Since Time - Breaking, {} {}".format(exch_name, coin_name))
                    break
                since_time = int(new_since)
                
            print("{:<6} - {}: {} / {}".format(exch_name, coin_name, len(data), n_data))
        #--------------------------------------------------------------------
        # Data extract finished!
        
        # rename the columns -> Each columns is prefixed by the exchange name, followed by the coin name without the "/"
        data.columns = ['date', 'open_{}{}'.format(exch_name, coin_name), 
                        'high_{}{}'.format(exch_name, coin_name), 
                        'low_{}{}'.format(exch_name, coin_name), 
                        'close_{}{}'.format(exch_name, coin_name), 
                        'volume_{}{}'.format(exch_name, coin_name)]
        
        # Ensure the data is sorted by date, ascending
        data = data.sort_values('date').reset_index(drop=True)
        #print(exch_name, coin_name, max(data.date), bitmex.milliseconds() - max(data.date), bitmex.milliseconds()//TimeStep * TimeStep,
        #      "*" * 5 if bitmex.milliseconds() - max(data.date) > TimeStep else "")
        
        # Store the data extract for this specific coin
        datasets[exch_name][coin_name] = copy.deepcopy(data)
        
        # if no coins have been added to the joined dataset, then set it to the current one, otherwise inner join on date
        if i == 0:
            df = copy.deepcopy(data)
        else:
            df = df.merge(data, 'inner', 'date')
            
        print(exch_name, coin_name, len(data) )
        
    # Store the final joined dataset in the "all_data" dictionary
    all_data[exch_name] = df
    # Done!
    return df

# Create a dictionary that we will loop through. This decides what will be extracted, and from which exchange
extract_dict = {"BMX" : (bitmex, ["BTC/USD"], 95),
                "BFX" : (bitfinex, ["BTC/USDT", "XRP/USDT", "IOTA/USDT", "EOS/USDT"], 950),
                "BIN" : (binance, ["BTC/USDT", "ETH/USDT", "NEO/USDT", "IOTA/BTC", "EOS/BTC", "XRP/BTC"], 950),
                "GDX" : (gdax, ["BTC/USD", "ETH/USD", "LTC/USD"], 290),
                #"OKX" : (okex, ["BTC/USDT", "ETH/USDT", "XRP/USDT", "IOTA/USDT"], 250),
                #"HUO" : (huobi, ["BTC/USDT", "BCH/USDT", "XRP/USDT"], 1000),
                #"HIT" : (hit, ["BTC/USDT", "BCH/USDT", "XRP/USDT", "LTC/USDT"], 950),
                #"BTX" : (bittrex, ["BTC/USD", "ETH/USD"], 350),
                #"POL" : (poloniex, ["BTC/USDT"], 450),
               } 


datasets    = {} # Reset datasets dict before running the extract code
threads     = [] # Run each extract in its own thread - much faster!

for k, v in extract_dict.items():
    Exchange, Coin, Limit = v[0], v[1], v[2]
    params = {}
    
    # GDAX exchange requires the 'granularity' parameter - should be set to the equivalent number of seconds of 1 time-step
    if type(Exchange) is ccxt.gdax:
        params = {'granularity' : int( TimeStep//1000) }
        
    threads.append( threading.Thread( target=getData, args=(Exchange, k, Limit, timeframe, Coin, params) ) )
    threads[-1].start()
    
for t in threads:
    t.join()

# Refer to the all_data dictionary that was built within each thread, and join each dataset together for our final data
finalData = None
for k in extract_dict:
    if k not in all_data:
        print(k, "is missing in dict")
        continue
    if finalData is None:
        finalData = all_data[k]
    elif k in all_data:
        finalData = finalData.merge(all_data[k], 'inner', 'date')
        
# Remove the very last record from the data - there will duplicates in this due to live-data 
# having the same date stamp, but new numbers.
finalData = finalData[finalData.date < max(finalData.date)]
# Again, make sure data is sorted, and reset the index
finalData = finalData.drop_duplicates().sort_values('date').reset_index(drop=True)

try:
    os.mkdir("{}/{}".format(EXPORT_PATH, timeframe))
except FileExistsError as e1:
    pass
except OSError as e2:
    print('Failed to create directory {} - Incorrect syntax?'.format(timeframe))
except:
    print('Error occurred - {}.'.format(sys.exc_info()[0]))
            
# Output final data
finalData.to_csv("{}/{}/ALL.csv".format(EXPORT_PATH, timeframe), index=False)
# Done!
print(finalData)
