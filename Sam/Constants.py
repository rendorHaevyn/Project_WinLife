import pickle
import re

# Return BTC for BTC/USD pair
def PairToCoin(pair):
    return pair[:pair.index("/")]
# Utility Function to return True / False regex matching
def pattern_match(patt, string):
    return re.findall(patt, string) != []
# Utility Function to save objects in memory to a file
def save_memory(obj, path):
    return pickle.dump(obj, open(path, "wb"))
# Utility Function to load objects from the harddisk
def load_memory(path):
    return pickle.load(open(path, "rb"))


COMMISSION = 0.003

TF_1D  = "1d"
TF_1H  = "1h"
TF_30M = "30m"
TF_15M = "15m"
TF_5M  = "5m"
TF_1M  = "1m"

TIME_FRAMES  = [TF_1D, TF_1H, TF_30M, TF_15M, TF_5M, TF_1M]


# Booleans
HP_COMMISSION      = "COMMISSION"
HP_USE_VOLUME      = "USE_VOLUME"
HP_SHORTS          = "ALLOW_SHORTS"
HP_DISCOUNT_N      = "DISCOUNT_STEPS"
HP_DISCOUNT_FACTOR = "GAMMA"

ALL_PAIRS = sorted(["BTC/USD",
             "ETH/USD",
             "BCH/USD",
             "EOS/USD",
             "XRP/USD",
             "IOTA/USD",
             "LTC/USD",
             "ETC/USD",
             "XMR/USD",
             "NEO/USD",
             "ZEC/USD",
             "OMG/USD",
             "DASH/USD"
             ])
    
ALL_COINS = [PairToCoin(c) for c in ALL_PAIRS]

TF_TO_MIN = {"1d"  : 1440,
             "1h"  : 60,
             "30m" : 30,
             "15m" : 15,
             "5m"  : 5,
             "1m"  : 1}

TF_TO_HOUR = dict([(TF, MINS / 60.0) for TF, MINS in TF_TO_MIN.items()])
TF_TO_SEC = dict([(TF, MINS * 60) for TF, MINS in TF_TO_MIN.items()])
