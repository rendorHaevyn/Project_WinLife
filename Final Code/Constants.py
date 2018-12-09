import pickle
import re
import time

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


SAVE_PATH = "C:\CryptoAI\SavedModel"
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

TF_TO_HOUR = { TF : MINS / 60.0 for TF, MINS in TF_TO_MIN.items() }
TF_TO_SEC  = { TF : MINS * 60.0 for TF, MINS in TF_TO_MIN.items() }
TF_TO_MS   = { TF : SEC * 1000  for TF, SEC in TF_TO_SEC.items() }

# Easy way to convert Q values into weighted decision probabilities via softmax.
# This is useful if we probablistically choose actions based on their values rather
# than always choosing the max.

# eg Q[s,0] = -1
#    Q[s,1] = -2
#    softmax([-1,-2]) = [0.731, 0.269] --> 73% chance of standing, 27% chance of hitting
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class Stopwatch:
    
    def __init__(self):
        
        self.watch = {}
        self.totals =  {}
        
    def start(self, name):
        
        if name not in self.watch:
            self.watch[name] = []
        self.watch[name].append( [time.time(), time.time(), 0] )
    
    def end(self, name):
        
        self.watch[name][-1][1] = time.time()
        self.watch[name][-1][2] = self.watch[name][-1][1] - self.watch[name][-1][0]
        if name not in self.totals:
            self.totals[name] = self.watch[name][-1][2]
        else:
            self.totals[name] += self.watch[name][-1][2]
    
    def get_info(self, name):
        total_time = self.totals[name]
        calls = len(self.watch[name])
        avg_time = total_time / calls
        return total_time, avg_time, calls
    
    def display(self):
        
        totals = {}
        tt = 0
        
        for name in self.watch:
            tot, avg, calls = self.get_info(name)
            tt += tot
            totals[name] = [tot, avg, calls]
        
        items = sorted(totals.items(), reverse=True)
        #print(items)
        print("{:<19}  {:<11}  {:<11}  {:<11}{:>6}".format(
                "Block", "TotalTime", "AvgTime", "Calls", "Percent"))
        for nm, (T, A, C) in items:
            print("{:<19}  {:<11.3f}  {:<11.6f}  {:<11}{:>6.2f}%".format(
                    nm,T,A,C,100*T/tt))
