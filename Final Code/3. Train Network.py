from __future__ import print_function
import numpy as np
import random
import pandas as pd
import sklearn
import sklearn.decomposition
import sklearn.ensemble
import sklearn.preprocessing
import sklearn.cluster
from sklearn import *
import math
import gc
import os
import sys
import itertools
import threading
from matplotlib import pyplot as plt
import matplotlib.colors
import tensorflow as tf
import re
import time
import pickle
import Constants
import gym
import copy
import gc

import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist, pdist

# Utility Function to return True / False regex matching
def pattern_match(patt, string):
    return re.findall(patt, string) != []
# Utility Function to save objects in memory to a file
def save_memory(obj, path):
    return pickle.dump(obj, open(path, "wb"))
# Utility Function to load objects from the harddisk
def load_memory(path):
    return pickle.load(open(path, "rb"))

gc.collect()

try:
    os.mkdir(Constants.SAVE_PATH)
except FileExistsError as e1:
    pass
except OSError as e2:
    print('Failed to create directory {} - Incorrect syntax?'.format(Constants.SAVE_PATH))
except:
    print('Error occurred - {}.'.format(sys.exc_info()[0]))

############################ START BLACKJACK CLASS ############################
class Market(gym.Env):
    
    """Trading Market environment"""
    
    def randomIndex(self):
        return random.randint(0, len(self.TRAIN)-self.DISCOUNT_STEPS - 10)
    
    def __init__(self, dataFile = None, COINS_IN = [], COINS_OUT = [], short = False):
        
        gc.collect()
        
        self.data = dataFile
        self.data = self.data.dropna(axis=0, how='any').reset_index(drop=True)
        cut_off_date = int(time.mktime(time.strptime('01/04/2018', "%d/%m/%Y"))) * 1000
        self.data = self.data[self.data.date > cut_off_date].reset_index(drop=True)

        self.data['reward_USD'] = 0
        if COINS_OUT == []:
            COINS_OUT = ['USD'] + [x.replace('close_','') for x in self.data.columns if "close_" in x]
        self.COINS = COINS_OUT
        print("{} rows & {} columns".format(len(self.data), len(self.data.columns)))
        #--------------------------------------------------------------------------------------
        # Manual Options
        #--------------------------------------------------------------------------------------
        self.COMMISSION     = 1e-10  # Commision % as a decimal to use in loss function
        self.NORMALIZE      = True   # Normalize Data
        self.ALLOW_SHORTS   = True   # Allow Shorts or not
        self.GAMMA          = 0.5   # The discount factor
        self.DISCOUNT_STEPS = 5     # Number of periods to look ahead for discounting
        self.TRAIN_PERCENT  = 0.75    # Percentage of data to use as training
        self.MULTS          = 1      # How many future rewards to include in output
        #--------------------------------------------------------------------------------------
        # List of coins data to use as input variables. Set to [] to use all coins
        #--------------------------------------------------------------------------------------
        self.N_COINS     = len(self.COINS)#( len(self.COINS) * 2 - 1 ) if self.ALLOW_SHORTS else len(self.COINS)
        #--------------------------------------------------------------------------------------
        # Create a list of X column names to use for modelling
        #--------------------------------------------------------------------------------------
        the_coins = []
        if COINS_IN == []:
            for c in self.data.columns:
                if "reward_" in c and c != "reward_USD" and not c.endswith("_S") and c.replace("reward_","") not in the_coins:
                    the_coins.append(c.replace("reward_",""))
        else:
            for c in self.data.columns:
                if "reward_" in c and c != "reward_USD" and not c.endswith("_S") and c.replace("reward_","") not in the_coins:
                    the_coin = c.replace("reward_","")
                    if the_coin in COINS_IN:
                        the_coins.append(the_coin)
                            
        self.COINS_IN = the_coins
        
        in_cols = []
        for c in self.data.columns:
            if "DAY_OF_WEEK" in c or "HOUR_OF_DAY" in c:
                in_cols.append(c)
                continue
            for a in sorted(set(the_coins)):
                if "_"+a in c:
                    in_cols.append(c)
        
        COLS_X = []
        for x in in_cols:
            if "reward_" in x or "train_" in x:
                continue
            COLS_X.append(x)
                
        #for c in self.data.columns:
        #    if "limit" in c:
        #        self.data[c.replace("limit","reward")] = self.data[c]
        #--------------------------------------------------------------------------------------
        # Create a list of Y column names to use for modelling
        #--------------------------------------------------------------------------------------
        COLS_Y = [] if "USD" not in self.COINS and "USDT" not in self.COINS else ["reward_USD"]

        for c in self.data.columns:
            added = False
            if 'reward' in c and (c != 'reward_USD' and c not in COLS_Y):

                if COINS_OUT == []:
                    COLS_Y += [c]
                    added = True

                else:
                    for a in sorted(set(self.COINS)):
                        if c == "reward_" + a and c not in COLS_Y:
                            COLS_Y += [c]
                            print("added reward:", c)
                            added = True
                if added:
                    #self.data[c+"_S"] = self.data[c].apply(lambda x : math.log10(2-10**x))
                    self.data[c+"_S"] = self.data[c].apply(lambda x : -x)
                    
        if self.ALLOW_SHORTS:
            COLS_Y += ["{}_S".format(y) for y in COLS_Y if y != "reward_USD"]
            
        current_ys = copy.deepcopy(COLS_Y)
        for ahead in range(1, self.MULTS):
            for y in current_ys:
                c = y + "_" + str(ahead + 1)
                self.data[c] = self.data[y].shift(-ahead)
                COLS_Y.append(c)
        
        self.N_CRYPTO = len([1 for y in COLS_Y if y != "reward_USD" and not y.endswith("_S")])
        
        PORT_W = [w.replace("reward_", "MARGIN_") for w in COLS_Y]
        for p in PORT_W:
            self.data[p] = 0

        self.data["MARGIN_USD"] = 1
        if self.COMMISSION != 0:
            COLS_X += PORT_W
            
        # Hard-code in spread
        for x in COLS_Y:
            if x in ("train_USD", "reward_USD"):
                continue
            self.data[x] = self.data[x].apply(lambda x : x + math.log10(1-0.0/4000))
            
        COLS_Y_TRAIN = [x.replace("reward_","train_") for x in COLS_Y]
        print(COLS_Y)
        print(COLS_Y_TRAIN)
        
        for y_pos in range(len(COLS_Y_TRAIN)):
            
            train_col = COLS_Y_TRAIN[y_pos]
            orig_col = COLS_Y[y_pos]
            stmt = "self.data['{}'] = self.data['{}']".format(train_col, orig_col)
            for ahead in range(1,self.DISCOUNT_STEPS+1):
                stmt += "+(self.GAMMA**{}) * self.data['{}'].shift({})".format(ahead, orig_col, -ahead)
            #for ahead in range(1,self.DISCOUNT_STEPS+1):
            #    stmt += "+((0.25*self.GAMMA)**{}) * self.data['{}'].shift({})".format(ahead, orig_col, ahead)
            #stmt += "+ math.log10(1 - 0.0001)"
            print("Calculating Discount Rewards...", end="")
            exec(stmt)
            
        self.COLS_Y_TRAIN = COLS_Y_TRAIN
        self.data = self.data.dropna(axis=0, how='any').reset_index(drop=True)
        
        #for c in COLS_Y:
        #    if "USD" in c:
        #        continue
        #    self.data[c] = self.data[c] + math.log10(1 - self.COMMISSION)

        #self.data = self.data.dropna(axis=0, how='any').reset_index(drop=True)
        #--------------------------------------------------------------------------------------
        # Split Train/Test
        #--------------------------------------------------------------------------------------
        train_idx = int( self.TRAIN_PERCENT * len(self.data) )
        #--------------------------------------------------------------------------------------
        # Normalizing the X columns. Scale using training data only
        #--------------------------------------------------------------------------------------
        self.SCALE_DICT = {}
        if self.NORMALIZE:
            '''print("Normalizing Data...", end="")
            scaler = sklearn.preprocessing.StandardScaler()
            print("Fitting Scaler: {}".format(len(COLS_X)))
            scaler.fit( self.data[:train_idx][COLS_X] )
            print("Using Scaler: {}".format(len(COLS_X)))
            self.data[COLS_X] = scaler.transform(self.data[COLS_X])
            self.SAVE_SCALER = scaler
            self.SAVE_SCALER_COLS = COLS_X'''
            
            #def scale_col(dat, x, mu, sd):
            #    dat[x] = dat[x].apply(lambda x : (x-mu)/sd)
            #    print("Scaled {}".format(x))
                
            #norm_threads = []
            
            descriptions = self.data[:train_idx].describe()
            
            for i, x in enumerate(COLS_X):
                if "MARGIN" in x or 'date' in x or "close_" in x or "open_" in x or "low_" in x or "high_" in x:
                    continue
                
                mu, sd = descriptions[x]['mean'], descriptions[x]['std']
                
                print("Normalizing {} - {} / {}      {:.5f}, {:.5f}".format(x, (i+1), len(COLS_X), mu, sd))
                self.SCALE_DICT[x] = (mu, sd)
                self.data[x] = self.data[x].apply(lambda x : (x-mu)/sd)
                #thr = threading.Thread(target=scale_col, args=(self.data, x, mu, sd))
                #norm_threads.append(thr)
                #norm_threads[-1].start()
                
            #for thr in norm_threads:
            #    thr.join()
            
            print("Done")
            
        self.TRAIN = self.data[:train_idx].reset_index(drop=True)
        #self.TEST = self.TRAIN
        self.TEST = self.data[train_idx:].reset_index(drop=True)
        
        
        fee_rate = 0.002/100
        self.TRAIN_HOLD = copy.deepcopy(self.TRAIN)
        self.TRAIN_HOLD[PORT_W]        = 0 # Set all holdings to 0
        self.TRAIN_HOLD[PORT_W[0]]     = 1 # Set first holding to 1
        self.TRAIN_HOLD[COLS_Y_TRAIN]       += 2 * math.log10(1 - fee_rate) # Add transaction cost to all rewards
        self.TRAIN_HOLD[COLS_Y_TRAIN[0]]    -= 2 * math.log10(1 - fee_rate) # Remove it from the one we're holding
        for i, y in enumerate(COLS_Y):
            if i == 0:
                continue
            new = copy.deepcopy(self.TRAIN)
            new[PORT_W]        = 0 # Set all holdings to 0
            new[PORT_W[i]]     = 1 # Set first holding to 1
            new[COLS_Y_TRAIN]       += 2 * math.log10(1 - fee_rate) # Add transaction cost to all rewards
            new[COLS_Y_TRAIN[i]]    -= 2 * math.log10(1 - fee_rate) # Remove it from the one we're holding
            if COLS_Y[0] == "reward_USD":
                new[COLS_Y_TRAIN[0]]    -= 1 * math.log10(1 - fee_rate) # Remove it from the one we're holding
            self.TRAIN_HOLD = self.TRAIN_HOLD.append(new)
            
        self.TRAIN_HOLD.reset_index(inplace=True)
            
        self.COLS_X = COLS_X
        self.COLS_Y = COLS_Y
        self.N_IN   = len(COLS_X)
        self.N_OUT  = len(COLS_Y)
        
        self.holdings = {}
        for i, c in enumerate(sorted(self.COLS_Y)):
            self.holdings[c.replace("reward_","")] = 0
        self.holdings['USD'] = 1
        
        self.position = self.randomIndex()
        self.ACTIONS  = [x.replace("reward_","") for x in self.COLS_Y]
        self.PORT_W   = PORT_W
        
        self.N_CRYPTO_IN = len(self.COINS_IN)
        
        print("CRYPTO_IN:" + str(self.COINS_IN))
        
        self.PREV_W_COLS = PORT_W
        
        gc.collect()
        
        print("Market Data Loaded")
        
    def save(self):
        items = [ (self.SCALE_DICT, "{}\\SCALE_DICT.save".format(Constants.SAVE_PATH)),
                  (self.PRICE_TENSOR_COLS, "{}\\PRICE_TENSOR_COLS.save".format(Constants.SAVE_PATH)),
                  (self.PRICE_LAGS, "{}\\PRICE_LAGS.save".format(Constants.SAVE_PATH))]

        for i in items:
            try:
                save_memory(i[0], i[1])
            except:
                pass
        
    def step(self, action):
        
        rw = 0
        
        self.COMM_REWARD = math.log10(1 - self.COMMISSION)
        
        act_loc = M.ACTIONS.index(action)
        if self.TRAIN.at[self.position, self.PORT_W[act_loc]] == 1:
            rw = 0
        elif action in ("USD", "USDT") or self.TRAIN.at[self.position, "MARGIN_USD"] == 1:#\
        #(self.TRAIN.at[self.position, "MARGIN_USD"] == 1 and action not in ("USD", "USDT")):
            rw = 1 * self.COMM_REWARD
        else:
            rw = 2 * self.COMM_REWARD
        
        rw += self.TRAIN.at[self.position, "reward_{}".format(action)]
        self.position += 1
        
        for w in self.PORT_W:
            self.TRAIN.set_value(self.position, w, 0)
        self.TRAIN.set_value(self.position, self.PORT_W[act_loc], 1)
        
        if np.isnan(rw):
            print(self.position, action, self.holdings)
        
        return rw
            
############################ END MARKET CLASS ############################

raw_data = pd.read_csv("Data/Crypto/5m/ALL_MOD.csv")
#raw_data = pd.read_csv("Data/Forex/15m/ALL_MOD.csv")

#M = Market(raw_data,
#           COINS_IN  = ['BTC', 'EOS', 'ETC', 'ETH', 'IOTA', 'LTC', 'XRP'],
#           COINS_OUT = ['BTC', 'EOS', 'ETC', 'ETH', 'IOTA', 'LTC', 'XRP'])


fx_pairs_in = ['AUDCAD', 'AUDJPY', 'AUDNZD', 'AUDUSD', 'CADJPY', 'EURAUD', 'EURCAD', 'EURGBP', 
            'EURJPY', 'EURNZD', 'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPJPY', 'GBPNZD', 'GBPUSD', 
            'NZDCAD', 'NZDJPY', 'NZDUSD', 'USDCAD', 'USDJPY', 'USDOLLAR']

fx_pairs_out = ['AUDCAD', 'AUDJPY', 'AUDNZD', 'AUDUSD', 'CADJPY', 'EURAUD', 'EURCAD', 'EURGBP', 
            'EURJPY', 'EURNZD', 'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPJPY', 'GBPNZD', 'GBPUSD', 
            'NZDCAD', 'NZDJPY', 'NZDUSD', 'USDCAD', 'USDJPY']

fx_pairs = ['USD', 'AUDUSD', 'EURUSD', 'GBPJPY', 'AUDJPY', 'GBPUSD', 'USDJPY', 'EURAUD', 'EURJPY']

M = Market(raw_data,
           COINS_IN  = ["BMXBTCUSD", "BFXBTCUSDT", "BINBTCUSDT", "GDXBTCUSD", "BFXXRPUSDT", "BINETHUSDT"],
           COINS_OUT = ["BMXBTCUSD"])


#M = Market(raw_data,
#          COINS_IN  = ["AUDJPY", "AUDUSD", "GBPJPY", "GBPUSD", "EURUSD", "NZDUSD", "EURCAD", "USDJPY"],
#          COINS_OUT = ['USDJPY'])

X2 = []
for x in M.data.columns:
    
    channel_rank = 1000
    lag_rank     = 1000
    
    channel_rank = 0   if "L_CLOSE"        in x else channel_rank
    channel_rank = 1   if "L_LOW"          in x else channel_rank
    channel_rank = 2   if "L_HIGH"         in x else channel_rank
    channel_rank = 3   if "L_VOLUME"       in x else channel_rank
    channel_rank = 4   if "L_VOLPRICE"     in x else channel_rank
    
    channel_rank = 5   if "L2_CLOSE"       in x else channel_rank
    channel_rank = 6   if "L2_LOW"         in x else channel_rank
    channel_rank = 7   if "L2_HIGH"        in x else channel_rank
    
    channel_rank = 8   if "L3_CLOSE"       in x else channel_rank
    channel_rank = 9   if "L3_LOW"         in x else channel_rank
    channel_rank = 10  if "L3_HIGH"        in x else channel_rank
    
    channel_rank = 11  if "SMACLOSE1"           in x else channel_rank
    channel_rank = 12  if "SMACLOSE2"           in x else channel_rank
    channel_rank = 13  if "SMACLOSE3"           in x else channel_rank
    channel_rank = 14  if "SMACLOSE4"           in x else channel_rank
    channel_rank = 15  if "SMACLOSE5"           in x else channel_rank
    
    channel_rank = 16  if "SMALOW1"           in x else channel_rank
    channel_rank = 17  if "SMALOW2"           in x else channel_rank
    channel_rank = 18  if "SMALOW3"           in x else channel_rank
    channel_rank = 19  if "SMALOW4"           in x else channel_rank
    channel_rank = 20  if "SMALOW5"           in x else channel_rank
    
    channel_rank = 21  if "SMAHIGH1"           in x else channel_rank
    channel_rank = 22  if "SMAHIGH2"           in x else channel_rank
    channel_rank = 23  if "SMAHIGH3"           in x else channel_rank
    channel_rank = 24  if "SMAHIGH4"           in x else channel_rank
    channel_rank = 25  if "SMAHIGH5"           in x else channel_rank
    
    channel_rank = 26  if "RSI1"           in x else channel_rank
    channel_rank = 27  if "RSI2"           in x else channel_rank
    channel_rank = 28  if "RSI3"           in x else channel_rank
    channel_rank = 29  if "RSI4"           in x else channel_rank
    channel_rank = 30  if "RSI5"           in x else channel_rank
    
    #channel_rank = 29  if "SUPPORT1"       in x else channel_rank

    #channel_rank = 30  if "RESIST1"        in x else channel_rank
    
    channel_rank = 31 if "LINEAR1"         in x else channel_rank
    channel_rank = 32 if "LINEAR2"         in x else channel_rank
    channel_rank = 33 if "LINEAR3"         in x else channel_rank

        
    S_COINS = sorted(M.COINS_IN)
    coin_rank = -1
    for i, c in enumerate(S_COINS):
        if x.endswith("_"+c):
            coin_rank = i
            break
        
    try:
        lag_rank = int("".join([ch for ch in x[x.index("_"):] if ch in '0123456789']))
        lag_rank *= -1
    except:
        pass
    
    if coin_rank < 0:
        continue
    
    X2.append( (coin_rank, lag_rank, channel_rank, x) )
    
X2.sort(key = lambda x : (x[0], x[1], x[2]))

PRICE_TENSOR  = [(x[-1], x[-2], x[-3]) for x in X2 if 0 <= x[2] < 1000]

cols                = list(M.data.columns)
PRICE_LAGS          = len(set([x[2] for x in PRICE_TENSOR]))
PRICE_CHANNELS      = len(set([x[1] for x in PRICE_TENSOR]))
PRICE_TENSOR_COLS   = [x[0] for x in PRICE_TENSOR]
PRICE_TENSOR_IDX    = [cols.index(x) for x in PRICE_TENSOR_COLS]
M.PRICE_LAGS        = PRICE_LAGS
M.PRICE_TENSOR_COLS = PRICE_TENSOR_COLS

MU_SD_TABLE = M.TRAIN[PRICE_TENSOR_COLS].describe()

USE_SIGMOID     = True
N_COINS         = M.N_COINS
N_CRYPTO        = M.N_CRYPTO_IN
N_IN            = M.N_IN
N_OUT           = M.N_OUT
TIMESTEP_DAYS   = 86400000 / (M.data.date - M.data.date.shift(1)).describe()['50%']

with tf.device("/GPU:0"):
    
    # PrevW
    HOLD_W   = tf.placeholder(tf.float32, [None, N_OUT])
    HOLD_W   = tf.reshape(HOLD_W, [-1, N_OUT])
    # Actual Rewards
    Y_       = tf.placeholder(tf.float32, [None, N_OUT])
    
    Q_TARGET = tf.placeholder(tf.float32, [None, N_OUT])
    Q_TARGET = tf.reshape(Q_TARGET, [-1, N_OUT])
    
    keep_p1  = tf.placeholder(tf.float32, name = 'keep1')
    keep_p2  = tf.placeholder(tf.float32, name = 'keep2')
    keep_p3  = tf.placeholder(tf.float32, name = 'keep3')
    
    #--------------------------------------------------------------------------------------
    # Define Neural Network layers
    #--------------------------------------------------------------------------------------

    h_1         = 1
    w_1         = 1
    CH_OUT_1    = 20
    FILTER1     = [h_1, w_1, PRICE_CHANNELS, CH_OUT_1] # Filter 1 x 3 x 3, Input has 4 channels
    
    h_2         = 1
    w_2         = PRICE_LAGS - w_1 + 1
    CH_OUT_2    = 50
    FILTER2     = [h_2, w_2, CH_OUT_1, CH_OUT_2]
    
    # Final
    h_f         = N_CRYPTO
    w_f         = 1
    CH_OUT_f    = 100
    FILTERf     = [h_f, w_f, CH_OUT_2, CH_OUT_f]
    
    SDEV        = 1
    BIAS_MULT   = 0
    
    initializer     = tf.contrib.layers.xavier_initializer()
    initializer_cnn = tf.contrib.layers.xavier_initializer_conv2d()
    
    X_PRICE_TENSOR    = tf.placeholder(tf.float32, [None, len(PRICE_TENSOR_COLS)])
    X_PRICE_TENSOR_NN = tf.reshape(X_PRICE_TENSOR, [-1, N_CRYPTO, PRICE_LAGS, PRICE_CHANNELS])
    
    #X_PRICE_TENSOR_NN_AVG = tf.nn.avg_pool(X_PRICE_TENSOR_NN, [1,1,3,1], [1,1,3,1], 'VALID')
    
    X_SCALER = tf.Variable(tf.ones([N_CRYPTO, 1, PRICE_CHANNELS]))
    X_SCALER2 = tf.Variable(tf.ones([1, PRICE_LAGS, PRICE_CHANNELS]))
    
    X_PRICE_TENSOR_NN_AVG = X_PRICE_TENSOR_NN
    #X_PRICE_TENSOR_NN_AVG = tf.round(4 * X_PRICE_TENSOR_NN) / 4
    
    X_PRICE_TENSOR_NN_AVG = tf.multiply(X_PRICE_TENSOR_NN_AVG, X_SCALER)
    X_PRICE_TENSOR_NN_AVG = tf.multiply(X_PRICE_TENSOR_NN_AVG, X_SCALER2)
    
    LEAKY_ALPHA = 0.05
    
    # LAYER 1
    CW1 = tf.Variable(tf.random_normal(FILTER1, stddev = SDEV * (1/(h_1*w_1*PRICE_CHANNELS))**0.5 ))
    CB1 = tf.Variable(tf.zeros([CH_OUT_1]))
    CL1 = tf.nn.leaky_relu(tf.nn.conv2d(X_PRICE_TENSOR_NN_AVG, CW1, [1,1,1,1], padding="VALID") + CB1 * BIAS_MULT, LEAKY_ALPHA)
    CL1 = tf.nn.dropout(CL1, keep_p1)
    
    # LAYER 2
    CW2 = tf.Variable(tf.random_normal(FILTER2, stddev = SDEV * (1/(h_2*w_2*CH_OUT_1))**0.5))
    CB2 = tf.Variable(tf.zeros([CH_OUT_2]))
    CL2 = tf.nn.leaky_relu(tf.nn.conv2d(CL1, CW2, [1,1,1,1], padding="VALID") + CB2 * BIAS_MULT, LEAKY_ALPHA)
    
    CL2 = tf.nn.dropout(CL2, keep_p2)
    
    CW4 = tf.Variable(tf.random_normal(FILTERf, stddev = SDEV * (1/(h_f*w_f*CH_OUT_f))**0.5))
    CB4 = tf.Variable(tf.zeros([CH_OUT_f]))
    CL4 = tf.nn.relu(tf.nn.conv2d(CL2, CW4, [1,1,1,1], padding="VALID") + CB4 * BIAS_MULT)
    
    CL4 = tf.nn.dropout(CL4, keep_p3)
    
    CL_flat = tf.reshape(CL4, (-1, CH_OUT_f * N_CRYPTO//h_f))
    CL_flat = tf.concat( [CL_flat, HOLD_W], -1)
    
    fc_w = tf.Variable( initializer([int(CL_flat.shape[-1]), 100]) )
    fc_b = tf.Variable( initializer([100]) )
    
    fc_w2 = tf.Variable( initializer([100, N_OUT]) )
    fc_b2 = tf.Variable( initializer([N_OUT]) )
    
    LOSS_L2 = tf.nn.l2_loss(fc_w)
    
    Q_UNSCALED1 = tf.nn.relu(tf.matmul(CL_flat, fc_w) + fc_b * BIAS_MULT)
    Q_UNSCALED1 = tf.nn.dropout(Q_UNSCALED1, keep_p3)
    
    Q_UNSCALED = tf.matmul(Q_UNSCALED1, fc_w2) + fc_b2 * BIAS_MULT
    Q_UNSCALED = tf.nn.dropout(Q_UNSCALED, keep_p3)
    
    if USE_SIGMOID:
        Q_PREDICT = tf.nn.sigmoid(Q_UNSCALED)
    else:
        #Q_PREDICT = Q_UNSCALED
        Q_PREDICT = tf.nn.softmax(Q_UNSCALED, 1)
        
    #--------------------------------------------------------------------------------------
    # Define Loss Functions
    #--------------------------------------------------------------------------------------
    
    q_predict_mean, q_predict_var = tf.nn.moments(Q_PREDICT, axes=[1])
    all_returns   = tf.reduce_sum(Q_PREDICT * Q_TARGET, 1)
    all_returns2  = tf.nn.relu(  tf.reduce_sum(Q_PREDICT * Q_TARGET, 1) ) ** 0.8 - \
                   tf.nn.relu( -tf.reduce_sum(Q_PREDICT * Q_TARGET, 1) ) ** 0.8
                   
    loss_func     = -tf.reduce_mean(all_returns)
    r_mean, r_var = tf.nn.moments(all_returns, axes=[0])
    sharpe_loss   = -r_mean / (r_var**0.5)
    
    winning_trades = tf.nn.relu(all_returns)
    winning_trades_mean = tf.reduce_mean(winning_trades)
    losing_trades  = tf.nn.relu(-all_returns)
    losing_trades_mean = tf.reduce_mean(losing_trades)
    
    winning_trades2 = tf.nn.relu(all_returns2)
    winning_trades_mean2 = tf.reduce_mean(winning_trades2)
    losing_trades2  = tf.nn.relu(-all_returns2)
    losing_trades_mean2 = tf.reduce_mean(losing_trades2)
    
    #min_func = -tf.reduce_mean(tf.reduce_sum(Q_PREDICT * Q_TARGET, 1) ) * math.e**-r_stdev
    opt_func = (winning_trades_mean2/losing_trades_mean2) * (-tf.reduce_mean(all_returns2) + 0.1 * tf.reduce_mean(losing_trades2))# - \
    #opt_func = -tf.reduce_mean(winning_trades) / tf.reduce_mean(losing_trades)# - \
               #1e-7 * tf.reduce_min(all_returns)
               
    #opt_func = -tf.reduce_mean(tf.reduce_sum(Q_PREDICT * Q_TARGET, 1) ) * math.e**-r_var

    #opt_func = -tf.reduce_sum(all_returns)# + 0.5*tf.reduce_sum(losing_trades)
    #opt_func = tf.reduce_sum(tf.square(Q_PREDICT - Q_TARGET), 0)
    opt_func = -tf.reduce_mean(all_returns2)

               
    #loss_func = -tf.reduce_sum(Q_PREDICT * Q_TARGET) \
    #            - math.log10(1-M.COMMISSION)*tf.reduce_sum( tf.abs(tf.reduce_sum(Q_PREDICT[1:,:] - Q_PREDICT[:-1,:], 1) ) )

    LR_START = 0.0005
    
    # Optimizer
    LEARNING_RATE    = tf.Variable(LR_START, trainable=False)
    optimizer        = tf.train.AdamOptimizer(LEARNING_RATE)#(LEARNING_RATE)
    train_step       = optimizer.minimize(1e2 * opt_func)
    
    #--------------------------------------------------------------------------------------
    # Begin Tensorflow Session
    #--------------------------------------------------------------------------------------
    
    init = tf.global_variables_initializer()
    
    config                              = tf.ConfigProto()
    config.intra_op_parallelism_threads = 32
    config.log_device_placement         = True
    
    sess = tf.Session(config=config)
    sess.run(init)

# probability of picking a random action. This decays over time
epsilon  = 0.1

all_rewards  = [] # Holds all observed rewards. The rolling mean of rewards should improve as the network learns
all_Qs       = [] # Holds all predicted Q values. Useful as a sanity check once the network is trained
all_losses   = [] # Holds all the (Q_TARGET - Q_PREDICTED) values. The rolling mean of this should decrease
Q_TARGETS    = []
Q_PREDS      = []
PRICE_STATES = []
H_WEIGHTS    = []
Q_CONVERGE   = {} # Not used yet
projections  = []
watch        = Constants.Stopwatch()

train_losses, test_losses, transf_losses, opt_losses = [], [], [], []
gc.collect()

episode = 0
smallest_loss = 1e6
while episode < 1000000:
    
    init_pos = episode % (len(M.TRAIN)-50)#
    #init_pos = M.randomIndex()
    M.position = init_pos
    
    USD_STATE = None
    USD_PRICE_STATE = None
    Q_USD = 0
    W_USD = 0 
    
    '''if episode == 100:
        update_LR = tf.assign(LEARNING_RATE, 0.001)
        sess.run(update_LR)'''
    
    for w_index, starting_w in enumerate(M.PORT_W):
        
        watch.start('update_W')
        M.position = init_pos
        for w in M.PORT_W:
            M.TRAIN.set_value(M.position, w, 0)
        M.TRAIN.set_value(M.position, starting_w, 1)
        watch.end('update_W')
        
        watch.start('set_state')
        init_price_state = np.array(M.TRAIN.iloc[M.position, PRICE_TENSOR_IDX])
        watch.end('set_state')
        
        watch.start('Q_PREDICT')
        Q1 = sess.run(Q_PREDICT, feed_dict = {
                      X_PRICE_TENSOR : np.reshape(init_price_state,(-1, len(PRICE_TENSOR_COLS)) ),
                      HOLD_W : np.array(M.TRAIN.ix[M.position, M.PORT_W]).reshape( (-1, N_OUT) ),
                      keep_p1 : 1, keep_p2 : 1, keep_p3 : 1} )
        watch.end('Q_PREDICT')
        if w_index == 0:
            USD_PRICE_STATE = init_price_state
            Q_USD = Q1
            W_USD = np.array(M.TRAIN.ix[M.position, M.PORT_W]).reshape( (-1, N_OUT) )
            
        targetQ = list(Q1[0])
        
        for act_num, begin_act in enumerate(M.ACTIONS):
            
            M.position = init_pos
            for w in M.PORT_W:
                M.TRAIN.set_value(M.position, w, 0)
            M.TRAIN.set_value(M.position, starting_w, 1)
            #print(M.TRAIN.loc[M.position, M.PORT_W])
            
            watch.start("market_step")
            #G = M.step(begin_act)
            #Gpercent = 100*(10**G-1)
            #G = math.log10(1+int(Gpercent*8)/800)
            profit = M.TRAIN.at[M.position, M.COLS_Y_TRAIN[act_num]]
            G = profit
            M.position += 1
            
            watch.end("market_step")
            
            for t in range(0):#M.DISCOUNT_STEPS):
                
                state = np.array(M.TRAIN.loc[M.position, M.COLS_X])
                price_state = np.array(M.TRAIN.loc[M.position, PRICE_TENSOR_COLS])
                
                if random.random() < epsilon:
                    act = random.choice(M.ACTIONS)
                else:
                    Q    = sess.run(Q_PREDICT, feed_dict = {
                                X_PRICE_TENSOR : price_state.reshape(-1, len(PRICE_TENSOR_COLS)),
                                HOLD_W : np.array(M.TRAIN.ix[M.position, M.PORT_W]).reshape( (-1, N_OUT) ),
                                keep_p1 : 1, keep_p2 : 1, keep_p3 : 1} )
                    
                    act = M.ACTIONS[np.argmax(Q)]
                
                if t == M.DISCOUNT_STEPS-1 and episode > 1000:
                    G += M.GAMMA ** (t+1) * max(Q[0])
                else:
                    G += M.GAMMA ** (t+1) * M.step(act)
                    
                #for w in M.PORT_W:
                #    M.TRAIN.set_value(M.position, w, 0)
                #M.TRAIN.set_value(M.position, M.PORT_W[M.ACTIONS.index(act)], 1)
            
            targetQ[act_num] = G
        
        PRICE_STATES.append(init_price_state)
        Q_PREDS.append(Q1)
        Q_TARGETS.append(targetQ)
        H_WEIGHTS.append(M.TRAIN.ix[init_pos, M.PORT_W])
        
        if w_index == 0:
            usd_target = copy.deepcopy(targetQ)
            break
    
    num_depth = 1+max(0, math.log(episode+1)-2)+len(M.TRAIN)#*0.15
    num_depth = len(M.TRAIN)
    #num_depth = 1024
    if len(Q_TARGETS) >= num_depth or True:
        
        COL_W  = '\033[0m'  # white (normal)
        COL_R  = '\033[41m' # red
        COL_G  = '\033[42m' # green
        COL_O  = '\033[33m' # orange
        COL_B  = '\033[34m' # blue
        COL_P  = '\033[35m' # purple
        
        #update_drop_rt = tf.assign(tf_keep_prob, 0.7)
        #sess.run(update_drop_rt)
        
        #the_x = np.reshape( np.array(X_STATES),  (-1, N_IN) )
        the_p = np.reshape( np.array(PRICE_STATES), (-1, len(PRICE_TENSOR_COLS)))
        the_q = np.reshape( np.array(Q_TARGETS), (-1, N_OUT))
        the_w = np.reshape( np.array(H_WEIGHTS), (-1, N_OUT))
        
        the_p = np.reshape(np.array(M.TRAIN_HOLD[PRICE_TENSOR_COLS]), (-1, len(PRICE_TENSOR_COLS)) )
        the_q = np.reshape(np.array(M.TRAIN_HOLD[M.COLS_Y_TRAIN]), (-1, len(M.COLS_Y_TRAIN)) )
        the_w = np.reshape(np.array(M.TRAIN_HOLD[M.PORT_W]), (-1, len(M.PORT_W)) )
        
        #for i in range(int(num_depth+0.5)):
        i = 0
        PR_KEEP_1, PR_KEEP_2, PR_KEEP_3 = 0.70, 0.70, 0.70
        use_sample = True
        while i < 2000000000:
            
            rates = {0 :   0.0005, 
                     1e4 : 0.0001, 
                     3e4 : 0.00003, 
                     1e6 : 0.00001}
            
            if i in rates:
                update_LR = tf.assign(LEARNING_RATE, rates[i])
                sess.run(update_LR)

            opt = train_step

            #opt = train_step_start if i < 200 or random.random() < 0.02 else train_step
            #l_func = loss_func_start if i < 200 else loss_func
            #opt = train_step
            watch.start("Gradient_Update")
            if use_sample:
                
                n_samples = min(i//100+500, round(0.2 * len(the_p)) )
                #n_samples = 50
                #samples = [int(random.random()**0.5 * len(the_p)) for _ in range(n_samples)]
                samples = random.sample(range(len(the_p)), n_samples)
                x_noise   = np.random.normal(0, 0.15, the_p[samples,:].shape)
                #y_noise   = np.random.normal(-1e-9, 1e-9, the_q[samples,:].shape)
                #samples = random.sample(range(len(the_p)), round(0.3*len(the_p)))
                sess.run(opt, 
                              feed_dict = {X_PRICE_TENSOR : the_p[samples,:] + x_noise,
                                           Q_TARGET : the_q[samples,:],
                                           HOLD_W : the_w[samples,:],
                                           keep_p1 : PR_KEEP_1, keep_p2 : PR_KEEP_2, keep_p3 : PR_KEEP_3})
    
            else:
                sess.run(opt, 
                              feed_dict = {X_PRICE_TENSOR : the_p,
                                           Q_TARGET : the_q,
                                           HOLD_W : the_w,
                                           keep_p1 : PR_KEEP_1, keep_p2 : PR_KEEP_2, keep_p3 : PR_KEEP_3}  )
    
            watch.end("Gradient_Update")
            if i % 100 == 0:
                
                train_loss = sess.run(loss_func, 
                                      feed_dict = {X_PRICE_TENSOR : the_p,
                                                   Q_TARGET : the_q,
                                                   HOLD_W : the_w,
                                                   keep_p1 : 1, keep_p2 : 1, keep_p3 : 1}  )
                
                price_state = np.reshape(M.TEST[PRICE_TENSOR_COLS], (-1, len(PRICE_TENSOR_COLS)) )
                truth = np.reshape(M.TEST[M.COLS_Y], (-1, len(M.COLS_Y)) )
                w = np.reshape(M.TEST[M.PORT_W], (-1, len(M.PORT_W)) )
                
                test_loss, losing_mean, opt_loss = sess.run([loss_func, losing_trades_mean, opt_func], 
                                      feed_dict = {X_PRICE_TENSOR : price_state,
                                                   Q_TARGET : truth,
                                                   HOLD_W : w,
                                                   keep_p1 : 1, keep_p2 : 1, keep_p3 : 1}  )
                
                if test_loss < smallest_loss and i > 1000:
                    # Add ops to save and restore all the variables.
                    saver = tf.train.Saver()
                    # Save the variables to disk.
                    saver.save(sess, "{}\\model.ckpt".format(Constants.SAVE_PATH))
                    print("Model saved in path: {}".format(Constants.SAVE_PATH))
                    M.save()
                    smallest_loss = test_loss
                
                '''test_loss_trans = sess.run(l_func, 
                                      feed_dict = {X_PRICE_TENSOR : price_state,
                                                   Q_TARGET : np.reshape(M.TEST[M.COLS_Y_TRAIN], (-1, len(M.COLS_Y_TRAIN)) ),
                                                   HOLD_W : w,
                                                   keep_p1 : 1, keep_p2 : 1, keep_p3 : 1}  )'''
    
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                transf_losses.append(losing_mean)
                opt_losses.append(opt_loss)
    
                fig, ax1 = plt.subplots()
                
                plot_window = 1000
                
                train_plot_data = pd.Series(train_losses[-plot_window:]).rolling(5).mean()
                test_plot_data  = pd.Series(test_losses[-plot_window:]).rolling(5).mean()
                transf_plot_data = pd.Series(transf_losses[-plot_window:]).rolling(5).mean()
                opt_plot_data = pd.Series(opt_losses[-plot_window:]).rolling(5).mean()
                
                color = 'tab:red'
                ax1.set_xlabel('iteration')
                ax1.set_ylabel('train loss', color=color)
                
                ax1.plot(range(1, len(train_plot_data)+1), train_plot_data, color=color)
                ax1.tick_params(axis='y', labelcolor=color)
                
                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                
                color = 'tab:blue'
                ax2.set_ylabel('test loss', color=color)  # we already handled the x-label with ax1
                ax2.plot(range(1, len(test_plot_data)+1), test_plot_data, color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                
                ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
                
                color = 'tab:green'
                ax3.set_ylabel('L2 Loss', color=color)  # we already handled the x-label with ax1
                ax3.plot(range(1, len(transf_plot_data)+1), transf_plot_data, color=color)
                ax3.tick_params(axis='y', labelcolor=color)
                
                ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
                
                color = 'tab:orange'
                ax4.set_ylabel('Loss Value', color=color)  # we already handled the x-label with ax1
                ax4.plot(range(1, len(opt_plot_data)+1), opt_plot_data, color=color)
                ax4.tick_params(axis='y', labelcolor=color)
                
                fig.tight_layout()  # otherwise the right y-label is slightly clipped
                plt.show()
                
                DailyReturnTrain = 100 * (10**(-train_losses[-1] * TIMESTEP_DAYS) - 1)
                DailyReturnTest  = 100 * (10**(-test_losses[-1] * TIMESTEP_DAYS) - 1)  
                
                #DailyReturnTrain = -train_losses[-1] * TIMESTEP_DAYS
                #DailyReturnTest  = -test_losses[-1] * TIMESTEP_DAYS
                
                print("Iteration: {:<10}, Train Loss: {:<.8f}, Test Loss: {:<.8f}, "
                      "Test Daily Return: {}{:<.2f}%{}".
                      format(i,train_loss, test_loss, (COL_G if DailyReturnTest > 0 else COL_R), DailyReturnTest, COL_W))

            if i % 1000 == 0:
                gc.collect()
                watch.display()
            if i % 100000 == 0 and i > 0:
                
                '''M.TEST = D
                M.TEST['MARGIN_USD'] = 0
                M.TEST['MARGIN_BMXBTCUSD'] = 0
                M.TEST['MARGIN_BMXBTCUSD_S'] = 0
                
                M.TEST['reward_USD'] = 0
                M.TEST['reward_BMXBTCUSD'] = M.TEST['close_BMXBTCUSD'].shift(-1) / M.TEST['close_BMXBTCUSD']
                M.TEST['reward_BMXBTCUSD'] = M.TEST['reward_BMXBTCUSD'].apply(lambda x : math.log10(x))
                M.TEST['reward_BMXBTCUSD_S'] = M.TEST['reward_BMXBTCUSD'].apply(lambda x : -x)
                '''
                
                gc.collect()
                dat = M.TEST
                '''dat = M.data
                state = np.array(dat[M.COLS_X])
                price_state = np.array(dat[PRICE_TENSOR_COLS])
                w = np.array(dat[M.PORT_W])
                nn_outs, Q_pred     = sess.run([CL_flat, Q_PREDICT], feed_dict = {
                              X_PRICE_TENSOR : price_state.reshape(-1, len(PRICE_TENSOR_COLS) ),
                              keep_p1 : 1, keep_p2 : 1, keep_p3 : 1
                              } )
    
                lst = []
                out_data = dat.copy()
                out_cols = []
                act_cols = []
                
                for idx in range(len(nn_outs[0])):
                    lst = [x[idx] for x in nn_outs]
                    c = "NN_OUT_{}".format(idx+1)
                    out_data[c] = lst
                    out_cols.append(c)
                
                for idx, action in enumerate(M.ACTIONS):
                    print(idx, action)
                    lst = [x[idx] for x in Q_pred]
                    c = "ACT_{}".format(action)
                    out_data[c] = lst
                    out_cols.append(c)
                    act_cols.append(c)
                    if idx >= len(M.ACTIONS) / M.MULTS - 1:
                        break
                    
                out_cols += M.COLS_Y[ : len(M.COLS_Y) // M.MULTS ]
                out_data[out_cols].to_csv("Crypto Q Data.csv",index=False)
                
                
                C = sklearn.cluster.KMeans(10)
                C.fit(out_data[:len(M.TRAIN)][act_cols])
                plt.plot(C.cluster_centers_, 'o')
                out_data['state'] = C.predict(out_data[act_cols])
                out_cols.append('state')
                out_data[out_cols].to_csv("Crypto Q Data.csv",index=False)
                #(C.cluster_centers_ - out_data[act_cols])**2
                
                tr = out_data[:len(M.TRAIN)][act_cols]
                kMeansVar = [KMeans(n_clusters=k).fit(tr) for k in range(1, 20)]
                centroids = [X.cluster_centers_ for X in kMeansVar]
                k_euclid = [cdist(tr, cent) for cent in centroids]
                dist = [np.min(ke, axis=1) for ke in k_euclid]
                wcss = [sum(d**2) for d in dist]
                tss = sum(pdist(tr)**2)/tr.shape[0]
                bss = tss - wcss
                plt.plot(bss)
                plt.show()
                
                tr = out_data[:len(M.TRAIN)]
                
                Q = {}
                for st in set(out_data.state):
                    for a in act_cols:
                        Q[(st,a)] = {}
                        for a2 in act_cols:
                            Q[(st,a)][a2] = 0
                        
                
                def getAction(state, epsilon=0.05, bestAct=False):
                    if random.random() < epsilon:
                        return random.choice((act_cols))
                    elif bestAct == False:
                        return np.random.choice(list(Q[state].keys()), p=softmax(list(Q[state].values())))
                    else:
                        best, best_v = None, 0
                        for k,v in Q[state].items():
                            if best is None:
                                best = k
                                best_v = v
                                continue
                            if v > best_v:
                                best = k
                                best_v = v
                        return best
                        
                num_iter = 0
                
                loop_forever = True
                while loop_forever:
                    
                    try:
                        H = random.choice(act_cols)
                        pos = random.randint(0, len(M.TRAIN)-2)
                        current_state = tr.at[pos, "state"], H
                        current_action = getAction(current_state, 0.1, False)
                        
                        reward = tr.ix[pos, current_action.replace("ACT","reward")]
                        if H != current_action:
                            reward += math.log10( 1 - 0.000 )
                        
                        new_state = tr.at[pos+1, "state"], current_action
                        next_best_rw = max(Q[new_state].values())
                        
                        td_target = reward + 0.99 * next_best_rw
                        td_error = td_target - Q[current_state][current_action]
                        Q[current_state][current_action] += 0.1 * td_error
                        
                        num_iter += 1
                        if num_iter % 20000 == 0:
                            print(num_iter)
                            #for k, v in Q[(3,"ACT_IOTA")].items():
                            #    print(k, v)
                                
                        if num_iter % 100000 == 0:
                            
                            H = "ACT_USD"
                            tst = out_data[len(M.TRAIN):].reset_index(drop=True)
                            raws, tcs, rewards = [], [], []
                            
                            for pos in range(0, len(tst)-1):
                                
                                current_state = tst.at[pos, "state"], H
                                current_action = getAction(current_state, 0, True)
                                
                                reward = tr.ix[pos, current_action.replace("ACT","reward")]
                                
                                if H != current_action:
                                    tc = math.log10( 1 - 0.002 )
                                else:
                                    tc = 0
                                
                                raws.append(reward)
                                tcs.append(tc)
                                rewards.append(reward+tc)
                                    
                                H = current_action
                        
                            plt.plot(pd.Series(raws).cumsum())
                            print(list(pd.Series(raws).cumsum())[-1])
                            gc.collect()
                            #plt.plot(pd.Series(rewards).cumsum())
                            plt.show()
                            
                    except KeyboardInterrupt:
                        loop_forever = False
                        break'''
                    
                print( len(dat) )
                M.position = 0
                dat[M.PORT_W] = 0
                dat["MARGIN_USD"] = 1
                prevHoldings = None
                all_qs_out = []
                
                G = []
                profits, scaled_profits = [], []
                costs, n_switch = [], []
                Vs = []
                
                price_states = np.array(dat[PRICE_TENSOR_COLS])
                
                for test_pos in range(0, len(dat)-1):
                    
                    w = np.array(dat.loc[M.position, M.PORT_W]).reshape(-1, len(M.PORT_W))
        
                    Q, V  = sess.run([Q_PREDICT, Q_UNSCALED], feed_dict = {
                              X_PRICE_TENSOR : price_states[test_pos].reshape(-1, len(PRICE_TENSOR_COLS) ),
                              HOLD_W : w,
                              keep_p1 : 1, keep_p2 : 1, keep_p3 : 1
                              } )
                    
                    all_qs_out.append(np.round(Q[0], 3))
                    act = M.ACTIONS[np.argmax(Q)]
                    
                    if USE_SIGMOID:
                        binaries = np.apply_along_axis(lambda x : 1 if x > 0.5 else 0, 0, Q)
                    else:
                        binaries = [0] * len(M.ACTIONS)
                        binaries[np.argmax(Q)] = 1
                        binaries = np.array(binaries)
                        
                    profit = sum(binaries * dat.ix[M.position, M.COLS_Y])
                    
                    #if profits:
                        #profit *= ( 10 ** pd.Series(profits).cumsum()[len(profits)-1] )
                    
                    tc = 0
                    if prevHoldings is None:
                        prevHoldings = binaries
                        n_switch.append(0)
                    else:
                        chng = np.abs(binaries - prevHoldings)
                        n_switch.append(chng.sum() > 0)
                        chng = chng * math.log10(1-0.075/100)
                        #chng = chng * -1
                        tc = sum(chng)
                        prevHoldings = binaries
                        
                    costs.append(tc)
                    profits.append(profit)
                    G.append(profit+tc)
                    M.position += 1
                    
                    Vs.append( max(0, max(V[0]) ) )
                    
                    scaled_profits.append(profit * Vs[-1]**0.5 )
                    
                    #act = M.ACTIONS[np.random.choice(range(len(M.ACTIONS)), 
                    #                                      p = softmax(Q[0]))]
                    #G.append( M.stepTest(act) )
                    
                    
                    
                    for w in M.PORT_W:
                        dat.set_value(M.position, w, 0)
                    dat.set_value(M.position, 
                                          M.PORT_W[M.ACTIONS.index(act)], 
                                          1)
                    if test_pos % 1000 == 0 and test_pos > 0:
                        print("Switch Rate: {:.2f}%".format( 100.0 * sum(n_switch) / len(n_switch) ))
                        plt.plot(pd.Series(profits).cumsum())
                        #plt.plot(pd.Series(G).cumsum())
                        plt.show()
                    
                plt.plot(pd.Series(profits).cumsum())
                print("Switch Rate: {:.2f}%".format( 100.0 * sum(n_switch) / len(n_switch) ))
                projections.append(pd.Series(G).cumsum())
                
                for num_p, p in enumerate(projections[::-1]):
                    plt.plot(p)
                    print(p[len(p)-1])
                    if num_p >= 10:
                        break
                plt.show()
                
                for idx in range(len(all_qs_out[0])):
                    hold_data = [x[idx] for x in all_qs_out]
                    plt.plot(pd.Series(hold_data).rolling(200).mean())
                #for c in M.PORT_W:
                #    plt.plot(pd.rolling_mean(dat[c], 10))
                plt.legend(M.PORT_W)
                plt.show()
            i += 1
        watch.end("Gradient_Update")
        all_losses.append(train_loss)
        rolling_window = 2000
        watch.start("rolling_loss")
        rolling_loss = np.mean( all_losses[-rolling_window:]  )
        watch.end("rolling_loss")
        #update_drop_rt = tf.assign(tf_keep_prob, 1)
        #sess.run(update_drop_rt)
        
        Q_NEW = sess.run(Q_PREDICT, feed_dict = {
                      X_PRICE_TENSOR : np.reshape(USD_PRICE_STATE,(-1, len(PRICE_TENSOR_COLS)) ),
                      keep_p1 : 1, keep_p2 : 1, keep_p3 : 1
                      } )
    
        print("Episode: {:<12}, Rolling Loss: {:.6f}, Position: {}".format(
                episode, rolling_loss*10**5, init_pos))
        print("Target: {:<24}, Pred: {:<24}, Upd: {:<24}, Epsilon: {:.2f}%".format(
                "["+"".join(["{}{:<6.3f}%\033[0m ".format(COL_R if x < 0 else G, 100*(10**x-1)) 
                for x in usd_target])+"]",
                "["+"".join(["{}{:<6.3f}%\033[0m ".format(COL_R if x < 0 else G, 100*(10**x-1)) 
                for x in Q_USD[0]])+"]",
                "["+"".join(["{}{:<6.3f}%\033[0m ".format(COL_R if x < 0 else G, 100*(10**x-1))  
                for x in (Q_NEW-Q_USD)[0]])+"]",
                100*epsilon))
        #print(episode, targetQ[0], Q1[0], (Q_NEW-Q1)[0], loss, "{:.6f}".format(epsilon))
        
        X_STATES, PRICE_STATES, Q_PREDS, Q_TARGETS = [], [], [], []
        
    epsilon = 10/((episode/500) + 10)
    epsilon = max(0.001, epsilon)
    epsilon = 0
    
    if episode % 500 == 0:
        watch.display()
    
    episode += 1
