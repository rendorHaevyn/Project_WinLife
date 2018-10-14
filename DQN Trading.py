from __future__ import print_function
import numpy as np
import random
import pandas as pd
import sklearn
import sklearn.decomposition
import sklearn.ensemble
import sklearn.preprocessing
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

############################ START BLACKJACK CLASS ############################
class Market(gym.Env):
    
    """Trading Market environment"""
    
    def randomIndex(self):
        return random.randint(0, len(self.TRAIN)-self.DISCOUNT_STEPS - 10)
    
    def __init__(self, dataFile = None, COINS = [], short = False):
        
        gc.collect()
        
        self.data = pd.read_csv(dataFile)
        self.data = self.data.dropna(axis=0, how='any').reset_index(drop=True)
        cut_off_date = int(time.mktime(time.strptime('01/06/2018', "%d/%m/%Y"))) * 1000
        self.data = self.data[self.data.date > cut_off_date].reset_index(drop=True)
        self.data['reward_USD'] = 0
        if COINS == []:
            COINS = [x[len("close_"):] for x in self.data.columns if "close_" in x]
        self.COINS = COINS
        print("{} rows & {} columns".format(len(self.data), len(self.data.columns)))
        #--------------------------------------------------------------------------------------
        # Manual Options
        #--------------------------------------------------------------------------------------
        self.COMMISSION     = 1e-10  # Commision % as a decimal to use in loss function
        self.USE_PCA        = False   # Use PCA Dimensionality Reduction
        self.NORMALIZE      = True
        self.PCA_COMPONENTS = 200    # Number of Principle Components to reduce down to
        self.INCLUDE_VOLUME = True   # Include Volume as a feature
        self.ALLOW_SHORTS   = True   # Allow Shorts or not
        self.GAMMA          = 1    # The discount factor
        self.DISCOUNT_STEPS = 12     # Number of periods to look ahead for discounting
        self.TRAIN_PERCENT  = 0.7    # Percentage of data to use as training
        #--------------------------------------------------------------------------------------
        # List of coins data to use as input variables. Set to [] to use all coins
        #--------------------------------------------------------------------------------------
        self.N_CHANNEL   = 3 + (1 if self.INCLUDE_VOLUME else 0)
        self.N_COINS     = len(self.COINS)#( len(self.COINS) * 2 - 1 ) if self.ALLOW_SHORTS else len(self.COINS)
        #--------------------------------------------------------------------------------------
        # Create a list of X column names to use for modelling
        #--------------------------------------------------------------------------------------
        in_cols = []
        for c in self.data.columns:
            for a in sorted(set(self.COINS)):
                if a in c:
                    in_cols.append(c)
        
        COLS_X = []
        for x in in_cols:
            if "L_" in x or "L2_" in x or "REG" in x or "PIECE" in x or "MOV" in x or "RSI" in x:
            #if "MOV" in x:
                if "VOLUME" in x and self.INCLUDE_VOLUME == False:
                    continue
                COLS_X.append(x)
                
        #for c in self.data.columns:
        #    if "limit" in c:
        #        self.data[c.replace("limit","reward")] = self.data[c]
        #--------------------------------------------------------------------------------------
        # Create a list of Y column names to use for modelling
        #--------------------------------------------------------------------------------------
        COLS_Y = [] if "USD" not in COINS and "USDT" not in COINS else ["reward_USD"]
        for c in self.data.columns:
            added = False
            if 'reward' in c and 'USD' not in c:
                if self.COINS == []:
                    COLS_Y += [c]
                    added = True
                else:
                    for a in sorted(set(self.COINS)):
                        if a in c:
                            COLS_Y += [c]
                            added = True
                if added:
                    self.data[c+"_S"] = self.data[c].apply(lambda x : math.log10(2-10**x))
                    
        if self.ALLOW_SHORTS:
            COLS_Y += ["{}_S".format(y) for y in COLS_Y if y != "reward_USD"]
        
        self.N_CRYPTO = len([1 for y in COLS_Y if y != "reward_USD" and not y.endswith("_S")])
        
        PORT_W = [w.replace("reward_", "MARGIN_") for w in COLS_Y]
        for p in PORT_W:
            self.data[p] = 0

        self.data["MARGIN_USD"] = 1
        if self.COMMISSION != 0:
            COLS_X += PORT_W
            
        #stmt = "self.data[COLS_Y] = self.data[COLS_Y]"
        #for ahead in range(1,self.DISCOUNT_STEPS+1):
        #    stmt += "+(self.GAMMA**{}) * self.data[COLS_Y].shift({})".format(ahead, -ahead)
        #print("Calculating Discount Rewards...", end="")
        #exec(stmt)
        
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
        if self.NORMALIZE:
            print("Normalizing Data...", end="")
            scaler = sklearn.preprocessing.StandardScaler()
            print("Fitting Scaler: {}".format(len(COLS_X)))
            scaler.fit( self.data[:train_idx][COLS_X] )
            print("Using Scaler: {}".format(len(COLS_X)))
            self.data[COLS_X] = scaler.transform(self.data[COLS_X])
            print("Done")
        #--------------------------------------------------------------------------------------
        # Apply PCA if set to True. Principle Components calculated using training data only
        #--------------------------------------------------------------------------------------
        if self.USE_PCA:
        
            print("PCA...",end="")
            PCA_MODEL = sklearn.decomposition.PCA(self.PCA_COMPONENTS)
            PCA_MODEL.fit(self.data[:train_idx][COLS_X])
            Xs = pd.DataFrame(PCA_MODEL.transform(self.data[COLS_X]))
            
            Xs.columns = ["PCA_"+str(x) for x in range(1,len(Xs.columns)+1)]
            self.data[Xs.columns] = Xs
            COLS_X = list(Xs.columns) + (PORT_W if self.COMMISSION != 0 else [])
        
            print("Done")
            print("Variance explained: {}".format(100*PCA_MODEL.explained_variance_ratio_.cumsum()[-1]))
            #print(PCA_MODEL.explained_variance_)
            #print(PCA_MODEL.explained_variance_ratio_)
            
        self.TRAIN = self.data[:train_idx].reset_index(drop=True)
        #self.TEST = self.TRAIN
        self.TEST = self.data[train_idx:].reset_index(drop=True)
            
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
        
        all_coins = []
        for c in self.data.columns:
            if "reward_" in c and "USD" not in c and not c.endswith("_S"):
                all_coins.append(c.replace("reward_",""))
        #all_coins = ['BTC', 'IOTA']
        
        self.N_CRYPTO_IN = len(all_coins)

        X2 = []
        for x in self.data.columns:
            
            channel_rank = 100
            lag_rank = 100
            
            if "L_LOW" in x:
                channel_rank = 0
            if "L_CLOSE" in x:
                channel_rank = 1
            if "L_HIGH" in x:
                channel_rank = 2
            if "MOV" in x:
                channel_rank = 3        
            if "RSI" in x:
                channel_rank = 4
                
            if "L2_LOW" in x:
                channel_rank = 5
            if "L2_CLOSE" in x:
                channel_rank = 6
            if "L2_HIGH" in x:
                channel_rank = 7
                
            if "L_VOLUME" in x:
                channel_rank = 8
            #if "L2_VOLUME" in x:
            #    channel_rank = 9
                
            if "REG_CLOSE" in x:
                channel_rank = 10
            if "PIECE_CLOSE" in x:
                channel_rank = 11
            #if "REG_VOLUME" in x:
            #    channel_rank = 12
                
            S_COINS = sorted(all_coins)
            coin_rank = -1
            for i, c in enumerate(S_COINS):
                if x.endswith(c):
                    coin_rank = i
                    break
                
            if channel_rank in (10, 11):
                lag_rank = int("".join([ch for ch in x if ch in '0123456789']))
            else:
                try:
                    lag_rank = int("".join([ch for ch in x[x.index("_"):] if ch in '0123456789']))
                    if pattern_match("(RSI|MOV)[0-9]+_*", x):
                        channel_rank += 1 / int("".join([ch for ch in x[:x.index("_")] if ch in '0123456789']))
                    if pattern_match("L2?_(LOW|CLOSE|HIGH|VOLUME)", x) or \
                       pattern_match("(RSI|MOV)[0-9]+_*", x):
                        lag_rank *= -1
                except:
                    pass
            if coin_rank < 0:
                continue
            
            X2.append( (coin_rank, lag_rank, channel_rank, x) )
            
        X2.sort(key = lambda x : (x[0], x[1], x[2]))
        
        PRICE_TENSOR  = [(x[-1], x[-2], x[-3]) for x in X2 if 0 <= x[2] <= (9 if self.INCLUDE_VOLUME else 7)]
        REG_TENSOR    = [(x[-1], x[-2], x[-3]) for x in X2 if 9 <= x[2] <= (12 if self.INCLUDE_VOLUME else 11)]
        
        cols = list(self.data.columns)
        self.PRICE_LAGS = len(set([x[2] for x in PRICE_TENSOR]))
        self.PRICE_CHANNELS = len(set([x[1] for x in PRICE_TENSOR]))
        self.PRICE_TENSOR_COLS = [x[0] for x in PRICE_TENSOR]
        self.PRICE_TENSOR_IDX = [cols.index(x) for x in self.PRICE_TENSOR_COLS]
        
        self.REG_LAGS = len(set([x[2] for x in REG_TENSOR]))
        self.REG_CHANNELS = len(set([x[1] for x in REG_TENSOR]))
        self.REG_TENSOR_COLS = [x[0] for x in REG_TENSOR]
        self.REG_TENSOR_IDX = [cols.index(x) for x in self.REG_TENSOR_COLS]
        
        self.COLS_X_IDX  = [cols.index(x) for x in self.COLS_X]
        
        self.PREV_W_COLS = PORT_W
        self.PREV_W_IDX  = [cols.index(x) for x in self.PREV_W_COLS]
        
        gc.collect()
        
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
    
    def stepTest(self, action):
        
        rw = 0
        
        self.COMM_REWARD = math.log10(1 - self.COMMISSION)
        
        act_loc = M.ACTIONS.index(action)
        if self.TEST.at[self.position, self.PORT_W[act_loc]] == 1:
            rw = 0
        elif action in ("USD", "USDT") or self.TEST.at[self.position, "MARGIN_USD"] == 1:#\
        #(self.TEST.at[self.position, "MARGIN_USD"] == 1 and action not in ("USD", "USDT")):
            rw = 1 * self.COMM_REWARD
        else:
            rw = 2 * self.COMM_REWARD
        
        rw += self.TEST.at[self.position, "reward_{}".format(action)]
        self.position += 1
        
        for w in self.PORT_W:
            self.TEST.set_value(self.position, w, 0)
        self.TEST.set_value(self.position, self.PORT_W[act_loc], 1)
        
        if np.isnan(rw):
            print(self.position, action, self.holdings)
        
        return rw
            
############################ END BLACKJACK CLASS ############################

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

plt.ion()

M = Market("5m/ALL_MOD.csv", ['USD', 'BCH', 'BTC', 'EOS', 'ETH', 'IOTA', 'NEO', 'XRP'])

N_COINS         = M.N_COINS
N_CRYPTO        = M.N_CRYPTO_IN
N_IN            = M.N_IN
N_OUT           = M.N_OUT
PRICE_LAGS      = M.PRICE_LAGS
PRICE_CHANNELS  = M.PRICE_CHANNELS
REG_LAGS        = M.REG_LAGS
REG_CHANNELS    = M.REG_CHANNELS

MAX_PROFIT = True

with tf.device("/cpu:0"):
    # Input / Output place holders
    X = tf.placeholder(tf.float32, [None, N_IN])
    X = tf.reshape(X, [-1, N_IN])
    # PrevW
    PREV_W = tf.placeholder(tf.float32, [None, N_OUT])
    # Actual Rewards
    Y_     = tf.placeholder(tf.float32, [None, N_OUT])
    
    Q_TARGET = tf.placeholder(tf.float32, [None, N_OUT])
    Q_TARGET = tf.reshape(Q_TARGET, [-1, N_OUT])
    
    dropout_prob = tf.placeholder(tf.float32, name = 'dropout_probability')
    #--------------------------------------------------------------------------------------
    # Define hidden layers
    #--------------------------------------------------------------------------------------
    
    h_0         = N_CRYPTO
    w_0         = 1
    CH_OUT_0    = PRICE_CHANNELS
    FILTER0     = [h_0, w_0, PRICE_CHANNELS, CH_OUT_0] # Filter 1 x 3 x 3, Input has 4 channels
    
    h_1         = 1
    w_1         = 5
    CH_OUT_1    = 4
    FILTER1     = [h_1, w_1, PRICE_CHANNELS, CH_OUT_1] # Filter 1 x 3 x 3, Input has 4 channels
    
    h_1A         = N_CRYPTO
    w_1A         = w_1
    CH_OUT_1A    = CH_OUT_1
    FILTER1A     = [h_1A, w_1A, PRICE_CHANNELS, CH_OUT_1A]
    
    h_2         = 1
    w_2         = PRICE_LAGS - w_1 + 1
    CH_OUT_2    = 20
    FILTER2     = [h_2, w_2, CH_OUT_1*1, CH_OUT_2]
    
    # Final
    h_3         = 1
    w_3         = 1
    CH_OUT_3    = 1
    FILTER3     = [h_3, w_3, CH_OUT_2, CH_OUT_3]
    
    # Reg Tensor
    h_R         = 1
    w_R         = REG_LAGS
    CH_OUT_R    = 3
    FILTER_REG  = [h_R, w_R, REG_CHANNELS, CH_OUT_R]
    
    h_R2         = N_COINS-1
    w_R2         = 2
    CH_OUT_R2    = 3
    FILTER_REG2  = [h_R2, w_R2, REG_CHANNELS, CH_OUT_R2]
    
    h_R3         = 1
    w_R3         = REG_CHANNELS
    CH_OUT_R3    = 3
    FILTER_REG3  = [h_R3, w_R3, CH_OUT_R2, CH_OUT_R3]
    
    SDEV        = 1
    BIAS_MULT   = 0
    USE_COMB_L  = 0
    
    initializer     = tf.contrib.layers.xavier_initializer()
    initializer_cnn = tf.contrib.layers.xavier_initializer_conv2d()
    
    X_PRICE_TENSOR = tf.placeholder(tf.float32, [None, len(M.PRICE_TENSOR_COLS)])
    X_PRICE_TENSOR_NN = tf.reshape(X_PRICE_TENSOR, [-1, N_CRYPTO, PRICE_LAGS, PRICE_CHANNELS])
    
    X_REG_TENSOR   = tf.placeholder(tf.float32, [None, len(M.REG_TENSOR_COLS)])
    X_REG_TENSOR_NN   = tf.reshape(X_REG_TENSOR, [-1, N_CRYPTO, REG_LAGS, REG_CHANNELS])
    
    #CW1Z = tf.Variable(tf.random_normal([N_COINS-1,2,N_CHANNEL,N_CHANNEL], stddev = SDEV))
    #CB1Z = tf.Variable(tf.zeros([N_CHANNEL]))
    #CL1Z = tf.nn.leaky_relu(tf.nn.conv2d(X_PRICE_TENSOR, CW1Z, [1,1,1,1], padding="SAME") + CB1Z * BIAS_MULT)
    
    CW0 = tf.Variable(tf.random_normal(FILTER0, stddev = SDEV * (1/(h_0*w_0*PRICE_CHANNELS))**0.5 ))
    CB0 = tf.Variable(tf.zeros([CH_OUT_0]))
    CL0 = tf.nn.leaky_relu(tf.nn.conv2d(X_PRICE_TENSOR_NN, CW0, [1,1,1,1], 
                                  padding="VALID") + CB0 * BIAS_MULT)
    
    X_PRICE_TENSOR_NEW = tf.concat([X_PRICE_TENSOR_NN, CL0], 1)
    
    CW1 = tf.Variable(tf.random_normal(FILTER1, stddev = SDEV * (1/(h_1*w_1*PRICE_CHANNELS))**0.5 ))
    CB1 = tf.Variable(tf.zeros([CH_OUT_1]))
    CL1 = tf.nn.leaky_relu(tf.nn.conv2d(X_PRICE_TENSOR_NEW if USE_COMB_L else X_PRICE_TENSOR_NN, CW1, [1,1,1,1], 
                                  padding="VALID") + CB1 * BIAS_MULT)
    
    CW1A = tf.Variable(tf.random_normal(FILTER1A, stddev = SDEV * (1/(h_1*w_1*PRICE_CHANNELS))**0.5 ))
    CB1A = tf.Variable(tf.zeros([CH_OUT_1A]))
    CL1A = tf.nn.leaky_relu(tf.nn.conv2d(X_PRICE_TENSOR_NN, CW1A, [1,1,1,1], 
                                  padding="VALID") + CB1A * BIAS_MULT)
    
    CL1A = tf.nn.dropout(CL1A, dropout_prob)
    CL1 = tf.nn.dropout(CL1, dropout_prob)
    #CL1 = tf.concat([CL1A, CL1], 1)
    #CL1 = tf.layers.batch_normalization(CL1)
    
    # Shape is N_COINS x (LAGS - 2) x 6
    
    CW2 = tf.Variable(tf.random_normal(FILTER2, stddev = SDEV * (1/(h_2*w_2*CH_OUT_1))**0.5))
    CB2 = tf.Variable(tf.zeros([CH_OUT_2]))
    CL2 = (tf.nn.conv2d(CL1, CW2, [1,1,1,1], padding="VALID") + CB2 * BIAS_MULT)
    
    CL2 = tf.nn.dropout(CL2, dropout_prob)
    
    #CL2 = tf.layers.batch_normalization(CL2)
    #CL3 = tf.concat([CL3, tf.reshape(PREV_W[:,1:], (-1,N_COINS-1,1,1))], -1)
    # Shape is N_COINS x 1 x 30
    
    #CWREG = tf.Variable(tf.random_normal(FILTER_REG, stddev = SDEV))
    #CBREG = tf.Variable(tf.zeros([CH_OUT_R]))
    #CLREG = tf.nn.leaky_relu(tf.nn.conv2d(X_REG_TENSOR, CWREG, [1,1,1,1], padding="VALID") + CBREG * BIAS_MULT)
    
    #CWREG2 = tf.Variable(tf.random_normal(FILTER_REG2, stddev = SDEV))
    #CBREG2 = tf.Variable(tf.zeros([CH_OUT_R2]))
    #CLREG2 = tf.nn.leaky_relu(tf.nn.conv2d(X_REG_TENSOR, CWREG2, [1,1,1,1], padding="SAME") + CBREG2 * BIAS_MULT)
    
    #CWREG3 = tf.Variable(tf.random_normal(FILTER_REG3, stddev = SDEV))
    #CBREG3 = tf.Variable(tf.zeros([CH_OUT_R3]))
    #CLREG3 = tf.nn.leaky_relu(tf.nn.conv2d(CLREG2, CWREG3, [1,1,1,1], padding="VALID") + CBREG3 * BIAS_MULT)
    
    #CL3 = tf.concat([CL3, CLREG, CLREG3], -1)
    CL3 = CL2
    
    CW4 = tf.Variable(tf.random_normal(FILTER3, stddev = SDEV * (1/(h_3*w_3*CH_OUT_2))**0.5))
    CB4 = tf.Variable(tf.zeros([CH_OUT_3]))
    CL4A = (tf.nn.conv2d(CL3, CW4, [1,1,1,1], padding="SAME") + CB4 * BIAS_MULT)
    CL4 = tf.reshape( CL4A, (-1, N_COINS) )
    
    coin_shape = N_COINS if USE_COMB_L else N_COINS-1
    coin_shape = N_CRYPTO
    
    CL_flat = tf.reshape(CL3, (-1, (coin_shape)*1*CH_OUT_2))
    #CL_flat = tf.layers.batch_normalization(CL_flat)
    
    initializer = tf.contrib.layers.xavier_initializer()
    
    in_shape = (coin_shape)*1*CH_OUT_2
    
    #fc_w = tf.Variable(tf.random_normal([in_shape, 200], stddev = SDEV), trainable=True)
    #fc_w = fc_w * (2/(coin_shape)*1*CH_OUT_2)**0.5
    #fc_b = tf.Variable(tf.zeros([200]))
    
    fc_w = tf.Variable( initializer([in_shape, 200]) )
    fc_b = tf.Variable( initializer([200]) )
    
    fc_w2 = tf.Variable( initializer([200, N_OUT]) )
    fc_b2 = tf.Variable( initializer([N_OUT]) )
    
    
    #fc_w = tf.ones( [in_shape, 200] )
    #fc_b = tf.ones( [200] )
    
    #fc_w2 = tf.ones( [200, N_OUT] )
    #fc_b2 = tf.ones( [N_OUT] )
    
    CL_flat2 = tf.nn.leaky_relu( tf.matmul(CL_flat, fc_w) + 0*fc_b)
    CL_flat2 = tf.nn.dropout(CL_flat2, dropout_prob)
    #CL_flat2 = tf.layers.batch_normalization(CL_flat2)
    #CL_flat2 = tf.nn.dropout(CL_flat2, dropout_prob)
    Q_PREDICT = CL4
    Q_PREDICT = tf.concat([Q_PREDICT, -1*Q_PREDICT], axis=1)
    Q_PREDICT = (tf.matmul(CL_flat2, fc_w2) + 0*fc_b2)
    
    
    if MAX_PROFIT:
        Q_PREDICT = tf.nn.softmax(Q_PREDICT, 1)
    
    
    #reg_losses = tf.nn.l2_loss(fc_w) + tf.nn.l2_loss(fc_w2)
    #reg_losses = tf.nn.l2_loss(CW1) + tf.nn.l2_loss(CW2) + tf.nn.l2_loss(CW4)
    
    #CW4 = tf.Variable(tf.random_normal(FILTER3, stddev = SDEV))
    #CB4 = tf.Variable(tf.zeros([CH_OUT_3]))
    #CL4A = (tf.nn.conv2d(CL3, CW4, [1,1,1,1], padding="SAME") + CB4 * BIAS_MULT)
    #CL4 = tf.reshape( CL4A, (-1, N_COINS) )
    
    #tf_mean = tf.Variable(0.0)
    #tf_var  = tf.Variable(1.0)
    #CL4 = tf.nn.batch_normalization(CL4, tf_mean, tf_var, None, None, variance_epsilon=1e-6)
    
    #fc_w = tf.Variable(tf.random_normal([N_COINS, N_OUT], stddev = SDEV), trainable=True)
    #fc_b = tf.Variable(tf.zeros([N_OUT]))
    
    #Q_PREDICT = tf.matmul(CL4, fc_w) + fc_b * BIAS_MULT
    #Q_PREDICT = CL4
    
    #cnn_keep_prob = tf.Variable(0.85, trainable=False)
    #fc_w_dropped = tf.nn.dropout(fc_w, cnn_keep_prob)
    
    #Y_scores = tf.matmul(CL4, fc_w_dropped) + fc_b * BIAS_MULT
    #Y = tf.nn.softmax(Y_scores)
    '''
    # Define number of Neurons per layer
    K = 100 # Layer 1
    L = 40 # Layer 2
    M = 20 # Layer 2
    
    SDEV  = 0.1
    
    # Input / Output place holders
    X = tf.placeholder(tf.float32, [None, N_IN])
    X = tf.reshape(X, [-1, N_IN])
    
    # This will be the observed reward + decay_factor * max(Q[s+1, 0], Q[s+1, 1]).
    # This should be an estimate of the 'correct' Q-value with the ony caveat being that
    # the Q-value of the next state is a biased estimate of the true value.
    
    # LAYER 1
    W1 = tf.Variable(tf.random_normal([N_IN, K], stddev = SDEV))
    B1 = tf.Variable(tf.random_normal([K], stddev = SDEV))
    
    # LAYER 2
    W2 = tf.Variable(tf.random_normal([K, L], stddev = SDEV))
    B2 = tf.Variable(tf.random_normal([L], stddev = SDEV))
    
    # LAYER 3
    W3 = tf.Variable(tf.random_normal([L, M], stddev = SDEV))
    B3 = tf.Variable(tf.random_normal([M], stddev = SDEV))
    
    # LAYER 4
    W4 = tf.Variable(tf.random_normal([M, N_OUT], stddev = SDEV))
    B4 = tf.Variable(tf.random_normal([N_OUT], stddev = SDEV))
    
    BIAS_MULT = 0
    
    H1  = tf.nn.leaky_relu(tf.matmul(X,  W1) + BIAS_MULT * B1)
    #H1  = tf.layers.batch_normalization(H1)
    #H1  = tf.nn.dropout(H1, tf_keep_prob)
    
    H2  = tf.nn.leaky_relu(tf.matmul(H1,  W2) + BIAS_MULT * B2)
    #H2  = tf.layers.batch_normalization(H2)
    #H2  = tf.nn.dropout(H2, tf_keep_prob)
    
    H3  = tf.nn.leaky_relu(tf.matmul(H2,  W3) + BIAS_MULT * B3)
    #H3  = tf.layers.batch_normalization(H3)
    #H3  = tf.nn.dropout(H3, tf_keep_prob)
    
    # The predicted Q value, as determined by our network (function approximator)
    # outputs expected reward for standing and hitting in the form [stand, hit] given the
    # current game state
    Q_PREDICT  = (tf.matmul(H3,  W4) + BIAS_MULT * B4)'''
    
    # Is this correct? The Q_TARGET should be a combination of the real reward and the discounted
    # future rewards of the future state as predicted by the network. Q_TARGET - Q_PREDICT should be
    # the error in prediction, which we want to minimise. Does this loss function work to help the network
    # converge to the true Q values with sufficient training?
    q_predict_mean, q_predict_var = tf.nn.moments(Q_PREDICT, axes=[1])
    loss_func_start = tf.reduce_mean(q_predict_var)
    loss_func = tf.reduce_sum(tf.reduce_sum(tf.abs(Q_TARGET - Q_PREDICT), axis=1))
    if MAX_PROFIT:
        loss_func = -tf.reduce_sum(Q_PREDICT * Q_TARGET)
        #loss_func = -tf.reduce_sum(Q_PREDICT * Q_TARGET) \
        #            - math.log10(1-M.COMMISSION)*tf.reduce_sum( tf.abs(tf.reduce_sum(Q_PREDICT[1:,:] - Q_PREDICT[:-1,:], 1) ) )
    #loss_func = tf.reduce_mean(tf.losses.huber_loss(Q_TARGET, Q_PREDICT))
    #losses_func = (tf.square(Q_TARGET - Q_PREDICT))
    
    # This are some placeholder values to enable manually set decayed learning rates. For now, use
    # the same learning rate all the time.
    LR_START = 0.00003
    #LR_END   = 0.000002
    #LR_DECAY = 0.999
    
    # Optimizer
    LEARNING_RATE = tf.Variable(LR_START, trainable=False)
    optimizer     = tf.train.RMSPropOptimizer(LEARNING_RATE)#(LEARNING_RATE)
    train_step_start    = optimizer.minimize(loss_func_start)
    train_step    = optimizer.minimize(loss_func)
    
    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(init)

# Number of episodes (games) to play
num_eps  = 100000000
# probability of picking a random action. This decays over time
epsilon  = 0.1

all_rewards = [] # Holds all observed rewards. The rolling mean of rewards should improve as the network learns
all_Qs      = [] # Holds all predicted Q values. Useful as a sanity check once the network is trained
all_losses  = [] # Holds all the (Q_TARGET - Q_PREDICTED) values. The rolling mean of this should decrease
hands       = [] # Holds a summary of all hands played. (game_state, Q[stand], Q[hit], action_taken)


Q_TARGETS    = []
Q_PREDS      = []
X_STATES     = []
PRICE_STATES = []
STR_STATE    = []
ACTS         = []

Q_CONVERGE = {}

IDEAL_RESULTS = {}
last_update = 0
projections = []
episode = 0

class stopwatch:
    
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

watch = stopwatch()

train_losses, test_losses = [], []

gc.collect()
while episode < 1000000:
    
    init_pos = episode % (len(M.TRAIN)-50)#
    #init_pos = M.randomIndex()
    M.position = init_pos
    
    USD_STATE = None
    USD_PRICE_STATE = None
    Q_USD = 0
    
    '''if episode == 100:
        update_LR = tf.assign(LEARNING_RATE, 0.0005)
        sess.run(update_LR)
        
    if episode == 5000:
        update_LR = tf.assign(LEARNING_RATE, 0.0001)
        sess.run(update_LR)
    
    if episode == 25000:
        update_LR = tf.assign(LEARNING_RATE, 0.00003)
        sess.run(update_LR)
        
    if episode == 100000:
        update_LR = tf.assign(LEARNING_RATE, 0.000002)
        sess.run(update_LR)'''
    
    for w_index, starting_w in enumerate(M.PORT_W):
        
        watch.start('update_W')
        M.position = init_pos
        for w in M.PORT_W:
            M.TRAIN.set_value(M.position, w, 0)
        M.TRAIN.set_value(M.position, starting_w, 1)
        watch.end('update_W')
        
        watch.start('set_state')
        init_state = M.TRAIN.iloc[M.position, M.COLS_X_IDX]
        init_price_state = M.TRAIN.iloc[M.position, M.PRICE_TENSOR_IDX]
        watch.end('set_state')
        
        watch.start('Q_PREDICT')
        Q1 = sess.run(Q_PREDICT, feed_dict = {X : np.reshape(init_state,(-1, N_IN)),
                      X_PRICE_TENSOR : np.reshape(init_price_state,(-1, len(M.PRICE_TENSOR_COLS)) ), 
                      dropout_prob : 1} )
        watch.end('Q_PREDICT')
        if w_index == 0:
            USD_STATE = init_state
            USD_PRICE_STATE = init_price_state
            Q_USD = Q1
            
        targetQ = list(Q1[0])
        
        for act_num, begin_act in enumerate(M.ACTIONS):
            
            M.position = init_pos
            for w in M.PORT_W:
                M.TRAIN.set_value(M.position, w, 0)
            M.TRAIN.set_value(M.position, starting_w, 1)
            #print(M.TRAIN.loc[M.position, M.PORT_W])
            
            watch.start("market_step")
            G = M.step(begin_act)
            Gpercent = 100*(10**G-1)
            #G = math.log10(1+int(Gpercent*8)/800)
            watch.end("market_step")
            #for w in M.PORT_W:
            #    M.TRAIN.set_value(M.position, w, 0)
            #M.TRAIN.set_value(M.position, M.PORT_W[act_num], 1)
            
            for t in range(0):#M.DISCOUNT_STEPS):
                
                state = np.array(M.TRAIN.loc[M.position, M.COLS_X])
                price_state = np.array(M.TRAIN.loc[M.position, M.PRICE_TENSOR_COLS])
                
                if random.random() < epsilon:
                    act = random.choice(M.ACTIONS)
                else:
                    Q    = sess.run(Q_PREDICT, feed_dict = {X : state.reshape(-1, N_IN),
                                X_PRICE_TENSOR : price_state.reshape(-1, len(M.PRICE_TENSOR_COLS)),
                                dropout_prob : 1} )
                    
                    act = M.ACTIONS[np.argmax(Q)]
                
                if t == M.DISCOUNT_STEPS-1 and episode > 1000:
                    G += M.GAMMA ** (t+1) * max(Q[0])
                else:
                    G += M.GAMMA ** (t+1) * M.step(act)
                    
                #for w in M.PORT_W:
                #    M.TRAIN.set_value(M.position, w, 0)
                #M.TRAIN.set_value(M.position, M.PORT_W[M.ACTIONS.index(act)], 1)
            
            targetQ[act_num] = G
        
        X_STATES.append(init_state)
        PRICE_STATES.append(init_price_state)
        Q_PREDS.append(Q1)
        Q_TARGETS.append(targetQ)
        
        if w_index == 0:
            usd_target = copy.deepcopy(targetQ)
            break
    
    num_depth = 1+max(0, math.log(episode+1)-2)+len(M.TRAIN)#*0.15
    num_depth = len(M.TRAIN)
    #num_depth = 1024
    if len(Q_TARGETS) >= num_depth:
        
        W  = '\033[0m'  # white (normal)
        R  = '\033[41m' # red
        G  = '\033[42m' # green
        O  = '\033[33m' # orange
        B  = '\033[34m' # blue
        P  = '\033[35m' # purple
        
        #update_drop_rt = tf.assign(tf_keep_prob, 0.7)
        #sess.run(update_drop_rt)
        
        the_x = np.reshape( np.array(X_STATES),  (-1, N_IN) )
        the_p = np.reshape( np.array(PRICE_STATES), (-1, len(M.PRICE_TENSOR_COLS)))
        the_q = np.reshape( np.array(Q_TARGETS), (-1, N_OUT))
        watch.start("Gradient_Update")
        #for i in range(int(num_depth+0.5)):
        i = 0
        while i < 2000000000:
            
            use_sample = False
            tmp_drop_prob = max(0.5, min(1, 1- i/10000))
            tmp_drop_prob = 0.5
            opt = train_step_start if i < 200 or random.random() < 0.02 else train_step
            l_func = loss_func_start if i < 200 else loss_func
            #opt = train_step
            if use_sample:
                samples = random.sample(range(len(the_x)), min(i//2+1,len(the_x)))
                sess.run(opt, 
                                      feed_dict = {X_PRICE_TENSOR : the_p[samples,:],
                                                   Q_TARGET : the_q[samples,:],
                                                   dropout_prob : tmp_drop_prob}  )
    
                
            else:
                sess.run(opt, 
                                      feed_dict = {X_PRICE_TENSOR : the_p,
                                                   Q_TARGET : the_q,
                                                   dropout_prob : tmp_drop_prob}  )
    
            
            if i%25==0:
                train_loss = sess.run(l_func, 
                                      feed_dict = {X_PRICE_TENSOR : the_p,
                                                   Q_TARGET : the_q,
                                                   dropout_prob : 1}  )
                #state = np.reshape(M.TEST[M.COLS_X], (-1, N_IN) )
                price_state = np.reshape(M.TEST[M.PRICE_TENSOR_COLS], (-1, len(M.PRICE_TENSOR_COLS)) )
                truth = np.reshape(M.TEST[M.COLS_Y], (-1, len(M.COLS_Y)) )
                test_loss = sess.run(l_func, 
                                      feed_dict = {X_PRICE_TENSOR : price_state,
                                                   Q_TARGET : truth,
                                                   dropout_prob : 1}  )
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                plt.plot(train_losses[-20:])
                plt.plot(test_losses[-20:])
                plt.legend(["Train", "Test"])
                plt.show()
                print("Iteration: {:<10}, Train Loss: {:<.8f}, Test Loss: {:<.8f}".format(i,train_loss, test_loss))

            if i % 20000 == 0 and i > 0:
                gc.collect()
                print( i )
                M.position = 0
                M.TEST[M.PORT_W] = 0
                M.TEST["MARGIN_USD"] = 1
                
                G = []
                for test_pos in range(0, len(M.TEST)-1):
                    
                    state = np.array(M.TEST.loc[M.position, M.COLS_X])
                    price_state = np.array(M.TEST.loc[M.position, M.PRICE_TENSOR_COLS])
        
                    Q     = sess.run(Q_PREDICT, feed_dict = {X : state.reshape(-1, N_IN),
                              X_PRICE_TENSOR : price_state.reshape(-1, len(M.PRICE_TENSOR_COLS) ),
                              dropout_prob : 1
                              } )
                    act = M.ACTIONS[np.argmax(Q)]
                    #act = M.ACTIONS[np.random.choice(range(len(M.ACTIONS)), 
                    #                                      p = softmax(Q[0]))]
                    G.append( M.stepTest(act) )
                    
                    #for w in M.PORT_W:
                    #    M.TEST.set_value(M.position, w, 0)
                    #M.TEST.set_value(M.position, 
                    #                      M.PORT_W[M.ACTIONS.index(act)], 
                    #                      1)
                    
                projections.append(pd.Series(G).cumsum())
                for num_p, p in enumerate(projections[::-1]):
                    plt.plot(p)
                    print(p[len(p)-1])
                    if num_p >= 10:
                        break
                plt.show()
                for c in M.PORT_W:
                    plt.plot(pd.rolling_mean(M.TEST[c], 500))
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
        
        Q_NEW = sess.run(Q_PREDICT, feed_dict = {X : np.reshape(USD_STATE,(-1, N_IN)),
                      X_PRICE_TENSOR : np.reshape(USD_PRICE_STATE,(-1, len(M.PRICE_TENSOR_COLS)) ),
                      dropout_prob : 1
                      } )
    
        print("Episode: {:<12}, Rolling Loss: {:.6f}, Position: {}".format(
                episode, rolling_loss*10**5, init_pos))
        print("Target: {:<24}, Pred: {:<24}, Upd: {:<24}, Epsilon: {:.2f}%".format(
                "["+"".join(["{}{:<6.3f}%\033[0m ".format(R if x < 0 else G, 100*(10**x-1)) 
                for x in usd_target])+"]",
                "["+"".join(["{}{:<6.3f}%\033[0m ".format(R if x < 0 else G, 100*(10**x-1)) 
                for x in Q_USD[0]])+"]",
                "["+"".join(["{}{:<6.3f}%\033[0m ".format(R if x < 0 else G, 100*(10**x-1))  
                for x in (Q_NEW-Q_USD)[0]])+"]",
                100*epsilon))
        #print(episode, targetQ[0], Q1[0], (Q_NEW-Q1)[0], loss, "{:.6f}".format(epsilon))
        
        X_STATES, PRICE_STATES, Q_PREDS, Q_TARGETS = [], [], [], []
        
    epsilon = 10/((episode/500) + 10)
    epsilon = max(0.001, epsilon)
    epsilon = 0
    
    if episode % 500 == 0:
        watch.display()
    
    all_l = []
    if episode % 20000 == 0 and episode > 0:
        
        #update_drop_rt = tf.assign(tf_keep_prob, 1)
        #sess.run(update_drop_rt)
        
        #M.TEST = M.data[int(M.TRAIN_PERCENT*len(M.data)):].reset_index(drop=True)
        for CUT in range(1,11):
            M.position = 0
            M.TEST[M.PORT_W] = 0
            M.TEST["MARGIN_USD"] = 1
            
            Qs, G, A = [], [], []
            
            action_time, lastAct = 0, None
            
            for test_pos in range(0, len(M.TEST)-1):
                
                #state = np.array(M.TEST.loc[M.position, M.COLS_X])
                price_state = np.array(M.TEST.loc[M.position, M.PRICE_TENSOR_COLS])
    
                Q,L     = sess.run([Q_PREDICT, loss_func], feed_dict = #{X : state.reshape(-1, N_IN),
                          {X_PRICE_TENSOR : price_state.reshape(-1, len(M.PRICE_TENSOR_COLS) ),
                          dropout_prob : 1,
                          Q_TARGET : np.reshape((M.TEST.loc[M.position, M.COLS_Y]), (-1, len(M.COLS_Y)))
                          } )
        
                act = M.ACTIONS[np.argmax(Q)]
                #act = M.ACTIONS[np.random.choice(range(len(M.ACTIONS)), 
                #                                      p = softmax(Q[0]))]
                #print(Q)
                #print(M.TEST.loc[M.position, M.COLS_Y])
                #print(L)
                all_l.append(L)
                
                if test_pos - action_time < CUT and lastAct != None:
                    act = lastAct
                else:
                    lastAct = act
                    action_time = test_pos
                
                
                if MAX_PROFIT:
                    #G.append(sum(Q[0] * M.TEST.loc[M.position, M.COLS_Y]))
                    rwd = M.stepTest(act)
                    G.append( rwd )
                else:
                    rwd = M.stepTest(act)
                    G.append( rwd )
                    
                Qs.append(Q)
                A.append(act)
                
                #for w in M.PORT_W:
                #    M.TEST.set_value(M.position, w, 0)
                #M.TEST.set_value(M.position, 
                #                      M.PORT_W[M.ACTIONS.index(act)], 
                #                      1)
                
            projections.append(pd.Series(G).cumsum())
            for num_p, p in enumerate(projections[::-1]):
                plt.plot(p)
                print(p[len(p)-1])
                if num_p >= 10:
                    break
            plt.show()
            for c in M.PORT_W:
                plt.plot(pd.rolling_mean(M.TEST[c], 500))
            plt.legend(M.PORT_W)
            plt.show()
    
    episode += 1