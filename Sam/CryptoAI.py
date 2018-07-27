from __future__ import print_function
import numpy as np
import random
import pandas as pd
import sklearn
import sklearn.decomposition
import sklearn.ensemble
import sklearn.preprocessing
import math
import os
import sys
import itertools
import threading
from matplotlib import pyplot as plt
import tensorflow as tf
import re
import time
import pickle
import Constants
#tensorboard logs path
LOG_PATH = "logs"

# Utility Function to return True / False regex matching
def pattern_match(patt, string):
    return re.findall(patt, string) != []
# Utility Function to save objects in memory to a file
def save_memory(obj, path):
    return pickle.dump(obj, open(path, "wb"))
# Utility Function to load objects from the harddisk
def load_memory(path):
    return pickle.load(open(path, "rb"))

#cut_off_date = int(time.mktime(time.strptime('01/01/2018', "%d/%m/%Y"))) * 1000

#--------------------------------------------------------------------------------------
# Read in the price data
#--------------------------------------------------------------------------------------
print("Loading Data...", end="")
data_raw = pd.read_csv("15m/ALL_MOD.csv").dropna(axis=0, how='any').reset_index(drop=True)
data     = data_raw[data_raw.date > cut_off_date].reset_index(drop=True)
data     = data.drop('date', axis=1)
data['reward_USD'] = 0
print("{} rows & {} columns".format(len(data), len(data.columns)))
#--------------------------------------------------------------------------------------
# Manual Options
#--------------------------------------------------------------------------------------
COMMISSION     = 0.000 # Commision % as a decimal to use in loss function
USE_PCA        = False   # Use PCA Dimensionality Reduction
PCA_COMPONENTS = 400     # Number of Principle Components to reduce down to
USE_SUPER      = False   # Create new features using supervised learning
INCLUDE_VOLUME = True    # Include Volume as a feature
ALLOW_SHORTS   = True   # Allow Shorts or not
DISCOUNT       = False   # Train on discounted rewards
DISCOUNT_STEPS = 24      # Number of periods to look ahead for discounting
GAMMA          = 0.75    # The discount factor

SAVE_MODELS    = True
TRADING_PATH   = "Live Trading"
SAVE_LENGTH    = 0.33    # Save all pre-processing models from this percentage of raw data onwards
#--------------------------------------------------------------------------------------
# Defining the batch size and test length
#--------------------------------------------------------------------------------------
BATCH_SZ_MIN = 50
BATCH_SZ_MAX = 50
TEST_LEN     = int(round(0.25*len(data)))
IDX_MAX      = int(max(0, len(data) - TEST_LEN - BATCH_SZ_MAX - 1))
SAVE_IDX     = int(round(SAVE_LENGTH * len(data_raw)))
#--------------------------------------------------------------------------------------
# List of coins to trade. Set to [] to use all coins
#--------------------------------------------------------------------------------------
COINS       = ['USD', 'BCH', 'BTC', 'ETH', 'IOTA']
#COINS = ['USD', 'IOTA', 'EOS']
# List of coins data to use as input variables. Set to [] to use all coins
#--------------------------------------------------------------------------------------
INPUT_COINS = []
N_CHANNEL   = 3# + (1 if INCLUDE_VOLUME else 0)
N_COINS     = len(COINS)#( len(COINS) * 2 - 1 ) if ALLOW_SHORTS else len(COINS)
#--------------------------------------------------------------------------------------
# Create fields to store "Previous Weights" - Only needed when commission is > 0
#--------------------------------------------------------------------------------------
PORT_W = []
if COMMISSION > 0:
    PORT_W.append('MARGIN_USD')
    data["MARGIN_USD"] = 1
    for i, a in enumerate(sorted(COINS)):
        data["MARGIN_{}".format(a)] = 1 if a == "USD" else 0
        if "MARGIN_{}".format(a) not in PORT_W:
            PORT_W.append("MARGIN_{}".format(a))
    if ALLOW_SHORTS:
        for i, a in enumerate(sorted(COINS)):
            if a in ["USD", "USDT"]:
                continue
            data["MARGIN_{}_S".format(a)] = 0

if ALLOW_SHORTS:
    x = list(PORT_W)
    for asset in x[1:]:
        PORT_W.append(asset+"_S")
#--------------------------------------------------------------------------------------
# Create a list of X column names to use for modelling
#--------------------------------------------------------------------------------------
in_cols = []
for c in data.columns:
    if INPUT_COINS == []:
        in_cols.append(c)
    else:
        for a in set(INPUT_COINS):
            if a in c:
                in_cols.append(c)

COLS_X = []
for x in in_cols:
    if "L_" in x or "L2_" in x or "REG" in x:
        if "VOLUME" in x and INCLUDE_VOLUME == False:
            continue
        COLS_X.append(x)
if COMMISSION != 0:
    COLS_X += PORT_W
#--------------------------------------------------------------------------------------
# Create a list of Y column names to use for modelling
#--------------------------------------------------------------------------------------

COLS_Y = ["reward_USD"]
for c in data.columns:
    added = False
    if 'reward' in c and 'USD' not in c:
        if COINS == []:
            COLS_Y += [c]
            added = True
        else:
            for a in set(COINS):
                if a in c:
                    COLS_Y += [c]
                    added = True
        if added:
            data[c+"_S"] = data[c].apply(lambda x : math.log10(2-10**x))
if ALLOW_SHORTS:
    COLS_Y += ["{}_S".format(y) for y in COLS_Y[1:]]
#--------------------------------------------------------------------------------------
# Normalizing the X columns. Scale using training data only
#--------------------------------------------------------------------------------------
print("Normalizing Data...", end="")
#for x in COLS_X:
    #print("Finding Median: {}".format(x))
    #median      = data[SAVE_IDX:][x].describe()[5]
    #print("Applying Median: {}".format(x))
    #data[x].fillna(median,inplace=True)
    #data[x]     = data[x].apply(lambda x : median if np.isinf(x) or np.isnan(x) else x)
scaler = sklearn.preprocessing.StandardScaler()
print("Fitting Scaler: {}".format(len(COLS_X)))
scaler.fit( data[:IDX_MAX+BATCH_SZ_MAX] [COLS_X] )
print("Using Scaler: {}".format(len(COLS_X)))
data[COLS_X] = scaler.transform(data[COLS_X])
if SAVE_MODELS:
    live_scaler = sklearn.preprocessing.StandardScaler()
    live_scaler.fit( data[SAVE_IDX:] [COLS_X] )
    save_memory(live_scaler, TRADING_PATH+"/Scaler.save")
    save_memory(COLS_X, TRADING_PATH+"/COLS_X_ORIG.save")
print("Done")
#--------------------------------------------------------------------------------------
# Apply PCA if set to True. Principle Components calculated using training data only
#--------------------------------------------------------------------------------------
if USE_PCA:

    print("PCA...",end="")
    PCA_MODEL = sklearn.decomposition.PCA(PCA_COMPONENTS)
    PCA_MODEL.fit(data[:IDX_MAX+BATCH_SZ_MAX][COLS_X])
    Xs = pd.DataFrame(PCA_MODEL.transform(data[COLS_X]))
    
    Xs.columns = ["PCA_"+str(x) for x in range(1,len(Xs.columns)+1)]
    data[Xs.columns] = Xs
    COLS_X = list(Xs.columns) + (PORT_W if COMMISSION != 0 else [])
    
    if SAVE_MODELS:
        live_pca = sklearn.decomposition.PCA(PCA_COMPONENTS)
        live_pca.fit( data[SAVE_IDX:] [COLS_X] )
        save_memory(live_pca, TRADING_PATH+"/PCA.save")

#    print(PCA_MODEL.explained_variance_)
#    print(PCA_MODEL.explained_variance_ratio_)
    print("Done")
    print("Variance explained: {}".format(100*PCA_MODEL.explained_variance_ratio_.cumsum()[-1]))
#--------------------------------------------------------------------------------------
# Generate Supervised Learning Predictions if set to True. This does not work for now
#--------------------------------------------------------------------------------------
if USE_SUPER:
    pass
    '''training = data[:IDX_MAX]
    cols_to_add = []
    for target in COLS_Y:
        model = sklearn.ensemble.RandomForestRegressor()
        model.fit(training[COLS_X], training[target])
        newcol = "RF_{}".format(target)
        data[newcol] = model.predict(data[COLS_X])
        cols_to_add.append(newcol)
    COLS_X += cols_to_add'''
    
#--------------------------------------------------------------------------------------
# Transform rewards into discounted reward if enabled. "data" uses transformed 
# rewards, "data_imm" uses raw, un-modified reward.
#--------------------------------------------------------------------------------------
if DISCOUNT:
    data_imm = data.copy()
    stmt = "data[COLS_Y] = data[COLS_Y]"
    for ahead in range(1,DISCOUNT_STEPS+1):
        stmt += "+(GAMMA**{}) * data[COLS_Y].shift({})".format(ahead, -ahead)
    print("Calculating Discount Rewards...", end="")
    exec(stmt)
    print("Done")
else:
    data_imm = data.copy()

data = data.dropna(axis=0, how='any').reset_index(drop=True)
if DISCOUNT:
    data_imm = data_imm[:-DISCOUNT_STEPS]
    
#for c in COLS_Y:
#    data[c] = data[c] - math.log10(1.004)
#data["reward_USD"] = 0
    
N_IN  = len(COLS_X)
N_OUT = len(COLS_Y)

if SAVE_MODELS:
    save_memory(COLS_X, TRADING_PATH+"/COLS_X.save")
    save_memory(COLS_Y, TRADING_PATH+"/COLS_Y.save")

#--------------------------------------------------------------------------------------
#  
#                                NEURAL NETWORK DESIGN
#
#--------------------------------------------------------------------------------------
    
X2 = []
for x in COLS_X:
    
    channel_rank = 100
    if "L_LOW" in x:
        channel_rank = 0
    if "L_CLOSE" in x:
        channel_rank = 1
    if "L_HIGH" in x:
        channel_rank = 2
        
    if "L2_LOW" in x:
        channel_rank = 3
    if "L2_CLOSE" in x:
        channel_rank = 4
    if "L2_HIGH" in x:
        channel_rank = 5
        
    if "L_VOLUME" in x:
        channel_rank = 6
    if "L2_VOLUME" in x:
        channel_rank = 7
        
        
    if "REG_CLOSE" in x:
        channel_rank = 8
    if "REG_VOLUME" in x:
        channel_rank = 9
        
    S_COINS = sorted(COINS)
    coin_rank = -1
    for i, c in enumerate(S_COINS):
        if x.endswith(c):
            coin_rank = i
            break
        
    lag_rank = 100
    try:
        lag_rank = int("".join([ch for ch in x if ch in '0123456789']))
        if pattern_match("L2?_(LOW|CLOSE|HIGH|VOLUME)", x):
            lag_rank *= -1
    except:
        pass
    if coin_rank < 0:
        continue
    X2.append( (coin_rank, lag_rank, channel_rank, x) )
    
N_LAGS      = 30
X2.sort(key = lambda x : (x[0], x[1], x[2]))

VOL_TENSOR  = [x[-1] for x in X2 if 6 <= x[2] <= 6]
VOL_TENSOR2 = [x[-1] for x in X2 if 7 <= x[2] <= 7]
PRICE_TENSOR  = [x[-1] for x in X2 if 0 <= x[2] <= 2]
PRICE_TENSOR2 = [x[-1] for x in X2 if 3 <= x[2] <= 5]
REG_TENSOR    = [x[-1] for x in X2 if 8 <= x[2] <= 9]


# Input / Output place holders
X = tf.placeholder(tf.float32, [None, N_IN])
X = tf.reshape(X, [-1, N_IN])
# PrevW
PREV_W = tf.placeholder(tf.float32, [None, N_OUT])
# Actual Rewards
Y_     = tf.placeholder(tf.float32, [None, N_OUT])
#--------------------------------------------------------------------------------------
# Define hidden layers
#--------------------------------------------------------------------------------------

h_1         = 1
w_1         = 3
CH_OUT_1    = 1
FILTER1     = [h_1, w_1, N_CHANNEL, CH_OUT_1] # Filter 1 x 3 x 3, Input has 4 channels

FILTERV     = [h_1, w_1, 1, CH_OUT_1]

h_2         = 1
w_2         = N_LAGS - w_1 + 1
CH_OUT_2    = 4
FILTER2     = [h_2, w_2, CH_OUT_1*1, CH_OUT_2]

h_3         = 1
w_3         = 1
CH_OUT_3    = 1
FILTER3     = [h_3, w_3, CH_OUT_2, CH_OUT_3]

SDEV        = 1
BIAS_MULT   = 0

X_PRICE_TENSOR  = tf.placeholder(tf.float32, [None, N_COINS-1, N_LAGS, N_CHANNEL])
X_PRICE_TENSOR2 = tf.placeholder(tf.float32, [None, N_COINS-1, N_LAGS, N_CHANNEL])

X_VOL_TENSOR  = tf.placeholder(tf.float32, [None, N_COINS-1, N_LAGS, 1])
X_VOL_TENSOR2 = tf.placeholder(tf.float32, [None, N_COINS-1, N_LAGS, 1])

#CW1Z = tf.Variable(tf.random_normal([N_COINS-1,2,N_CHANNEL,N_CHANNEL], stddev = SDEV))
#CB1Z = tf.Variable(tf.random_normal([N_CHANNEL], stddev = SDEV))
#CL1Z = tf.nn.relu(tf.nn.conv2d(X_PRICE_TENSOR, CW1Z, [1,1,1,1], padding="SAME") + CB1Z * BIAS_MULT)

CW1A = tf.Variable(tf.random_normal(FILTER1, stddev = SDEV))
CB1A = tf.Variable(tf.random_normal([CH_OUT_1], stddev = SDEV))
CL1A = tf.nn.relu(tf.nn.conv2d(X_PRICE_TENSOR, CW1A, [1,1,1,1], padding="VALID") + CB1A * BIAS_MULT)
'''
CW1B = tf.Variable(tf.random_normal(FILTER1, stddev = SDEV))
CB1B = tf.Variable(tf.random_normal([CH_OUT_1], stddev = SDEV))
CL1B = tf.nn.relu(tf.nn.conv2d(X_PRICE_TENSOR2, CW1B, [1,1,1,1], padding="VALID") + CB1B * BIAS_MULT)

CW1C = tf.Variable(tf.random_normal(FILTERV, stddev = SDEV))
CB1C = tf.Variable(tf.random_normal([CH_OUT_1], stddev = SDEV))
CL1C = tf.nn.relu(tf.nn.conv2d(X_VOL_TENSOR, CW1C, [1,1,1,1], padding="VALID") + CB1C * BIAS_MULT)

CW1D = tf.Variable(tf.random_normal(FILTERV, stddev = SDEV))
CB1D = tf.Variable(tf.random_normal([CH_OUT_1], stddev = SDEV))
CL1D = tf.nn.relu(tf.nn.conv2d(X_VOL_TENSOR2, CW1D, [1,1,1,1], padding="VALID") + CB1D * BIAS_MULT)'''

#CL2 = tf.concat([CL1A, CL1B, CL1C, CL1D], -1)
#CL2 = tf.concat([CL1A, CL1B, CL1C], -1)

#tf_mean = tf.Variable(0.0)
#tf_var  = tf.Variable(1.0)
#CL2 = tf.nn.batch_normalization(CL1A, tf_mean, tf_var, None, None, variance_epsilon=1e-6)

CL2 = CL1A
#CL2 = tf.concat([tf.nn.batch_normalization(CL1A, tf_mean1, tf_var1, None, None, 1e-6),\
#                 tf.nn.batch_normalization(CL1B, tf_mean2, tf_var2, None, None, 1e-6),
#                 tf.nn.batch_normalization(CL1C, tf_mean3, tf_var3, None, None, 1e-6)], -1)
# Shape is N_COINS x (LAGS - 2) x 6

CW3 = tf.Variable(tf.random_normal(FILTER2, stddev = SDEV))
CB3 = tf.Variable(tf.random_normal([CH_OUT_2], stddev = SDEV))
CL3 = tf.nn.relu(tf.nn.conv2d(CL2, CW3, [1,1,1,1], padding="VALID") + CB3 * BIAS_MULT)
#CL3 = tf.concat([CL3, tf.reshape(PREV_W[:,1:], (-1,N_COINS-1,1,1))], -1)
# Shape is N_COINS x 1 x 30

CW4 = tf.Variable(tf.random_normal(FILTER3, stddev = SDEV))
CL4A = tf.nn.relu(tf.nn.conv2d(CL3, CW4, [1,1,1,1], padding="SAME"))
CL4 = tf.reshape( CL4A, (-1, N_COINS-1) )

#tf_mean = tf.Variable(0.0)
#tf_var  = tf.Variable(1.0)
#CL4 = tf.nn.batch_normalization(CL4, tf_mean, tf_var, None, None, variance_epsilon=1e-6)

fc_w = tf.Variable(tf.random_normal([N_COINS-1, N_OUT], stddev = SDEV), trainable=True)
fc_b = tf.Variable(tf.random_normal([N_OUT], stddev = SDEV), trainable=True)

cnn_keep_prob = tf.Variable(0.85, trainable=False)
fc_w_dropped = tf.nn.dropout(fc_w, cnn_keep_prob)

Y_scores = tf.matmul(CL4, fc_w_dropped)# + fc_b * BIAS_MULT
Y = tf.nn.softmax(Y_scores)

cnn_lambda_reg = 0.00000001

#reg_losses = tf.nn.l2_loss(CW1A) + tf.nn.l2_loss(CW1B) + tf.nn.l2_loss(CW1C)# + tf.nn.l2_loss(CW1D)
#reg_losses += tf.nn.l2_loss(CW3) + tf.nn.l2_loss(CW4) + tf.nn.l2_loss(fc_w)

reg_losses = tf.nn.l2_loss(CW1A) + tf.nn.l2_loss(CW3) + tf.nn.l2_loss(CW4) + 4*tf.nn.l2_loss(fc_w)

#USD_BIAS = tf.Variable(tf.random_normal([1], stddev = SDEV))
#CL5 = tf.concat([USD_BIAS, CL4], axis=0)
#Y2  = tf.nn.softmax(CL4)
# Shape is N_COINS x 1 x 30
'''
# Define number of Neurons per layer
K = 100 # Layer 1
L = 100 # Layer 2
M = 100 # Layer 3
N = 100 # Layer 4

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

#reg_losses =  tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)
#reg_losses += tf.nn.l2_loss(B1) + tf.nn.l2_loss(B2) + tf.nn.l2_loss(B3)

# Magic number is around 0.0001
lambda_reg = 0.000001

#--------------------------------------------------------------------------------------
# Define Computation Graph
#--------------------------------------------------------------------------------------
# Feed forward. Output of previous is input to next
# Activation function for final layer is Softmax for portfolio weights in the range [0,1]
H1  = tf.nn.relu(tf.matmul(X,  W1) + B1)
DH1 = tf.nn.dropout(H1, 0.95)
H2  = tf.nn.relu(tf.matmul(DH1, W2) + B2)
DH2 = tf.nn.dropout(H2, 0.9)
bn_mean = tf.Variable(0.0)
bn_dev  = tf.Variable(1.0)
DH2 = tf.nn.batch_normalization(DH2, bn_mean, bn_dev, None, None, 1e-6)
H3  = tf.nn.relu(tf.matmul(DH2, W3) + B3)
DH3 = tf.nn.dropout(H3, 0.85)
#Y   = tf.nn.softmax(tf.matmul(DH3, W4) + B4)
#Y_MAX = tf.sign(Y - tf.reduce_max(Y,axis=1,keep_dims=True)) + 1'''
#--------------------------------------------------------------------------------------
# Define Loss Function
#--------------------------------------------------------------------------------------
if COMMISSION == 0:
    weight_moves = tf.reduce_mean(tf.reduce_sum(tf.abs(Y[1:] - Y[:-1]), axis=1))
    tensor_rwds = tf.log (10**tf.reduce_sum(Y * Y_, axis=1) )
    reward      = tf.reduce_sum(tensor_rwds)
    loss        = -tf.reduce_mean( tensor_rwds ) + cnn_lambda_reg * reg_losses
else:
    weight_moves = tf.reduce_mean(tf.reduce_sum(tf.abs(Y[1:] - Y[:-1]), axis=1))
    tensor_rwds = tf.log (tf.reduce_sum( ( 1-COMMISSION*tf.abs(Y-PREV_W) ) * (Y * 10**Y_), axis=1))
    reward      = tf.reduce_sum( tensor_rwds )
    loss        = -tf.reduce_mean( tensor_rwds )# + lambda_reg * reg_losses

tf.summary.scalar("loss",loss)
#tf.summary.scalar("tensor_rwds",[tensor_rwds])
summary_op = tf.summary.merge_all()

LR_START = 0.0001
LR_END   = 0.00005
LR_DECAY = 0.999

# Optimizer
LEARNING_RATE = tf.Variable(LR_START)
optimizer     = tf.train.AdamOptimizer(LEARNING_RATE)
train_step    = optimizer.minimize(loss)

test_imm   = data_imm.iloc[len(data_imm)-TEST_LEN:, :].reset_index(drop=True)
test_dat   = data.iloc[len(data)-TEST_LEN:, :].reset_index(drop=True)

feed_dat = {X:  np.reshape(test_dat[COLS_X], (-1,N_IN)), 
            Y_: np.reshape(test_dat[COLS_Y], (-1, N_OUT))}
                                   
feed_imm = {X:  np.reshape(test_imm[COLS_X], (-1,N_IN)), 
            Y_: np.reshape(test_imm[COLS_Y], (-1, N_OUT))}


# === Tensorboard - start-0 === #
# Create a summary to monitor cost tensor
tf.summary.scalar("loss",loss)
# Create a summary to monitor rewards tensor
tf.summary.scalar("reward",reward)
#tf.summary.scalar("tensor_rwds",[tensor_rwds])
# Merge all summaries into a single op
summary_op = tf.summary.merge_all()
# === Tensorboard - end-0 === #

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# === Tensorboard - start-1 === #
# op to write logs to Tensorboard
writer = tf.summary.FileWriter(LOG_PATH,graph=tf.get_default_graph()) #sess.graph ?

dat_rwds, imm_rwds, dat_losses, imm_losses = [], [], [], []
print("Begin Learning...")
#---------------------------------------------------------------------------------------------------
for epoch in range(100000):
    
    new_lr = max(LR_DECAY**epoch*LR_START, LR_END)
    
    # Measure loss on validation set every 100 epochs
    if epoch % 20 == 0:
        
        update_drop_rt = tf.assign(cnn_keep_prob, 1)
        sess.run(update_drop_rt)
        
        if COMMISSION != 0:
            prev_weights = [[1 if idx == 0 else 0 for idx in range(N_OUT)]]
            stime = time.time()
            test_dat.at[0,PORT_W] = prev_weights[-1]
            test_imm.at[0,PORT_W] = prev_weights[-1]
            b_x = np.reshape(test_dat[COLS_X], (-1,N_IN))
            b_y = np.reshape(test_dat[COLS_Y], (-1,N_OUT))
            pr_tens1 = np.reshape(np.array(test_dat[PRICE_TENSOR]), (-1, N_COINS-1, N_LAGS, N_CHANNEL))
            pr_tens2 = np.reshape(np.array(test_dat[PRICE_TENSOR2]),(-1, N_COINS-1, N_LAGS, N_CHANNEL))
            v_tens1 = np.reshape(np.array(test_dat[VOL_TENSOR]), (-1, N_COINS-1, N_LAGS, 1))
            v_tens2 = np.reshape(np.array(test_dat[VOL_TENSOR2]),(-1, N_COINS-1, N_LAGS, 1))
            #---------------------------------------------------------
            for r in range(len(test_dat) - 1):

                feed_row = {X: np.reshape(np.array(b_x.iloc[r,:]), (-1,N_IN)),
                            
                            Y_: np.reshape(np.array(b_y.iloc[r,:]), (-1,N_OUT)),
                            
                            PREV_W: np.reshape(prev_weights[-1], (-1, N_OUT)),
                            
                            X_PRICE_TENSOR  : pr_tens1[r].reshape(-1,N_COINS-1,N_LAGS,N_CHANNEL),
                            
                            X_PRICE_TENSOR2 : pr_tens2[r].reshape(-1,N_COINS-1,N_LAGS,N_CHANNEL),
                            
                            X_VOL_TENSOR  : v_tens1[r].reshape(-1,N_COINS-1,N_LAGS,1),
                            
                            X_VOL_TENSOR2 : v_tens2[r].reshape(-1,N_COINS-1,N_LAGS,1)
                            }
                                               
                weights, y_vec  = sess.run([Y, Y_], feed_dict=feed_row)
                y_vec_10 = (weights * 10 ** y_vec)
                w = y_vec_10 / np.sum(y_vec_10)
                prev_weights.append(w[0])
                
                b_x.at[r+1,PORT_W]      = w[0]
                test_dat.at[r+1,PORT_W] = w[0]
                test_imm.at[r+1,PORT_W] = w[0]
                #print(r / (time.time() - stime))
            #---------------------------------------------------------
                
            feed_dat = {X:  np.reshape(test_dat[COLS_X], (-1, N_IN)), 
                        Y_: np.reshape(test_dat[COLS_Y], (-1, N_OUT)),
                        PREV_W: np.reshape(prev_weights, (-1, N_OUT)),
                        
                        X_PRICE_TENSOR  : np.reshape(np.array(test_dat[PRICE_TENSOR]), \
                                        (-1, N_COINS-1, N_LAGS, N_CHANNEL)),
                            
                        X_PRICE_TENSOR2 : np.reshape(np.array(test_dat[PRICE_TENSOR2]), \
                                        (-1, N_COINS-1, N_LAGS, N_CHANNEL)),
                                                     
                        X_VOL_TENSOR  : np.reshape(np.array(test_dat[VOL_TENSOR]), \
                                        (-1, N_COINS-1, N_LAGS, 1)),
                            
                        X_VOL_TENSOR2 : np.reshape(np.array(test_dat[VOL_TENSOR2]), \
                                        (-1, N_COINS-1, N_LAGS, 1))
                        }
                                   
            feed_imm = {X:  np.reshape(test_imm[COLS_X], (-1, N_IN)), 
                        Y_: np.reshape(test_imm[COLS_Y], (-1, N_OUT)),
                        PREV_W: np.reshape(prev_weights, (-1, N_OUT)),
                        
                        X_PRICE_TENSOR  : np.reshape(np.array(test_imm[PRICE_TENSOR]), 
                                        (-1, N_COINS-1, N_LAGS, N_CHANNEL)),
                            
                        X_PRICE_TENSOR2 : np.reshape(np.array(test_imm[PRICE_TENSOR2]), 
                                        (-1, N_COINS-1, N_LAGS, N_CHANNEL)),
                                        
                        X_VOL_TENSOR  : np.reshape(np.array(test_imm[VOL_TENSOR]), \
                                        (-1, N_COINS-1, N_LAGS, 1)),
                            
                        X_VOL_TENSOR2 : np.reshape(np.array(test_imm[VOL_TENSOR2]), \
                                        (-1, N_COINS-1, N_LAGS, 1))
                        }
        else:
            feed_dat = {X:  np.reshape(test_dat[COLS_X], (-1, N_IN)), 
                        Y_: np.reshape(test_dat[COLS_Y], (-1, N_OUT)),
                        
                        X_PRICE_TENSOR  : np.reshape(np.array(test_dat[PRICE_TENSOR]), \
                                        (-1, N_COINS-1, N_LAGS, N_CHANNEL)),
                            
                        X_PRICE_TENSOR2 : np.reshape(np.array(test_dat[PRICE_TENSOR2]), \
                                        (-1, N_COINS-1, N_LAGS, N_CHANNEL)),
                                                     
                        X_VOL_TENSOR  : np.reshape(np.array(test_dat[VOL_TENSOR]), \
                                        (-1, N_COINS-1, N_LAGS, 1)),
                            
                        X_VOL_TENSOR2 : np.reshape(np.array(test_dat[VOL_TENSOR2]), \
                                        (-1, N_COINS-1, N_LAGS, 1))
                        }
                                   
            feed_imm = {X:  np.reshape(test_imm[COLS_X], (-1, N_IN)), 
                        Y_: np.reshape(test_imm[COLS_Y], (-1, N_OUT)),
                        
                        X_PRICE_TENSOR  : np.reshape(np.array(test_imm[PRICE_TENSOR]), 
                                        (-1, N_COINS-1, N_LAGS, N_CHANNEL)),
                            
                        X_PRICE_TENSOR2 : np.reshape(np.array(test_imm[PRICE_TENSOR2]), 
                                        (-1, N_COINS-1, N_LAGS, N_CHANNEL)),
                                                     
                        X_VOL_TENSOR  : np.reshape(np.array(test_imm[VOL_TENSOR]), \
                                        (-1, N_COINS-1, N_LAGS, 1)),
                            
                        X_VOL_TENSOR2 : np.reshape(np.array(test_imm[VOL_TENSOR2]), \
                                        (-1, N_COINS-1, N_LAGS, 1))
                        }
        d_rwd, d_loss = sess.run([reward, loss], feed_dict=feed_dat)
        i_rwd, i_loss = sess.run([reward, loss], feed_dict=feed_imm)
        
        dat_rwds.append(math.exp(d_rwd))
        imm_rwds.append(math.exp(i_rwd))
        dat_losses.append(d_loss)
        imm_losses.append(i_loss)
        print("Epoch {:<9} Loss: {:<12.6f} {:<12.6f} Reward: {:<12.6f} {:<12.6f}".format\
              (epoch, dat_losses[-1], dat_losses[-1], dat_rwds[-1], imm_rwds[-1]))

    #-----------------------------------------------------------------
    
    update_drop_rt = tf.assign(cnn_keep_prob, 0.99 + 0.01 * random.random())
    sess.run(update_drop_rt)
        
    idx      = int(round(random.random()**0.5*IDX_MAX))
    batch_sz = random.randint(BATCH_SZ_MIN, BATCH_SZ_MAX)
    sub_data = data.iloc[idx:idx+batch_sz, :].reset_index(drop=True)
    sub_data = data.iloc[:IDX_MAX,:]
    #sub_data = data.iloc[IDX_MAX:,:].reset_index(drop=True)
    #sub_data = sub_data.sample(batch_sz).reset_index(drop=True)
    #sub_data = test_dat
    batch_X, batch_Y = (sub_data[COLS_X], sub_data[COLS_Y])
    
    if COMMISSION != 0:
        prev_weights = []
        rand = np.random.random(N_OUT)
        rand /= rand.sum()
        prev_weights.append(rand)
        batch_X.at[0,PORT_W] = prev_weights[-1]
        b_x = np.reshape(batch_X, (-1,N_IN))
        b_y = np.reshape(batch_Y, (-1,N_OUT))
        pr_tens1 = np.reshape(np.array(b_x[PRICE_TENSOR]), (len(b_x), N_COINS-1, N_LAGS, N_CHANNEL))
        pr_tens2 = np.reshape(np.array(b_x[PRICE_TENSOR2]),(len(b_x), N_COINS-1, N_LAGS, N_CHANNEL))
        v_tens1 = np.reshape(np.array(b_x[VOL_TENSOR]), (-1, N_COINS-1, N_LAGS, 1))
        v_tens2 = np.reshape(np.array(b_x[VOL_TENSOR2]),(-1, N_COINS-1, N_LAGS, 1))
        for r in range(len(batch_X) - 1):
            if random.random() < 0.03:
                rand = np.random.random(N_OUT)
                rand /= rand.sum()
                prev_weights.append(rand)
                b_x.at[r+1,PORT_W] = rand
            else:
                feed_row = {X: np.reshape(np.array(b_x.iloc[r,:]), (-1,N_IN)),
                            
                            Y_: np.reshape(np.array(b_y.iloc[r,:]), (-1,N_OUT)),
                            
                            PREV_W: np.reshape(prev_weights[-1], (-1, N_OUT)),
                            
                            X_PRICE_TENSOR  : pr_tens1[r].reshape(-1,N_COINS-1,N_LAGS,N_CHANNEL),
                            
                            X_PRICE_TENSOR2 : pr_tens2[r].reshape(-1,N_COINS-1,N_LAGS,N_CHANNEL),
                            
                            X_VOL_TENSOR  : v_tens1[r].reshape(-1,N_COINS-1,N_LAGS,1),
                            
                            X_VOL_TENSOR2 : v_tens2[r].reshape(-1,N_COINS-1,N_LAGS,1)
                            }
                weights, y_vec  = sess.run([Y, Y_], feed_dict=feed_row)
                w = (weights * 10** y_vec) / np.sum(weights * 10 ** y_vec)
                prev_weights.append(w[0])
                b_x.at[r+1,PORT_W] = w[0]
                
        batch_X = b_x
        train_data = {X:  np.reshape(batch_X, (-1,N_IN)), 
                      Y_: np.reshape(batch_Y, (-1,N_OUT)),
                      PREV_W: np.reshape(prev_weights, (-1, N_OUT)),
                      
                      X_PRICE_TENSOR  : np.reshape(np.array(batch_X[PRICE_TENSOR]), \
                                        (-1, N_COINS-1, N_LAGS, N_CHANNEL)),
                            
                      X_PRICE_TENSOR2 : np.reshape(np.array(batch_X[PRICE_TENSOR2]), \
                                        (-1, N_COINS-1, N_LAGS, N_CHANNEL)),
                                                   
                      X_VOL_TENSOR  : np.reshape(np.array(batch_X[VOL_TENSOR]), \
                                        (-1, N_COINS-1, N_LAGS, 1)),
                            
                      X_VOL_TENSOR2 : np.reshape(np.array(batch_X[VOL_TENSOR2]), \
                                        (-1, N_COINS-1, N_LAGS, 1))
                                        
                      }
        
    else:
        train_data = {X:  np.reshape(batch_X, (-1,N_IN)), 
                      Y_: np.reshape(batch_Y, (-1,N_OUT)),
                      
                      X_PRICE_TENSOR  : np.reshape(np.array(batch_X[PRICE_TENSOR]), \
                                        (-1, N_COINS-1, N_LAGS, N_CHANNEL)),
                            
                      X_PRICE_TENSOR2 : np.reshape(np.array(batch_X[PRICE_TENSOR2]), \
                                        (-1, N_COINS-1, N_LAGS, N_CHANNEL)),
                                                   
                      X_VOL_TENSOR  : np.reshape(np.array(batch_X[VOL_TENSOR]), \
                                        (-1, N_COINS-1, N_LAGS, 1)),
                            
                      X_VOL_TENSOR2 : np.reshape(np.array(batch_X[VOL_TENSOR2]), \
                                        (-1, N_COINS-1, N_LAGS, 1))
                      }
        
    #_, summary = sess.run([train_step,summary_op], feed_dict=train_data)
    #a_rwd = sess.run(reward, feed_dict=train_data)
    step, r_wd, lss, reg_lss = sess.run([train_step,reward,loss,reg_losses], feed_dict=train_data)
    #a_rwd2 = sess.run(reward, feed_dict=train_data)
    #print("Epoch {:<12} Reward: {:<12.6f} ---> {:<12.6f}".format(epoch, a_rwd, a_rwd2))
    #writer.add_summary(summary,epoch)
    
    # === Tensorboard - start-2 === #
     # Run optimization / cost op (backprop / to get loss value) and summary nodes
    #_, summary = sess.run([train_step,summary_op], feed_dict=train_data)
    # Write logs at every iteration
    
    
    print("Epoch: {:<10} LR: {:<12.8f} Loss: {:<12.8f} Rwd: {:<12.8f}".format(epoch, reg_lss, lss, math.exp(r_wd)))
    #update_lr = tf.assign(LEARNING_RATE, new_lr)
    #sess.run(update_lr)
    
    if epoch == 0:
        update_lr = tf.assign(LEARNING_RATE, 0.01)
        sess.run(update_lr)
    
    if epoch == 100:
        update_lr = tf.assign(LEARNING_RATE, 0.003)
        sess.run(update_lr)
    
    if epoch == 200:
        update_lr = tf.assign(LEARNING_RATE, 0.001)
        sess.run(update_lr)
    
    if epoch == 300:
        update_lr = tf.assign(LEARNING_RATE, 0.0001)
        sess.run(update_lr)
    
    #update_lr = tf.assign(LEARNING_RATE, new_lr)
    #sess.run(update_lr)
    
    #print("Learning Rate: {}, reward: {}".format(new_lr, math.exp(r_wd)))
#---------------------------------------------------------------------------------------------------

plt.plot(dat_rwds)
plt.plot(imm_rwds)
plt.legend(['Discount Test Reward', 'Actual Test Reward'], loc=4)
plt.show()

update_drop_rt = tf.assign(cnn_keep_prob, 1)
sess.run(update_drop_rt)

if COMMISSION != 0:
    prev_weights = [[1 if idx == 0 else 0 for idx in range(N_OUT)]]
    stime = time.time()
    test_imm.at[:,PORT_W] = prev_weights * len(test_imm)
    b_x = np.reshape(test_imm[COLS_X], (-1,N_IN))
    b_y = np.reshape(test_imm[COLS_Y], (-1,N_OUT))
    pr_tens1 = np.reshape(np.array(b_x[PRICE_TENSOR]), (len(b_x), N_COINS-1, N_LAGS, N_CHANNEL))
    pr_tens2 = np.reshape(np.array(b_x[PRICE_TENSOR2]),(len(b_x), N_COINS-1, N_LAGS, N_CHANNEL))
    v_tens1 = np.reshape(np.array(b_x[VOL_TENSOR]), (-1, N_COINS-1, N_LAGS, 1))
    v_tens2 = np.reshape(np.array(b_x[VOL_TENSOR2]),(-1, N_COINS-1, N_LAGS, 1))
    for r in range(len(test_imm) - 1):
        if r % 1000 == 0:
            print("{:.2f}%".format(100.0 * r / len(test_imm)))
        feed_row = {X: np.reshape(np.array(b_x.iloc[r,:]), (-1,N_IN)),
                            
                    Y_: np.reshape(np.array(b_y.iloc[r,:]), (-1,N_OUT)),
                            
                    PREV_W: np.reshape(prev_weights[-1], (-1, N_OUT)),
                            
                    X_PRICE_TENSOR  : pr_tens1[r].reshape(-1,N_COINS-1,N_LAGS,N_CHANNEL),
                            
                    X_PRICE_TENSOR2 : pr_tens2[r].reshape(-1,N_COINS-1,N_LAGS,N_CHANNEL),

                    X_VOL_TENSOR  : v_tens1[r].reshape(-1,N_COINS-1,N_LAGS,1),
                            
                    X_VOL_TENSOR2 : v_tens2[r].reshape(-1,N_COINS-1,N_LAGS,1)        
                    }
        weights, y_vec  = sess.run([Y, Y_], feed_dict=feed_row)
        y_vec_10 = (weights * 10 ** y_vec)
        w = y_vec_10 / np.sum(y_vec_10)
        prev_weights.append(w[0])
        b_x.at[r+1,PORT_W]      = w[0]
        test_dat.at[r+1,PORT_W] = w[0]
        test_imm.at[r+1,PORT_W] = w[0]
        #print(r / (time.time() - stime))
                           
    feed_imm = {X:  np.reshape(test_imm[COLS_X], (-1, N_IN)), 
                Y_: np.reshape(test_imm[COLS_Y], (-1, N_OUT)),
                PREV_W: np.reshape(prev_weights, (-1, N_OUT)),
                        
                X_PRICE_TENSOR  : np.reshape(np.array(test_imm[PRICE_TENSOR]), 
                                  (-1, N_COINS-1, N_LAGS, N_CHANNEL)),
                            
                X_PRICE_TENSOR2 : np.reshape(np.array(test_imm[PRICE_TENSOR2]), 
                                  (-1, N_COINS-1, N_LAGS, N_CHANNEL)),

                X_VOL_TENSOR    : np.reshape(np.array(test_imm[VOL_TENSOR]), \
                                        (-1, N_COINS-1, N_LAGS, 1)),
                            
                X_VOL_TENSOR2   : np.reshape(np.array(test_imm[VOL_TENSOR2]), \
                                        (-1, N_COINS-1, N_LAGS, 1))
                }

    y1, y2, pw, f_rewards, f_loss = sess.run([Y, Y_, PREV_W, tensor_rwds, loss], 
                                         feed_dict = feed_imm)
else:
    
    feed_imm = {X:  np.reshape(test_imm[COLS_X], (-1, N_IN)), 
                Y_: np.reshape(test_imm[COLS_Y], (-1, N_OUT)),
                        
                X_PRICE_TENSOR  : np.reshape(np.array(test_imm[PRICE_TENSOR]), 
                                  (-1, N_COINS-1, N_LAGS, N_CHANNEL)),
                            
                X_PRICE_TENSOR2 : np.reshape(np.array(test_imm[PRICE_TENSOR2]), 
                                  (-1, N_COINS-1, N_LAGS, N_CHANNEL)),

                X_VOL_TENSOR    : np.reshape(np.array(test_imm[VOL_TENSOR]), \
                                        (-1, N_COINS-1, N_LAGS, 1)),
                            
                X_VOL_TENSOR2   : np.reshape(np.array(test_imm[VOL_TENSOR2]), \
                                        (-1, N_COINS-1, N_LAGS, 1))
                }
    y1, y2, f_rewards, f_loss = sess.run([Y, Y_, tensor_rwds, loss], 
                                         feed_dict = feed_imm)


long_short = [[] for _ in range(3 if ALLOW_SHORTS else 2)]
props = [list(y1[:,i]) for i in range(len(y1[0]))]

for val in y1:
    val = list(val)
    long_short[0].append(val[0])
    if ALLOW_SHORTS:
        long_short[1].append(sum(val[1:N_COINS//2+1]))
        long_short[2].append(sum(val[N_COINS//2+1:]))
    else:
        long_short[1].append(sum(val[1:]))

rolling_window = 100
for i in range(len(long_short)):
    dat = pd.rolling_mean(pd.Series(long_short[i]),rolling_window)
    plt.plot(dat)
plt.legend(['USD', 'Long', 'Short'] if ALLOW_SHORTS else ['USD', 'Long'], loc=4)
plt.show()

for i in range(len(props)):
    dat = pd.rolling_mean(pd.Series(props[i]),rolling_window)
    plt.plot(dat)
plt.legend([x[x.index("_")+1:] for x in COLS_Y], loc=4)
plt.show()

result = pd.concat(
        
        [pd.DataFrame(y2, columns=COLS_Y),
         pd.DataFrame(y1, columns=["{}_%".format(x.replace("reward_","")) for x in COLS_Y])
         ]
        ,axis=1
        )

def showArray(arr, decimals = 3):
    return "[" + " ".join(["{1:.{0}f}".format(decimals, x) for x in arr]) + "]"

w = list(y1)

rewards     = []    # All Rewards     (Multiplicative)
log_rewards = []    # All Log Rewards (Additive)
prevW       = [1] + [0] * (N_OUT - 1) # Weights from previous period

STEP = 1

print("Iteration, PrevW, Action, PriceChange, NewW, Reward")
#------------------------------------------------------------------------------
for i in range(len(y2)):
    
    c = 0.0025
    
    for j in range(len(w[i])):
        w[i][j] = max(w[i][j],0)
        
    if i % STEP == 0:
        cw = [x for x in w[i]]
        
    rw     = 0    # Reward for this time step

    # Iterate through each asset and add each reward to the net reward
    #----------------------------------------------------------------
    for asset in range(len(cw)):

        # Transaction Cost
        tc       = (1 - c * abs((cw[asset] - prevW[asset])**1))
        if i % STEP != 0:
            tc = 1
        mult     = (10**y2[i][asset] - 1) + 1
            
        rw_asset = tc * (cw[asset]) * mult 
        rw      += rw_asset
    #----------------------------------------------------------------

    # Calculate what new weights will be after price move
    newW = [cw[A] * 10**y2[i][A] for A in range(len(cw))]
    newW = [x/sum(newW) for x in newW]
    
    #print(i, showArray(prevW), "-->", showArray(w[i]), "*", showArray(y[i]), "=", showArray(newW), " {{ {}{:.3f}% }}".format("+" if rw >= 1 else "", 100*rw-100))
    
    prevW = newW
    cw    = newW
    rewards.append(rw)
    log_rewards.append(math.log10(rw))
    
plt.plot(pd.Series(log_rewards).cumsum())
plt.plot(pd.Series(f_rewards).apply(lambda x : math.log10(math.exp(x))).cumsum())
plt.plot(pd.Series(test_imm.close_BTC / test_imm.close_BTC[0]).apply(lambda x : math.log10(x)))
#plt.plot(prof)
plt.show()

plt.plot(pd.Series(test_imm.close_BTC / test_imm.close_BTC[0]).apply(lambda x : math.log10(x)),
         pd.Series(log_rewards).cumsum(), 'ob')
plt.show()
