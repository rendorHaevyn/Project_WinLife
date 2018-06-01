from __future__ import print_function
import numpy as np
import random
import pandas as pd
import sklearn
import sklearn.decomposition
import sklearn.ensemble
import sklearn.preprocessing
import math
import itertools
import threading
from matplotlib import pyplot as plt
import tensorflow as tf
import re
import time

# Utility Function to return True / False regex matching
def pattern_match(patt, string):
    return re.findall(patt, string) != []

#--------------------------------------------------------------------------------------
# Read in the price data
#--------------------------------------------------------------------------------------
print("Loading Data...", end="")
data_raw = pd.read_csv("M15/ALL.csv").dropna(axis=0, how='any').reset_index(drop=True)
data     = data_raw[data_raw['date'] > 1514466000]
data     = data_raw.drop('date', axis=1)
data['reward_USD'] = 0
print("{} rows & {} columns".format(len(data), len(data.columns)))
#--------------------------------------------------------------------------------------
# Manual Options
#--------------------------------------------------------------------------------------
COMMISSION     = 0.0025  # Commision % as a decimal to use in loss function
USE_PCA        = True   # Use PCA Dimensionality Reduction
PCA_COMPONENTS = 400     # Number of Principle Components to reduce down to
USE_SUPER      = False   # Create new features using supervised learning
INCLUDE_VOLUME = True    # Include Volume as a feature
ALLOW_SHORTS   = True   # Allow Shorts or not
DISCOUNT       = True   # Train on discounted rewards
DISCOUNT_STEPS = 24      # Number of periods to look ahead for discounting
GAMMA          = 0.66    # The discount factor
#--------------------------------------------------------------------------------------
# List of coins to trade. Set to [] to use all coins
#--------------------------------------------------------------------------------------
COINS       = ['USD', 'BCH', 'XRP', 'XMR', 'LTC']
# List of coins data to use as input variables. Set to [] to use all coins
#--------------------------------------------------------------------------------------
INPUT_COINS = []
N_VEC       = 3 + 1 if INCLUDE_VOLUME else 0
N_COINS     = ( len(COINS) * 2 - 1 ) if ALLOW_SHORTS else len(COINS)
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
    if "L_" in x or "REG" in x:
        if "VOLUME" in x and INCLUDE_VOLUME == False:
            continue
        COLS_X.append(x)
if COMMISSION != 0:
    COLS_X += PORT_W
#--------------------------------------------------------------------------------------
# Create a list of Y column names to use for modelling
#--------------------------------------------------------------------------------------             
out_cols = []
for c in data.columns:
    if COINS == []:
        out_cols.append(c)
    else:
        for a in set(COINS):
            if a in c:
                out_cols.append(c)
            
if ALLOW_SHORTS:
    short_cols = []
    for c in out_cols:
        if 'reward' in c:
            data[c+"_S"] = data[c].apply(lambda x : -x)
            short_cols.append(c+"_S")
    out_cols += short_cols 

if ALLOW_SHORTS:
    COLS_Y = [x.replace("MARGIN", "reward") for x in PORT_W]
else:
    COLS_Y = ["reward_USD"] + sorted([y for y in out_cols if 'reward' in y and "USD" not in y])
#--------------------------------------------------------------------------------------
# Defining the batch size and test length
#--------------------------------------------------------------------------------------
BATCH_SZ_MIN = 25
BATCH_SZ_MAX = 50
TEST_LEN     = int(round(0.2*len(data)))
IDX_MAX      = int(max(0, len(data) - TEST_LEN - BATCH_SZ_MAX - 1))
#--------------------------------------------------------------------------------------
# Normalizing the X columns. Scale using training data only
#--------------------------------------------------------------------------------------
print("Normalizing Data...", end="")
for x in COLS_X:
    median      = data[x].describe()[5]
    data[x]     = data[x].apply(lambda x : median if np.isinf(x) or np.isnan(x) else x)
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit( data[:IDX_MAX+BATCH_SZ_MAX] [COLS_X] )
data[COLS_X] = scaler.transform(data[COLS_X])
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
    
'''for c in COLS_Y:
    data[c] = data[c] - math.log10(1.004)
data["reward_USD"] = 0'''
    
N_IN  = len(COLS_X)
N_OUT = len(COLS_Y)

#--------------------------------------------------------------------------------------
#  
#                                NEURAL NETWORK DESIGN
#
#--------------------------------------------------------------------------------------

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
# Define number of Neurons per layer
K = 100 # Layer 1
L = 100 # Layer 2
M = 100 # Layer 3
N = 100 # Layer 4

SDEV = 0.1

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

#--------------------------------------------------------------------------------------
# Define Computation Graph
#--------------------------------------------------------------------------------------
# Feed forward. Output of previous is input to next
# Activation function for final layer is Softmax for portfolio weights in the range [0,1]
H1 = tf.nn.relu(tf.matmul(X,  W1) + B1)
H2 = tf.nn.relu(tf.matmul(H1, W2) + B2)
H3 = tf.nn.relu(tf.matmul(H2, W3) + B3)
Y  = tf.nn.log_softmax(tf.matmul(H3, W4) + B4)
Y_MAX = tf.sign(Y - tf.reduce_max(Y,axis=1,keep_dims=True)) + 1
#--------------------------------------------------------------------------------------
# Define Loss Function
#--------------------------------------------------------------------------------------
if COMMISSION == 0:
    tensor_rwds = tf.log (10**tf.reduce_sum(Y * Y_, axis=1) )
    loss        = -tf.reduce_mean( tensor_rwds )
else:
    tensor_rwds = tf.log (tf.reduce_sum( ( 1-COMMISSION*tf.abs(Y-PREV_W) ) * (Y * 10**Y_), axis=1))
    loss        = -tf.reduce_mean( tensor_rwds )

# Optimizer
LEARNING_RATE 	= 0.0002
optimizer 		= tf.train.AdamOptimizer(LEARNING_RATE)
train_step 	= optimizer.minimize(loss)

test_imm   = data_imm.iloc[len(data_imm)-TEST_LEN:, :].reset_index(drop=True)
test_dat   = data.iloc[len(data)-TEST_LEN:, :].reset_index(drop=True)

feed_dat = {X:  np.reshape(test_dat[COLS_X], (-1,N_IN)), 
            Y_: np.reshape(test_dat[COLS_Y], (-1, N_OUT))}
                                   
feed_imm = {X:  np.reshape(test_imm[COLS_X], (-1,N_IN)), 
            Y_: np.reshape(test_imm[COLS_Y], (-1, N_OUT))}

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

dat_rwds, imm_rwds = [], []
def eval_nn(lst, feed, len_test=1):
    rwd = sess.run(loss, feed_dict=feed)
    rwd = 100 * math.exp(-rwd * len_test) - 100
    lst.append(rwd)

print("Begin Learning...")
#---------------------------------------------------------------------------------------------------
for epoch in range(10000000):
    
    # Measure loss on validation set every 100 epochs
    if epoch % 100 == 0 or epoch < 10:
        
        if COMMISSION != 0:
            prev_weights = [[1 if idx == 0 else 0 for idx in range(N_OUT)]]
            stime = time.time()
            test_dat.at[0,PORT_W] = prev_weights[-1]
            test_imm.at[0,PORT_W] = prev_weights[-1]
            b_x = np.reshape(test_dat[COLS_X], (-1,N_IN))
            b_y = np.reshape(test_dat[COLS_Y], (-1,N_OUT))
            #---------------------------------------------------------
            for r in range(len(test_dat) - 1):
                feed_row = {X:  np.reshape(np.array(b_x.iloc[r,:]), (-1,N_IN)),
                            Y_: np.reshape(np.array(b_y.iloc[r,:]), (-1,N_OUT)),
                            PREV_W: np.reshape(prev_weights[-1], (-1, N_OUT))}
                                               
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
                        PREV_W: np.reshape(prev_weights, (-1, N_OUT))}
                                   
            feed_imm = {X:  np.reshape(test_imm[COLS_X], (-1, N_IN)), 
                        Y_: np.reshape(test_imm[COLS_Y], (-1, N_OUT)),
                        PREV_W: np.reshape(prev_weights, (-1, N_OUT))}
                
        threading.Thread(target=eval_nn,args=(dat_rwds,feed_dat,len(test_dat))).start()
        threading.Thread(target=eval_nn,args=(imm_rwds,feed_imm,len(test_imm))).start()
        if dat_rwds and imm_rwds:
            print("{:<16} {:<16.6f} {:<16.6f}%".format(epoch, dat_rwds[-1], imm_rwds[-1]))

    #-----------------------------------------------------------------
        
    idx      = int(round(random.random()**0.8*IDX_MAX))
    batch_sz = random.randint(BATCH_SZ_MIN, BATCH_SZ_MAX)
    sub_data = data.iloc[idx:idx+batch_sz, :].reset_index(drop=True)
    batch_X, batch_Y = (sub_data[COLS_X], sub_data[COLS_Y])
    
    if COMMISSION != 0:
        prev_weights = []
        rand = np.random.random(N_OUT)
        rand /= rand.sum()
        prev_weights.append(list(rand))
        batch_X.at[0,PORT_W] = prev_weights
        b_x = np.reshape(batch_X, (-1,N_IN))
        b_y = np.reshape(batch_Y, (-1,N_OUT))
        for r in range(len(batch_X) - 1):
            feed_row = {X: np.reshape(np.array(b_x.iloc[r,:]), (-1,N_IN)),
                        Y_: np.reshape(np.array(b_y.iloc[r,:]), (-1,N_OUT)),
                        PREV_W: np.reshape(prev_weights[-1], (-1, N_OUT))}
            weights, y_vec  = sess.run([Y, Y_], feed_dict=feed_row)
            w = (weights * 10** y_vec) / np.sum(weights * 10 ** y_vec)
            prev_weights.append(w[0])
            b_x.at[r+1,PORT_W] = w[0]
                
        batch_X = b_x
        train_data = {X:  np.reshape(batch_X, (-1,N_IN)), 
                      Y_: np.reshape(batch_Y, (-1,N_OUT)),
                      PREV_W: np.reshape(prev_weights, (-1, N_OUT))}
        
    else:
        train_data = {X:  np.reshape(batch_X, (-1,N_IN)), 
                      Y_: np.reshape(batch_Y, (-1,N_OUT))}
        
    sess.run(train_step, feed_dict=train_data)
#---------------------------------------------------------------------------------------------------

plt.plot(dat_rwds)
plt.plot(imm_rwds)
plt.legend(['Discount Test Reward', 'Actual Test Reward'], loc=4)
plt.show()

if COMMISSION != 0:
    prev_weights = [[1 if idx == 0 else 0 for idx in range(N_OUT)]]
    stime = time.time()
    test_imm.at[:,PORT_W] = prev_weights * len(test_imm)
    b_x = np.reshape(test_imm[COLS_X], (-1,N_IN))
    b_y = np.reshape(test_imm[COLS_Y], (-1,N_OUT))
    for r in range(len(test_imm) - 1):
        feed_row = {X:  np.reshape(b_x.iloc[r,:], (-1,N_IN)),
                    Y_: np.reshape(b_y.iloc[r,:], (-1,N_OUT)),
                    PREV_W: np.reshape(prev_weights[-1], (-1, N_OUT))}
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
                PREV_W: np.reshape(prev_weights, (-1, N_OUT))}

    y1, y2, pw, f_rewards, f_loss = sess.run([Y, Y_, PREV_W, tensor_rwds, loss], 
                                         feed_dict = feed_imm)
else:
    
    y1, y2, f_rewards, f_loss = sess.run([Y, Y_, tensor_rwds, loss], 
                                         feed_dict = feed_imm)
y3 = y1 * y2
prof = [0]
for x in y3:
    prof.append(prof[-1]+sum(x))
    
plt.plot(prof)
plt.legend(['Actual Reward'], loc=4)
plt.show()

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
prevW       = pw[0] # Weights from previous period

STEP = 1

print("Iteration, PrevW, Action, PriceChange, NewW, Reward")
#------------------------------------------------------------------------------
for i in range(len(y2)):
    
    c = 0.002
    
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
