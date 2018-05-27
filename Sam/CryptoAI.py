import numpy as np
import random
import pandas as pd
import sklearn
import sklearn.decomposition
import math
import itertools
import threading
from matplotlib import pyplot as plt
import tensorflow as tf
import re
import time

def pattern_match(patt, string):
    return re.findall(patt, string) != []

print("Loading Data...", end="")
data_raw = pd.read_csv("M15/ALL.csv").dropna(axis=0, how='any').reset_index(drop=True)
data     = data_raw[data_raw['date'] > 1514466000]
data     = data_raw.drop('date', axis=1)
print("{} rows & {} columns".format(len(data), len(data.columns)))

COMMISSION     = 0.004
USE_PCA        = True
INCLUDE_VOLUME = True
ALLOW_SHORTS   = False
DISCOUNT       = True
DISCOUNT_STEPS = 12
GAMMA          = 0.8

ASSETS      = ['USD', 'BCH', 'XRP', 'XMR', 'LTC']
INPUT_ASSET = ['BTC', 'BCH', 'XRP', 'XMR', 'LTC']
N_VEC       = 3 + 1 if INCLUDE_VOLUME else 0
N_ASSETS    = ( len(ASSETS) * 2 - 1 ) if ALLOW_SHORTS else len(ASSETS)

MARGIN_VEC = []
if COMMISSION > 0:
    MARGIN_VEC.append('MARGIN_USD')
    data["MARGIN_USD"] = 1
    for i, a in enumerate(sorted(ASSETS)):
        data["MARGIN_{}".format(a)] = 1 if a == "USD" else 0
        if "MARGIN_{}".format(a) not in MARGIN_VEC:
            MARGIN_VEC.append("MARGIN_{}".format(a))
    if ALLOW_SHORTS:
        for i, a in enumerate(sorted(ASSETS)):
            if a == "USD":
                continue
            data["MARGIN_{}_S".format(a)] = 0

if ALLOW_SHORTS:
    x = list(MARGIN_VEC)
    for asset in x[1:]:
        MARGIN_VEC.append(asset+"_S")

stmt  = "data['market_lag'] = (0"
n_rws = 0
cols2 = []
for c in data.columns:
    if not INPUT_ASSET:
        cols2.append(c)
        if pattern_match("L_CLOSE_1_.*",c):
            stmt += "+data['{}']".format(c)
            n_rws += 1
    else:
        for a in set(INPUT_ASSET):
            if a in c:
                cols2.append(c)
                if pattern_match("L_CLOSE_1_.*",c):
                    stmt += "+data['{}']".format(c)
                    n_rws += 1
                    
stmt += ")/{}".format(n_rws)
exec(stmt)

cols = []
for c in data.columns:
    if not ASSETS:
        cols.append(c)
    else:
        for a in set(ASSETS):
            if a in c:
                cols.append(c)
            
if ALLOW_SHORTS:
    short_cols = []
    for c in cols:
        if 'reward' in c:
            data[c+"_S"] = data[c].apply(lambda x : -x)
            short_cols.append(c+"_S")
    cols += short_cols 
    
'''        
all_reward_cols = [c for c in data.columns if pattern_match("reward_.*", c)]
usd_rewards = []
for i in range(len(data)):
    row = data.iloc[i,:]
    avg_mkt = row[all_reward_cols].describe()[1]
    if avg_mkt < 0:
        usd_rewards.append(-avg_mkt*0.5)
    else:
        usd_rewards.append(0)'''
        
data['reward_USD'] = 0

if INCLUDE_VOLUME:
    COLS_X = [x for x in cols2 if 'L_' in x or x == 'market_lag']
    if COMMISSION != 0:
        COLS_X = COLS_X + MARGIN_VEC
else:
    COLS_X = [x for x in cols2 if ('L_' in x and "VOLUME" not in x) in x or x == 'market_lag']
    if COMMISSION != 0:
        COLS_X = COLS_X + MARGIN_VEC

if ALLOW_SHORTS:
    COLS_Y = [x.replace("MARGIN", "reward") for x in MARGIN_VEC]
else:
    COLS_Y = ["reward_USD"] + sorted([y for y in cols if 'reward' in y and "USD" not in y])

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

#if not DISCOUNT or True:
#    for c in COLS_Y:
#        data[c] = data[c] - math.log10(1.001)
#    data["reward_USD"] = 0

print("Normalizing Data...", end="")
for x in COLS_X:
    
    data[x] = data[x].apply(lambda x : 0 if np.isinf(x) else x)
    data[x] = (data[x] - data[x].describe()[1])/(data[x].describe()[2]+1e-10)
    
    data_imm[x] = data_imm[x].apply(lambda x : 0 if np.isinf(x) else x)
    data_imm[x] = (data_imm[x] - data_imm[x].describe()[1])/(data_imm[x].describe()[2]+1e-10)
print("Done")

if USE_PCA:
    PCA_MODEL = sklearn.decomposition.PCA(15)
    PCA_MODEL.fit(data[COLS_X])
    Xs = pd.DataFrame(PCA_MODEL.transform(data[COLS_X]))
    Xs.columns = ["PCA_"+str(x) for x in range(1,len(Xs.columns)+1)]
    data[Xs.columns] = Xs
    data_imm[Xs.columns] = Xs
    COLS_X = list(Xs.columns) + MARGIN_VEC
    print(PCA_MODEL.explained_variance_)
    print(PCA_MODEL.explained_variance_ratio_)
    print(PCA_MODEL.explained_variance_ratio_.cumsum())

N_IN  = len(COLS_X)
N_OUT = len(COLS_Y)

# Define number of Neurons per layer
K = 300 # Layer 1
L = 300 # Layer 2
M = 300 # Layer 3
N = 300 # Layer 4
O = 300  # Layer 5

SDEV = 0.1

# LAYER 1
# Initialize weights, normal dist.
W1 = tf.Variable(tf.random_normal([N_IN, K], stddev = SDEV))
# Bias terms initialized to zero
B1 = tf.Variable(tf.random_normal([K]))

# LAYER 2
W2 = tf.Variable(tf.random_normal([K, L], stddev = SDEV))
B2 = tf.Variable(tf.random_normal([L]))

# LAYER 3
W3 = tf.Variable(tf.random_normal([L, M], stddev = SDEV))
B3 = tf.Variable(tf.random_normal([M]))

# LAYER 4
W4 = tf.Variable(tf.random_normal([M, N_OUT], stddev = SDEV))
B4 = tf.Variable(tf.random_normal([N_OUT]))

# LAYER 5
W5 = tf.Variable(tf.random_normal([N, N_OUT], stddev = SDEV))
B5 = tf.Variable(tf.random_normal([N_OUT]))

# LAYER 6
#W6 = tf.Variable(tf.random_normal([O, N_OUT], stddev = SDEV))
#B6 = tf.Variable(tf.random_normal([N_OUT]))

X = tf.placeholder(tf.float32, [None, N_IN])
X = tf.reshape(X, [-1, N_IN])

# Feed forward. Output of previous is input to next
# Activation function for final layer is Softmax for probabilties in the range [0,1]
Y1 = tf.nn.relu(tf.matmul(X,  W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y = tf.nn.softmax(tf.matmul(Y3, W4) + B4)
#Y5 = tf.nn.leaky_relu(tf.matmul(Y4, W5) + B5)
#Y =  tf.nn.softmax(tf.matmul(Y4, W5) + B5)
#Y  = (tf.matmul(Y4, W5) + B5)

# PrevW
PREV_W = tf.placeholder(tf.float32, [None, N_OUT])
Y_     = tf.placeholder(tf.float32, [None, N_OUT])

if COMMISSION == 0:
    loss = -tf.reduce_mean( tf.log(10**tf.reduce_sum (Y * Y_, axis=1) ) )
    tf_rewards = ( tf.log(tf.reduce_sum(Y * Y_, axis=1) ) )
else:
    #loss = -tf.reduce_mean( 10000000 *   (  (1-COMMISSION*(Y-PREV_W)**2) * (Y * 10**Y_)  ) )
    loss = -tf.reduce_mean  (   tf.log (
                                          tf.reduce_sum( ( 1-COMMISSION*(Y-PREV_W)**2 ) * (Y * 10**Y_), axis=1)
                                       )
                            )
    tf_rewards = (  tf.log (
                              tf.reduce_sum( ( 1-COMMISSION*tf.abs(Y-PREV_W) ) * (Y * 10**Y_), axis=1)
                           )
                 )

# Optimizer
LEARNING_RATE 	= 0.000005
optimizer 		= tf.train.AdamOptimizer(LEARNING_RATE)
train_step 		= optimizer.minimize(loss)

BATCH_SZ_MIN = 50#round(0.05*len(data))
BATCH_SZ_MAX = 50#round(0.2*len(data))
TEST_LEN     = round(0.2*len(data))
IDX_MAX      = len(data) - TEST_LEN - BATCH_SZ_MAX - 1

test_imm   = data_imm.iloc[len(data_imm)-TEST_LEN:, :].reset_index(drop=True)
test_dat   = data.iloc[len(data)-TEST_LEN:, :].reset_index(drop=True)

feed_dat = {X: np.reshape(test_dat[COLS_X], (-1,N_IN)), 
            Y_: np.reshape(test_dat[COLS_Y], (-1, N_OUT))}
                                   
feed_imm = {X: np.reshape(test_imm[COLS_X], (-1,N_IN)), 
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
for i in range(10000000):
    
    if i % 200 == 0 and i != 0:
        
        if COMMISSION != 0:
            prev_weights = [[1 if idx == 0 else 0 for idx in range(N_OUT)]]
            stime = time.time()
            test_dat.at[:,MARGIN_VEC] = prev_weights * len(test_dat)
            test_imm.at[:,MARGIN_VEC] = prev_weights * len(test_imm)
            b_x = np.reshape(test_dat[COLS_X], (-1,N_IN))
            b_y = np.reshape(test_dat[COLS_Y], (-1,N_OUT))
            for r in range(len(test_dat) - 1):
                feed_row = {X:  np.reshape(b_x.iloc[r,:], (-1,N_IN)),
                            Y_: np.reshape(b_y.iloc[r,:], (-1,N_OUT)),
                            PREV_W: np.reshape(prev_weights[-1], (-1, N_OUT))}
                weights, y_vec  = sess.run([Y, Y_], feed_dict=feed_row)
                w = (weights * 10** y_vec) / np.sum(weights * 10 ** y_vec)
                prev_weights.append(w[0])
                for index, a in enumerate(MARGIN_VEC):
                    b_x.set_value(r+1, a, w[0][index])
                    test_dat.set_value(r+1, a, w[0][index])
                    test_imm.set_value(r+1, a, w[0][index])
                #print(r / (time.time() - stime))
                
            feed_dat = {X:  np.reshape(test_dat[COLS_X], (-1, N_IN)), 
                        Y_: np.reshape(test_dat[COLS_Y], (-1, N_OUT)),
                        PREV_W: np.reshape(prev_weights, (-1, N_OUT))}
                                   
            feed_imm = {X:  np.reshape(test_imm[COLS_X], (-1, N_IN)), 
                        Y_: np.reshape(test_imm[COLS_Y], (-1, N_OUT)),
                        PREV_W: np.reshape(prev_weights, (-1, N_OUT))}
                
        threading.Thread(target=eval_nn,args=(dat_rwds,feed_dat,len(test_dat))).start()
        threading.Thread(target=eval_nn,args=(imm_rwds,feed_imm,len(test_imm))).start()
        while dat_rwds == [] or imm_rwds == []:
            pass
        print("{:<16} {:<16.6f} {:<16.6f}%".format(i, dat_rwds[-1], imm_rwds[-1]))
        
    else:
        
        idx      = round(random.random()**0.75*IDX_MAX)
        batch_sz = random.randint(BATCH_SZ_MIN, BATCH_SZ_MAX)
        sub_data = data.iloc[idx:idx+batch_sz, :].reset_index(drop=True)
        #sub_data = data.iloc[:IDX_MAX,:].sample(batch_sz)
        batch_X, batch_Y = (sub_data[COLS_X], sub_data[COLS_Y])
        
        if COMMISSION != 0:
            prev_weights = [[1 if idx == 0 else 0 for idx in range(N_OUT)]]
            batch_X.at[:,MARGIN_VEC] = prev_weights * len(batch_X)
            b_x = np.reshape(batch_X, (-1,N_IN))
            b_y = np.reshape(batch_Y, (-1,N_OUT))
            for r in range(len(batch_X) - 1):
                feed_row = {X: np.reshape(b_x.iloc[r,:], (-1,N_IN)),
                            Y_: np.reshape(b_y.iloc[r,:], (-1,N_OUT)),
                            PREV_W: np.reshape(prev_weights[-1], (-1, N_OUT))}
                weights, y_vec  = sess.run([Y, Y_], feed_dict=feed_row)
                w = (weights * 10** y_vec) / np.sum(weights * 10 ** y_vec)
                prev_weights.append(w[0])
                for index, a in enumerate(MARGIN_VEC):
                    b_x.set_value(r+1, a, w[0][index])
                    
            batch_X = b_x
            train_data = {X:  np.reshape(batch_X, (-1,N_IN)), 
                          Y_: np.reshape(batch_Y, (-1,N_OUT)),
                          PREV_W: np.reshape(prev_weights, (-1, N_OUT))}
            
            sess.run(train_step, feed_dict=train_data)
            
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
    test_dat.at[:,MARGIN_VEC] = prev_weights * len(test_dat)
    test_imm.at[:,MARGIN_VEC] = prev_weights * len(test_imm)
    b_x = np.reshape(test_dat[COLS_X], (-1,N_IN))
    b_y = np.reshape(test_dat[COLS_Y], (-1,N_OUT))
    for r in range(len(test_dat) - 1):
        feed_row = {X:  np.reshape(b_x.iloc[r,:], (-1,N_IN)),
                    Y_: np.reshape(b_y.iloc[r,:], (-1,N_OUT)),
                    PREV_W: np.reshape(prev_weights[-1], (-1, N_OUT))}
        weights, y_vec  = sess.run([Y, Y_], feed_dict=feed_row)
        w = (weights * 10** y_vec) / np.sum(weights * 10 ** y_vec)
        prev_weights.append(w[0])
        for index, a in enumerate(MARGIN_VEC):
            b_x.set_value(r+1, a, w[0][index])
            test_dat.set_value(r+1, a, w[0][index])
            test_imm.set_value(r+1, a, w[0][index])
        #print(r / (time.time() - stime))
        
    feed_dat = {X:  np.reshape(test_dat[COLS_X], (-1, N_IN)), 
                Y_: np.reshape(test_dat[COLS_Y], (-1, N_OUT)),
                PREV_W: np.reshape(prev_weights, (-1, N_OUT))}
                           
    feed_imm = {X:  np.reshape(test_imm[COLS_X], (-1, N_IN)), 
                Y_: np.reshape(test_imm[COLS_Y], (-1, N_OUT)),
                PREV_W: np.reshape(prev_weights, (-1, N_OUT))}

    y1, y2, pw, f_rewards, f_loss = sess.run([Y, Y_, PREV_W, tf_rewards, loss], 
                                         feed_dict = feed_imm)
else:
    
    y1, y2, f_rewards, f_loss = sess.run([Y, Y_, tf_rewards, loss], 
                                         feed_dict = feed_imm)
y3 = y1 * y2
prof = [0]
for x in y3:
    prof.append(prof[-1]+sum(x))
    
prof2 = prof
    
prof = [0]
for x in f_rewards:
    prof.append(prof[-1]+math.log(math.exp(x)))
    
plt.plot(prof)
plt.legend(['Actual Reward'], loc=4)
plt.show()

long_short = [[] for _ in range(3 if ALLOW_SHORTS else 2)]
props = [list(y1[:,i]) for i in range(len(y1[0]))]

for val in y1:
    val = list(val)
    long_short[0].append(val[0])
    if ALLOW_SHORTS:
        long_short[1].append(sum(val[1:N_ASSETS//2+1]))
        long_short[2].append(sum(val[N_ASSETS//2+1:]))
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

assets    = len(w[0])   # Number of assets
n_actions = len(w)      # Number of actions
y = list(y2)

rewards     = []   # All Rewards     (Multiplicative)
log_rewards = []   # All Log Rewards (Additive)
prevW       = w[0] # Weights from previous period

STEP = 1

print("Iteration, PrevW, Action, PriceChange, NewW, Reward")
#------------------------------------------------------------------------------
for i, reward in enumerate(y):
    
    c = 0.0025
    
    if i >= len(y):
        break
    
    for j in range(len(w[i])):
        w[i][j] = max(w[i][j],0)
        
    if i % STEP == 0:
        cw = [x for x in w[i]]
        
    action = w[i] # The new weights are our action
    rw     = 0    # Reward for this time step

    # Iterate through each asset and add each reward to the net reward
    #----------------------------------------------------------------
    for asset in range(assets):

        # Transaction Cost
        tc       = (1 - c * abs((cw[asset] - prevW[asset])**1))
        if i % STEP != 0:
            tc = 1
        mult     = (10**y[i][asset] - 1) + 1
            
        rw_asset = tc * (cw[asset]) * mult 
        rw      += rw_asset
    #----------------------------------------------------------------

    # Calculate what new weights will be after price move
    newW = [cw[A] * 10**y[i][A] for A in range(assets)]
    newW = [x/sum(newW) for x in newW]
    
    #print(i, showArray(prevW), "-->", showArray(w[i]), "*", showArray(y[i]), "=", showArray(newW), " {{ {}{:.3f}% }}".format("+" if rw >= 1 else "", 100*rw-100))
    
    prevW = newW
    cw = newW
    rewards.append(rw)
    log_rewards.append(math.log(rw))
    
plt.plot(pd.Series(log_rewards).cumsum())
plt.plot(pd.Series(test_imm.close_BTC / test_imm.close_BTC[0]).apply(lambda x : math.log10(x)))
#plt.plot(prof)
plt.show()
