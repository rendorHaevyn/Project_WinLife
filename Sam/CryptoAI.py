import numpy as np
import random
import pandas as pd
import sklearn
import math
import itertools
import threading
from matplotlib import pyplot as plt
import tensorflow as tf
import re

random.seed(1)

print("Loading Data...", end="")
data = pd.read_csv("M5/ALL.csv").dropna(axis=0, how='any').reset_index(drop=True)
data.drop('date', axis=1, inplace=True)
print("{} rows & {} columns".format(len(data), len(data.columns)))
data = data[int(len(data)*0.2):len(data)].reset_index(drop=True)

INCLUDE_VOLUME = True
ALLOW_SHORTS   = False
DISCOUNT       = True
DISCOUNT_STEPS = 24
GAMMA          = 0.95

ASSETS      = ['USD', 'BTC', 'ETH', 'BCH', 'XRP', 'DASH', 'XMR', 'ZEC', 'LTC']
INPUT_ASSET = []#['BTC', 'BCH', 'ETH']
N_VEC       = 3 + 1 if INCLUDE_VOLUME else 0
N_ASSETS    = ( len(ASSETS) * 2 - 1 ) if ALLOW_SHORTS else len(ASSETS)

#for i, a in enumerate(ASSETS):
#    data["M_{}".format(a)] = 1 if i == 0 else 0

cols = []
for c in data.columns:
    if not ASSETS:
        cols.append(c)
    else:
        for a in set(ASSETS):
            if a in c:
                cols.append(c)
                break
            
if ALLOW_SHORTS:
    short_cols = []
    for c in cols:
        if 'reward' in c:
            data[c+"_S"] = data[c].apply(lambda x : -x)
            short_cols.append(c+"_S")
    cols += short_cols   
        
cols2 = []
for c in data.columns:
    if not INPUT_ASSET:
        cols2.append(c)
    else:
        for a in set(INPUT_ASSET):
            if a in c:
                cols2.append(c)
                break
        
data['reward_USD'] = 0
#data['reward_BCH'] = data['reward_BCH'] - 0.001
#data['reward_BCH_S'] = data['reward_BCH_S'] - 0.001

if INCLUDE_VOLUME:
    COLS_X = [x for x in cols2 if 'L_' in x or "M_" in x]
else:
    COLS_X = [x for x in cols2 if ('L_' in x and "VOLUME" not in x) or "M_" in x]
COLS_Y = ["reward_USD"] + [y for y in cols if 'reward' in y and "USD" not in y]

if DISCOUNT:
    data_imm = data.copy()
    stmt = "data[COLS_Y] = data[COLS_Y]"
    for ahead in range(1,DISCOUNT_STEPS+1):
        stmt += "+(GAMMA**{}) * data[COLS_Y].shift({})".format(ahead, -ahead)
    print("Calculating Discount Rewards...", end="")
    exec(stmt)
    print("Done")
else:
    data_imm = data

data = data.dropna(axis=0, how='any').reset_index(drop=True)

for x in COLS_X:
    
    data[x] = data[x].apply(lambda x : 0 if np.isinf(x) else x)
    data[x] = (data[x] - data[x].describe()[1])/(data[x].describe()[2]+1e-10)
    
    data_imm[x] = data_imm[x].apply(lambda x : 0 if np.isinf(x) else x)
    data_imm[x] = (data_imm[x] - data_imm[x].describe()[1])/(data_imm[x].describe()[2]+1e-10)

N_IN  = len(COLS_X)
N_OUT = len(COLS_Y)

# Define number of Neurons per layer
K = 700 # Layer 1
L = 600  # Layer 2
M = 500  # Layer 3
N = 400  # Layer 4
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
W4 = tf.Variable(tf.random_normal([M, N], stddev = SDEV))
B4 = tf.Variable(tf.random_normal([N]))

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
Y1 = tf.nn.leaky_relu(tf.matmul(X,  W1) + B1)
Y2 = tf.nn.leaky_relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.leaky_relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.leaky_relu(tf.matmul(Y3, W4) + B4)
#Y5 = tf.nn.leaky_relu(tf.matmul(Y4, W5) + B5)
Y =  tf.nn.softmax(tf.matmul(Y4, W5) + B5)
#Y  = (tf.matmul(Y4, W5) + B5)

# Placeholder for correct answers, digit recognition so theres 10 output probabilities
Y_ = tf.placeholder(tf.float32, [None, N_OUT])

loss = -tf.reduce_mean( 10000000 * (Y * Y_) )

# Optimizer
LEARNING_RATE 	= 0.00001
optimizer 		= tf.train.AdamOptimizer(LEARNING_RATE)
train_step 		= optimizer.minimize(loss)

BATCH_SZ_MIN = 4#round(0.05*len(data))
BATCH_SZ_MAX = 24#round(0.2*len(data))
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

# Start with a 'good' initialization. This is just for fun
init_rewards = []
while True:
    
    sess.run(init)
    init_rewards.append(-sess.run(loss,feed_dict=feed_imm))
    if len(init_rewards) > 50:
        if init_rewards[-1] >= pd.Series(init_rewards).describe()[6]:
            break
        
plt.plot(init_rewards)
plt.show()
dat_rwds, imm_rwds = [], []

def eval_nn(lst, feed):
    rwd = sess.run(loss, feed_dict=feed)
    lst.append(-rwd)

for i in range(1000000):
    
    if i % 100 == 0:
        
        threading.Thread(target=eval_nn,args=(dat_rwds,feed_dat)).start()
        threading.Thread(target=eval_nn,args=(imm_rwds,feed_imm)).start()
        if dat_rwds and imm_rwds:
            print("{:<16} {:<16.6f} {:<16.6f}".format(i, dat_rwds[-1], imm_rwds[-1]))
        
    else:
        
        idx      = round(random.random()**0.5*IDX_MAX)
        batch_sz = random.randint(BATCH_SZ_MIN, BATCH_SZ_MAX)
        sub_data = data.iloc[idx:idx+batch_sz, :].reset_index(drop=True)
        sub_data = data.iloc[:IDX_MAX,:].sample(batch_sz)
        batch_X, batch_Y = (sub_data[COLS_X], sub_data[COLS_Y])
        train_data = {X:  np.reshape(batch_X, (-1, N_IN)), 
                      Y_: np.reshape(batch_Y, (-1, N_OUT))}
        sess.run(train_step, feed_dict=train_data)
        
    #batch_X = np.matrix(batch_X)
    #batch_Y = np.matrix(batch_Y)

    #for index, a in enumerate(ASSETS):
    #    batch_X["M_{}".format(a)] = 1 if index == 0 else 0 
    
    #for item in range(len(batch_X) - 1): 
    #    row_data = {X: np.matrix(batch_X.iloc[item,:]),
    #                Y_: np.matrix(batch_Y.iloc[item,:])}
    #    weights, y_vec  = sess.run([Y, Y_], feed_dict=row_data)
    #    w = (weights * 10** y_vec) / np.sum(weights * 10 ** y_vec)
    #    for index, a in enumerate(ASSETS):
    #        batch_X.loc[item+1, "M_{}".format(a)] = w[0][index]

plt.plot(dat_rwds)
plt.plot(imm_rwds)
plt.legend(['Discount Test Reward', 'Actual Test Reward'])
plt.show()

y1, y2 = sess.run([Y, Y_], feed_dict = feed_imm)
y3 = y1 * y2
prof = [0]

for x in y3:
    prof.append(prof[-1]+sum(x))
    
plt.plot(prof)
plt.legend(['Actual Reward'])
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
plt.legend(['USD', 'Long', 'Short'] if ALLOW_SHORTS else ['USD', 'Long'])
plt.show()

for i in range(len(props)):
    dat = pd.rolling_mean(pd.Series(props[i]),rolling_window)
    plt.plot(dat)
plt.legend([x[x.index("_")+1:] for x in COLS_Y])
plt.show()
plt.ion()
