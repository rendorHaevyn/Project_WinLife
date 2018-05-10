import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
import pandas as pd
import sklearn
import math
import itertools
from matplotlib import pyplot as plt
import tensorflow as tf
import re

data = pd.read_csv("M5/ALL.csv").dropna(axis=0, how='any').reset_index(drop=True)
data.drop('date', axis=1, inplace=True)
data = data[100:len(data)-4000].reset_index(drop=True)
#data['usd'] = [0] * len(data)

ASSETS      = ['BTC', 'ETH', 'XRP']
ASSETS2     = ['BTC', 'ETH', 'XRP']
N_LAGS      = 10
N_VEC       = 3
N_ASSETS    = len(ASSETS)
N_BATCH     = 3000

cols = []
for c in data.columns:
    for a in ASSETS:
        if a in c:
            cols.append(c)
            break
        
cols2 = []
for c in data.columns:
    for a in ASSETS2:
        if a in c:
            cols2.append(c)
            break

X_cols = [x for x in cols2 if 'L_' in x and "VOLUME" not in x]
Y_cols = [y for y in cols if 'reward' in y]

N_X = len(ASSETS2)

batches = []
batch_size = 20

test_df  = data.iloc[len(data)-2000:, :]
test_x   = test_df[X_cols]
test_y   = test_df[Y_cols]

for i in range(N_BATCH):
    min_index = len(data)-5000
    max_index = len(data) - batch_size
    idx = round(random.random()**0.5*len(data))
    #idx = random.randint(min_index, max_index)
    sub_data = data.iloc[idx:idx+batch_size, :]
    batches.append( (sub_data[X_cols], sub_data[Y_cols]) )
    
X_df = data[X_cols]
Y_df = data[Y_cols]

# Define number of Neurons per layer
K = 200 # Layer 1
L = 100 # Layer 2
M = 60  # Layer 3
N = 30  # Layer 4

# LAYER 1
# Initialize weights, normal dist.
W1 = tf.Variable(tf.truncated_normal([N_LAGS * N_VEC * N_X, K], stddev = 0.1))
# Bias terms initialized to zero
B1 = tf.Variable(tf.zeros([K]))

# LAYER 2
W2 = tf.Variable(tf.truncated_normal([K, L], stddev = 0.1))
B2 = tf.Variable(tf.zeros([L]))

# LAYER 3
W3 = tf.Variable(tf.truncated_normal([L, M], stddev = 0.1))
B3 = tf.Variable(tf.zeros([M]))

# LAYER 4
W4 = tf.Variable(tf.truncated_normal([M, N], stddev = 0.1))
B4 = tf.Variable(tf.zeros([N]))

# LAYER 5 (Output Layer)
W5 = tf.Variable(tf.truncated_normal([N, N_ASSETS], stddev = 0.1))
B5 = tf.Variable(tf.zeros([N_ASSETS]))

X = tf.placeholder(tf.float32, [None, N_LAGS, N_VEC, N_X])
X = tf.reshape(X, [-1, N_LAGS * N_VEC * N_X])

# Feed forward. Output of previous is input to next
# Activation function for final layer is Softmax for probabilties in the range [0,1]
Y1 = tf.nn.relu(tf.matmul(X,  W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Y  = tf.nn.tanh(tf.matmul(Y4, W5) + B5)
#Y  = (tf.matmul(Y4, W5) + B5)

init = tf.initialize_all_variables()

# Placeholder for correct answers, digit recognition so theres 10 output probabilities
Y_ = tf.placeholder(tf.float32, [None, N_ASSETS])

# Loss Function
# : reduce_sum just sums up the vector 
loss = tf.reduce_mean((Y - Y_)**2)

loss = -tf.reduce_mean((Y * Y_))

# Optimizer
LEARNING_RATE 	= 0.003
optimizer 		= tf.train.GradientDescentOptimizer(LEARNING_RATE)
train_step 		= optimizer.minimize(loss)

# : tf.XXX does not produce results. It just creates a 'computation graph' in memory
# Need to define a session first to run things
sess = tf.Session()
sess.run(init)

errs = []

test_data = {X: test_x, Y_: test_y}

for i in range(N_BATCH):
    batch_X, batch_Y = batches[i]
    batch_X = np.matrix(batch_X)
    batch_Y = np.matrix(batch_Y)
    train_data = {X: batch_X, Y_: batch_Y}
    sess.run(train_step, feed_dict=train_data)
    if i % 20 == 0:
        err = sess.run([loss], feed_dict=test_data)
        print(i, err)
        errs.append(err)

plt.plot(errs)

out = pd.DataFrame((sess.run([Y], feed_dict=test_data))[0], columns=["gain_"+x for x in ASSETS])

y1, y2 = sess.run([Y, Y_], feed_dict = test_data)