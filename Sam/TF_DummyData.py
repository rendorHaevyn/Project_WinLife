import numpy as np
import random
import pandas as pd
import sklearn
import math
import time
from matplotlib import pyplot as plt
import tensorflow as tf

# Goal: Create a NN that uses softmax to determine the weights to assign to assets
#       y, y2, y3. y is +-1 depending on x. The NN should learn that if x > 0.5, then
#       all the weight should go to y (reward=1), 
#       otherwise all the weight should go to y2 or y3 since reward for y = -1 if
#       x <= 0.5

N = 10000
data_points = []
random.seed(1)

#---------------------------------
for i in range(N):
    
    x = random.random()               # Random Number - Our only input variable
    y1 = 1 if x > 0.5 else -1         # Huge reward, completely dependent on x
    y2 = 0.01 * (random.random()-0.5) # Negligible Reward 1
    y3 = -y2
    
    data_points.append( (x, y1, y2, y3) )
#---------------------------------
    
# Create Dataframe of data
data = pd.DataFrame(data_points, columns=['x', 'y1', 'y2', 'y3'])
data['x'] = (data['x'] - data['x'].describe()[1])/(data['x'].describe()[2])
data = data.sort_values('x')
data = data.reset_index(drop=True)

ys = ['y1','y2', 'y3']

plt.plot(data.x, data.y1, 'ob') # Plot step
plt.show()

X = tf.placeholder(tf.float32, [None, 1])
X = tf.reshape(X, [-1, 1]) # I don't know why but it doesnt work without this reshape

# Define number of Neurons per layer
K        = 20  # Layer 1
L        = 10  # Layer 2
N_INPUT  = 1
N_OUTPUT = len(ys)

shp_x = np.reshape(data.x, (-1, N_INPUT))       # Input data
shp_y = np.reshape(data[ys], (-1, N_OUTPUT))    # Output 'Labels

                   
SDEV = 5

# LAYER 1
# Initialize weights, normal dist.
W1 = tf.Variable(tf.random_normal([N_INPUT, K], stddev = SDEV))
# Bias terms initialized to zero
B1 = tf.Variable(tf.random_normal([K]))

# LAYER 2
W2 = tf.Variable(tf.random_normal([K, L], stddev = SDEV))
B2 = tf.Variable(tf.random_normal([L]))

# LAYER 3
W3 = tf.Variable(tf.random_normal([L, N_OUTPUT], stddev = SDEV))
B3 = tf.Variable(tf.random_normal([N_OUTPUT]))

# Feed forward. Output of previous is input to next
# Activation function for final layer is Softmax for probabilties in the range [0,1]
L1 = tf.nn.relu(tf.matmul(X,  W1) + B1)
L2 = tf.nn.relu(tf.matmul(L1, W2) + B2)

Y  = tf.nn.softmax(tf.matmul(L2, W3) + B3) # Softmax activation

Y_ = tf.placeholder(tf.float32, [None, N_OUTPUT])

# Loss Function
# : reduce_sum just sums up the vector 

# Y is the predicted weights from softmax. Y_ is the reward. Multiplying them together
# gives you the expected reward...in theory.
neg_reward = -tf.reduce_sum( Y * Y_ )

# Learning Rate
LR = 0.0001
optimizer 		= tf.train.AdamOptimizer(LR)
train_step        = optimizer.minimize(neg_reward)

init = tf.global_variables_initializer()
sess = tf.Session()

# Start with a 'good' initialization. This is just for fun
init_rewards = []
while True:
    
    sess.run(init)
    init_rewards.append(-sess.run(neg_reward, 
                                  feed_dict={X: shp_x, Y_: shp_y}))
    if len(init_rewards) > 100:
        if init_rewards[-1] >= pd.Series(init_rewards).describe()[6]:
            break
        
plt.plot(init_rewards)
plt.show()

# A perfect model would produce this output
perfect_reward = -sum(data.y1[data.y1==1])

use_mini_batch   = True
min_batch_sz     = 100
max_batch_sz     = 500
training_rewards = []
N_ITERATIONS     = 20000

#---------------------------------------------------------------------------
for i in range(N_ITERATIONS):
    
    if use_mini_batch and i % 100 != 0:
        smpl_size = random.randint(min_batch_sz, max_batch_sz)
        smpl_size *= 2
        zeros = data[data.y1 == -1].sample(smpl_size//2)
        ones = data[data.y1 == 1].sample(smpl_size//2)
        sample = zeros.append(ones)
        #sample = data.sample()
        train_data = {X: np.reshape(sample.x, (-1, N_INPUT)), \
                      Y_: np.reshape(sample[ys], (-1, N_OUTPUT))}
        sess.run(train_step, feed_dict=train_data)
    else:
        train_data = {X: shp_x, Y_: shp_y}
    
    if i % 100 == 0:
        pred, real, reward = sess.run([Y, Y_, neg_reward], feed_dict = train_data)
        training_rewards.append(reward)
        print("Iteration: {:<10.0f} Reward = {:<16.6f} Ideal Reward = {:<16.6f}". \
              format(i, reward, perfect_reward))
        
#---------------------------------------------------------------------------

plt.plot(training_rewards[:])
plt.show()

train_data = {X: shp_x, Y_: shp_y}
yhat, y_real = sess.run([Y, Y_], feed_dict = train_data)
plt.plot(yhat)
plt.show()

# if learning properly, the first column of y1 should alternate between 1 and 0,
# depending purely on the value of x. The loss function should equal to the negative sum
# of the occurance of +1s in the data. A rule based engine would apply the simple logic:
    
# if x > 0.5, w = [1, 0, 0]
# else        w = [0, a, b] where a + b = 1

rule_based_reward = 0
for i in range(len(data)):
    observation = data.iloc[i,:]
    if observation.x > 0.5:
        weights = [1, 0, 0]
    else:
        rands = [random.random() for _ in range(2)]
        rands = [x/sum(rands) for x in rands]
        weights = [0, rands[0], rands[1]]
    rule_based_reward += weights[0] * observation.y1
    rule_based_reward += weights[1] * observation.y2
    rule_based_reward += weights[2] * observation.y3

print(-rule_based_reward)
