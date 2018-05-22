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

data_points = []
N = 10000

random.seed(1)

#---------------------------------
for i in range(N):
    
    x = random.random()               # Random Number - Our only input variable
    y = 1 if x > 0.5 else -1          # Huge reward, completely dependent on x
    y2 = 0.01 * (random.random()-0.5) # Negligible Reward 1
    y3 = 0.01 * (random.random()-0.5) # Negligible Reward 2
    
    data_points.append( (x, y, y2, y3) )
#---------------------------------
    
# Create Dataframe of data
data = pd.DataFrame(data_points, columns=['x', 'y', 'y2', 'y3'])

data = data.reset_index(drop=True)

ys = ['y','y2','y3']

plt.plot(data.x, data.y, 'ob') # Plot step

X = tf.placeholder(tf.float32, [None, 1])
X = tf.reshape(X, [-1, 1]) # I don't know why but it doesnt work without this reshape

# Define number of Neurons per layer
K        = 100 # Layer 1
L        = 50  # Layer 2
N_INPUT  = 1
N_OUTPUT = len(ys)

# LAYER 1
# Initialize weights, normal dist.
W1 = tf.Variable(tf.truncated_normal([N_INPUT, K], stddev = 1))
# Bias terms initialized to zero
B1 = tf.Variable(tf.zeros([K]))

# LAYER 2
W2 = tf.Variable(tf.truncated_normal([K, L], stddev = 1))
B2 = tf.Variable(tf.zeros([L]))

# LAYER 2
W3 = tf.Variable(tf.truncated_normal([L, N_OUTPUT], stddev = 1))
B3 = tf.Variable(tf.zeros([N_OUTPUT]))

# Feed forward. Output of previous is input to next
# Activation function for final layer is Softmax for probabilties in the range [0,1]
Y1 = tf.nn.relu(tf.matmul(X,  W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y  = tf.nn.softmax(tf.matmul(Y2, W3) + B3) # Softmax activation

Y_ = tf.placeholder(tf.float32, [None, N_OUTPUT])

# Loss Function
# : reduce_sum just sums up the vector 
#loss = tf.reduce_mean((Y - Y_)**2)
# Y is the predicted weights from softmax. Y_ is the reward. Multiplying them together
# gives you the expected reward...in theory.
loss = -tf.reduce_sum( Y * Y_ )

# Learning Rate
LR = 0.0002
optimizer   = tf.train.AdamOptimizer(LR)
train_step  = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

errs = []

shp_x = np.reshape(data.x, (-1, N_INPUT))       # Input data
shp_y = np.reshape(data[ys], (-1, N_OUTPUT))    # Output 'Labels'

# A perfect model would produce this loss
desired_loss = -sum(data.y[data.y==1])

t_1 = time.time()

for i in range(100000):
    
    batch = data.sample(random.randint(5,5000))
    train_data = {X: np.reshape(batch.x, (-1, N_INPUT)), Y_: np.reshape(batch[ys], (-1, N_OUTPUT))}
    #train_data = {X: shp_x, Y_: shp_y}
    sess.run(train_step, feed_dict=train_data)
    
    if i % 100 == 0:
        train_data = {X: shp_x, Y_: shp_y}
        pred, real, error = sess.run([Y, Y_, loss], feed_dict = train_data)
        errs.append(error)
        print("Iteration: {:<10.0f} Loss = {:<16.6f} Perfect Loss = {:<16.6f}".format(i, errs[-1], desired_loss))

plt.plot(errs)

y1, y2 = sess.run([Y, Y_], feed_dict = train_data)
plt.plot(y1)

# if learning properly, the first column of y1 should alternate between 1 and 0,
# depending purely on the value of x. The loss function should equal to the negative sum
# of the occurance of +1s in the data. A rule based engine would apply the simple logic:
    
# if x > 0.5, w = [1, 0, 0]
# else        w = [0, a, b] where a + b = 1

rule_based_loss = 0
for i in range(len(data)):
    observation = data.iloc[i,:]
    if observation.x > 0.5:
        weights = [1, 0, 0]
    else:
        rands = [random.random() for _ in range(2)]
        rands = [x/sum(rands) for x in rands]
        weights = [0, random.random(), rands[0], rands[1]]
    rule_based_loss += weights[0] * observation.y
    rule_based_loss += weights[1] * observation.y2
    rule_based_loss += weights[2] * observation.y3

print(-rule_based_loss)
