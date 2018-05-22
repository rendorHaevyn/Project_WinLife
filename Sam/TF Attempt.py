import numpy as np
import random
import pandas as pd
import sklearn
import math
from matplotlib import pyplot as plt
import tensorflow as tf

data_points = []
N = 10000

random.seed(1)

#---------------------------------
for i in range(N):
    
    x = random.random()
    y = 1 if x > 0.5 else -1
    y2 = 0.01 * (random.random()-0.5)
    y3 = 0.01 * (random.random()-0.5)
    
    data_points.append( (x, y, y2, y3) )
#---------------------------------
    
data = pd.DataFrame(data_points, columns=['x', 'y', 'y2', 'y3'])
#data = test_df[X_cols+Y_cols]
#data.columns = ['x', 'y', 'y2', 'y3']
data = data.reset_index(drop=True)

ys = ['y','y2','y3']

plt.plot(data.x, data.y, 'ob')

X = tf.placeholder(tf.float32, [None, 1])
X = tf.reshape(X, [-1, 1])

# Define number of Neurons per layer
K        = 15 # Layer 1
L        = 5 # Layer 2
N_INPUT  = 1
N_OUTPUT = len(ys)

# LAYER 1
# Initialize weights, normal dist.
W1 = tf.Variable(tf.truncated_normal([N_INPUT, K], stddev = 0.1))
# Bias terms initialized to zero
B1 = tf.Variable(tf.zeros([K]))

# LAYER 2
W2 = tf.Variable(tf.truncated_normal([K, L], stddev = 0.1))
B2 = tf.Variable(tf.zeros([L]))

# LAYER 2
W3 = tf.Variable(tf.truncated_normal([L, N_OUTPUT], stddev = 0.1))
B3 = tf.Variable(tf.zeros([N_OUTPUT]))

# Feed forward. Output of previous is input to next
# Activation function for final layer is Softmax for probabilties in the range [0,1]
Y1 = tf.nn.relu(tf.matmul(X,  W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y  = tf.nn.tanh(tf.matmul(Y2, W3) + B3)

Y_ = tf.placeholder(tf.float32, [None, N_OUTPUT])

init = tf.initialize_all_variables()

# Loss Function
# : reduce_sum just sums up the vector 
#loss = tf.reduce_sum((Y - Y_)**2)
#loss = tf.losses.mean_squared_error(Y_, Y)
#loss = tf.reduce_mean((Y - Y_)**2)
loss = tf.reduce_mean((Y - Y_)**2)

#def loss_f(lab, pred):
#    return tf.reduce_mean((lab-pred)**2)

LR = 0.01
optimizer 		= tf.train.GradientDescentOptimizer(LR)
train_step        = optimizer.minimize(loss)

sess = tf.Session()
sess.run(init)

errs = []

shp_x = np.reshape(data.x, (-1, N_INPUT))
shp_y = np.reshape(data[ys], (-1, N_OUTPUT))

t_1 = time.time()

for i in range(100000):
       
    #idx1 = random.randint(0, len(shp_x-1-batch_size))
    #b_x = shp_x[idx1:(idx1+batch_size)]
    #b_y = shp_y[idx1:(idx1+batch_size)]
    #batch = data.sample(500)
    batch = data
    #batch = data
    train_data = {X: shp_x, Y_: shp_y}
    #train_data = {X: np.reshape(batch.x, (-1, N_INPUT)), Y_: np.reshape(batch[ys], (-1,N_OUTPUT))}
    sess.run(train_step, feed_dict=train_data)
    #pred, real, error = sess.run([Y, Y_, loss], feed_dict = train_data)
    if i % 100 == 0:
        train_data = {X: shp_x, Y_: shp_y}
        pred, real, error = sess.run([Y, Y_, loss], feed_dict = train_data)
        errs.append(error)
        print(errs[-1], 1*0.999**i, i / (time.time() - t_1))

plt.plot(errs)

y1, y2 = sess.run([Y, Y_], feed_dict = train_data)
plt.plot(y1)
    
#plt.plot(prof)