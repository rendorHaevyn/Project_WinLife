import tensorflow as tf

# Placeholder is the input data. The "None" is the number of images in the batch,
# The shape is 28 x 28 x 1, representing the pixels
X = tf.placeholder(tf.float32, [None, 28, 28, 1])

# Variables are parameters to be trained
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

init = tf.initialize_all_variables()

# Model
Y  = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b)
# Placeholder for correct answers, digit recognition so theres 10 output probabilities
Y_ = tf.placeholder(tf.float32, [None, 10])

# Loss Function
# : reduce_sum just sums up the vector 
cross_entropy 	= -tf.reduce_sum(Y_ * tf.log(Y))

# % of correct answers found in batch
is_correct 		= tf.equal(tf.argMax(Y, 1), tf.argmax(Y_, 1))
accuracy 		= tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Optimizer
LEARNING_RATE 	        = 0.003
optimizer 		= tf.train.GradientDescentOptimizer(LEARNING_RATE)
train_step 		= optimizer.minimize(cross_entropy)

# : tf.XXX does not produce results. It just creates a 'computation graph' in memory
# Need to define a session first to run things
sess = tf.Session()
sess.run(init)

for i in range(1000):
	# load batch of images and correct answers
	batch_X, batch_Y = mnist.train.next_batch(100)
	
	# Note X, Y_ are the PLACEHOLDERS we defined earlier
	train_data = {X: batch_X, Y_: batch_Y}
	
	# train
	sess.run(train_step, feed_dict=train_data)
	
	# success ?
	a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
	
	# success on test data ?
	test_data = {X: mnist.test.images, Y_: mnist.test.labels}
	a,c = sess.run([accuracy, cross_entropy], feed=test_data)
	
	
###################
#  DEEP LEARNING  #
###################


# Define number of Neurons per layer
K = 200 # Layer 1
L = 100 # Layer 2
M = 60  # Layer 3
N = 30  # Layer 4

# LAYER 1
# Initialize weights, normal dist.
W1 = tf.Variable(tf.truncated_normal([28*28, K], stddev = 0.1))
# Bias terms initialized to zero
B1 = tf.Variable(tf.zeroes([K]))

# LAYER 2
W2 = tf.Variable(tf.truncated_normal([K, L], stddev = 0.1))
B2 = tf.Variable(tf.zeroes([L]))

# LAYER 3
W3 = tf.Variable(tf.truncated_normal([L, M], stddev = 0.1))
B3 = tf.Variable(tf.zeroes([M]))

# LAYER 4
W4 = tf.Variable(tf.truncated_normal([M, N], stddev = 0.1))
B4 = tf.Variable(tf.zeroes([N]))

# LAYER 5 (Output Layer)
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev = 0.1))
B5 = tf.Variable(tf.zeroes([10]))

X = tf.reshape(X, [-1, 28*28])

# Feed forward. Output of previous is input to next
# Activation function for final layer is Softmax for probabilties in the range [0,1]
Y1 = tf.nn.relu(tf.matmul(X,  W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Y  = tf.nn.softmax(tf.matmul(Y4, W5) + B5)

####################
#  REGULARIZATION  #
####################

# May overfit -> Use Regularization with "Dropout". In dropout, only some
# neurons are kept, others are removed at random. This is done on each iteration
# of the training loop. Freezes weights for a given neuron on a particular iteration.

# Define the probability of keeping a neuron
pkeep = tf.placeholder(tf.float32)

Yf = tf.nn.relu(tf.matmul(X, W) + B)
Y  = tf.nn.dropout(Yf, pkeep) # Applied to the previous layer

###################
#  CONVOLUTIONAL  #
###################

K = 4
L = 8
M = 12

# CONV LAYER 1
# [filter size, filter size, input channels, output channels]
# eg 5x5 patch, 1 input channel, K output channels
W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev = 0.1))
B1 = tf.Variable(tf.ones([K])/10)

# CONV LAYER 2
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev = 0.1))
B2 = tf.Variable(tf.ones([L])/10)

# CONV LAYER 3
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev = 0.1))
B3 = tf.Variable(tf.ones([M])/10)

N = 200

# FULLY CONNECTED LAYER
W4 = tf.Variable(tf.truncated_normal([7*7*M, N]), stddev = 0.1)
B4 = tf.Variable(tf.ones([N])/10)

W4 = tf.Variable(tf.truncated_normal([N, 10]), stddev = 0.1)
B4 = tf.Variable(tf.ones([10])/10)

Y1 = tf.nn.relu(tf.nn.conv2d(X,  W1, strides=[1,1,1,1], padding="SAME") + B1)
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1,2,2,1], padding="SAME") + B2)
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1,2,2,1], padding="SAME") + B3)

YY = tf.reshape(Y3, shape=[-1, 7*7*M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
Y  = tf.nn.softmax(tf.matmul(Y4, W5) + B5)


###############
#  WHITENING  #
###############

# Batch Normalizing / Scaling Data

def batchnorm_layer(YLogits, is_test, Offset, Scale, iteration, convolutional=False):
	exp_moving_avg = tf.train.ExponentialMovingAverage(0.9999, iteration)
	mean, variance = tf.nn.moments(YLogits, [0,1,2] if convolutional else [0])
	update_moving_averages = exp_moving_avg.apply([mean, variance])
	m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda : mean)
	v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
	Ybn = tf.nn.batch_normalization(Ylogits, m, v, Offset, Scale, variance_epsilon=1e-5)
	return Ybn, update_moving_averages
