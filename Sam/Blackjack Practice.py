import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
import pandas as pd
import sklearn
import math
import itertools

class Blackjack(gym.Env):
    
    """Simple Blackjack environment"""
    
    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(2)
        self._seed()
        # Start the first game
        self.prevState = self.reset()
        self.nA = 4
        self.highest = 4

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getTotal(cards):
        running_total = 0
        softs = 0
        for c in cards:
            running_total += c
            if c == 11:
                softs += 1
            if running_total > 21 and softs > 0:
                softs -= 1
                running_total -= 10

        return "H" if softs == 0 else "S", running_total

    def drawCard():
        return random.choice([2,3,4,5,6,7,8,9,10,10,10,10,11])

    def reset(self):
        self.soft = "H"
        self.dealer = [Blackjack.drawCard()]
        self.player = [Blackjack.drawCard() for _ in range(2)]
        pstate, ptotal = Blackjack.getTotal(self.player)
        dstate, dtotal = Blackjack.getTotal(self.dealer)
        state = "{}_{}".format("BJ" if Blackjack.isBlackjack(self.player) else pstate+str(ptotal), dtotal)
        return state

    def isBlackjack(cards):
        return sum(cards) == 21 and len(cards) == 2

    def step(self, action):
        
        assert self.action_space.contains(action)

        # Stand
        if action == 0:
            
            pstate, ptotal = Blackjack.getTotal(self.player)
            dstate, dtotal = Blackjack.getTotal(self.dealer)
            
            while dtotal < 17:
                self.dealer.append(Blackjack.drawCard())
                dstate, dtotal = Blackjack.getTotal(self.dealer)

            if Blackjack.isBlackjack(self.player) and not Blackjack.isBlackjack(self.dealer):
                rw = 1.5
            elif dtotal > 21 or (dtotal <= 21 and ptotal > dtotal):
                rw = 1
            elif dtotal == ptotal:
                rw = 0
            else:
                rw = -1

            state = "{}_{}".format("BJ" if Blackjack.isBlackjack(self.player) else pstate+str(ptotal), dtotal)
                
            return state, rw, True, {}

        else:

            self.player.append(Blackjack.drawCard())
            pstate, ptotal = Blackjack.getTotal(self.player)
            dstate, dtotal = Blackjack.getTotal(self.dealer)

            state = "{}_{}".format("BJ" if Blackjack.isBlackjack(self.player) else pstate+str(ptotal), dtotal)

            if ptotal > 21:
                return state, -1, True, {}

            else:
                return state, 0, False, {}
            
def cardsToX(cards):
    ans = [0] * 12
    for c in cards:
        ans[c] += 1
    ans = ans[2:12]
    return ans

import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.ion()

# Define number of Neurons per layer
K = 10 # Layer 1
L = 10 # Layer 2

N_IN  = 2#10 * 2
N_OUT = 2
SDEV  = 0.01

# Input / Output place holders
X = tf.placeholder(tf.float32, [None, N_IN])
X = tf.reshape(X, [-1, N_IN])

nextQ = tf.placeholder(tf.float32, [None, N_OUT])

# LAYER 1
W1 = tf.Variable(tf.random_normal([N_IN, K], stddev = SDEV))
B1 = tf.Variable(tf.random_normal([K], stddev = SDEV))

# LAYER 2
W2 = tf.Variable(tf.random_normal([K, L], stddev = SDEV))
B2 = tf.Variable(tf.random_normal([L], stddev = SDEV))

# LAYER 3
W3 = tf.Variable(tf.random_normal([L, N_OUT], stddev = SDEV))
B3 = tf.Variable(tf.random_normal([N_OUT], stddev = SDEV))

H1     = tf.nn.relu(tf.matmul(X,  W1) + B1)
H2     = (tf.matmul(H1,  W2) + B2)
Q_OUT  = (tf.matmul(H2,  W3) + B3)

loss_func = tf.reduce_sum(tf.square(nextQ - Q_OUT))

LR_START = 0.00001
LR_END   = 0.000002
LR_DECAY = 0.999

# Optimizer
LEARNING_RATE = tf.Variable(LR_START, trainable=False)
optimizer     = tf.train.GradientDescentOptimizer(LEARNING_RATE)
train_step    = optimizer.minimize(loss_func)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

game = Blackjack()
num_eps = 10000000
epsilon = 0.1

all_rewards = []
all_Qs = []

hands = []

for ep in range(num_eps):
    game.reset()
    while True:
        
        x = cardsToX(game.player) + cardsToX(game.dealer)
        x = [sum(game.player), sum(game.dealer)]
        Q1 = sess.run(Q_OUT, feed_dict = {X : np.reshape( np.array(x), (-1, N_IN) )})
        all_Qs.append(Q1)
        act = np.argmax(Q1)
        
        hands.append( (sum(game.player), sum(game.dealer), Q1[0][0], Q1[0][1], act) )
        
        if random.random() < epsilon:
            act = random.randint(0, N_OUT-1)
        state, reward, done, _ = game.step(act)
        #Obtain the Q' values by feeding the new state through our network
        x2 = cardsToX(game.player) + cardsToX(game.dealer)
        x2 = [sum(game.player), sum(game.dealer)]
        Q2 = sess.run(Q_OUT,feed_dict = {X : np.reshape( np.array(x2), (-1, N_IN) )})
        #Obtain maxQ' and set our target value for chosen action.
        maxQ1 = np.max(Q2)
        targetQ = Q1
        targetQ[0,act] = reward + 1.0 * maxQ1
        
        loss, _ = sess.run([loss_func, train_step], feed_dict = {X : np.reshape( np.array(x), (-1, N_IN) ),
                                          nextQ : targetQ})
        print(targetQ, Q1, loss)

        if done:
            #Reduce chance of random action as we train the model.
            epsilon = 1./((ep/500) + 10)
            all_rewards.append(reward)
            if ep % 1000 == 0 and ep > 0:
                #print(sum(all_rewards[-100:])/100)
                roll = pd.rolling_mean(pd.Series(all_rewards),  5000)
                plt.plot(roll)
                plt.pause(0.02)
                plt.show()
            break
    #Train our network using target and predicted Q values

print(cardsToX(game.player))
print(game.dealer)

'''
for ep in range(num_eps):
    game.reset()
    while True:
        
        x = cardsToX(game.player) + cardsToX(game.dealer)
        x = [sum(game.player), sum(game.dealer)]
        Q1 = sess.run(Q_OUT, feed_dict = {X : np.reshape( np.array(x), (-1, N_IN) )})
        all_Qs.append(Q1)
        act = np.argmax(Q1)
        
        hands.append( (sum(game.player), sum(game.dealer), Q1[0][0], Q1[0][1], act) )
        
        if random.random() < epsilon:
            act = random.randint(0, N_OUT-1)
        state, reward, done, _ = game.step(act)
        #Obtain the Q' values by feeding the new state through our network
        x2 = cardsToX(game.player) + cardsToX(game.dealer)
        x2 = [sum(game.player), sum(game.dealer)]
        Q2 = sess.run(Q_OUT,feed_dict = {X : np.reshape( np.array(x2), (-1, N_IN) )})
        #Obtain maxQ' and set our target value for chosen action.
        maxQ1 = np.max(Q2)
        targetQ = Q1
        targetQ[0,act] = reward + 1.0 * maxQ1
        
        sess.run(train_step, feed_dict = {X : np.reshape( np.array(x), (-1, N_IN) ),
                                          nextQ : targetQ,
                                          Q_OUT : Q1})

        if done:
            #Reduce chance of random action as we train the model.
            epsilon = 1./((ep/500) + 10)
            all_rewards.append(reward)
            if ep % 1000 == 0 and ep > 0:
                #print(sum(all_rewards[-100:])/100)
                roll = pd.rolling_mean(pd.Series(all_rewards),  5000)
                plt.plot(roll)
                plt.pause(0.02)
                plt.show()
            break
    #Train our network using target and predicted Q values

print(cardsToX(game.player))
print(game.dealer)'''


