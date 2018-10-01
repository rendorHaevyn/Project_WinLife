# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 12:33:34 2018

@author: edwba
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
import pandas as pd
import sklearn
import math
import itertools
import tensorflow as tf
from matplotlib import pyplot as plt

############################ START BLACKJACK CLASS ############################

class Blackjack(gym.Env):
    
    """Simple Blackjack environment"""
    
    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(2)
        self._seed()
        # Start the first game
        self.prevState = self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    # Returns a tuple of the form (str, int) where str is "H" or "S" depending on if its a
    # Soft or Hard hand and int is the sum total of the cards in hand
    # Example output: ("H", 15)
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
        # Draw a random card from the deck with replacement. 11 is ACE
        # I've set it to always draw a 5. In theory this should be very easy to learn and
        # The only possible states, and their correct Q values should be:
        # Q[10_5, stand] = -1  Q[10_5, hit] = 0
        # Q[15_5, stand] = -1  Q[15_5, hit] = 0
        # Q[20_5, stand] =  0  Q[20_5, hit] = -1
        # The network can't even learn this!
        return 5
        return random.choice([5,6])
        return random.choice([2,3,4,5,6,7,8,9,10,10,10,10,11])
    
    def isBlackjack(cards):
        return sum(cards) == 21 and len(cards) == 2
    
    def getState(self):
        # Defines the state of the current game
        pstate, ptotal = Blackjack.getTotal(self.player)
        dstate, dtotal = Blackjack.getTotal(self.dealer)
        return "{}_{}".format("BJ" if Blackjack.isBlackjack(self.player) else pstate+str(ptotal), dtotal)

    def reset(self):
        # Resets the game - Dealer is dealt 1 card, player is dealt 2 cards
        # The player and dealer are represented by an array of numbers, which are the cards they were
        # dealt in order
        self.soft = "H"
        self.dealer = [Blackjack.drawCard()]
        self.player = [Blackjack.drawCard() for _ in range(2)]
        pstate, ptotal = Blackjack.getTotal(self.player)
        dstate, dtotal = Blackjack.getTotal(self.dealer)
        
        # Returns the current state of the game
        return self.getState()

    def step(self, action):
        
        assert self.action_space.contains(action)
        
        # Action should be 0 or 1.
        # If standing, the dealer will draw all cards until they are >= 17. This will end the episode
        # If hitting, a new card will be added to the player, if over 21, reward is -1 and episode ends

        # Stand
        if action == 0:
            
            pstate, ptotal = Blackjack.getTotal(self.player)
            dstate, dtotal = Blackjack.getTotal(self.dealer)
            
            while dtotal < 17:
                self.dealer.append(Blackjack.drawCard())
                dstate, dtotal = Blackjack.getTotal(self.dealer)

            # if player won with blackjack
            if Blackjack.isBlackjack(self.player) and not Blackjack.isBlackjack(self.dealer):
                rw = 1.5
            # if dealer bust or if the player has a higher number than dealer
            elif dtotal > 21 or (dtotal <= 21 and ptotal > dtotal and ptotal <= 21):
                rw = 1
            # if theres a draw
            elif dtotal == ptotal:
                rw = 100
            # player loses in all other situations
            else:
                rw = -1

            state = self.getState()
                
            # Returns (current_state, reward, boolean_true_if_episode_ended, empty_dict)
            return state, rw, True, {}

        # Hit
        else:

            # Player draws another card
            self.player.append(Blackjack.drawCard())
            # Calc new total for player
            pstate, ptotal = Blackjack.getTotal(self.player)

            state = self.getState()

            # Player went bust and episode is over
            if ptotal > 21:
                return state, -1, True, {}
            # Player is still in the game, but no observed reward yet
            else:
                return state, 0, False, {}
            
############################ END BLACKJACK CLASS ############################
            
# Converts a player or dealers hand into an array of 10 cards
# that keep track of how many of each card are held. The card is identified
# through its index:
                
# Index: 0 1 2 3 4 5 6 7 9 10
# Card:  2 3 4 5 6 7 8 9 T A
                
def cardsToX(cards):
    ans = [0] * 12
    for c in cards:
        ans[c] += 1
    ans = ans[2:12]
    return ans

# Easy way to convert Q values into weighted decision probabilities via softmax.
# This is useful if we probablistically choose actions based on their values rather
# than always choosing the max.

# eg Q[s,0] = -1
#    Q[s,1] = -2
#    softmax([-1,-2]) = [0.731, 0.269] --> 73% chance of standing, 27% chance of hitting
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

plt.ion()

# Define number of Neurons per layer
K = 20 # Layer 1
L = 10 # Layer 2
M =  5 # Layer 2

# Option to use binary states of the form 
#[Soft/Hard, total==2, total==3, total==4...total==31]. 
# Player can never have total <= 4, dealer can never have total >= 27 
# -> Length = 55
USE_BINARY_STATES = True

if USE_BINARY_STATES:
    N_IN = 55
else:
    N_IN = 20 # 10 unique cards for player, and 10 for dealer = 20 total inputs
    
N_OUT = 2
SDEV  = 0.01

# Input / Output place holders
X = tf.placeholder(tf.float32, [None, N_IN])
X = tf.reshape(X, [-1, N_IN])

# This will be the observed reward + decay_factor * max(Q[s+1, 0], Q[s+1, 1]).
# This should be an estimate of the 'correct' Q-value with the ony caveat being that
# the Q-value of the next state is a biased estimate of the true value.
Q_TARGET = tf.placeholder(tf.float32, [None, N_OUT])

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

H1     = tf.nn.relu(tf.matmul(X,  W1) + B1)
H2     = tf.nn.relu(tf.matmul(H1,  W2) + B2)
H3     = tf.nn.relu(tf.matmul(H2,  W3) + B3)

# The predicted Q value, as determined by our network (function approximator)
# outputs expected reward for standing and hitting in the form [stand, hit] given the
# current game state
Q_PREDICT  = (tf.matmul(H3,  W4) + B4)

# Is this correct? The Q_TARGET should be a combination of the real reward and the discounted
# future rewards of the future state as predicted by the network. Q_TARGET - Q_PREDICT should be
# the error in prediction, which we want to minimise. Does this loss function work to help the network
# converge to the true Q values with sufficient training?
loss_func = tf.reduce_sum(tf.square(Q_TARGET - Q_PREDICT))

# This are some placeholder values to enable manually set decayed learning rates. For now, use
# the same learning rate all the time.
LR_START = 0.001
#LR_END   = 0.000002
#LR_DECAY = 0.999

# Optimizer
LEARNING_RATE = tf.Variable(LR_START, trainable=False)
optimizer     = tf.train.GradientDescentOptimizer(LEARNING_RATE)#(LEARNING_RATE)
train_step    = optimizer.minimize(loss_func)

#init = tf.global_variables_initializer()
#sess = tf.Session()
#sess.run(init)

# Initialise the game environment
game = Blackjack()

# Number of episodes (games) to play
num_eps  = 100000
# probability of picking a random action. This decays over time
epsilon  = 0.1
# discount factor. For blackjack, future rewards are equally important as immediate rewards.
discount = 1.0

all_rewards = [] # Holds all observed rewards. The rolling mean of rewards should improve as the network learns
all_Qs      = [] # Holds all predicted Q values. Useful as a sanity check once the network is trained
all_losses  = [] # Holds all the (Q_TARGET - Q_PREDICTED) values. The rolling mean of this should decrease
hands       = [] # Holds a summary of all hands played. (game_state, Q[stand], Q[hit], action_taken)

# boolean switch to use the highest action value instead of a stochastic decision via softmax on Q-values
use_argmax  = True 

import nn_arch_1
nn = nn_arch_1.nn(N_IN, N_OUT)
nn.initialise_nn()

eta = 0.0001
gamma = 1


game.reset()

if USE_BINARY_STATES:
            
        pstate, ptotal = Blackjack.getTotal(game.player)
        dstate, dtotal = Blackjack.getTotal(game.dealer)

        x                  = [0] * N_IN
        x[0]               = int(pstate == "S")
        x[ptotal - 1 - 2]  = 1
        x[29]              = int(dstate == "S")
        x[29 + dtotal - 1] = 1

    else:

        # x is the array of 20 numbers. The player cards, and the dealer cards.
        x = cardsToX(game.player) + cardsToX(game.dealer)
        
q = nn.output_nn(x)

if use_argmax:
    # action is selected to be the one with the highest Q-value
    act = np.argmax(q)      
if random.random() < epsilon:
    # action is selected randomly
    act = random.randint(0, N_OUT-1)
        
# Begin generating episodes
for ep in range(num_eps):
    
    game.reset()

    # Keep looping until the episode is not over
    while True:
        
        q_old = q
        x_old = x
        act_old = act
        
        # Q1 refers to the predicted Q-values before any action was taken
        #Q1 = sess.run(Q_PREDICT, feed_dict = {X : np.reshape( np.array(x), (-1, N_IN) )})
        #all_Qs.append(Q1)
        
        # Get game state before action is taken
        game_state = game.getState()
        
        # Take action! Observe new state, reward, and if the game is over
        game_state_new, reward, done, _ = game.step(act)
        
        if USE_BINARY_STATES:
            
            pstate, ptotal = Blackjack.getTotal(game.player)
            dstate, dtotal = Blackjack.getTotal(game.dealer)
            
            x                  = [0] * N_IN
            x[0]               = int(pstate == "S")
            x[ptotal - 1 - 2]  = 1
            x[29]              = int(dstate == "S")
            x[29 + dtotal - 1] = 1

        else:
            
            # x is the array of 20 numbers. The player cards, and the dealer cards.
            x = cardsToX(game.player) + cardsToX(game.dealer)
            
        act = np.argmax(q)      
        if random.random() < epsilon:
            # action is selected randomly
            act = random.randint(0, N_OUT-1)

        q = nn.output_nn(x)
        
        # targetQ is the same as our predicted one initially. The index of the action we took will be
        # updated to be [observed reward] + [discount_factor] * max(Q[s+1])
        
        # If the game is done, then there is no future state
        #if done:
        #    targetQ[0,act] = reward
        #    all_rewards.append(reward)
        #else:
        #    targetQ[0,act] = reward
        
        delta = eta * (reward + gamma * q[act] - q_old[act_old])
        
        # update weights
        
        nn.update_nn(x_old, act_old + 1, delta)
        
        # Perform one gradient descent update, filling the placeholder value for Q_TARGET with targetQ.
        # The returned loss is the difference between the predicted Q-values and the targetQ we just calculated
        #loss, _, _ = sess.run([loss_func, Q_PREDICT, train_step], 
        #                      feed_dict = {X        : np.reshape( np.array(x), (-1, N_IN) ),
        #                                   Q_TARGET : targetQ}
        #                      )
            
        #all_losses.append(loss)
        
        # Every 1000 episodes, show how the q-values moved after the gradient descent update
        if ep % 1000 == 0 and ep > 0:
            print(ep, reward, q, act, x, "...")
            print(q_old, act_old, x_old, "\n\n")
            #Q_NEW = sess.run(Q_PREDICT, feed_dict = {X        : np.reshape( np.array(x), (-1, N_IN) ),
            #                                         Q_TARGET : targetQ})
        
            #print(game_state, targetQ[0], Q1[0], (Q_NEW-Q1)[0], loss, ep, epsilon, act)
            #rolling_window = 1000
            #rolling_mean = np.mean( all_rewards[-rolling_window:] )
            #rolling_loss = np.mean( all_losses[-rolling_window:]  )
            #print("Rolling mean reward: {:<10.4f}, Rolling loss: {:<10.4f}".format(rolling_mean, rolling_loss))
        
        if done:
            # Reduce chance of random action as we train the model.
            epsilon = 2/((ep/500) + 10)
            epsilon = max(0.02, epsilon)
            # rolling mean of rewards should increase over time!
            
            if ep % 1000 == 0 and ep > 0:
                pass# Show the rolling mean of all losses. This should decrease over time!
                #plt.plot(pd.rolling_mean(pd.Series(all_losses),  5000))
                #plt.pause(0.02)
                #plt.show()
            break

nn.close_nn()

ax1.plot(xPlot,rPlot)
for j in range(nn.nW1):
    for k in range(ZX):
        ax2.plot(xPlot,wPlot1[j][k])
for j in range(nn.nW2):
    for k in range(nn.nW1):
        ax3.plot(xPlot,wPlot2[j][k])
for j in range(nn.nW3):
    for k in range(nn.nW2):
        ax4.plot(xPlot,wPlot3[j][k])
for j in range(nn.nW4):
    for k in range(nn.nW3):
        ax5.plot(xPlot,wPlot4[j][k])
for j in range(A):
    for k in range(nn.nW4):
        ax6.plot(xPlot,wPlot5[j][k])

print(cardsToX(game.player))
print(game.dealer)
