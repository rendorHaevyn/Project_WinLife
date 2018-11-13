from __future__ import print_function
import numpy as np
import random
import pandas as pd
import sklearn
import sklearn.decomposition
import sklearn.ensemble
import sklearn.preprocessing
import sklearn.cluster
from sklearn import *
import math
import gc
import os
import sys
import itertools
import threading
from matplotlib import pyplot as plt
import matplotlib.colors
import tensorflow as tf
import re
import time
import pickle
import Constants
import gym
import copy
import gc

import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist, pdist

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

out_data = pd.read_csv("Crypto Q Data.csv")
nn_cols  = [x for x in out_data.columns if "NN_OUT_" in x]
act_cols = [x for x in out_data.columns if "ACT" in x]
rew_cols = [x for x in out_data.columns if "reward_" in x]

out_data['ACT_LONG'] = 0
out_data['ACT_SHORT'] = 0
out_data['reward_LONG'] = 0
out_data['reward_SHORT'] = 0

for c in act_cols:
    if c.endswith("_S"):
        out_data['ACT_SHORT'] += out_data[c]
        out_data['reward_SHORT'] += out_data[c.replace("ACT_", "reward_")]
    else:
        out_data['ACT_LONG'] += out_data[c]
        out_data['reward_LONG'] += out_data[c.replace("ACT_", "reward_")]

TRAIN_SPLIT = 0.6
TRAIN_LOC = int(0.6 * len(out_data) )

#scaler = sklearn.preprocessing.StandardScaler()
#scaler.fit( out_data[:TRAIN_LOC][nn_cols] )
#out_data[nn_cols] = scaler.transform(out_data[nn_cols])

clust_cols = nn_cols

C = sklearn.cluster.KMeans(200)
C.fit(out_data[:TRAIN_LOC][clust_cols])
plt.plot(C.cluster_centers_, 'o')
plt.show()

'''
states = []
for col in nn_cols:
    C.fit( np.array( out_data[:TRAIN_LOC][col] ).reshape(-1, 1) )
    p = C.predict( np.array( out_data[col] ).reshape(-1, 1) )
    if not states:
        states = [str(x) for x in p]
    else:
        for i in range(len(states)):
            states[i] += str(p[i])
            '''
#out_data['state'] = states
out_data['state'] = C.predict(out_data[clust_cols])
#out_data['state'] = out_data['state'].shift(0).astype('str')+\
#                    out_data['state'].shift(1).astype('str')#+\
                    #out_data['state'].shift(2).astype('str')


'''
tr = out_data[:TRAIN_LOC][nn_cols]
kMeansVar = [KMeans(n_clusters=k).fit(tr) for k in range(2, 20, 2)]
centroids = [X.cluster_centers_ for X in kMeansVar]
k_euclid = [cdist(tr, cent) for cent in centroids]
dist = [np.min(ke, axis=1) for ke in k_euclid]
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(tr)**2)/tr.shape[0]
bss = tss - wcss
plt.plot(bss)
plt.show()'''

class stopwatch:
    
    def __init__(self):
        
        self.watch = {}
        self.totals =  {}
        
    def start(self, name):
        
        if name not in self.watch:
            self.watch[name] = []
        self.watch[name].append( [time.time(), time.time(), 0] )
    
    def end(self, name):
        
        self.watch[name][-1][1] = time.time()
        self.watch[name][-1][2] = self.watch[name][-1][1] - self.watch[name][-1][0]
        if name not in self.totals:
            self.totals[name] = self.watch[name][-1][2]
        else:
            self.totals[name] += self.watch[name][-1][2]
    
    def get_info(self, name):
        total_time = self.totals[name]
        calls = len(self.watch[name])
        avg_time = total_time / calls
        return total_time, avg_time, calls
    
    def display(self):
        
        totals = {}
        tt = 0
        
        for name in self.watch:
            tot, avg, calls = self.get_info(name)
            tt += tot
            totals[name] = [tot, avg, calls]
        
        items = sorted(totals.items(), reverse=True)
        #print(items)
        print("{:<19}  {:<11}  {:<11}  {:<11}{:>6}".format(
                "Block", "TotalTime", "AvgTime", "Calls", "Percent"))
        for nm, (T, A, C) in items:
            print("{:<19}  {:<11.3f}  {:<11.6f}  {:<11}{:>6.2f}%".format(
                    nm,T,A,C,100*T/tt))

watch = stopwatch()

tr = out_data[:TRAIN_LOC].reset_index(drop=True)

VALID_ACTIONS = ['ACT_USD', 'ACT_IOTA', 'ACT_IOTA_S']
#VALID_ACTIONS = ['ACT_LONG', 'ACT_SHORT']
#VALID_ACTIONS = act_cols

Q = {}
for st in set(out_data.state):
    for a in VALID_ACTIONS:
        Q[(st,a)] = {}
        for a2 in VALID_ACTIONS:
            Q[(st,a)][a2] = math.log10(1.0001)*(random.random() - 0.5)
        

def getAction(state, epsilon=0.05, bestAct=False):
    try:
        if random.random() < epsilon:
            return random.choice((VALID_ACTIONS))
        elif bestAct == False:
            weights = list(Q[state].values())
            offset = min(weights)
            fixed = sum( [abs(w) for w in weights] ) / len(weights)
            pos_weights = np.array([(z - offset + fixed / 2)**3 for z in weights])
            pos_weights = pos_weights / sum(pos_weights)
            #probs = softmax(list(Q[state].values()))
            if random.random() < 0.0001:
                print(["{:<5.3}".format(round(x,3)) for x in pos_weights])
            return np.random.choice(list(Q[state].keys()), p=pos_weights)
        else:
            slot = np.argmax(list(Q[state].values()))
            return list(Q[state].keys())[slot]
    except KeyboardInterrupt:
        return
        
num_iter = 0
projections_train, projections_test = [], []


TRAIN_COST = math.log10(1 - 0.0025)
TEST_COST  = math.log10(1 - 0.0020)

return_sums = {}
return_counts = {}
return_list = {}

USE_MONTE = 1

loop_forever = True
while loop_forever:
    
    GAMMA = 1
    USE_MONTE = num_iter > 1e6
    SAMPLE_STEPS = 10 if USE_MONTE else 0

    try:
        for H in VALID_ACTIONS:
            
            pos = 1 + round( random.random() ** 0.5 * (len(tr) - SAMPLE_STEPS - 4) )
            current_state = tr.at[pos, "state"], H
            #ep = max(0.01,  (1 - ((num_iter+1) / 1000000)**0.5) )
            #ep = 0.01
            
            current_action = getAction(current_state, 1, True)
            
            reward = tr.at[pos, current_action.replace("ACT","reward")]
            if H != current_action:
                reward += TRAIN_COST
                
            H = current_action
            
            for step in range(1, SAMPLE_STEPS+1):
            
                new_state = tr.at[pos+step, "state"], H
                new_action = getAction(new_state, 0, False if num_iter < 1e5 else True)
                #new_action = getAction(new_state, 0, True)
                
                reward += GAMMA ** step * tr.at[pos+step, new_action.replace("ACT","reward")]
                
                if H != new_action:
                    reward += TRAIN_COST
                H = new_action
    
            new_state = tr.at[pos+SAMPLE_STEPS+1, "state"], H
            next_best_rw = max(Q[new_state].values())
            
            sa_pair = (current_state, current_action)
            if sa_pair not in return_sums:
                return_sums[sa_pair] = 0
                return_counts[sa_pair] = 0
                return_list[sa_pair] = []
            return_sums[sa_pair]   += reward
            return_counts[sa_pair] += 1
            return_list[sa_pair].append(reward)
            
            if USE_MONTE:
                
                window = 10000
                sub_list = return_list[sa_pair][-window:]
                if len(sub_list) < window:
                    new_q = np.mean(return_list[sa_pair]) / (np.var(return_list[sa_pair]) + 1e-6)
                    #new_q = pd.Series(return_list[sa_pair]).describe()['50%']
                    if len(sub_list) == window - 1:
                        print(sa_pair, new_q)
                else:
                    new_q = np.mean(sub_list) / np.var(sub_list)
                    #new_q = pd.Series(sub_list).describe()['50%']
                
                Q[current_state][current_action] = new_q
            
            else:
                
                td_target = reward + GAMMA ** (SAMPLE_STEPS+1) * next_best_rw
                td_error = td_target - Q[current_state][current_action]
                Q[current_state][current_action] += 1e-5 * td_error
        
        num_iter += 1

        if num_iter % 10000 == 0:
            print(num_iter)
            #for k, v in Q[(3,"ACT_IOTA")].items():
            #    print(k, v)
                
        if num_iter % 100000 == 0:
            
            for plot_iter in range(2):
                
                H = VALID_ACTIONS[0]
                if plot_iter == 0:
                    tst = out_data[:TRAIN_LOC].reset_index(drop=True)
                else:
                    tst = out_data[TRAIN_LOC:].reset_index(drop=True)
                    
                raws, tcs, rewards = [], [], []
                
                for pos in range(1, len(tst)-1):
                    
                    current_state = tst.at[pos, "state"], H
                    current_action = getAction(current_state, 0, True)
                    
                    reward = tst.at[pos, current_action.replace("ACT","reward")]
                    
                    if H != current_action:
                        tc = TRAIN_COST if plot_iter == 0 else TEST_COST
                    else:
                        tc = 0
                    
                    raws.append(reward)
                    tcs.append(tc)
                    rewards.append(reward+tc)
                        
                    H = current_action
            
                if plot_iter == 0:
                    projections = projections_train
                else:
                    projections = projections_test
                print("On {}: ".format("Train" if plot_iter == 0 else "Test"), 
                      list(pd.Series(rewards).cumsum())[-1])
                projections.append(pd.Series(rewards).cumsum())
                for pnum, p in enumerate(projections[::-1]):
                    plt.plot(p)
                    if pnum >= 10:
                        break
                gc.collect()

                plt.show()
            
    except KeyboardInterrupt:
        loop_forever = False
        break