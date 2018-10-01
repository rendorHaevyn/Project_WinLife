# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:04:16 2018

@author: ebarker
"""

# ipython only (?)
#%reset 

##########################################
#  Code to store and update CNN weights  #
##########################################

import numpy as np
import random
import matplotlib.pyplot as plt
import math

# neural network approximation architecture code
import nn_arch_1


## GENERAL FUNCTIONS

def intToBool(num, digits):
    return np.asarray([bool(num & (1<<n)) for n in range(digits)],dtype=np.int32)

def boolToInt(array):
    x = 0
    for i in range(len(array)):
        x = array[i] << i | x
    return x

def epsilonGreedy(q, epsilon):
    if (random.random() < epsilon):
        return random.randint(1,len(q))
    else:
        return np.argmax(q) + 1


## TEMP ENVIRONMENT (usually provided externally)

# global parameters

ZX = 6
ZY = 1
T = 10000
intT = T/100
beta = 0.2

# environment

S = 2**ZX
A = 2**ZY
zeta = 6 # for Bernoilli style reward function
#R = np.random.normal(0,1,[S,A])
R = np.random.binomial(1,zeta/S,[S,A]) * S
P = np.random.randint(1,S,[S,A])

Gamma = np.empty((0,ZX), int)

for i in range(S):
    Gamma = np.append(Gamma,np.reshape(intToBool(i,ZX),[1,ZX]),axis=0)

# environment update

def genInput(y, old_s):
    a = boolToInt(y) + 1
    new_s = P[old_s - 1, a - 1]
    x = intToBool(new_s - 1,ZX)
    r = R[old_s - 1, a - 1]
    return new_s, x, r

# create agent approximation architecture
    
nn = nn_arch_1.nn(ZX, A)

## AGENT PARAMETERS

# arguments provided externally

eta = 0.001
epsilon = 0.05
gamma = 0.99

# initialise environment

s = np.random.randint(1,S)
x = intToBool(s - 1,ZX)
r = 0

# session start

nn.initialise_nn()


## RUN ITERATIONS

# initialise other variables

q = nn.output_nn(x)
a = epsilonGreedy(q.flatten(), epsilon)
y = intToBool(a - 1, ZY)
    
# for plotting

xPlot = range(math.floor(T/intT))
rPlot = [0 for i in range(math.floor(T/intT))]
wPlot1 = [[[0 for i in range(math.floor(T/intT))] for j in range(ZX)] for k in range(nn.nW1)]
wPlot2 = [[[0 for i in range(math.floor(T/intT))] for j in range(nn.nW1)] for k in range(nn.nW2)]
wPlot3 = [[[0 for i in range(math.floor(T/intT))] for j in range(nn.nW2)] for k in range(nn.nW3)]
wPlot4 = [[[0 for i in range(math.floor(T/intT))] for j in range(nn.nW3)] for k in range(nn.nW4)]
wPlot5 = [[[0 for i in range(math.floor(T/intT))] for j in range(nn.nW4)] for k in range(A)]
counter = 0
rAvg = 0

fig1 = plt.figure()
ax1 = plt.axes()
fig2 = plt.figure()
ax2 = plt.axes()
fig3 = plt.figure()
ax3 = plt.axes()
fig4 = plt.figure()
ax4 = plt.axes()
fig5 = plt.figure()
ax5 = plt.axes()
fig6 = plt.figure()
ax6 = plt.axes()

for t in range(T):
    
    # save old variables
    
    q_old = q
    a_old = a
    x_old = x

    # generate update

    s, x, r = genInput(y, s)
    q = nn.output_nn(x)
    a = epsilonGreedy(q, epsilon)
    y = intToBool(a - 1, ZY)
    
    # create tensor from inputs
    
    delta = eta * (r + gamma * q[a - 1] - q_old[a_old - 1])
    
    # update weights
    
    nn.update_nn(x_old, a_old, delta)
    
    # tracking for debugging
    
    #print(session.run(Q, feed_dict={X: np.reshape(x,[1,ZX]), D: delta}))
    #print(session.run(nambla1, feed_dict={X: np.reshape(x,[1,ZX]), OA: a_old - 1, D: delta}))
    
    rAvg = (rAvg * counter + r) / (counter + 1)
    
    counter = counter + 1
    
    if t%intT==0: 
        rPlot[math.floor(t/intT)] = rAvg
        temp = nn.get_weights_nn()
        tempW1 = temp[0]
        tempW2 = temp[1]
        tempW3 = temp[2]
        tempW4 = temp[3]
        tempW5 = temp[4]
        for j in range(nn.nW1):
            for k in range(ZX):
                wPlot1[j][k][math.floor(t/intT)] = tempW1[k,j]
        for j in range(nn.nW2):
            for k in range(nn.nW1):
                wPlot2[j][k][math.floor(t/intT)] = tempW2[k,j]
        for j in range(nn.nW3):
            for k in range(nn.nW2):
                wPlot3[j][k][math.floor(t/intT)] = tempW3[k,j]
        for j in range(nn.nW4):
            for k in range(nn.nW3):
                wPlot4[j][k][math.floor(t/intT)] = tempW4[k,j]
        for j in range(A):
            for k in range(nn.nW4):
                wPlot5[j][k][math.floor(t/intT)] = tempW5[k,j]
        counter = 0
        print("iteration: ", t, " reward:", round(rAvg,5), " state:", s, " action:", a, " temporal difference: ", delta)
    

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
    
print("\nMean performance: ",np.mean(rPlot[math.floor(len(rPlot) * (1-beta)):len(rPlot)]), " ... compare sum(R)/SA:", sum(sum(R))/S/A)

