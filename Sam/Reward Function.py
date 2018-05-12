import pandas as pd
import numpy  as np
import random
import math

random.seed(2231)          # y_t, y_t+1... are generated randomly. Use seed to get same results each run

c        = 0.0025       # Transaction cost (%)
discount = 0.00         # Discount Factor
d_len    = 1            # Number of time steps to discount

def showArray(arr, decimals = 3):
    return "[" + " ".join(["{1:.{0}f}".format(decimals, x) for x in arr]) + "]"

# Random actions. These are the weights we are changing to. First asset is USD
w = [[0.5, 0.3, 0.2],
     [1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0],
     [0.0, 0.0, 1.0],
     [0.5, 0.5, 0.0],
     [0.0, 0.5, 0.5],
     [1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0],
     [1.0, 0.0, 0.0]]

assets    = len(w[0])   # Number of assets
n_actions = len(w)      # Number of actions
y         = []          # y_t, y_t+1 vector

#----------------------------------
for i in range(n_actions):
    # Some random price change in the range +- 10%
    y.append([1] + [1+round((random.random()-0.5)/5,2) for _ in range(assets-1)])
#----------------------------------

rewards     = []   # All Rewards     (Multiplicative)
log_rewards = []   # All Log Rewards (Additive)
prevW       = w[0] # Weights from previous period

print("Iteration, PrevW, Action, PriceChange, NewW, Reward")
#------------------------------------------------------------------------------
for i, reward in enumerate(y):
    
    if i + d_len - 1 >= len(y): # Can't look forward further than our data
        break
    
    action = w[i] # The new weights are our action
    rw     = 0    # Reward for this time step

    # Iterate through each asset and add each reward to the net reward
    #----------------------------------------------------------------
    for asset in range(assets):

        # Transaction Cost
        tc       = (1 - c * abs(w[i][asset] - prevW[asset]))
        mult     = 1
        
        # Accomodate for Discounted Rewards
        for d in range(d_len):
            mult *= (y[i+d][asset] - 1) * (discount ** d) + 1
            
        rw_asset = tc * (w[i][asset]) * mult 
        rw      += rw_asset
    #----------------------------------------------------------------

    # Calculate what new weights will be after price move
    newW = [w[i][A] * y[i][A] for A in range(assets)]
    newW = [round(x/sum(newW),4) for x in newW]
    
    print(i, showArray(prevW), "-->", showArray(w[i]), "*", showArray(y[i]), "=", showArray(newW), " {{ {}{:.3f}% }}".format("+" if rw >= 1 else "", 100*rw-100))
    
    prevW = newW
    rewards.append(rw)
    log_rewards.append(math.log(rw))
