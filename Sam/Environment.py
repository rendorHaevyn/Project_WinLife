import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
import pandas as pd
import sklearn
import math
import itertools

class Price(gym.Env):
    """Simple Price environment"""

    data = pd.read_csv("ALL2.csv")
    returns = []
    for i in range(len(data)):
        returns.append(sum(data.iloc[i,1:]) / 12)
    data['Reward'] = returns
    MAX_STEPS = 100
    
    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(13)
        self._seed()
        # Start the first game
        self.prevState = self.reset()
        self.nA = 4
        self.highest = 4
        self.prevAction = 0
        self.assets  = "1000000000000"

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.index = random.randint(0, len(Price.data)-10000-self.MAX_STEPS)
        #self.index = 5000
        self.steps_taken = 0
        self.assets  = "1000000000000"
        state = Price.data.ix[self.index, "State"]
        state = "{}_{}".format(Price.data.ix[self.index, "State"],self.assets)
        return state

    def step(self, action):
        
        assert self.action_space.contains(action)

        self.steps_taken += 1

        rw = Price.data.iloc[self.index,1+action]
        if self.assets[action] != "1":
            rw = math.log( (10 ** rw) - 0.003, 10)

        holdings = ["0"]*13
        holdings[action] = "1"
        self.assets = "".join(holdings)
                
        self.index += 1
        
        if self.steps_taken >= Price.MAX_STEPS:
            state = "terminal"
        else:
            try:
                state = Price.data.iloc[self.index,0]
                state = "{}_{}".format(state, self.assets)
            except:
                state = "terminal"
            
        return state, rw, state == "terminal", {}

    def _get_obs(self):
        return tuple(self.grid)

class PriceMargin(gym.Env):
    """Simple Price environment"""

    data = pd.read_csv("ALL2.csv")
    MAX_STEPS = 100
    actionMap = {0:-0.3, 1:-0.2, 2:-0.1, 3:0, 4:0.1, 5:0.2, 6:0.3}
    action_list = []
    for i in range(20):
        A = tuple([random.random()*0.6-0.3 for _ in range(12)])
        action_list.append(A)
    
    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(len(PriceMargin.action_list))
        self._seed()
        # Start the first game
        self.reset()
        self.nA = 4
        self.highest = 4
        self.margin = tuple([0]*12)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.index = random.randint(0, len(PriceMargin.data)-self.MAX_STEPS)
        #self.index = 5000
        self.steps_taken = 0
        return "{}_{}".format(PriceMargin.data.ix[self.index, "State"], 0)

    def step(self, action):
        
        assert self.action_space.contains(action)

        self.steps_taken += 1

        m = PriceMargin.action_list[action]
        rw = 0
        for idx in range(len(m)):
            rw += m[idx] * PriceMargin.data.iloc[self.index,1+idx]
            chng = abs(self.margin[idx] - m[idx])
            #rw = math.log( (10 ** rw) - 0.002 * chng, 10)
        #rw = m * PriceMargin.data.iloc[self.index,1:13]
        #print(rw)
        #rw = math.log( (10 ** rw) - 0.002 * chng, 10)
        self.margin = m
        self.index += 1
        state = str("{}_{}".format(PriceMargin.data.iloc[self.index,0], m))
        if self.steps_taken >= PriceMargin.MAX_STEPS:
            done = True
        else:
            done = False
            
        return str(state)+str(m), rw, done, {}

    def _get_obs(self):
        return tuple(self.grid)


class Blackjack(gym.Env):
    
    """Simple Blackjack environment"""

    data = [0] * 10000
    
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

