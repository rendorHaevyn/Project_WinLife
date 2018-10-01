import gym
import matplotlib
import numpy as np
import sys
import random
import plotting
import pandas as pd
from matplotlib import pyplot as plt
from gym import spaces
from gym.utils import seeding

from collections import defaultdict
if "../" not in sys.path:
  sys.path.append("../") 
  
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
        #return 5
        #return random.choice([5,6])
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
                rw = 0
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

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

env = Blackjack()
stats = None

E_LAST = False


def make_epsilon_greedy_policy(Q, epsilon, nA):

    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    global E_LAST
    
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


convergence = {}

def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """

    global E_LAST
    
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    # Keeps track of useful statistics
    global stats
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes)) 
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            wind = min(1000, i_episode)
            r_mean = sum(stats.episode_rewards[i_episode-wind:i_episode])/wind
            print("\rEpisode {}/{} ({:.2f}%) Mean Rwd {}.\n".format(i_episode, num_episodes, 
                                                        100*i_episode/num_episodes, r_mean), end="")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []

        LAST_EPISODE = i_episode == num_episodes
          
        state = env.reset()
        
        #if LAST_EPISODE:
        #  E_LAST = True
        #  env.index = 0
        #  environment.Price.MAX_STEPS = len(env.data) - 2

        e_len = 0
        cum_profit = [0]

        while True:

          try:
            e_len += 1
            probs = policy(state)
            action = 0
            if True:#random.random() < 2 * i_episode/num_episodes:
                for prob_i in range(len(probs)):
                    if probs[prob_i] == max(probs):
                        action = prob_i
                        break
            else:
                action = np.random.choice(np.arange(len(probs)), p=softmax(probs))
            if LAST_EPISODE:
              maxP = np.max(probs)
              for idx in range(len(probs)):
                if probs[idx] == maxP:
                  action = idx
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            stats.episode_rewards[i_episode-1] += reward
            cum_profit.append(stats.episode_rewards[i_episode-1])
            if done:
                break
            state = next_state
          except KeyboardInterrupt:
            return Q, policy
        
        stats.episode_lengths[i_episode-1] = e_len

        if i_episode == num_episodes:
          print(stats.episode_rewards[-1])
          plt.plot(cum_profit)
          plt.show()
        

        # Find all (state, action) pairs we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        sa_in_episode = set([(x[0], x[1]) for x in episode])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            # Find the first occurance of the (state, action) pair in the episode
            first_occurence_idx = next(i for i,x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
            # Sum up all rewards since the first occurance
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
            if (state,action) not in convergence:
                convergence[(state,action)] = []
            #convergence[(state,action)].append(Q[state][action])
        
        # The policy is improved implicitly by changing the Q dictionary
    
    return Q, policy, returns_count

for k,v in sorted(convergence.items(), key=lambda x : (x[0], x[1])):
    plt.plot(v)
plt.show()


Q, policy, counts = mc_control_epsilon_greedy(env, num_episodes=20000000, discount_factor=1, epsilon=0.01)

# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
qitems = []
for state, actions in Q.items():
    lst = [state]+list(actions)
    for a in actions:
      lst += [counts[(state, a)]]
    qitems.append(tuple(lst))
    action_value = np.max(actions)
    V[state] = actions
#pd.DataFrame(qitems).to_clipboard()
#plotting.plot_value_function(V, title="Optimal Value Function
plotting.plot_episode_stats(stats, smoothing_window=500)

for k,v in V.items():
    print(k, v[0], v[1])
