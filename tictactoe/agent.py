import collections
import numpy as np
import os
import pickle
import random


class Learner(object):
    """
    Parent class for Q-learning and Sarsa-learning agents.
    """
    def __init__(self, alpha, gamma, epsilon):
        # Reward accumulator
        self.total_reward = 0
        # Keep a list of accumulated award for each episode
        self.rewards = []
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Possible actions correspond to the set of all x,y coordinate pairs
        self.actions = []
        for i in range(3):
            self.actions += [(0, i), (1, i), (2, i)]
        self.Q = {}
        for action in self.actions:
            # Initialize Q values of all state-action pairs to 0
            self.Q[action] = collections.defaultdict(int)

    def get_action(self, s):
        # Make sure we only consider empty board spaces
        possible_actions = [a for a in self.actions if s[a[0]*3 + a[1]] == '-']
        if random.random() < self.epsilon:
            # Random choose.
            action = possible_actions[random.randint(0,len(possible_actions)-1)]
        else:
            # Greedy choose. At least one action will always be possible
            # when this function is called.
            Q_max = -np.inf
            for a in possible_actions:
                if self.Q[a][s] > Q_max:
                    Q_max = self.Q[a][s]
                    action = a
        return action

    def save_agent(self, path):
        """ Pickle the agent object instance to save the agent's state. """
        if os.path.isfile(path):
            os.remove(path)
        f = open(path, 'wb')
        pickle.dump(self, f)
        f.close()


class QLearner(Learner):
    """
    A class to implement the Q-learning agent.
    """
    def __init__(self, alpha, gamma, epsilon):
        Learner.__init__(self, alpha, gamma, epsilon)

    def update(self, s, s_, a, r):
        """ Perform the Q-Learning step update of Q values. """
        # Update Q(s,a)
        if s_ is not None:
            # Hold list of Q values for all a_,s_ pairs so we can access max later
            Q_options = []
            for action in self.actions:
                Q_options += [self.Q[action][s_]]
            self.Q[a][s] = (1 - self.alpha)*self.Q[a][s] + self.alpha*(r + self.gamma*max(Q_options))
        else:
            self.Q[a][s] = (1 - self.alpha)*self.Q[a][s] + self.alpha*r


class SarsaLearner(Learner):
    """
    A class to implement the Sarsa-learning agent.
    """
    def __init__(self, alpha, gamma, epsilon):
        Learner.__init__(self, alpha, gamma, epsilon)

    def update(self, s, s_, a, a_, r):
        """ Perform the Sarsa step update of Q values. """
        # Update Q(s,a)
        if s_ is not None:
            self.Q[a][s] = (1 - self.alpha)*self.Q[a][s] + self.alpha*(r + self.gamma*self.Q[a_][s_])
        else:
            self.Q[a][s] = (1 - self.alpha)*self.Q[a][s] + self.alpha*r

