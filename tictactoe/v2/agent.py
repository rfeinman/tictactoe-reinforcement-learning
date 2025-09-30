import numpy as np
import random
from abc import ABC, abstractmethod
from collections import defaultdict

from .environment import Environment


class Agent(ABC):
    """
    Parent class for Q-learning and SARSA agents.

    Parameters
    ----------
    alpha : float
        learning rate
    gamma : float
        temporal discounting rate
    eps : float
        probability of random action vs. greedy action
    eps_decay : float
        epsilon decay rate. Larger value = more decay
    """
    actions: list

    def __init__(self, alpha=0.5, gamma=0.9, eps=0.1, eps_decay=0.0):
        # Agent parameters
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay

        # Initialize Q values to 0 for all state-action pairs. Accessed as Q[s][a]
        self.Q = defaultdict(lambda: defaultdict(float))

        # Keep a list of reward received at each episode
        self.rewards = []

    @abstractmethod
    def allowed_actions(self, s) -> list:
        pass

    def select_action(self, s):
        """
        Select an action given the current game state.

        Parameters
        ----------
        s : string
            state
        """
        # Only consider the allowed actions (empty board spaces)
        allowed_actions = self.allowed_actions(s)

        if random.random() < self.eps:
            # Random choice.
            action_idx = random.randrange(len(allowed_actions))
        else:
            # Greedy choice.
            # If multiple options are equally good, sample from them
            values = np.array([self.Q[s][a] for a in allowed_actions])
            idx_max, = np.where(values == np.max(values))
            action_idx = np.random.choice(idx_max) if len(idx_max) > 1 else idx_max[0]

        # select
        action = allowed_actions[action_idx]

        # update epsilon; geometric decay
        if self.eps_decay > 0:
            self.eps *= 1 - self.eps_decay

        return action

    @abstractmethod
    def update(self, s, s_, a, a_, r) -> None:
        """
        Perform the update of Q(s,a).

        Parameters
        ----------
        s : string
            previous state
        s_ : string
            new state
        a : (i,j) tuple
            previous action
        a_ : (i,j) tuple
            new action. NOT used by Q-learner!
        r : int
            reward received after executing action "a" in state "s"
        """
        pass

    def run(self, env: Environment, episodes=1, progress_bar=False, verbose=False):
        from tqdm import tqdm
        # def callback(it, s, a):
        #     if callbacks is not None:
        #         for callback_fn in callbacks:
        #             callback_fn(it, s, a)

        if verbose:
            callback = lambda it, s, a: print(f'it={it}\n{s}\n')
        else:
            callback = lambda *args, **kwargs: None

        s = env.initialize()
        a = self.select_action(s)

        with tqdm(total=episodes, ncols=120, disable=not progress_bar) as pbar:
            it = 0
            ep = 0
            metrics = defaultdict(list)
            while True:
                callback(it, s, a)

                # execute action and observe reward + new state
                r, s_ = env.execute(s, a)
                metrics['reward'].append(r)
                q_vals = np.array([q for qs in self.Q.values() for q in qs.values()])
                metrics['q_min'].append(q_vals.min())
                metrics['q_max'].append(q_vals.max())
                metrics['q_mean'].append(q_vals.mean())

                if env.is_terminal(s_):
                    # Perform final update
                    self.update(s, None, a, None, r)

                    #callback(it, s, a)
                    ep += 1
                    pbar.update(1)

                    if ep >= episodes:
                        break
                    else:
                        if verbose:
                            print('-------- RESTART --------\n')
                        #it = 0
                        s = env.initialize()
                        a = self.select_action(s)
                        continue

                # select new action
                a_ = self.select_action(s_)

                # update action-value function
                self.update(s, s_, a, a_, r)
                s, a = s_, a_
                it += 1

        return dict(metrics)


class TicTacToeAgent(Agent):
    # Possible actions correspond to the set of all x,y coordinate pairs
    actions = [(i,j) for i in range(3) for j in range(3)]

    def allowed_actions(self, s) -> list:
        # Allowed actions = empty board spaces in TicTacToe
        if isinstance(s, str):
            return [a for a in self.actions if s[a[0] * 3 + a[1]] == '-']
        return [a for a in self.actions if s[a[0]][a[1]] == '-']


class QLearningAgent(Agent):
    """ Implementation of the Q-learning agent """
    def update(self, s, s_, a, a_, r):
        """
        Perform the Q-Learning update of Q(s,a).

        Parameters
        ----------
        s : string
            previous state
        s_ : string
            new state
        a : (i,j) tuple
            previous action
        a_ : (i,j) tuple
            new action. NOT used by Q-learner!
        r : int
            reward received after executing action "a" in state "s"
        """
        if s_ is not None:
            maxQ = max(self.Q[s_][aa] for aa in self.allowed_actions(s_))
            self.Q[s][a] += self.alpha * (r + self.gamma * maxQ - self.Q[s][a])
        else:
            # terminal state update
            self.Q[s][a] += self.alpha * (r - self.Q[s][a])


class SarsaAgent(Agent):
    """ Implementation of the SARSA-learning agent """
    def update(self, s, s_, a, a_, r):
        """
        Perform the SARSA update of Q(s,a).

        Parameters
        ----------
        s : string
            previous state
        s_ : string
            new state
        a : (i,j) tuple
            previous action
        a_ : (i,j) tuple
            new action
        r : int
            reward received after executing action "a" in state "s"
        """
        if s_ is not None:
            self.Q[s][a] += self.alpha * (r + self.gamma * self.Q[s_][a_] - self.Q[s][a])
        else:
            # terminal state update
            self.Q[s][a] += self.alpha * (r - self.Q[s][a])


class TicTacToeQLearningAgent(QLearningAgent, TicTacToeAgent):
    pass


class TicTacToeSarsaAgent(SarsaAgent, TicTacToeAgent):
    pass


if __name__ == '__main__':
    #agent = TicTacToeAgent()
    agent = TicTacToeQLearningAgent()
    print(f'type(agent) = {type(agent).__name__}')
