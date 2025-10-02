from typing import Optional, Union, Any, List, Tuple
import numpy as np
import random
from collections import defaultdict

from .environment import Environment


class Agent:
    """
    Class for Q-learning and SARSA agents.

    Parameters
    ----------
    sarsa : bool
        whether to use the SARSA update rule (vs Q-learning rule)
    alpha : float
        learning rate
    gamma : float
        temporal discounting rate
    eps : float
        probability of random action vs. greedy action
    eps_decay : float
        epsilon decay rate. Larger value = more decay
    """

    def __init__(self, sarsa=False, alpha=0.5, gamma=0.9, eps=0.1, eps_decay=0.0):
        # Agent parameters
        self.sarsa = sarsa
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay

        # Initialize Q values to 0 for all state-action pairs. Accessed as Q[s][a]
        self.Q = defaultdict(lambda: defaultdict(float))

        # Keep a list of reward received at each episode
        self.metrics = defaultdict(list)

    def select_action(self, s: Any, env: Environment) -> Tuple[Any, float]:
        allowed_actions = env.get_actions(s)

        # Greedy choice
        Q_vals = np.array([self.Q[s][aa] for aa in allowed_actions])
        Q_max = Q_vals.max()

        if random.random() < self.eps:
            # Random choice.
            a = random.choice(allowed_actions)
            #a = allowed_actions[random.randrange(len(allowed_actions))]
        else:
            # Greedy choice
            # If multiple options are equally good, sample from them
            idx_max, = np.where(Q_vals == Q_max)
            a = allowed_actions[random.choice(idx_max)]
            #a = allowed_actions[np.random.choice(idx_max) if len(idx_max) > 1 else idx_max[0]]

        # update epsilon; geometric decay
        if self.eps_decay > 0:
            self.eps *= 1 - self.eps_decay

        # keep track of important metrics (TODO: update this)
        self.metrics['q(s,a)'].append(self.Q[s][a])
        # qs = np.array([q for qs in self.Q.values() for q in qs.values()])
        # #qs = np.array([self.Q[s][a] for s in self.Q.keys() for a in env.get_actions(s)])
        # metrics['q_min'].append(qs.min())
        # metrics['q_max'].append(qs.max())
        # metrics['q_mean'].append(qs.mean())

        return a, float(Q_max)

    def update(
            self,
            s: Any,
            a: Any,
            r: float,
            *,
            s_: Optional[Any] = None,
            a_: Optional[Any] = None,
            Q_max: Optional[float] = None,
            ) -> None:
        """Perform the update of Q(s,a)

        Parameters
        ----------
        s : string
            previous state
        a : (i,j) tuple
            previous action
        r : float
            reward received after executing action "a" in state "s"
        s_ : string
            new state
        a_ : (i,j) tuple
            new action. Only used by SARSA agent (sarsa=True)
        Q_max : float
            Q value of the greedy action from state s_. Only used
            by Q-learning agent (sarsa=False)
        """
        assert (s_ is not None) == (a_ is not None) == (Q_max is not None)

        if s_ is None:
            # Perform the *terminal* update of Q(s, a).
            # A terminal update occurs when, after taking action `a` from
            # state `s`, we land in a *terminal* state `s_`.
            self.Q[s][a] += self.alpha * (r - self.Q[s][a])
        elif self.sarsa:
            self.Q[s][a] += self.alpha * (r + self.gamma * self.Q[s_][a_] - self.Q[s][a])
        else:
            self.Q[s][a] += self.alpha * (r + self.gamma * Q_max - self.Q[s][a])

        self.metrics['reward'].append(r)

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
        a, _ = self.select_action(s, env)

        with tqdm(total=episodes, ncols=120, disable=not progress_bar) as pbar:
            it = 0
            ep = 0
            while True:
                callback(it, s, a)

                # execute action and observe reward + new state
                r, s_ = env.execute(s, a)

                if env.is_terminal(s_):
                    # Perform final update
                    self.update(s, a, r)

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
                        a, _ = self.select_action(s, env)
                        continue

                # select new action
                a_, Q_max = self.select_action(s_, env)
                self.update(s, a, r, s_=s_, a_=a_, Q_max=Q_max)

                # update action-value function
                s, a = s_, a_
                it += 1


if __name__ == '__main__':
    agent = Agent()
    print(f'type(agent) = {type(agent).__name__}')
