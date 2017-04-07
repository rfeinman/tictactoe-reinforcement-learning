# Tic-Tac-Toe with Reinforcement Learning
This is a repository for training an AI agent to play Tic-tac-toe using
reinforcement learning. Both the Sarsa learning and Q-learning RL
algorithms are implemented. A user may teach the agent himself by
playing the game for a couple of rounds, or he may apply an automated
teacher agent. 

## Code Structure

There are 3 main code files associated with this project:
1. game.py
1. agent.py
1. test.py

I have implemented two RL agents that learn to play the game of tic-tac-toe:
one follows the Sarsa learning algorithm, and the other follows Q-learning.
These agents are trained by a teacher agent that knows the optimal strategy;
however, the teacher only follows this strategy with a given probability
p at each turn. The rest of the time this teacher chooses randomly
from the moves that are available, so that the agents are able to win on
occasion and learn from these wins. To initialize the learning agent Q values,
I make use of default dictionaries with default values of 0 such that the
value for every state-action pair is initialized to 0.

The Q-learning agent, Sarsa-learning agent, and Teacher agent are all
implemented in agent.py. Each of the two learning agents inherit from a
parent learner class; the key difference between the two is their Q-value
update function. There is also a slight difference between the game loop
for the two, found in game.py. The teacher knows the optimal policy for
each state presented; however, this agent only takes the optimal choice
with a set probability (I have included RL reward plots for the cases of
p = 0.6, p = 0.7, and p = 0.9).

In game.py, the main game classes can be found. GameLearner holds the state
of the current game sequence, which will continue until the player choses
to stop or the teacher has hit its designated number of iterations. The Game
class holds the state of each particular game instance, and it contains the
majority of the main game functionality. The main game loop can be found in
the function playGame(), where the Q-learning and Sarsa-learning algorithms
are implemented for each game.

Unit tests can be found in test.py.

## Running the Program

### Train a new agent manually
To initialize a new agent and begin a new game loop, simply run:

    python game.py q                (Q-learner)
    python game.py s                (Sarsa-learner)

This will initialize the game and allow you to train the agent manually
by playing against the agent repeatedly. Be careful, however, as initializing
a new agent will delete the state of the previous agents that were stored for
the selected agent type. In the process of playing, you will be storing the
new agent state with each game iteration.

### Train a new agent automatically (via teacher agent)
To initialize a new RL agent and train it automatically with a teacher agent,
use the flag '-t' followed by the number of game iterations you would like to
train for:

    python game.py q -t 5000        (Q-learner)
    python game.py s -t 5000        (Sarsa-learner)

Again, be careful as this will overwrite previously-existing agents.

### Load a trained agent
To load existing agents and play against them, run:

    python game.py q -l             (Q-learner)
    python game.py s -l             (Sarsa-learner)

I have trained an instance of each the Q-learner and Sarsa-learner agents
and pickled them into .pkl files that are included here. These agents were
trained by a teacher of level 0.9 for 100000 episodes, and they have learned
to play considerably well. You can make use of these if you like, but they
will be overwritten if you have initialized new agents.

### Load a trained agent and train it further
You can train existing agents further by loading them and teaching them, via
a combination of '-t' and '-l':

    python game.py q -l -t 5000     (Q-learner)
    python game.py s -l -t 5000     (Sarsa-learner)

### Load a trained agent and view reward history plot
Finally, to load a stored agent and view a plot of its cumulative reward
history, use '-l' in combination with '-p':

    python game.py q -l -p          (Q-learner)
    python game.py s -l -p          (Sarsa-learner)
