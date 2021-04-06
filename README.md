# Tic-Tac-Toe with Reinforcement Learning
This is a repository for training an AI agent to play Tic-tac-toe using reinforcement learning. Both the SARSA and Q-learning RL algorithms are implemented. A user may teach the agent themself by playing against it or apply an automated teacher agent. 

## Code Structure

#### Source Code

The directory `tictactoe` contains the core source code for this project.
There are 3 main source code files:
1. game.py
2. agent.py
3. teacher.py

I have implemented two RL agents that learn to play the game of tic-tac-toe:
one follows the SARSA algorithm, and the other follows Q-learning.
These agents are trained by a teacher agent that knows the optimal strategy;
however, the teacher only follows this strategy with a given probability
p at each turn. The rest of the time this teacher chooses randomly
from the moves that are available, so that the agents are able to win on
occasion and learn from these wins. To initialize the learning agent Q values,
I make use of default dictionaries with default values of 0 such that the
value for every state-action pair is initialized to 0.

The Q-learning and SARSA agents are implemented in `agent.py`.
Each of the two learning agents inherit from a parent learner class; the key difference between the two is their Q-value update function. 

The Teacher agent is implemented in `teacher.py`. 
The teacher knows the optimal policy for each state presented; however, this agent only takes the optimal choice with a set probability.

In `game.py`, the main game class is found. 
The Game class holds the state of each particular game instance, and it contains the majority of the main game functionality. 
The main game loop can be found in the class's function playGame().

#### Game Script

To play the game (see "Running the Program" below for instructions) you will use the script called `play.py`.
The GameLearner class holds the state of the current game sequence, which will continue until the player choses to stop or the teacher has finished the designated number of episodes.
See instructions below on how to use this script.

## Running the Program

#### Train a new agent manually
To initialize a new agent and begin a game loop, simply run:

    python play.py -a q                (Q-learner)
    python play.py -a s                (Sarsa-learner)

This will initialize the game and allow you to train the agent manually by playing against the agent yourself. In the process of playing, you will be storing the new agent state with each game iteration. Use the argument `-p` to specify a path where the agent pickle should be saved:


    python play.py -a q -p my_agent_path.pkl


When unspecified, the path is set to either "q_agent.pkl" or "sarsa_agent.pkl" depending on agent type. If the file already exists, you'll be asked to overwrite.

#### Train a new agent automatically via teacher
To initialize a new RL agent and train it automatically with a teacher agent, use the flag `-t` followed by the number of game iterations you would like to train for:

    python play.py -a q -t 5000

Again, specify the pickle save path with the `-p` option.

#### Load an existing agent and continue training
To load an existing agent and continue training, use the `-l` flag:

    python play.py -a q -l             (load agent and train manually)
    python play.py -a q -l -t 5000     (load agent and train via teacher)

The agent will continue to learn and its pickle file will be overwritten. 

For this use case, the argument `-a` is only used to define a default agent path (if not specified by `-p`); otherwise, the agent type is determined by the contents of the loaded pickle.


#### Load a trained agent and view reward history plot
Finally, to load a stored agent and view a plot of its cumulative reward history, use the script plot_agent_reward.py:

    python plot_agent_reward.py -p q_agent.pkl
