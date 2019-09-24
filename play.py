import argparse
import os
import pickle
import sys
import numpy as np
import matplotlib.pylab as plt

from tictactoe.agent import Qlearner, SARSAlearner
from tictactoe.teacher import Teacher
from tictactoe.game import Game


def plot_agent_reward(rewards):
    """ Function to plot agent's accumulated reward vs. iteration """
    plt.plot(np.cumsum(rewards))
    plt.title('Agent Cumulative Reward vs. Iteration')
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.show()

class GameLearning(object):
    """
    A class that holds the state of the learning process. Learning
    agents are created/loaded here, and a count is kept of the
    games that have been played.
    """
    def __init__(self, args, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.games_played = 0

        if args.load:
            # load agent
            if args.agent_type == 'q':
                # QLearner
                try:
                    f = open('./qlearner_agent.pkl','rb')
                except IOError:
                    print("The agent file does not exist. Quitting.")
                    sys.exit(0)
            else:
                # SarsaLearner
                try:
                    f = open('./sarsa_agent.pkl','rb')
                except IOError:
                    print("The agent file does not exist. Quitting.")
                    sys.exit(0)
            self.agent = pickle.load(f)
            f.close()
            # If plotting, show plot and quit
            if args.plot:
                plot_agent_reward(self.agent.rewards)
                sys.exit(0)
        else:
            # check if agent state file already exists, and ask user whether to overwrite if so
            if ((args.agent_type == "q" and os.path.isfile('./qlearner_agent.pkl')) or
                    (args.agent_type == "s" and os.path.isfile('./qlearner_agent.pkl'))):
                while True:
                    response = input("An agent state is already saved for this type. "
                                         "Are you sure you want to overwrite? [y/n]: ")
                    if response == 'y' or response == 'yes':
                        break
                    elif response == 'n' or response == 'no':
                        print("OK. Quitting.")
                        sys.exit(0)
                    else:
                        print("Invalid input. Please choose 'y' or 'n'.")
            if args.agent_type == "q":
                self.agent = Qlearner(alpha,gamma,epsilon)
            else:
                self.agent = SARSAlearner(alpha,gamma,epsilon)

    def beginPlaying(self):
        """ Loop through game iterations with a human player. """
        print("Welcome to Tic-Tac-Toe. You are 'X' and the computer is 'O'.")

        def play_again():
            print("Games played: %i" % self.games_played)
            while True:
                play = input("Do you want to play again? [y/n]: ")
                if play == 'y' or play == 'yes':
                    return True
                elif play == 'n' or play == 'no':
                    return False
                else:
                    print("Invalid input. Please choose 'y' or 'n'.")

        while True:
            game = Game(self.agent)
            game.start()
            self.games_played += 1
            if not play_again():
                print("OK. Quitting.")
                break

    def beginTeaching(self, episodes):
        """ Loop through game iterations with a teaching agent. """
        teacher = Teacher()
        # Train for alotted number of episodes
        while self.games_played < episodes:
            game = Game(self.agent, teacher=teacher)
            game.start()
            self.games_played += 1
            # Monitor progress
            if self.games_played % 1000 == 0:
                print("Games played: %i" % self.games_played)

        plot_agent_reward(self.agent.rewards)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Play Tic-Tac-Toe.")
    parser.add_argument('-a', "--agent_type", type=str, default="q",
                        help="Specify the computer agent learning algorithm. "
                             "AGENT_TYPE='q' for Q-learning and ='s' for Sarsa-learning")
    parser.add_argument("-l", "--load", action="store_true",
                        help="whether to load trained agent")
    parser.add_argument("-t", "--teacher_episodes", default=None, type=int,
                        help="employ teacher agent who knows the optimal "
                             "strategy and will play for TEACHER_EPISODES games")
    parser.add_argument("-p", "--plot", action="store_true",
                        help="whether to plot reward vs. episode of stored agent "
                             "and quit")
    args = parser.parse_args()
    assert args.agent_type == 'q' or args.agent_type == 's', \
        "learner type must be either 'q' or 's'."
    if args.plot:
        assert args.load, "Must load an agent to plot reward."
        assert args.teacher_episodes is None, \
            "Cannot plot and teach concurrently; must chose one or the other."

    gl = GameLearning(args)
    if args.teacher_episodes is not None:
        gl.beginTeaching(args.teacher_episodes)
    else:
        gl.beginPlaying()