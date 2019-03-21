import argparse
import os
import pickle
import sys
import matplotlib.pylab as plt

from tictactoe.agent import QLearner, SarsaLearner
from tictactoe.teacher import Teacher
from tictactoe.game import Game


def plot_agent_reward(rewards, agent_type):
    """ Function to plot agent's accumulated reward vs. episode """
    plt.plot(rewards)
    if agent_type == 'q':
        plt.title('Q-Learning Agent Cumulative Reward vs. Episode')
    else:
        plt.title('Sarsa Agent Cumulative Reward vs. Episode')
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
            if args.learner_type == 'q':
                # QLearner
                try:
                    f = open('./qlearner_agent.pkl','rb')
                except IOError:
                    print("The agent file does not exist. Quitting.")
                    sys.exit(0)
                self.type = 'q'
            else:
                # SarsaLearner
                try:
                    f = open('./sarsa_agent.pkl','rb')
                except IOError:
                    print("The agent file does not exist. Quitting.")
                    sys.exit(0)
                self.type = 's'
            self.agent = pickle.load(f)
            f.close()
            # If plotting, show plot and quit
            if args.plot:
                plot_agent_reward(self.agent.rewards, self.type)
                sys.exit(0)
        else:
            # check if agent state file already exists, and ask user whether to overwrite if so
            if ((args.learner_type == "q" and os.path.isfile('./qlearner_agent.pkl')) or
                    (args.learner_type == "s" and os.path.isfile('./qlearner_agent.pkl'))):
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
            if args.learner_type == "q":
                self.agent = QLearner(alpha,gamma,epsilon)
                self.type = 'q'
            else:
                self.agent = SarsaLearner(alpha,gamma,epsilon)
                self.type = 's'

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
            game.start(self.type)
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
            game.start(self.type)
            self.games_played += 1
            # Monitor progress
            if self.games_played % 500 == 0:
                print("Games played: %i" % self.games_played)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Play Tic-Tac-Toe.")
    parser.add_argument("learner_type", help="Specify the computer agent learning algorithm."
                                             "'q' for Q-learning and 's' for Sarsa-learning",
                        type=str, default="q")
    parser.add_argument("-l", "--load", help="load trained agent", action="store_true")
    parser.add_argument("-t", "--teacher", help="employ teacher agent who knows the optimal strategy",
                        default=None, type=int)
    parser.add_argument("-p", "--plot", help="plot reward vs. episode of stored agent and quit",
                        action="store_true")
    args = parser.parse_args()
    assert args.learner_type == 'q' or args.learner_type == 's', \
        "learner type must be either 'q' or 's'."
    if args.plot:
        assert args.load, "Must load an agent to plot reward."
        assert args.teacher is None, \
            "Cannot plot and teach concurrently; must chose one or the other."

    gl = GameLearning(args)
    if args.teacher is not None:
        gl.beginTeaching(args.teacher)
    else:
        gl.beginPlaying()