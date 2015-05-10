import sys
import os
import time
import random
import argparse
import cPickle

import matplotlib.pylab as plt

from agent import QLearner, SarsaLearner, Teacher

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

class Game:
    """ The game class. New instance created for each new game. """
    def __init__(self, agent, teacher=None):
        self.computer = agent
        self.teacher = teacher
        # initialize the game board
        self.board = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]

    def printBoard(self):
        """ Prints the game board as text output to the terminal. """
        print "    0   1   2\n"
        row_num = 0
        for row in self.board:
            print row_num, ' ',
            for elt in row:
                print elt, ' ',
            print '\n'
            row_num += 1
        print '\n'

    def playerMove(self):
        """ Querry player for a move and update the board accordingly. """
        if self.teacher is not None:
            action = self.teacher.makeMove(self.board)
            self.board[action[0]][action[1]] = 'X'
        else:
            self.printBoard()
            while True:
                move = raw_input("Your move! Please select a row and column from 0-2 "\
                                    "in the format row,col: ")
                try:
                    row, col = int(move[0]), int(move[2])
                except ValueError:
                    print "INVALID INPUT! Please use the correct format."
                    continue
                if row not in range(3) or col not in range(3) or not self.board[row][col] == '-':
                    print "INVALID MOVE! Choose again."
                    continue
                self.board[row][col] = 'X'
                break

    def computerMove(self, action):
        """ Update board according to computer move. """
        self.board[action[0]][action[1]] = 'O'

    def checkForWin(self, key):
        """
        Check to see whether the player/agent with token 'key' has won.
        Returns a boolean holding truth value.
        """
        #check for player win on diagonals
        a = [self.board[0][0], self.board[1][1], self.board[2][2]]
        b = [self.board[0][2], self.board[1][1], self.board[2][0]]
        if a.count(key) == 3 or b.count(key) == 3:
            return True
        #check for player win on rows/columns
        for i in range(3):
            col = [self.board[0][i], self.board[1][i], self.board[2][i]]
            row = [self.board[i][0], self.board[i][1], self.board[i][2]]
            if col.count(key) == 3 or row.count(key) == 3:
                return True
        return False

    def checkForDraw(self):
        """
        Check to see whether the game has ended in a draw. Returns a
        boolean holding truth value.
        """
        draw = True
        for row in self.board:
            for elt in row:
                if elt == '-':
                    draw = False
        return draw

    def checkForEnd(self, key):
        """
        Checks if player/agent with token 'key' has ended the game. Returns -1
        if the game is still going, 0 if it is a draw, and 1 if the player/agent
        has won.
        """
        if self.checkForWin(key):
            if self.teacher is None:
                self.printBoard()
                if key == 'X':
                    print "Player wins!"
                else:
                    print "RL agent wins!"
            return 1
        elif self.checkForDraw():
            if self.teacher is None:
                self.printBoard()
                print "It's a draw!"
            return 0
        return -1

    def getStateKey(self):
        """
        Converts 2D list representing the board state into a string key
        for that state. Keys are used for Q-value hashing.
        """
        key = ''
        for row in self.board:
            for elt in row:
                key += elt
        return key

    def playGame(self, agent_type, player_first):
        """ Begin the tic-tac-toe game loop. """
        # initialize the agent's state and action
        if player_first:
            self.playerMove()
        oldState = self.getStateKey()
        if agent_type == 's':
            oldAction = self.computer.get_action(oldState)

        if agent_type == 'q':
            # Dealing with QLearner agent
            while True:
                action = self.computer.get_action(oldState)
                self.computerMove(action)
                check = self.checkForEnd('O')
                if not check == -1:
                    reward = check
                    break
                self.playerMove()
                state = self.getStateKey()
                check = self.checkForEnd('X')
                if not check == -1:
                    reward = -1*check
                    break
                else:
                    reward = 0
                self.computer.update(oldState, state, action, reward)
                oldState = state
        else:
            # Dealing with Sarsa agent
            while True:
                self.computerMove(oldAction)
                check = self.checkForEnd('O')
                if not check == -1:
                    reward = check
                    break
                self.playerMove()
                check = self.checkForEnd('X')
                if not check == -1:
                    reward = -1*check
                    break
                else:
                    reward = 0
                state = self.getStateKey()
                action = self.computer.get_action(state)
                self.computer.update(oldState, state, oldAction, action, reward)
                oldState = state
                oldAction = action

        self.computer.total_reward += reward
        self.computer.rewards += [self.computer.total_reward]
        # final update and save
        if agent_type == 'q':
            self.computer.update(oldState, None, action, reward)
            self.computer.save_agent('./qlearner_agent.pkl')
        else:
            self.computer.update(oldState, None, oldAction, None, reward)
            self.computer.save_agent('./sarsa_agent.pkl')

    def start(self, agent_type):
        """
        Function to determine how to play. Options include whether to employ
        teacher and whether to have computer or player go first.
        """
        if self.teacher is not None:
            # During teaching, chose who goes first randomly with equal probability
            if random.random() < 0.5:
                self.playGame(agent_type,False)
            else:
                self.playGame(agent_type,True)
        else:
            while True:
                response = raw_input("Would you like to go first? [y/n]: ")
                if response == 'n' or response == 'no':
                    #self.playComputerFirst(agent_type)
                    self.playGame(agent_type,False)
                    break
                elif response == 'y' or response == 'yes':
                    #self.playPlayerFirst(agent_type)
                    self.playGame(agent_type,True)
                    break
                else:
                    print "Invalid input. Please enter 'y' or 'n'."



class GameLearning:
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
                    print "The agent file does not exist. Quitting."
                    sys.exit(0)
                self.type = 'q'
            else:
                # SarsaLearner
                try:
                    f = open('./sarsa_agent.pkl','rb')
                except IOError:
                    print "The agent file does not exist. Quitting."
                    sys.exit(0)
                self.type = 's'
            self.agent = cPickle.load(f)
            f.close()
            # if plotting, show plot and quit
            if args.plot:
                plot_agent_reward(self.agent.rewards, self.type)
                sys.exit(0)
        else:
            # check if agent state file already exists, and ask user whether to overwrite if so
            if ((args.learner_type == "q" and os.path.isfile('./qlearner_agent.pkl')) or
                (args.learner_type == "s" and os.path.isfile('./qlearner_agent.pkl'))):
                while True:
                    response = raw_input("An agent state is already saved for this type. " \
                                        "Are you sure you want to overwrite? [y/n]: ")
                    if response == 'y' or response == 'yes':
                        break
                    elif response == 'n' or response == 'no':
                        print "OK. Quitting."
                        sys.exit(0)
                    else:
                        print "Invalid input. Please choose 'y' or 'n'."
            if args.learner_type == "q":
                self.agent = QLearner(alpha,gamma,epsilon)
                self.type = 'q'
            else:
                self.agent = SarsaLearner(alpha,gamma,epsilon)
                self.type = 's'

    def beginPlaying(self):
        """ Loop through game iterations with a human player. """
        print "Welcome to Tic-Tac-Toe. You are 'X' and the computer is 'O'."
        def play_again():
            print "Games played: ",self.games_played
            while True:
                play_again = raw_input("Do you want to play again? [y/n]: ")
                if play_again == 'y' or play_again == 'yes':
                    return True
                elif play_again == 'n' or play_again == 'no':
                    return False
                else:
                    print "Invalid input. Please choose 'y' or 'n'."
        while True:
            game = Game(self.agent)
            game.start(self.type)
            self.games_played += 1
            if not play_again():
                print "OK. Quitting."
                break

    def beginTeaching(self, episodes):
        """ Loop through game iterations with a teaching agent. """
        teacher = Teacher()
        # train for alotted number of episodes
        while self.games_played < episodes:
            game = Game(self.agent,teacher=teacher)
            game.start(self.type)
            self.games_played += 1
            # monitor progress
            if self.games_played % 500 == 0:
                print "Games played: ", self.games_played


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Play Tic-Tac-Toe.")
    parser.add_argument("learner_type", help="Specify the computer agent lerning algorithm.\
                        'q' for qlearning and 's' for sarsa", type=str, default="q")
    parser.add_argument("-l", "--load", help="load trained agent", action="store_true")
    parser.add_argument("-t", "--teacher", help="employ teacher agent who knows the optimal strategy", default=None, type=int)
    parser.add_argument("-p", "--plot", help="plot reward vs. episode of stored agent and quit", action="store_true")
    args = parser.parse_args()
    # agent type must be q or s
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

