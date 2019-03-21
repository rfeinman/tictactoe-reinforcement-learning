import random


class Game(object):
    """ The game class. New instance created for each new game. """
    def __init__(self, agent, teacher=None):
        self.computer = agent
        self.teacher = teacher
        # initialize the game board
        self.board = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]

    def printBoard(self):
        """ Prints the game board as text output to the terminal. """
        print('    0   1   2\n')
        row_num = 0
        for row in self.board:
            print('%i   ' % row_num, end='')
            for elt in row:
                print('%s   ' % elt, end='')
            print('\n')
            row_num += 1
        print('\n')

    def playerMove(self):
        """ Querry player for a move and update the board accordingly. """
        if self.teacher is not None:
            action = self.teacher.makeMove(self.board)
            self.board[action[0]][action[1]] = 'X'
        else:
            self.printBoard()
            while True:
                move = input("Your move! Please select a row and column from 0-2 "
                                 "in the format row,col: ")
                try:
                    row, col = int(move[0]), int(move[2])
                except ValueError:
                    print("INVALID INPUT! Please use the correct format.")
                    continue
                if row not in range(3) or col not in range(3) or not self.board[row][col] == '-':
                    print("INVALID MOVE! Choose again.")
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
        # check for player win on diagonals
        a = [self.board[0][0], self.board[1][1], self.board[2][2]]
        b = [self.board[0][2], self.board[1][1], self.board[2][0]]
        if a.count(key) == 3 or b.count(key) == 3:
            return True
        # check for player win on rows/columns
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
                    print("Player wins!")
                else:
                    print("RL agent wins!")
            return 1
        elif self.checkForDraw():
            if self.teacher is None:
                self.printBoard()
                print("It's a draw!")
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
        # Initialize the agent's state and action
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
        # Final update and save
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
                self.playGame(agent_type, False)
            else:
                self.playGame(agent_type, True)
        else:
            while True:
                response = input("Would you like to go first? [y/n]: ")
                if response == 'n' or response == 'no':
                    #self.playComputerFirst(agent_type)
                    self.playGame(agent_type, False)
                    break
                elif response == 'y' or response == 'yes':
                    #self.playPlayerFirst(agent_type)
                    self.playGame(agent_type, True)
                    break
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")

