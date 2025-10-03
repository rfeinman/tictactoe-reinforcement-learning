from typing import Optional, Union, List, Tuple
import numbers
import random
from abc import ABC, abstractmethod

from .base import Environment, State, Action, Reward


# -------------------------------------------------------

class TicTacToeAction(list, Action):
    def __init__(self, *args):
        data = args[0] if len(args) == 1 else args
        assert len(data) == 2
        assert all(isinstance(coord, numbers.Integral) for coord in data)
        assert all(0 <= coord < 3 for coord in data)
        super().__init__(data)

    def __hash__(self):
        return hash(tuple(self))


class TicTacToeState(list, State):
    _allowed_keys = {'-', 'X', 'O'}

    def __init__(self, data=None):
        """
        Parameters
        ----------
        data : list of lists
            The current game board
        """
        if data is None:
            data = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]
        elif isinstance(data, str):
            assert len(data) == 9
            assert all(elt in self._allowed_keys for elt in data)
            data = [[data[i * 3 + j] for j in range(3)] for i in range(3)]
        else:
            #data = [list(row) for row in data]
            data = list(map(list, data))
            assert len(data) == 3
            for row in data:
                assert len(row) == 3
                assert all(elt in self._allowed_keys for elt in row)

        super().__init__(data)

    def to_string(self) -> str:
        """
        Converts 2D list representing the board state into a string key
        for that state. Keys are used for Q-value hashing.
        """
        return ''.join(elt for row in self for elt in row)

    def __hash__(self):
        data = tuple(tuple(row) for row in self)
        #data = self.to_string()
        return hash(data)

    def update(self, loc: Union[TicTacToeAction, list, tuple], key: str) -> None:
        assert key in self._allowed_keys
        i, j = loc
        self[i][j] = key

    def check_for_win(self, key: str) -> bool:
        """
        Check to see whether the player/agent with token 'key' has won.
        Returns a boolean holding truth value.

        Parameters
        ----------
        key : string
            token of most recent player. Either 'O' or 'X'
        """
        assert key in self._allowed_keys.difference(['-'])

        # check for player win on diagonals
        a = [self[0][0], self[1][1], self[2][2]]
        b = [self[0][2], self[1][1], self[2][0]]
        if a.count(key) == 3 or b.count(key) == 3:
            return True
        # check for player win on rows/columns
        for i in range(3):
            col = [self[0][i], self[1][i], self[2][i]]
            row = [self[i][0], self[i][1], self[i][2]]
            if col.count(key) == 3 or row.count(key) == 3:
                return True
        return False

    def check_for_draw(self) -> bool:
        """
        Check to see whether the game has ended in a draw. Returns a
        boolean holding truth value.
        """
        draw = True
        for row in self:
            for elt in row:
                if elt == '-':
                    draw = False
        return draw

    def __repr__(self):
        """ Format the game board as text output for terminal """
        # lines = []
        # for i, row in enumerate(self):
        #     lines.append(' '.join(row))
        # s = '\n'.join(lines)

        lines = [repr(row) for row in self]
        s = f'[{lines[0]},\n'
        s += f' {lines[1]},\n'
        s += f' {lines[2]}]'

        return s


class TicTacToe(Environment):
    """ The game class. New instance created for each new game. """

    # NOTE: for now, player == teacher
    def __init__(self, player):
        self.player = player

    def get_actions(self, s: Optional[TicTacToeState] = None) -> list:
        # Possible actions: set of all (x,y) grid locations
        actions = [TicTacToeAction(i,j) for i in range(3) for j in range(3)]

        # Allowed actions: empty board spaces in TicTacToe
        if s is not None:
            if isinstance(s, str):
                actions = [a for a in actions if s[a[0] * 3 + a[1]] == '-']
            else:
                actions = [a for a in actions if s[a[0]][a[1]] == '-']

        return actions

    def initialize(self, agent_first: Optional[bool] = None) -> TicTacToeState:
        """ return initial state s """
        if agent_first is None:
            agent_first = random.random() < 0.5

        board = TicTacToeState()
        if not agent_first:
            xy = self.player.makeMove(board)
            board.update(xy, 'X')

        initial_state = board

        return initial_state

    def is_terminal(self, board: TicTacToeState) -> bool:
        return (
            board.check_for_win('O')
            or board.check_for_win('X')
            or board.check_for_draw()
        )

    def execute(self, board: TicTacToeState, action: TicTacToeAction) -> Tuple[Reward, TicTacToeState]:
        # NOTE: always make a copy to avoid modifying `state`
        board = TicTacToeState(board)

        # execute oldAction, observe reward and state
        board.update(action, 'O')

        if board.check_for_win('O'):
            # game is over (win). +1 reward
            reward = 1
            game_over = True
        elif board.check_for_draw():
            # game is over (draw). 0 reward
            reward = 0
            game_over = True
        else:
            xy = self.player.makeMove(board)
            board.update(xy, 'X')

            if board.check_for_win('X'):
                # game is over (lose). -1 reward
                reward = -1
                game_over = True
            elif board.check_for_draw():
                # game is over (draw). 0 reward
                reward = 0
                game_over = True
            else:
                # game continues. 0 reward
                reward = 0
                game_over = False

        if game_over and isinstance(self.player, Human):
            result = 'Draw' if reward == 0 else ('Player wins' if reward < 0 else 'Agent wins')
            print(f'Game over! Final result: {result}\n')
            self.player.printBoard(board)

        return reward, board


# -------------------------------------------------------

class Player(ABC):
    @abstractmethod
    def makeMove(self, board: TicTacToeState) -> TicTacToeAction:
        pass


class Teacher(Player):
    """
    A class to implement a teacher that knows the optimal playing strategy.
    Teacher returns the best move at any time given the current state of the game.
    Note: things are a bit more hard-coded here, as this was not the main focus of
    the exercise so I did not spend as much time on design/style. Everything works
    properly when tested.
    """

    def __init__(self, level=0.9):
        """
        Parameters
        ----------
        level : float
            Teacher ability level; a value between 0-1 that indicates the
            probability of making the optimal move at any given time.
        """
        self.ability_level = float(level)
        assert 0 <= self.ability_level <= 1

    def win(self, board, key='X'):
        """ If we have two in a row and the 3rd is available, take it. """
        # Check for diagonal wins
        a = [board[0][0], board[1][1], board[2][2]]
        b = [board[0][2], board[1][1], board[2][0]]
        if a.count('-') == 1 and a.count(key) == 2:
            ind = a.index('-')
            return ind, ind
        elif b.count('-') == 1 and b.count(key) == 2:
            ind = b.index('-')
            if ind == 0:
                return 0, 2
            elif ind == 1:
                return 1, 1
            else:
                return 2, 0
        # Now check for 2 in a row/column + empty 3rd
        for i in range(3):
            c = [board[0][i], board[1][i], board[2][i]]
            d = [board[i][0], board[i][1], board[i][2]]
            if c.count('-') == 1 and c.count(key) == 2:
                ind = c.index('-')
                return ind, i
            elif d.count('-') == 1 and d.count(key) == 2:
                ind = d.index('-')
                return i, ind
        return None

    def blockWin(self, board):
        """ Block the opponent if she has a win available. """
        return self.win(board, key='O')

    def fork(self, board):
        """ Create a fork opportunity such that we have 2 threats to win. """
        # Check all adjacent side middles
        if board[1][0] == 'X' and board[0][1] == 'X':
            if board[0][0] == '-' and board[2][0] == '-' and board[0][2] == '-':
                return 0, 0
            elif board[1][1] == '-' and board[2][1] == '-' and board[1][2] == '-':
                return 1, 1
        elif board[1][0] == 'X' and board[2][1] == 'X':
            if board[2][0] == '-' and board[0][0] == '-' and board[2][2] == '-':
                return 2, 0
            elif board[1][1] == '-' and board[0][1] == '-' and board[1][2] == '-':
                return 1, 1
        elif board[2][1] == 'X' and board[1][2] == 'X':
            if board[2][2] == '-' and board[2][0] == '-' and board[0][2] == '-':
                return 2, 2
            elif board[1][1] == '-' and board[1][0] == '-' and board[0][1] == '-':
                return 1, 1
        elif board[1][2] == 'X' and board[0][1] == 'X':
            if board[0][2] == '-' and board[0][0] == '-' and board[2][2] == '-':
                return 0, 2
            elif board[1][1] == '-' and board[1][0] == '-' and board[2][1] == '-':
                return 1, 1
        # Check all cross corners
        elif board[0][0] == 'X' and board[2][2] == 'X':
            if board[1][0] == '-' and board[2][1] == '-' and board[2][0] == '-':
                return 2, 0
            elif board[0][1] == '-' and board[1][2] == '-' and board[0][2] == '-':
                return 0, 2
        elif board[2][0] == 'X' and board[0][2] == 'X':
            if board[2][1] == '-' and board[1][2] == '-' and board[2][2] == '-':
                return 2, 2
            elif board[1][0] == '-' and board[0][1] == '-' and board[0][0] == '-':
                return 0, 0
        return None

    def blockFork(self, board):
        """ Block the opponents fork if she has one available. """
        corners = [board[0][0], board[2][0], board[0][2], board[2][2]]
        # Check all adjacent side middles
        if board[1][0] == 'O' and board[0][1] == 'O':
            if board[0][0] == '-' and board[2][0] == '-' and board[0][2] == '-':
                return 0, 0
            elif board[1][1] == '-' and board[2][1] == '-' and board[1][2] == '-':
                return 1, 1
        elif board[1][0] == 'O' and board[2][1] == 'O':
            if board[2][0] == '-' and board[0][0] == '-' and board[2][2] == '-':
                return 2, 0
            elif board[1][1] == '-' and board[0][1] == '-' and board[1][2] == '-':
                return 1, 1
        elif board[2][1] == 'O' and board[1][2] == 'O':
            if board[2][2] == '-' and board[2][0] == '-' and board[0][2] == '-':
                return 2, 2
            elif board[1][1] == '-' and board[1][0] == '-' and board[0][1] == '-':
                return 1, 1
        elif board[1][2] == 'O' and board[0][1] == 'O':
            if board[0][2] == '-' and board[0][0] == '-' and board[2][2] == '-':
                return 0, 2
            elif board[1][1] == '-' and board[1][0] == '-' and board[2][1] == '-':
                return 1, 1
        # Check all cross corners (first check for double fork opp using the corners array)
        elif corners.count('-') == 1 and corners.count('O') == 2:
            return 1, 2
        elif board[0][0] == 'O' and board[2][2] == 'O':
            if board[1][0] == '-' and board[2][1] == '-' and board[2][0] == '-':
                return 2, 0
            elif board[0][1] == '-' and board[1][2] == '-' and board[0][2] == '-':
                return 0, 2
        elif board[2][0] == 'O' and board[0][2] == 'O':
            if board[2][1] == '-' and board[1][2] == '-' and board[2][2] == '-':
                return 2, 2
            elif board[1][0] == '-' and board[0][1] == '-' and board[0][0] == '-':
                return 0, 0
        return None

    def center(self, board):
        """ Pick the center if it is available. """
        if board[1][1] == '-':
            return 1, 1
        return None

    def corner(self, board):
        """ Pick a corner move. """
        # Pick opposite corner of opponent if available
        if board[0][0] == 'O' and board[2][2] == '-':
            return 2, 2
        elif board[2][0] == 'O' and board[0][2] == '-':
            return 0, 2
        elif board[0][2] == 'O' and board[2][0] == '-':
            return 2, 0
        elif board[2][2] == 'O' and board[0][0] == '-':
            return 0, 0
        # Pick any corner if no opposites are available
        elif board[0][0] == '-':
            return 0, 0
        elif board[2][0] == '-':
            return 2, 0
        elif board[0][2] == '-':
            return 0, 2
        elif board[2][2] == '-':
            return 2, 2
        return None

    def sideEmpty(self, board):
        """ Pick an empty side. """
        if board[1][0] == '-':
            return 1, 0
        elif board[2][1] == '-':
            return 2, 1
        elif board[1][2] == '-':
            return 1, 2
        elif board[0][1] == '-':
            return 0, 1
        return None

    def randomMove(self, board):
        """ Chose a random move from the available options. """
        possibles = []
        for i in range(3):
            for j in range(3):
                if board[i][j] == '-':
                    possibles += [(i, j)]
        return possibles[random.randint(0, len(possibles)-1)]

    def makeMove(self, board):
        """
        Trainer goes through a hierarchy of moves, making the best move that
        is currently available each time. A touple is returned that represents
        (row, col).
        """
        # Chose randomly with some probability so that the teacher does not always win
        if random.random() > self.ability_level:
            return self.randomMove(board)
        # Follow optimal strategy
        a = self.win(board)
        if a is not None:
            return a
        a = self.blockWin(board)
        if a is not None:
            return a
        a = self.fork(board)
        if a is not None:
            return a
        a = self.blockFork(board)
        if a is not None:
            return a
        a = self.center(board)
        if a is not None:
            return a
        a = self.corner(board)
        if a is not None:
            return a
        a = self.sideEmpty(board)
        if a is not None:
            return a
        return self.randomMove(board)


class Human(Player):

    def printBoard(self, board: TicTacToeState) -> None:
        """
        Prints the game board as text output to the terminal.

        Parameters
        ----------
        board : list of lists
            the current game board
        """
        print('    0   1   2\n')
        for i, row in enumerate(board):
            print('%i   ' % i, end='')
            for elt in row:
                print('%s   ' % elt, end='')
            print('\n')

    def makeMove(self, board: TicTacToeState) -> TicTacToeAction:
        self.printBoard(board)

        while True:
            move = input(
                "Your move! Please select a row and column from 0-2 "
                "in the format row,col: "
            )
            print('\n')

            try:
                row, col = int(move[0]), int(move[2])
            except ValueError:
                print("INVALID INPUT! Please use the correct format.")
                continue

            if row not in range(3) or col not in range(3) or not board[row][col] == '-':
                print("INVALID MOVE! Choose again.")
                continue

            return TicTacToeAction(row, col)
