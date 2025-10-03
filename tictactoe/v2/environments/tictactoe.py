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

        return reward, board
