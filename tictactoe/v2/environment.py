import random
from abc import ABC, abstractmethod


class Environment(ABC):

    @abstractmethod
    def initialize(self):
        """ return initial state s """
        pass

    @abstractmethod
    def execute(self, s, a):
        """ execute action a from state s, return new state s' and reward r """
        pass

    def is_terminal(self, s):
        return False


# -------------------------------------------------------

class GameBoard(list):
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

    def update(self, loc, key: str) -> None:
        assert key in self._allowed_keys
        self[loc[0]][loc[1]] = key

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
    def __init__(self, player, use_string=False):
        #self.agent = agent
        self.player = player
        self.use_string = use_string

    def initialize(self, agent_first=None):
        """ return initial state s """
        if agent_first is None:
            agent_first = random.random() < 0.5

        board = GameBoard()
        if not agent_first:
            xy = self.player.makeMove(board)
            board.update(xy, 'X')

        return board.to_string() if self.use_string else board

    def is_terminal(self, state):
        board = GameBoard(state) if self.use_string else state

        game_over = (
            board.check_for_win('O')
            or board.check_for_win('X')
            or board.check_for_draw()
        )

        return game_over

    def execute(self, state, action):
        #board = GameBoard.from_string(state) if self.use_string else GameBoard(state)
        board = GameBoard(state)

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

        new_state = board.to_string() if self.use_string else board

        return reward, new_state

