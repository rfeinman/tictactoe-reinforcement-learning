from typing import Optional, Union, List, Tuple, TypeAlias
from abc import ABC, abstractmethod


class State(ABC):
    pass


class Action(ABC):
    pass


Reward: TypeAlias = float


class Environment(ABC):
    @abstractmethod
    def get_actions(self, s: Optional[State] = None) -> List[Action]:
        pass

    @abstractmethod
    def initialize(self) -> State:
        """ return initial state s """
        pass

    @abstractmethod
    def execute(self, s: State, a: Action) -> Tuple[Reward, State]:
        """ execute action a from state s, return reward r and new state s' """
        pass

    def is_terminal(self, s: State) -> bool:
        return False
