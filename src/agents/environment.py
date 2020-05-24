from agents.state import State
from agents.action import Action


class Environment:

    def __init__(self):
        self.__current = None

    def transition(self, act: Action) -> State:
        assert self.__current is not None
        return act(self.__current)

    def get_state(self) -> State:
        assert self.__current is not None
        return self.__current

    def set_state(self, s: State) -> None:
        self.__current = s