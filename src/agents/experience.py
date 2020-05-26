from collections import namedtuple
from agents.state import State
from agents.action import Action


class Experience:

    def __init__(self, src: State, dest: State, act: Action, r: float):
        """
        @param src: State where 'act' was applied
        @param dest: State resulting from applying 'act' in 'src'
        @param act: Action leading from 'src' to 'dest'
        @param r: reward derived from transitioning from 'src' to 'dst'
        """
        self.__reward = r
        self.__source = src
        self.__destination = dest
        self.__action = act

    def __repr__(self) -> str:
        return '(src: ' + repr(self.__source) + \
            ', dest: ' + repr(self.__destination) + \
            ', action: ' + repr(self.__action) + \
            ', reward: ' + str(self.__reward) + ')'

    def get_source(self) -> State:
        return self.__source

    def get_destination(self) -> State:
        return self.__destination

    def get_reward(self) -> float:
        return self.__reward

    def get_action(self) -> Action:
        return self.__action