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
        self.reward = r
        self.source = src
        self.destination = dest
        self.action = act