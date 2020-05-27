from typing import List, Tuple
from agents.state import State
from agents.action import Action
from agents.reward import Reward


class Environment:

    def __init__(self, act_space: List[Action], reward: Reward):
        assert len(act_space) > 0 and reward is not None
        self.__current = None
        self.__act_space = act_space
        self.__reward = reward

    def transition(self, act: Action) -> Tuple[State, float]:
        assert self.__current is not None and act is not None
        res = act(self.__current)
        rew = self.__reward(self.__current, res)
        #self.set_state(res)
        return res, rew

    def get_state(self) -> State:
        assert self.__current is not None
        return self.__current

    def set_state(self, s: State) -> None:
        assert s is not None
        self.__current = s

    def get_action(self, idx: int) -> Action:
        assert idx > 0
        return self.__act_space[idx]