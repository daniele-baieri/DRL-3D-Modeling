from agents.reward import Reward
from agents.state import State


class PrimReward(Reward):

    def __init__(self, alpha_1: float, alpha_2: float):
        self.__alpha_1 = alpha_1
        self.__alpha_2 = alpha_2

    def _forward(self, old: State, new: State) -> float:
        assert old is not None and new is not None
        iou = self.__iou(new) - self.__iou(old)
        iou_sum = self.__alpha_1 * (self.__iou_sum(new) - self.__iou_sum(old))
        parsimony = self.__alpha_2 * (self.__parsimony(new) - self.__parsimony(old))
        return iou + iou_sum + parsimony

    def __iou(self, s: State) -> float:
        raise NotImplementedError

    def __iou_sum(self, s: State) -> float:
        raise NotImplementedError

    def __parsimony(self, s: State) -> float:
        raise NotImplementedError