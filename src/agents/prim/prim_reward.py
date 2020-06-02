from agents.reward import Reward
from agents.state import State
from agents.prim.prim_state import PrimState


class PrimReward(Reward):

    def __init__(self, alpha_1: float, alpha_2: float):
        self.__alpha_1 = alpha_1
        self.__alpha_2 = alpha_2

    def _forward(self, old: PrimState, new: PrimState) -> float:
        assert old is not None and new is not None
        if len(new) == 0:
            return -1
        iou = self.__iou(new) - self.__iou(old)
        iou_sum = self.__alpha_1 * (self.__iou_sum(new) - self.__iou_sum(old))
        parsimony = self.__alpha_2 * (self.__parsimony(new) - self.__parsimony(old))
        return iou + iou_sum + parsimony

    def __iou(self, s: PrimState) -> float:
        # This is a complex method. First approach: compute intersection and union of meshes.
        #raise NotImplementedError
        return 0.0

    def __iou_sum(self, s: PrimState) -> float:
        #raise NotImplementedError
        return 0.0

    def __parsimony(self, s: PrimState) -> float:
        return len(s)