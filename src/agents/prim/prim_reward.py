from trimesh import Trimesh
from trimesh.boolean import union, intersection

from agents.reward import Reward
from agents.state import State
from agents.prim.prim_state import PrimState
from agents.prim.prim_model import PrimModel
from agents.base_model import BaseModel


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
        # NOTE: this may be *VERY* slow. If it is infeasible, turn to voxels
        target = BaseModel.get_model()
        curr = s.meshify()
        return self.__compute_iou(target, curr)

    def __iou_sum(self, s: PrimState) -> float:
        # NOTE: this may be *EVEN SLOWER*.
        P = s.get_primitives()
        target = BaseModel.get_model()
        res = sum(P, lambda c: self.__compute_iou(c.get_mesh(), target))
        return res / len(P)

    def __parsimony(self, s: PrimState) -> float:
        P = PrimState.num_primitives
        return P - len(s)

    def __compute_iou(self, m1: Trimesh, m2: Trimesh) -> float:
        v_int = intersection([m1, m2], engine='scad').volume
        v_union = union([m1, m2], engine='scad').volume
        return v_int / v_union