#from trimesh import Trimesh
#from trimesh.boolean import union, intersection
import torch
import math
from agents.reward import Reward
from agents.state import State
from agents.prim.prim_state import PrimState
from agents.prim.prim_model import PrimModel
from agents.base_model import BaseModel


class PrimReward(Reward):

    def __init__(self, alpha_1: float, alpha_2: float):
        self.__alpha_1 = alpha_1
        self.__alpha_2 = alpha_2
        self.__single_vox_volume = math.pow(PrimState.voxel_side, 3)
        self.__valid_cache = False

    def _forward(self, old: PrimState, new: PrimState) -> float:
        assert old is not None and new is not None
        if len(new) == 0:
            return -1
        self.__voxels_cache = new.voxelize(cubes=True)
        self.__valid_cache = True
        iou_new, iou_sum_new = self.iou(new), self.iou_sum(new)

        self.__voxels_cache = old.voxelize(cubes=True)
        iou_old, iou_sum_old = self.iou(old), self.iou_sum(old)
        self.__valid_cache = False

        iou = iou_new - iou_old
        iou_sum = self.__alpha_1 * (iou_sum_new - iou_sum_old)
        parsimony = self.__alpha_2 * (self.parsimony(new) - self.parsimony(old))
        return iou + iou_sum + parsimony

    def iou(self, s: PrimState) -> float:
        state = None
        if self.__valid_cache:
            state = torch.unique(torch.cat(self.__voxels_cache), sorted=False)
        else:
            state = s.voxelize(cubes=False)
        target = BaseModel.get_model()
        return self.__compute_iou(target, state)

    def iou_sum(self, s: PrimState) -> float:
        state = None
        if self.__valid_cache:
            state = self.__voxels_cache
        else:
            state = s.voxelize(cubes=True)
        #P = s.get_primitives()
        target = BaseModel.get_model()
        res = sum(self.__compute_iou(c, target) for c in state)
        return res / len(state)

    def parsimony(self, s: PrimState) -> float:
        P = PrimState.num_primitives
        return P - len(s)

    def __compute_iou(self, m1: torch.LongTensor, m2: torch.LongTensor) -> float:
        s1, s2 = set(m1.tolist()), set(m2.tolist())
        i = len(s1.intersection(s2)) * self.__single_vox_volume
        u = len(s1.union(s2)) * self.__single_vox_volume
        
        return i / u