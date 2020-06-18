import torch
import math, time
from typing import Tuple, Set
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
    '''
    def _forward(self, old: PrimState, new: PrimState) -> float:
        assert old is not None and new is not None
        if len(new) == 0:
            return -1
        target_voxels = set(BaseModel.get_model().tolist())

        self.__voxels_cache = new.voxelize(cubes=True)
        self.__valid_cache = True
        iou_new, iou_sum_new = self.iou(new, target_voxels), self.iou_sum(new, target_voxels)

        self.__voxels_cache = old.voxelize(cubes=True)
        iou_old, iou_sum_old = self.iou(old, target_voxels), self.iou_sum(old, target_voxels)
        self.__valid_cache = False

        iou = iou_new - iou_old
        iou_sum = self.__alpha_1 * (iou_sum_new - iou_sum_old)
        parsimony = self.__alpha_2 * (self.parsimony(new) - self.parsimony(old))
        return iou + iou_sum + parsimony

    def iou(self, s: PrimState, target: Set) -> float:
        state = None
        if self.__valid_cache:
            state = torch.unique(torch.cat(self.__voxels_cache), sorted=False)
        else:
            state = torch.unique(torch.cat(s.voxelize(cubes=True)), sorted=False)
        res = self.__compute_iou(target, set(state.tolist()))
        # print("IOU TIME: " + str(time.time() - t))
        return res

    def iou_sum(self, s: PrimState, target: Set) -> float:
        #t = time.time()
        state = None
        if self.__valid_cache:
            state = self.__voxels_cache
        else:
            state = s.voxelize(cubes=True)
        #target = BaseModel.get_model()
        res = sum(self.__compute_iou(set(c.tolist()), target) for c in state)
        #print("IOU SUM TIME: " + str(time.time() - t))
        return res / len(state)

    def parsimony(self, s: PrimState) -> float:
        P = PrimState.num_primitives
        return P - len(s)

    def __compute_iou(self, m1: Set, m2: Set) -> float:
        i = self.__volume_intersection(m1, m2)
        u = self.__volume_union(m1, m2)
        return i / u

    def __volume_intersection(self, s1: Set, s2: Set) -> float:
        return len(s1.intersection(s2)) * self.__single_vox_volume

    def __volume_union(self, s1: Set, s2: Set) -> float:
        return len(s1.union(s2)) * self.__single_vox_volume
    '''

    
    def _forward(self, old: PrimState, new: PrimState) -> float:
        assert old is not None and new is not None
        if len(new) == 0:
            return -1
        l = int(math.pow(PrimState.voxel_grid_side, 3))
        target = torch.zeros(l, dtype=torch.long)
        target[BaseModel.get_model()] = 1

        self.__voxels_cache = new.voxelize(cubes=True)
        self.__valid_cache = True
        iou_new, iou_sum_new = self.iou(new, target), self.iou_sum(new, target)

        self.__voxels_cache = old.voxelize(cubes=True)
        iou_old, iou_sum_old = self.iou(old, target), self.iou_sum(old, target)
        self.__valid_cache = False

        iou = iou_new - iou_old
        iou_sum = self.__alpha_1 * (iou_sum_new - iou_sum_old)
        parsimony = self.__alpha_2 * (self.parsimony(new) - self.parsimony(old))
        return iou + iou_sum + parsimony

    def iou(self, s: PrimState, target: torch.LongTensor) -> float:
        state = None
        if self.__valid_cache:
            state = torch.unique(torch.cat(self.__voxels_cache), sorted=False)
        else:
            state = torch.unique(torch.cat(s.voxelize(cubes=True)), sorted=False)
        #s.meshify().show()
        if torch.max(state) > 262144:
            s.meshify().show()
            return
        #t = time.time()
        res = self.__compute_iou(state, target)
        #print("IOU TIME: " + str(time.time() - t))
        return res

    def iou_sum(self, s: PrimState, target: torch.LongTensor) -> float:
        #t = time.time()
        state = None
        if self.__valid_cache:
            state = self.__voxels_cache
        else:
            state = s.voxelize(cubes=True)
        #target = BaseModel.get_model()
        #t = time.time()
        res = sum(self.__compute_iou(c, target) for c in state)
        #print("IOU SUM TIME: " + str(time.time() - t))
        return res / len(state)

    def parsimony(self, s: PrimState) -> float:
        P = PrimState.num_primitives
        return P - len(s)

    def __compute_iou(self, s: torch.LongTensor, t: torch.LongTensor) -> float:
        i = self.__volume_intersection(s, t)
        u = self.__volume_union(s, t)
        return i / u

    def __volume_intersection(self, s: torch.LongTensor, t: torch.LongTensor) -> float:
        l = int(math.pow(PrimState.voxel_grid_side, 3))
        x = torch.zeros(l, dtype=torch.long)
        x[s] = 1
        return (x & t).sum(dim=0).item() * self.__single_vox_volume

    def __volume_union(self, s: torch.LongTensor, t: torch.LongTensor) -> float:
        l = int(math.pow(PrimState.voxel_grid_side, 3))
        x = torch.zeros(l, dtype=torch.long)
        x[s] = 1   
        return (x | t).sum(dim=0).item() * self.__single_vox_volume
    