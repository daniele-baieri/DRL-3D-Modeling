import torch
import math, time, os
from typing import Tuple, Set

from agents.reward import Reward
from agents.state import State
from agents.prim.prim_state import PrimState
from agents.prim.prim_model import PrimModel
from agents.base_model import BaseModel


class PrimReward(Reward):

    def __init__(self, alpha_1: float, alpha_2: float, device: str):
        self.__alpha_1 = alpha_1
        self.__alpha_2 = alpha_2
        self.__single_vox_volume = math.pow(PrimState.voxel_side, 3)
        self.__valid_cache = False
        self.__device = device
    
    def _forward(self, old: PrimState, new: PrimState, model: torch.LongTensor) -> float:
        
        assert old is not None and new is not None
        if len(new) == 0:
            return -1
        cuda = self.__device == 'cuda'

        target = model.to(self.__device)

        cubes_vox_new = new.voxelize(cubes=True, use_cuda=cuda)
        cubes_vox_old = old.voxelize(cubes=True, use_cuda=cuda)
        vox_new = cubes_vox_new.sum(dim=0)
        vox_old = cubes_vox_old.sum(dim=0)

        iou_new, iou_sum_new = self.iou(vox_new, target), self.iou_sum(cubes_vox_new, target, len(new))
        iou_old, iou_sum_old = self.iou(vox_old, target), self.iou_sum(cubes_vox_old, target, len(old))

        iou = iou_new - iou_old
        iou_sum = self.__alpha_1 * (iou_sum_new - iou_sum_old)
        parsimony = self.__alpha_2 * (self.parsimony(new) - self.parsimony(old))
        res = iou + iou_sum + parsimony
        #assert abs(res.item()) < 1
        return res.item()

    def iou(self, s: torch.LongTensor, target: torch.LongTensor) -> torch.FloatTensor:
        #s.meshify().show()
        #if torch.max(state) > 262144:
        #    s.meshify().show()
        #    return
        #t = time.time()
        i = (s & target).sum().float()
        u = (s | target).sum().float()
        res = torch.div(i, u)
        #print("IOU TIME: " + str(time.time() - t))
        return res

    def iou_sum(self, s: torch.LongTensor, target: torch.LongTensor, live_prims: int) -> torch.FloatTensor:
        #t = time.time()
        i = (s & target).sum(dim=1).float()
        u = (s | target).sum(dim=1).float()
        res = torch.div(i, u).sum()
        #print("IOU SUM TIME: " + str(time.time() - t))
        return res / live_prims

    def parsimony(self, s: PrimState) -> float:
        P = PrimState.num_primitives
        return P - len(s)
