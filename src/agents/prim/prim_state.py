import sys, os, copy
#sys.path.insert(0, os.path.abspath('../../'))
import torch
from typing import List, Dict

from geometry.cuboid import Cuboid
from agents.state import State


class PrimState(State):

    num_primitives = 0

    def __init__(self, prim: List[Cuboid]):
        #assert len(prim) == pow(self.num_primitives, 3)
        self.__primitives = prim

    def tensorize(self) -> torch.FloatTensor:
        return torch.stack(  #doubt about this shape: how do we want to process this? voxels? all vertices?
            [x.get_pivots() for x in self.__primitives]
        )

    def get_primitives(self) -> List[Cuboid]:
        return copy.deepcopy(self.__primitives)

    @classmethod
    def init_state_space(cls, prim: int) -> None:
        cls.num_primitives = prim