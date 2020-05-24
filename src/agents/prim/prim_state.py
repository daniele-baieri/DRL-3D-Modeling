import sys, os
#sys.path.insert(0, os.path.abspath('../../'))
import torch
from typing import List

from geometry.primitive import Primitive
from agents.state import State


class PrimState(State):

    num_primitives = 3

    def __init__(self, prim: List[Primitive], ref: torch.FloatTensor, step: int):
        self.primitives = prim
        self.ref = ref
        self.step = step

    def tensorize(self) -> torch.Tensor:
        pass