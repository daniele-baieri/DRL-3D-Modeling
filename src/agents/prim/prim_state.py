import sys
sys.path.append('../../')
import torch

from typing import Set
from geometry.primitive import Primitive
from agents.state import State


class PrimState(State):

    num_primitives = 3

    def __init__(self, prim: Set[Primitive], ref: torch.FloatTensor, step: int):
        self.primitives = prim
        self.ref = ref
        self.step = step

    def tensorize() -> torch.Tensor:
        pass