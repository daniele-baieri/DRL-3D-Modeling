import torch
import sys
sys.path.append('./src/geometry/')
from typing import Set
from primitive import Primitive


class State:

    def __init__(self, prim: Set[Primitive], ref: torch.FloatTensor, step: int):
        self.primitives = prim
        self.ref = ref
        self.step = step
