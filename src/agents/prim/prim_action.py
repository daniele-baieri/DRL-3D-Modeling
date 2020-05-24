import sys, os
#sys.path.insert(0, os.path.abspath('../../'))
import torch

from typing import Set

from agents.prim.prim_state import PrimState
from geometry.primitive import Primitive
from agents.action import Action


class PrimAction(Action):

    primitives = 0
    vertices = 0

    def __init__(self, prim: int, vert: int, slide: int, delete: bool):
        assert prim >= 0 and prim < self.primitives and vert >= 0 and vert < self.vertices
        self.__delete = delete
        if not self.__delete:
            self.__prim = prim
            self.__vert = vert
            self.__slide = slide


    def _apply(self, s: PrimState) -> PrimState:
        pass

    @classmethod
    def init_action_space(cls, prims: int, verts: int):
        cls.primitives = prims
        cls.vertices = verts
