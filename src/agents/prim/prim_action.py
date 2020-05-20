import sys
sys.path.append('../../')
import torch

from typing import Set
from geometry.primitive import Primitive
from agents.state import State
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