from __future__ import annotations
import sys, os
#sys.path.insert(0, os.path.abspath('../../'))
import torch

from typing import Set, List
from itertools import product

from agents.prim.prim_state import PrimState
from geometry.primitive import Primitive
from agents.action import Action


class PrimAction(Action):

    act_space_size = 0
    primitives = 0
    vertices = 0
    slides = []

    def __init__(self, prim: int, vert: int=None, slide: float=None, axis: int=None, delete: bool=False):
        assert prim >= 0 and prim < self.primitives 

        self.__idx = -1
        self.__prim = prim
        self.__delete = delete
        if not delete:
            assert vert is not None and slide is not None
            assert vert >= 0 and vert < self.vertices
            assert axis >= 0 and axis < 3
            assert slide in set(self.slides)
            self.__vert = vert
            self.__axis = axis
            self.__slide = slide

    def __repr__(self) -> str:
        if self.__delete:
            return '<DEL primitive ' + str(self.__prim) + '>'
        else:
            return '<SLIDE vertex ' + str(self.__vert) + \
                ' of primitive ' + str(self.__prim) + \
                ' on axis ' + str(self.__axis) + \
                ' of amount ' + str(self.__slide) + '>'

    def _apply(self, s: PrimState) -> PrimState:
        prims = s.get_primitives()
        if self.__delete:
            prims.pop(self.__prim)
            return PrimState(prims)
        else:
            prims[self.__prim].slide(self.__vert, self.__axis, self.__slide)
            return PrimState(prims)

    def get_index(self) -> int:
        return self.__idx

    def set_index(self, idx: int) -> None:
        assert idx in range(0, self.act_space_size)
        self.__idx = idx

    @classmethod
    def init_action_space(cls, prims: int, verts: int, slides: List[float]) -> None:
        cls.primitives = prims
        cls.vertices = verts
        cls.slides = slides

    @classmethod
    def ground(cls) -> List[PrimAction]:
        res = []
        #ground sliding actions
        res.extend([
            PrimAction(p, vert=v, slide=s, axis=a) 
            for p, v, s, a in product(
                range(cls.primitives),
                range(cls.vertices),
                cls.slides,
                range(3)
            )
        ])
        
        #ground deleting actions
        res.extend([
            PrimAction(p, delete=True) 
            for p in range(cls.primitives)
        ])

        cls.act_space_size = len(res)

        return res