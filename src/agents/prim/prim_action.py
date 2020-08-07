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
            return 'DEL(p=' + str(self.__prim) + ')'
        else:
            return 'SLIDE(v=' + str(self.__vert) + \
                ', p=' + str(self.__prim) + \
                ', a=' + str(self.__axis) + \
                ', m=' + str(self.__slide * PrimState.unit) + ')'

    def _apply(self, s: PrimState) -> PrimState:
        prims = s.get_primitives()
        if prims[self.__prim] is None:
            return s
        if self.__delete:
            prims[self.__prim] = None
            #new_bb = None
        else:
            c = prims[self.__prim]
            prims[self.__prim] = c.slide(self.__vert, self.__axis, self.__slide * PrimState.unit)
            #v = prims[self.__prim].get_pivots()
            #bb = s.bounding_box
            #new_bb = torch.tensor(
            #    [[min(v[0,i], bb[0,i]) for i in range(3)], [max(v[0,i], bb[0,i]) for i in range(3)]]
            #)
        return PrimState(prims, s.get_reference(), s.get_step() + 1)#, bounding_box=new_bb)

    def get_index(self) -> int:
        return self.__idx

    def set_index(self, idx: int) -> None:
        assert idx in range(0, self.act_space_size)
        self.__idx = idx

    def get_primitive(self) -> int:
        return self.__prim

    def is_delete(self) -> bool:
        return self.__delete

    @classmethod
    def init_action_space(cls, prims: int, verts: int, slides: List[float]) -> None:
        cls.primitives = prims
        cls.vertices = verts
        cls.slides = slides

    @classmethod
    def ground(cls) -> List[PrimAction]:
        res = [
            [
                PrimAction(p, vert=v, slide=s, axis=a) 
                for v, s, a in product(range(cls.vertices), cls.slides, range(3))
            ] + 
            [
                PrimAction(p, delete=True)
            ]
            for p in range(cls.primitives)
        ]
        res = [x for l in res for x in l]

        cls.act_per_prim = cls.vertices * 3 * len(cls.slides) + 1
        cls.act_space_size = len(res)
        
        return res
