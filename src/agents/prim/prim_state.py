from __future__ import annotations
import copy
import torch
from typing import List
from itertools import product

from geometry.cuboid import Cuboid
from agents.state import State

from torch_geometric.data import Data


class PrimState(State):

    num_primitives = 0
    cube_side_len = 0
    __initial_state_cache = None
    __last_cube_size = 0
    __last_num_prims = 0

    def __init__(self, prim: List[Cuboid]):
        #assert len(prim) == pow(self.num_primitives, 3)
        self.__primitives = prim

    def __repr__(self) -> str:
        return repr(self.__primitives)

    def tensorize(self) -> torch.FloatTensor:
        if len(self.__primitives) == 0:
            return torch.empty(0)
        else:
            return torch.stack(  #doubt about this shape: how do we want to process this? 
                [x.get_pivots() for x in self.__primitives]
            )

    def to_geom_data(self) -> Data:
        return Cuboid.aggregate(self.__primitives)

    def get_primitives(self) -> List[Cuboid]:
        return copy.deepcopy(self.__primitives)

    @classmethod
    def init_state_space(cls, prim: int, side_len: float) -> None:
        cls.num_primitives = prim
        cls.cube_side_len = side_len

    @classmethod
    def initial(cls) -> PrimState:
        if cls.__initial_state_cache is not None and \
            cls.__last_num_prims == cls.num_primitives and \
            cls.__last_cube_size == cls.cube_side_len: 
            return cls.__initial_state_cache

        #top = cls.num_primitives# * cls.cube_side_len
        step = cls.cube_side_len
        r = range(0, cls.num_primitives)#, step)

        cubes = [
            Cuboid(
                torch.tensor([
                    [x * step for x in tup], 
                    [x * step + step for x in tup]
                ])
            )
            for tup in product(r, repeat=3)
        ]
        init = PrimState(cubes)
        if cls.__initial_state_cache is None:
            cls.__initial_state_cache = init
            cls.__last_cube_size = cls.cube_side_len
            cls.__last_num_prims = cls.num_primitives
        return init
        



        