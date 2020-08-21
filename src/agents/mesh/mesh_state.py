from __future__ import annotations
import copy, time, math, os
import torch
from typing import List, Union
from itertools import product

from agents.state import State
from agents.prim.prim_state import PrimState

from torch_geometric.data import Data
from torch_geometric.nn.pool import voxel_grid

from trimesh import Trimesh
from trimesh.boolean import union


class MeshState(State):


    def __init__(self, reference: torch.FloatTensor, step: int):
        self.__ref = reference
        self.__step = torch.zeros(self.episode_len + 1, dtype=torch.long)
        self.__step[step] = 1
        self.__step_idx = step

    def to_geom_data(self) -> Data:
        '''
        if self.__geom_cache is None:
            if len(self.__live_prims) == 0:
                self.__geom_cache = Data(
                    pos=torch.FloatTensor([[0,0,0],[0,0,0]]), 
                    edge_index=torch.LongTensor([[0, 0],[0, 1]])
                )
            else:
                self.__geom_cache = Cuboid.aggregate(self.__primitives)
            self.__geom_cache.reference = self.__ref
            self.__geom_cache.step = self.__step.unsqueeze(0)
            self.__geom_cache.prims = torch.FloatTensor([[0 if p is None else 1 for p in self.__primitives]])
        return self.__geom_cache
        '''
        return None

    '''
    def meshify(self) -> Trimesh: 
        # NOTE: Don't use during training. It's very slow. Just for visual purposes.
        if self.__mesh_cache is None:
            self.__mesh_cache = union([
                c.get_mesh() for c in self.__primitives if c is not None
                ], engine='scad'
            )
        return self.__mesh_cache
    '''

    def voxelize(self, cubes: bool=False, use_cuda: bool=False) -> torch.LongTensor:
        return None

    def get_reference(self) -> torch.FloatTensor:
        return self.__ref

    def get_step(self) -> int:
        return self.__step_idx

    def get_step_onehot(self) -> torch.LongTensor:
        return self.__step

    @classmethod
    def init_state_space(cls, prim: int=3, voxelization_grid: int=64, episode_len: int=300, max_coord_abs: float=1.0) -> None:
        assert prim > 0 and voxelization_grid > 0 and max_coord_abs > 0
        cls.num_primitives = prim ** 3
        cls.prims_per_side = prim
        cls.min_coord = -max_coord_abs
        cls.max_coord = max_coord_abs
        cls.cube_side_len = (cls.max_coord - cls.min_coord) / prim
        cls.voxel_grid_side = voxelization_grid
        cls.voxel_side = (cls.max_coord - cls.min_coord) / (cls.voxel_grid_side - 1)
        cls.episode_len = episode_len
        cls.unit = math.sqrt(12.0) / 16.0

    '''
    @classmethod
    def initial(cls, ref: torch.FloatTensor) -> PrimState:

        if cls.__initial_state_cache is not None and \
            cls.__last_num_prims == cls.num_primitives and \
            cls.__last_cube_size == cls.cube_side_len: 
            return cls.__initial_state_cache

        r = torch.linspace(cls.min_coord, cls.max_coord, cls.prims_per_side+1)[:-1]

        cubes = [
            Cuboid(torch.stack([tup, tup + cls.cube_side_len]))
            for tup in torch.cartesian_prod(r, r, r)
        ]
        init = PrimState(cubes, ref, 0)#, torch.tensor([[-1.0,-1.0,-1.0],[1.0,1.0,1.0]], dtype=torch.float))
        if cls.__initial_state_cache is None:
            cls.__initial_state_cache = init
            cls.__last_cube_size = cls.cube_side_len
            cls.__last_num_prims = cls.num_primitives
        return init
    '''
 