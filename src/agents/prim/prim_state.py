from __future__ import annotations
import copy, time, math, os
import torch
from typing import List, Union
from itertools import product

from geometry.cuboid import Cuboid
from agents.state import State

from torch_geometric.data import Data
from torch_geometric.nn.pool import voxel_grid

from trimesh import Trimesh
from trimesh.boolean import union


class PrimState(State):

    num_primitives = 0
    cube_side_len = 0
    __initial_state_cache = None
    __last_cube_size = 0
    __last_num_prims = 0

    def __init__(self, prim: List[Cuboid], reference: torch.FloatTensor, step: int, 
            bounding_box: torch.FloatTensor=None):
        #assert len(prim) == pow(self.num_primitives, 3)
        self.__primitives = prim
        self.__geom_cache = None
        self.__mesh_cache = None
        self.__vox_list_cache = None
        self.__vox_cache = None
        self.__ref = reference
        self.__step = torch.zeros(self.episode_len + 1, dtype=torch.long)
        self.__step[step] = 1
        self.__step_idx = step
        if bounding_box is not None:
            self.unit = torch.dist(bounding_box[1,:], bounding_box[0,:]).item() / 16.0
            self.bounding_box = bounding_box
        else:
            verts = torch.cat([c.get_pivots() for c in prim if c is not None])
            self.bounding_box = torch.stack([verts.min(dim=0)[0], verts.max(dim=0)[0]])
            self.unit = torch.dist(self.bounding_box[1,:], self.bounding_box[0,:]).item() / 16.0

    def __repr__(self) -> str:
        return repr(self.__primitives)

    def __len__(self) -> int:
        return len(self.get_live_primitives())

    def to_geom_data(self) -> Data:
        if self.__geom_cache is None:
            self.__geom_cache = Cuboid.aggregate(self.__primitives)
            self.__geom_cache.reference = self.__ref
            self.__geom_cache.step = self.__step.unsqueeze(0)
        return self.__geom_cache

    def meshify(self) -> Trimesh: 
        # NOTE: Don't use during training. It's very slow. Just for visual purposes.
        if self.__mesh_cache is None:
            self.__mesh_cache = union([
                c.get_mesh() for c in self.__primitives if c is not None
                ], engine='scad'
            )
        return self.__mesh_cache

    def voxelize(self, cubes: bool=False, use_cuda: bool=False) -> torch.LongTensor:
        """
        Efficient voxelization for union of cuboids shapes.
        """

        device = 'cuda' if use_cuda else 'cpu'

        prims = self.get_live_primitives()

        L = self.voxel_grid_side
        point_cloud = torch.cat([c.get_pivots() for c in prims])#.to(device)
        min_comp = point_cloud.min()
        max_comp = point_cloud.max()
        pitch = (max_comp - min_comp) / L #(self.max_coord - self.min_coord) / (self.voxel_grid_side) #

        if not cubes:
            G = torch.zeros(L, L, L, dtype=torch.long, device=device)   
        else:
            G = torch.zeros(len(prims), L, L, L, dtype=torch.long, device=device)

        idx = 0
        for c in prims:
            verts = c.get_pivots()#.to(device)   
            VOX = torch.floor((verts - min_comp) / pitch).long()
            if not cubes:
                G[VOX[0,0]:VOX[1,0], VOX[0,1]:VOX[1,1], VOX[0,2]:VOX[1,2]] = 1
            else:
                G[idx, VOX[0,0]:VOX[1,0], VOX[0,1]:VOX[1,1], VOX[0,2]:VOX[1,2]] = 1
            idx += 1
        
        if cubes:
            return G.view(len(prims), -1)
        else:
            return G.flatten()

    def get_live_primitives(self) -> List[Cuboid]:
        return [c for c in self.__primitives if c is not None]

    def get_primitives(self) -> List[Cuboid]:
        return copy.deepcopy(self.__primitives)

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
        init = PrimState(cubes, ref, 0, torch.tensor([[-1.0,-1.0,-1.0],[1.0,1.0,1.0]], dtype=torch.float))
        if cls.__initial_state_cache is None:
            cls.__initial_state_cache = init
            cls.__last_cube_size = cls.cube_side_len
            cls.__last_num_prims = cls.num_primitives
        return init
 