from __future__ import annotations
import copy, time, math
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

    def __init__(self, prim: List[Cuboid]):
        #assert len(prim) == pow(self.num_primitives, 3)
        self.__primitives = prim
        self.__geom_cache = None
        self.__mesh_cache = None
        self.__vox_list_cache = None
        self.__vox_cache = None

    def __repr__(self) -> str:
        return repr(self.__primitives)

    def __len__(self) -> int:
        return len(self.get_live_primitives())

    def to_geom_data(self) -> Data:
        if self.__geom_cache is None:
            self.__geom_cache = Cuboid.aggregate(self.__primitives)
        return self.__geom_cache

    def meshify(self) -> Trimesh: 
        # NOTE: Don't use during training. It's very slow. Just for visual purposes.
        if self.__mesh_cache is None:
            self.__mesh_cache = union([
                c.get_mesh() for c in self.__primitives if c is not None
                ], engine='scad'
            )
        return self.__mesh_cache

    def voxelize(self, cubes: bool=False) -> Union[torch.LongTensor, List[torch.LongTensor]]:
        if cubes and self.__vox_list_cache is not None:
            return self.__vox_list_cache
        elif not cubes and self.__vox_cache is not None:
            return self.__vox_cache

        prims = self.get_live_primitives()
        grid_side = self.voxel_grid_side
        voxel_side = self.voxel_side

        point_cloud = torch.cat([c.to_geom_data().pos for c in prims])
        min_xyz = torch.min(point_cloud, dim=0)[0]
        max_xyz = torch.max(point_cloud, dim=0)[0]
        X_space = torch.linspace(min_xyz[0].item(), max_xyz[0].item(), grid_side+1)[:-1]
        Y_space = torch.linspace(min_xyz[1].item(), max_xyz[1].item(), grid_side+1)[:-1]
        Z_space = torch.linspace(min_xyz[2].item(), max_xyz[2].item(), grid_side+1)[:-1] 

        subdivisions = [c.subdivide(X_space, Y_space, Z_space) for c in prims]

        if cubes:
            # NOTE: unique concatenation of all these voxel grids == (cubes == False)
            self.__vox_list_cache = [
                torch.unique(
                    voxel_grid(
                        pc, torch.zeros(len(pc)), voxel_side, self.min_coord, self.max_coord
                    ), sorted=False
                ) for pc in subdivisions
            ]
            return self.__vox_list_cache
        else:
            sub_point_cloud = torch.cat(subdivisions)
            voxelgrid = voxel_grid(
                sub_point_cloud, torch.zeros(len(sub_point_cloud)), 
                voxel_side, self.min_coord, self.max_coord
            )
            self.__vox_cache = torch.unique(voxelgrid, sorted=False)
            return self.__vox_cache

    def get_live_primitives(self) -> List[Cuboid]:
        return [c for c in self.__primitives if c is not None]

    def get_primitives(self) -> List[Cuboid]:
        return copy.deepcopy(self.__primitives)

    @classmethod
    def init_state_space(cls, prim: int, voxelization_grid: int) -> None:
        cls.num_primitives = int(math.pow(prim, 3))
        cls.prims_per_side = prim
        cls.min_coord = -1.0
        cls.max_coord = 1.0
        cls.cube_side_len = (cls.max_coord - cls.min_coord) / prim
        cls.voxel_grid_side = voxelization_grid
        cls.voxel_side = (cls.max_coord - cls.min_coord) / cls.voxel_grid_side

    @classmethod
    def initial(cls) -> PrimState:
        # NOTE: this occupies the space [[-1 1][-1 1][-1 1]]
        # Since ShapeNet meshes are normalized in [-1 1]
        if cls.__initial_state_cache is not None and \
            cls.__last_num_prims == cls.num_primitives and \
            cls.__last_cube_size == cls.cube_side_len: 
            return cls.__initial_state_cache

        r = torch.linspace(cls.min_coord, cls.max_coord, cls.prims_per_side+1)[:-1]

        cubes = [
            Cuboid(torch.stack([tup, tup + cls.cube_side_len]))
            for tup in torch.cartesian_prod(r, r, r)
        ]
        init = PrimState(cubes)
        if cls.__initial_state_cache is None:
            cls.__initial_state_cache = init
            cls.__last_cube_size = cls.cube_side_len
            cls.__last_num_prims = cls.num_primitives
        return init
        
    '''
    def tensorize(self) -> torch.FloatTensor: # NOTE: this might just go to the trash
        if len(self.__primitives) == 0:
            return torch.empty(0)
        else:
            return torch.stack(
                [x.get_pivots() for x in self.__primitives]
            )
    '''
