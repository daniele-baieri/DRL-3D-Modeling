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

    def __init__(self, prim: List[Cuboid], reference: torch.FloatTensor, step: int):
        #assert len(prim) == pow(self.num_primitives, 3)
        self.__primitives = prim
        self.__geom_cache = None
        self.__mesh_cache = None
        #self.__vox_list_cache = None
        #self.__vox_cache = None
        self.__ref = reference
        self.__step = torch.zeros(self.episode_len + 1, dtype=torch.long)
        self.__step[step] = 1
        self.__step_idx = step

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

    def voxelize(self, cubes: bool=False) -> Union[torch.LongTensor, List[torch.LongTensor]]:
        """
        Efficient voxelization for union of cuboids shapes.
        """
        #if cubes and self.__vox_list_cache is not None:
        #    return self.__vox_list_cache
        #elif not cubes and self.__vox_cache is not None:
        #    return self.__vox_cache

        prims = self.get_live_primitives()

        point_cloud = torch.cat([c.to_geom_data().pos for c in prims])        
        min_comp = point_cloud.min()
        max_comp = point_cloud.max()
        pitch = (max_comp - min_comp) / (self.voxel_grid_side)#(self.max_coord - self.min_coord) / (self.voxel_grid_side) #
        #offset = pitch / 2
        #vox_space = torch.linspace(min_comp, max_comp, self.voxel_grid_side)
        #Y_space = torch.linspace(self.min_coord+offset, self.max_coord-offset, self.voxel_grid_side)
        #Z_space = torch.linspace(self.min_coord+offset, self.max_coord-offset, self.voxel_grid_side)

        #subdivisions = [c.subdivide(vox_space, vox_space, vox_space) for c in prims]
        if not cubes:
            G = torch.zeros(self.voxel_grid_side, self.voxel_grid_side, self.voxel_grid_side, dtype=torch.long)
        else:
            G = []

        for c in prims:
            verts = c.get_pivots()
            VOX = torch.floor((verts - min_comp) / pitch).long()
            if not cubes:
                G[VOX[0,0]:VOX[1,0], VOX[0,1]:VOX[1,1], VOX[0,2]:VOX[1,2]] = 1
            else:
                H = torch.zeros(self.voxel_grid_side, self.voxel_grid_side, self.voxel_grid_side, dtype=torch.long)
                H[VOX[0,0]:VOX[1,0], VOX[0,1]:VOX[1,1], VOX[0,2]:VOX[1,2]] = 1
                G.append(H.flatten())
        
        if cubes:
            return torch.stack(G).to(os.environ['DEVICE'])
        else:
            return G.flatten().to(os.environ['DEVICE'])
        '''
        if cubes:
            # NOTE: unique concatenation of all these voxel grids == (cubes == False)
            self.__vox_list_cache = [
                torch.unique(
                    voxel_grid(
                        pc, torch.zeros(len(pc)), pitch, 
                        min_comp + offset, max_comp-offset#self.min_coord + offset, self.max_coord - offset#
                    ), 
                    sorted=False
                ) for pc in subdivisions if len(pc) > 0
            ]
            return self.__vox_list_cache
        else:
            sub_point_cloud = torch.cat(subdivisions).to(os.environ['DEVICE'])
            voxelgrid = voxel_grid(
                sub_point_cloud, torch.zeros(len(sub_point_cloud)), 
                pitch, min_comp, max_comp-offset#self.min_coord+offset, self.max_coord-offset#
            )
            self.__vox_cache = torch.unique(voxelgrid, sorted=False)
            return self.__vox_cache
        '''

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
        init = PrimState(cubes, ref, 0)
        if cls.__initial_state_cache is None:
            cls.__initial_state_cache = init
            cls.__last_cube_size = cls.cube_side_len
            cls.__last_num_prims = cls.num_primitives
        return init
 