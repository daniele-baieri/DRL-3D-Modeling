from __future__ import annotations
import copy, time, math, os
import torch
from typing import List, Union, Dict
from itertools import product

from geometry.cuboid import Cuboid

from agents.state import State

from agents.experience import Experience

from torch_geometric.data import Data, Batch
from torch_geometric.nn.pool import voxel_grid

from trimesh import Trimesh
from trimesh.boolean import union


class PrimState(State):

    num_primitives = 0
    cube_side_len = 0
    __initial_state_cache = None
    __last_cube_size = 0
    __last_num_prims = 0

    def __init__(self, prim: List[Cuboid], reference: torch.FloatTensor, step: int):#, bounding_box: torch.FloatTensor=None):
        #assert len(prim) == pow(self.num_primitives, 3)
        self.__primitives = prim
        self.__live_prims = [c for c in self.__primitives if c is not None]
        #self.__geom_cache = None
        self.__mesh_cache = None
        self.__ref = reference
        self.__step = torch.zeros(self.episode_len, dtype=torch.long)
        if step < self.episode_len:
            self.__step[step] = 1
        self.__step_idx = step

    def __repr__(self) -> str:
        return repr(self.__primitives)

    def __len__(self) -> int:
        return len(self.get_live_primitives())

    
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
        return Data(
            ref = self.__ref,
            pivots = self.get_cuboids_tensor().unsqueeze(0),
            step = self.get_step_onehot().float().unsqueeze(0),
            done = torch.tensor(self.__step_idx == self.episode_len)
        )
    

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

        prims = self.__primitives

        L = self.voxel_grid_side
        point_cloud = torch.cat([c.get_pivots() for c in self.__live_prims])#.to(device)
        min_comp = point_cloud.min()
        max_comp = point_cloud.max()
        pitch = (max_comp - min_comp) / L #(self.max_coord - self.min_coord) / (self.voxel_grid_side) #

        if not cubes:
            G = torch.zeros(L, L, L, dtype=torch.long, device=device)   
        else:
            G = torch.zeros(len(prims), L, L, L, dtype=torch.long, device=device)

        idx = 0
        for c in prims:
            if c is not None:
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

    def get_cuboids_tensor(self) -> torch.FloatTensor:
        return torch.cat([
            c.get_pivots().flatten() if c is not None
            else torch.zeros(6, dtype=torch.float)
            for c in self.__primitives
        ])

    def get_live_primitives(self) -> List[Cuboid]:
        return self.__live_prims

    def get_primitives(self) -> List[Cuboid]:
        return copy.deepcopy(self.__primitives)

    def get_reference(self) -> torch.FloatTensor:
        return self.__ref

    def get_step(self) -> int:
        return self.__step_idx

    def get_step_onehot(self) -> torch.LongTensor:
        return self.__step

    def is_deleted(self, prim: int) -> bool:
        return self.__primitives[prim] is None

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
        init = PrimState(cubes, ref, 0)
        if cls.__initial_state_cache is None:
            cls.__initial_state_cache = init
            cls.__last_cube_size = cls.cube_side_len
            cls.__last_num_prims = cls.num_primitives
        return init

    @classmethod
    def collate_prim_experiences(cls, exps: List[Experience]) -> Dict[str, torch.Tensor]:
        sources = Batch.from_data_list([e.get_source().to_geom_data() for e in exps])
        destinations = Batch.from_data_list([e.get_destination().to_geom_data() for e in exps])
        return {
            'src': sources,
            'dest': destinations,
            'act': torch.LongTensor([e.get_action().get_index() for e in exps]),
            'r': torch.FloatTensor([e.get_reward() for e in exps])
        }