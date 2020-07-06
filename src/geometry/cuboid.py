from __future__ import annotations
import torch, time, os

from itertools import product
from typing import List

from geometry.primitive import Primitive

from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_trimesh, from_trimesh
from torch_geometric.transforms import FaceToEdge

from trimesh import Trimesh
from trimesh.creation import box
from trimesh.repair import broken_faces


CUBE_MASK = [
    [0,0,0],
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,0],
    [1,0,1],
    [1,1,0],
    [1,1,1]
]

class Cuboid(Primitive):

    def __init__(self, v: torch.FloatTensor):
        """
        @param v: FloatTensor of shape (2,3) containing the coordinates of vertices V and V'
        """
        super().__init__()
        assert v.shape[0] == 2 and v.shape[1] == 3
        self.__vert = v
        
        # this creates a Trimesh.
        new_center = (v[1,:] + v[0,:]) / 2.0
        T = torch.eye(4)
        T[[0,1,2],3] = new_center
        self.__trimesh = box(extents=v[0,:]-v[1,:], transform=T)
        edge_maker = FaceToEdge()

        self.__shape = edge_maker(from_trimesh(self.__trimesh))


    def __repr__(self) -> str:
        return repr(self.__vert.tolist())

    def slide(self, vertex: int, axis: int, amount: float) -> Cuboid:
        new = self.__vert.clone().detach()
        new[vertex, axis] += amount
        return Cuboid(new)

    def get_pivots(self) -> torch.FloatTensor:
        return self.__vert

    def to_geom_data(self) -> Data:
        '''
        vertices = self.__vert[CUBE_MASK, [0,1,2]]

        edges = torch.tensor([ #vertices are adjacent if they differ for a single coordinate
            [p, q]
            for p, q in product(range(len(vertices)), repeat=2)
            if sum(map(lambda x,y: bool(x-y),vertices[p],vertices[q])) == 1
        ], dtype=torch.long)

        return Data(pos=vertices, edge_index=edges.t().contiguous())
        '''
        return self.__shape

    def get_mesh(self) -> Trimesh: 
        #NOTE: DO NOT use in training. Just for visualization/output
        return self.__trimesh #.copy()

    def subdivide(self, X: torch.FloatTensor, Y: torch.FloatTensor, Z: torch.FloatTensor) -> torch.Tensor:
        min_coord = torch.min(self.__vert, dim=0)[0]
        max_coord = torch.max(self.__vert, dim=0)[0]
        X_valid = X[(X >= min_coord[0]) & (X <= max_coord[0])]
        Y_valid = Y[(Y >= min_coord[1]) & (Y <= max_coord[1])]
        Z_valid = Z[(Z >= min_coord[2]) & (Z <= max_coord[2])]
        return torch.cartesian_prod(X_valid, Y_valid, Z_valid).to(os.environ['DEVICE'])

    #def voxelized(self, )

    @classmethod
    def aggregate(cls, C: List[Cuboid]) -> Data:
        data = [c.to_geom_data() for c in C if c is not None] #NOTE: check that this is right
        vertices = [d.pos for d in data]
        edges, offset = [], 0
        for d in data:
            edges.append(d.edge_index + offset)
            #faces.append(d.face + offset)
            offset += d.pos.shape[0]
        return Data(
            pos=torch.cat(vertices),
            edge_index=torch.cat(edges, dim=1)#,
            #face=torch.cat(faces, dim=1)
        )
        #return Batch.from_data_list(data)
