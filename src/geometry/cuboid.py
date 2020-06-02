from __future__ import annotations
import torch, time

from itertools import product
from typing import List

from geometry.primitive import Primitive

from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_trimesh, from_trimesh
from torch_geometric.transforms import FaceToEdge

from trimesh import Trimesh
from trimesh.creation import box


class Cuboid(Primitive):

    def __init__(self, v: torch.FloatTensor):
        """
        @param v: FloatTensor of shape (2,3) containing the coordinates of vertices V and V'
        """
        super().__init__()
        assert v.shape[0] == 2 and v.shape[1] == 3
        self.__vert = v
        new_center = (v[1,:] + v[0,:]) / 2.0
        T = torch.eye(4)
        T[[0,1,2],3] = new_center
        self.__shape = box(
            extents=v[0,:]-v[1,:],
            transform=T
        )
        #self.__shape.show()
        #print(self.__shape.center_mass)
        #print(self.__shape.centroid)
        #print(self.__shape.vertices)

    def __repr__(self) -> str:
        return repr(self.__vert.tolist())

    def slide(self, vertex: int, axis: int, amount: float) -> Cuboid:
        new = self.__vert
        new[vertex, axis] += amount
        return Cuboid(new)

    def get_pivots(self) -> torch.FloatTensor:
        return self.__vert

    def to_geom_data(self) -> Data:
        '''
        vertices = torch.stack([ #enumerate all 8 vertices from V and V'
            self.__vert[c,[0,1,2]]
            for c in product([0,1], repeat=3)
        ]).float()

        edges = torch.tensor([ #vertices are adjacent if they differ for a single coordinate
            [p, q]
            for p, q in product(range(len(vertices)), repeat=2)
            if sum(map(lambda x,y: bool(x-y),vertices[p],vertices[q])) == 1
        ], dtype=torch.long)

        return Data(pos=vertices, edge_index=edges.t().contiguous())
        '''
        T = FaceToEdge(remove_faces=False)
        return T(from_trimesh(self.__shape))

    def get_mesh(self) -> Trimesh:
        return self.__shape#.copy()

    @classmethod
    def aggregate(cls, C: List[Cuboid]) -> Data:
        data = [c.to_geom_data() for c in C if c is not None] #NOTE: check that this is right
        vertices = [d.pos for d in data]
        edges, faces, offset = [], [], 0
        for d in data:
            edges.append(d.edge_index + offset)
            faces.append(d.face + offset)
            offset += d.pos.shape[0]
        return Data(
            pos=torch.cat(vertices),
            edge_index=torch.cat(edges, dim=1),
            face=torch.cat(faces, dim=1)
        )
        #return Batch.from_data_list(data)

