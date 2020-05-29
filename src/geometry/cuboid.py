from __future__ import annotations
import torch
from itertools import product
from typing import List
from geometry.primitive import Primitive
from torch_geometric.data import Data, Batch


class Cuboid(Primitive):

    def __init__(self, v: torch.FloatTensor):
        """
        @param v: FloatTensor of shape (2,3) containing the coordinates of vertices V and V'
        """
        super().__init__()
        assert v.shape[0] == 2 and v.shape[1] == 3
        self.__vert = v
        # self.__points = 2

    def __repr__(self) -> str:
        return repr(self.__vert.tolist())

    def slide(self, vertex: int, axis: int, amount: float) -> None:
        self.__vert[vertex,axis] += amount

    def get_pivots(self) -> torch.FloatTensor:
        return self.__vert

    def to_geom_data(self) -> Data:
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

    @classmethod
    def aggregate(cls, C: List[Cuboid]) -> Data:
        data = [c.to_geom_data() for c in C]
        """
        vertices = [d.pos for d in data]
        edges, offset = [], 0
        for d in data:
            edges.append(d.edge_index + offset)
            offset += d.pos.shape[0]
        return Data(
            pos=torch.cat(vertices),
            edge_index=torch.cat(edges, dim=1)
        )
        """
        return Batch.from_data_list(data)

