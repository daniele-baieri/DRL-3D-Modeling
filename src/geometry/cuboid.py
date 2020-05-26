import torch
from geometry.primitive import Primitive


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

    '''
    def get_vertices(self) -> torch.FloatTensor:
        return torch.FloatTensor([
            []
            for i in range(3)
        ])
    '''
    