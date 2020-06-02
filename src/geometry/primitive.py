import typing

from trimesh import Trimesh


class Primitive:

    def __init__(self):
        pass

    def slide(self, vertex: int, axis: int, amount: float):
        """
        @param vertex: the vertex to slide along axis 'axis'
        @param axis: the axis along which to slide vertex 'vertex'
        @param amount: the additive factor of which to slide 'vertex' on 'axis'
        """
        raise NotImplementedError

    def meshify(self) -> Trimesh:
        raise NotImplementedError