import torch
from torch_geometric.data import Data


class State:

    '''
    def tensorize(self) -> torch.Tensor:
        raise NotImplementedError
    '''

    def to_geom_data(self) -> Data:
        raise NotImplementedError
