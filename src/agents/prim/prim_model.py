import sys
import torch

from torch_geometric.data import Data
from torch_geometric.transforms import Polar
from torch_geometric.nn import GMMConv, BatchNorm

from agents.base_model import BaseModel


class PrimModel(BaseModel):

    def __init__(self):
        super(PrimModel, self).__init__()
        
        # 1. Depth map processing stream
        # A standard convolutional image processing network. No batching: follows current episode.
         
        # 2. State processing stream
        # A GMM convolutional network. Actually processing the batched input to the network.
        self.__polar = Polar(cat=False)


        # 3. Step processing stream
        # Trivial FC layer. Also, no batching (follows current episode) 
        

    def forward(self, state_geom: Data) -> torch.Tensor:
        pos, edges = state_geom.pos, state_geom.edge_index
        pseudo = self.__polar(state_geom)
