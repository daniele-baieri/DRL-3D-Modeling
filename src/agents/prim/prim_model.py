import sys
import torch
from typing import List

from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Polar
from torch_geometric.nn import GMMConv, BatchNorm

from agents.base_model import BaseModel
from agents.prim.prim_state import PrimState


class PrimModel(BaseModel):

    def __init__(self):
        super(PrimModel, self).__init__()
        
        # 1. Depth map processing stream
        # A standard convolutional image processing network. No batching: follows current episode.
         
        # 2. State processing stream
        # A GMM convolutional network. Actually processing the batched input to the network.
        # self.__conv3D1 = GMMConv()

        # 3. Step processing stream
        # Trivial FC layer. Also, no batching (follows current episode) 


    def forward(self, state_batch: Batch) -> torch.Tensor:

        pos = state_batch.pos
        edges = state_batch.edge_index
        pseudo = state_batch.edge_attr

        print(pos.shape, edges.shape, pseudo.shape)