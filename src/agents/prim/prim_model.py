import sys
import torch
from typing import List

from torch.nn import Conv2d, ReLU, BatchNorm2d, MaxPool2d, Linear
#from torch.nn import ReLU

from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Polar
from torch_geometric.nn import GMMConv #, BatchNorm

from agents.base_model import BaseModel
from agents.prim.prim_state import PrimState


class PrimModel(BaseModel):

    def __init__(self, ep_len: int):
        super(PrimModel, self).__init__()

        self.set_episode_len(ep_len)
        
        self.relu = ReLU()

        # 1. Depth map processing stream
        # A standard convolutional image processing network. No batching: follows current episode.
        self.conv1 = Conv2d(1, 16, 3, padding=1)
        self.conv2 = Conv2d(16, 32, 3, padding=1)
        self.conv3 = Conv2d(32, 64, 3, padding=1)
        self.pool1 = MaxPool2d(5)
        self.pool2 = MaxPool2d(3)
        self.bn1 = BatchNorm2d(16)
        self.bn2 = BatchNorm2d(32)
        self.bn3 = BatchNorm2d(64)

        # 2. State processing stream
        # A GMM convolutional network. Actually processing the batched input to the network.
        # self.GMM1 = GMMConv()

        # 3. Step processing stream
        # Trivial FC layer. Also, no batching (follows current episode) 
        self.fc1 = Linear(ep_len, 256)

        # 4. Concatenation layer
        # 


    def forward(self, state_batch: Batch) -> torch.Tensor:

        x_1 = self.get_reference().unsqueeze(0).unsqueeze(1)
        x_1 = self.conv1(x_1)
        x_1 = self.relu(self.bn1(self.pool1(x_1)))
        x_1 = self.conv2(x_1)
        x_1 = self.relu(self.bn2(self.pool2(x_1)))
        x_1 = self.conv3(x_1)
        x_1 = self.relu(self.bn3(self.pool2(x_1))).flatten()
        x_1 = x_1.repeat(state_batch.num_graphs, 1)

        """self.get_step()
        pos = state_batch.pos
        edges = state_batch.edge_index
        pseudo = state_batch.edge_attr
        """
        #print(pos.shape, edges.shape, pseudo.shape)

        x_3 = torch.zeros(self.get_episode_len())
        x_3[self.get_step()] = 1
        x_3 = self.relu(self.fc1(x_3))
        x_3 = x_3.repeat(state_batch.num_graphs, 1)

        print(x_1.shape, x_3.shape)
        print(x_1)