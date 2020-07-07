import sys
import torch
from typing import List

from torch.nn import Conv2d, ReLU, BatchNorm2d, MaxPool2d, Linear, ELU, Softmax

from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Polar
from torch_geometric.nn import GMMConv, BatchNorm, global_mean_pool

from agents.base_model import BaseModel
from agents.prim.prim_state import PrimState


class PrimModel(BaseModel):

    def __init__(self, ep_len: int, act_space_size: int):
        super(PrimModel, self).__init__()

        self.set_episode_len(ep_len)

        #NOTE: what about some dropout?
        
        # 0. Activations
        self.relu = ReLU()
        self.elu = ELU()
        self.softmax = Softmax(dim=1)

        # 1. Depth map processing stream
        # A standard convolutional image processing network. No batching: follows current episode.
        self.conv1 = Conv2d(1, 16, 3, padding=1)
        self.conv2 = Conv2d(16, 32, 3, padding=1)
        self.conv3 = Conv2d(32, 64, 3, padding=1)
        self.pool1 = MaxPool2d(5)
        self.pool2 = MaxPool2d(3)
        self.flatbn1 = BatchNorm2d(16) #Do we REALLY need this? We always process 1 image at a time! B=1
        self.flatbn2 = BatchNorm2d(32)
        self.flatbn3 = BatchNorm2d(64)

        # 2. State processing stream
        # A GMM convolutional network. Actually processing the batched input to the network.
        self.GMM1 = GMMConv(3, 16, 2, 5)
        self.GMM2 = GMMConv(16, 32, 2, 5)
        self.GMM3 = GMMConv(32, 64, 2, 5)
        # Here we could use some batchnorm, but I don't know if it works well
        self.bn1 = BatchNorm(16)
        self.bn2 = BatchNorm(32)
        self.bn3 = BatchNorm(64)
        self.fc1 = Linear(64, 256)

        # 3. Step processing stream
        # Simple FC layer. Also, no batching (follows current episode) 
        self.fc2 = Linear(ep_len, 256)

        # 4. Concatenation layer
        self.fc3 = Linear(256 * 3, act_space_size)


    def forward(self, state_batch: Batch) -> torch.Tensor:

        batch_size = state_batch.num_graphs

        x_1 = BaseModel.get_reference().unsqueeze(0).unsqueeze(1)
        x_1 = self.conv1(x_1)
        x_1 = self.relu(self.flatbn1(self.pool1(x_1)))
        x_1 = self.conv2(x_1)
        x_1 = self.relu(self.flatbn2(self.pool2(x_1)))
        x_1 = self.conv3(x_1)
        x_1 = self.relu(self.flatbn3(self.pool2(x_1))).flatten()
        x_1 = x_1.repeat(batch_size, 1)

        pos = state_batch.pos
        edges = state_batch.edge_index
        pseudo = state_batch.edge_attr      
        x_2 = self.elu(self.GMM1(pos, edges, pseudo))
        x_2 = self.elu(self.GMM2(x_2, edges, pseudo))
        x_2 = self.elu(self.GMM3(x_2, edges, pseudo))
        #this makes shape right but might cause huge info loss
        x_2 = global_mean_pool(x_2, state_batch.batch) 
        #print(x_2.shape)
        x_2 = self.elu(self.fc1(x_2))
  
        x_3 = torch.zeros(self.get_episode_len())
        x_3[self.get_step()] = 1
        x_3 = self.relu(self.fc2(x_3))
        x_3 = x_3.repeat(batch_size, 1)

        #print(x_1.shape, x_2.shape, x_3.shape)

        x = torch.cat([x_1, x_2, x_3], dim=1)
        #print(x.shape)
        x = self.fc3(x)
        return self.softmax(x)