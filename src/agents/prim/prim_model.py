import sys
import torch
import torch.nn as nn
from typing import List

from torch.nn import Conv2d, ReLU, BatchNorm2d, MaxPool2d, Linear, ELU, Softmax, Dropout

from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Polar
from torch_geometric.nn import GMMConv, BatchNorm, global_max_pool, GlobalAttention

from agents.base_model import BaseModel
from agents.prim.prim_state import PrimState


class PrimModel(BaseModel):

    def __init__(self, act_space_size: int):
        super(PrimModel, self).__init__()

        #self.set_episode_len(ep_len)
        self.ep_len = PrimState.episode_len + 1

        self.dropout = Dropout(p=0.5)
        
        # 0. Activations
        self.relu = ReLU()
        self.elu = ELU()

        # 1. Depth map processing stream
        # A standard convolutional image processing network.
        self.conv1 = Conv2d(1, 16, 3, padding=1)
        self.conv2 = Conv2d(16, 32, 3, padding=1)
        self.conv3 = Conv2d(32, 64, 3, padding=1)
        self.pool1 = MaxPool2d(5)
        self.pool2 = MaxPool2d(3)
        self.flatbn1 = BatchNorm2d(16) 
        self.flatbn2 = BatchNorm2d(32)
        self.flatbn3 = BatchNorm2d(64)

        # 2. State processing stream
        # A GMM convolutional network. 
        self.GMM1 = GMMConv(3, 16, 2, 5)
        self.GMM2 = GMMConv(16, 64, 2, 5)
        self.GMM3 = GMMConv(64, 256, 2, 5)
        # Here we could use some batchnorm, but I don't know if it works well
        self.bn1 = BatchNorm(16)
        self.bn2 = BatchNorm(64)
        self.bn3 = BatchNorm(256)
        # Pooling
        #self.pool_geom = GlobalAttention(Linear(64, 1), nn=Linear(64, 256))
        #self.fc1 = Linear(64, 256)

        # 3. Step processing stream
        # Simple FC layer.
        self.fc2 = Linear(self.ep_len, 256)

        # 4. Primitive liveness check
        # An FC layer encoding which primitives are still in a PrimState
        # This should help the network not to execute actions on deleted primitives, which have 0 reward
        self.fc3 = Linear(PrimState.num_primitives, 256)

        # 4. Concatenation layer
        self.fc4 = Linear(256 * 4, act_space_size)


    def forward(self, state_batch: Batch) -> torch.Tensor:

        #batch_size = state_batch.num_graphs

        x_1 = state_batch.reference.unsqueeze(1)
        x_1 = self.conv1(x_1)
        x_1 = self.relu(self.flatbn1(self.pool1(x_1)))
        x_1 = self.conv2(x_1)
        x_1 = self.relu(self.flatbn2(self.pool2(x_1)))
        x_1 = self.conv3(x_1)
        x_1 = self.relu(self.flatbn3(self.pool2(x_1))).view(x_1.shape[0], -1)
        x_1 = self.dropout(x_1)
        #x_1 = x_1.repeat(batch_size, 1)

        pos = state_batch.pos
        edges = state_batch.edge_index
        pseudo = state_batch.edge_attr      
        x_2 = self.elu(self.GMM1(pos, edges, pseudo))
        x_2 = self.elu(self.GMM2(x_2, edges, pseudo))
        x_2 = self.elu(self.GMM3(x_2, edges, pseudo))
        #this makes shape right but might cause huge info loss
        x_2 = global_max_pool(x_2, state_batch.batch) 
        #x_2 = self.pool_geom(x_2, state_batch.batch)

        #x_2 = self.elu(self.fc1(self.dropout(x_2)))
  
        x_3 = self.relu(self.fc2(state_batch.step.float()))

        x_4 = self.relu(self.fc3(state_batch.prims))

        x = torch.cat([x_1, x_2, x_3, x_4], dim=1)

        x = self.fc4(self.dropout(x))
        return x

    
    def get_initial_state(self, ref: torch.FloatTensor) -> PrimState:
        return PrimState.initial(ref)