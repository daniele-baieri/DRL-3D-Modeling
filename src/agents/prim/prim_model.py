import sys
import torch
import torch.nn as nn
from typing import List

from torch.nn import Conv2d, ReLU, BatchNorm2d, MaxPool2d, Linear, ELU, Softmax, Dropout

from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Polar
from torch_geometric.nn import GMMConv, BatchNorm, global_mean_pool, GlobalAttention

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

        # 1. Reference processing stream
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
        self.sp1 = Linear(PrimState.num_primitives * 6, 128)
        self.sp2 = Linear(128, 256)

        #self.GMM1 = GMMConv(3, 16, 2, 3)
        #self.GMM2 = GMMConv(16, 32, 2, 3)
        #self.GMM3 = GMMConv(32, 64, 2, 3)
        #self.fc2 = Linear(64, 256)

        # Here we could use some batchnorm, but I don't know if it works well
        #self.bn1 = BatchNorm(3)
        #self.bn1 = BatchNorm(16)
        #self.bn2 = BatchNorm(32)
        #self.bn3 = BatchNorm(64)
        # Pooling
        #self.pool_geom = GlobalAttention(Linear(64, 1), nn=Linear(64, 256))
        #self.fc1 = Linear(64, 256)

        # 3. Step processing stream
        # Simple FC layer.
        self.fc3 = Linear(self.ep_len, 256)

        # 4. Primitive liveness check
        # An FC layer encoding which primitives are still in a PrimState
        # This should help the network not to execute actions on deleted primitives, which have 0 reward
        #self.fc4 = Linear(PrimState.num_primitives, 128)

        # 4. Concatenation layer
        self.fc4 = Linear(256 * 3, 1024)
        self.fc5 = Linear(1024, act_space_size)


    def forward(self, state_batch: Batch) -> torch.Tensor:
        #^ state_batch: Batch
        #batch_size = state_batch.num_graphs

        x_1 = state_batch.ref.unsqueeze(1)
        x_1 = self.conv1(x_1)
        x_1 = self.relu(self.flatbn1(self.pool1(x_1)))
        x_1 = self.conv2(x_1)
        x_1 = self.relu(self.flatbn2(self.pool2(x_1)))
        x_1 = self.conv3(x_1)
        x_1 = self.relu(self.flatbn3(self.pool2(x_1))).view(x_1.shape[0], -1)
        x_1 = self.dropout(x_1)
        #x_1 = x_1.repeat(batch_size, 1)

        '''
        pos = state_batch.pos
        edges = state_batch.edge_index
        pseudo = state_batch.edge_attr      
        x_2 = self.elu(self.GMM1(pos, edges, pseudo))
        x_2 = self.elu(self.GMM2(x_2, edges, pseudo))
        x_2 = self.elu(self.GMM3(x_2, edges, pseudo))
        x_2 = self.elu(self.fc2(x_2))
        #this makes shape right but might cause huge info loss
        x_2 = global_mean_pool(x_2, state_batch.batch) 
        '''
        x_2 = self.relu(self.sp1(state_batch.pivots))
        x_2 = self.relu(self.sp2(self.dropout(x_2)))

        x_3 = self.relu(self.fc3(state_batch.step))#state_batch.step.float()))

        #x_4 = self.relu(self.fc4(state_batch.prims))

        x = torch.cat([x_1, x_2, x_3], dim=1)

        x = self.relu(self.fc4(x))
        return self.fc5(x)

    
    def get_initial_state(self, ref: torch.FloatTensor) -> PrimState:
        return PrimState.initial(ref)