import torch
import trimesh
#from trimesh import Trimesh
from torch.nn import Module
from torch_geometric.data import Data

from agents.state import State


class BaseModel(Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def get_initial_state(self, ref: torch.FloatTensor) -> State:
        raise NotImplementedError

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError