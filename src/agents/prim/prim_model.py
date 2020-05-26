import sys
import torch
from agents.base_model import BaseModel


class PrimModel(BaseModel):

    def __init__(self):
        super(PrimModel, self).__init__()

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError