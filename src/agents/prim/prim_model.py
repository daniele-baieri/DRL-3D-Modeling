import sys
import torch
from torch.nn import Module


class PrimModel(Module):

    def __init__(self):
        super(PrimModel, self).__init__()