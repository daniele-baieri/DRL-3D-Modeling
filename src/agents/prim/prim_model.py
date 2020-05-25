import sys, os
#sys.path.insert(0, os.path.abspath('../../'))

import torch
from torch.nn import Module


class PrimModel(Module):

    def __init__(self):
        super(PrimModel, self).__init__()