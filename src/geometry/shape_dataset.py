import torch
from torch.utils.data import Dataset

from torch_geometric.datasets import ModelNet


class ShapeDataset(Dataset):

    def __init__(self, path: str):
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int): #-> ???
        # return 3D shape (mesh) and reference
        pass