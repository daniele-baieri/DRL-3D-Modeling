import torch
from torch.utils.data import Dataset


class ShapeDataset(Dataset):

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int): #-> ???
        # return 3D shape (mesh) and reference
        pass