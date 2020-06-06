import torch
import trimesh
import warnings

from typing import List, Dict
from pathlib import Path

from torch.utils.data import Dataset

from torch_geometric.datasets import ModelNet

# Synset to Label mapping (for ShapeNet core classes)
synset_to_label = {'04379243': 'table', '03211117': 'monitor', '04401088': 'phone',
                '04530566': 'watercraft', '03001627': 'chair', '03636649': 'lamp',
                '03691459': 'speaker', '02828884': 'bench', '02691156': 'plane',
                '02808440': 'bathtub', '02871439': 'bookcase', '02773838': 'bag',
                '02801938': 'basket', '02880940': 'bowl', '02924116': 'bus',
                '02933112': 'cabinet', '02942699': 'camera', '02958343': 'car',
                '03207941': 'dishwasher', '03337140': 'file', '03624134': 'knife',
                '03642806': 'laptop', '03710193': 'mailbox', '03761084': 'microwave',
                '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
                '04004475': 'printer', '04099429': 'rocket', '04256520': 'sofa',
                '04554684': 'washer', '04090263': 'rifle', '02946921': 'can'}

# Label to Synset mapping (for ShapeNet core classes)
label_to_synset = {v: k for k, v in synset_to_label.items()}


class ShapeDataset(Dataset):


    def __init__(self, path: str, categories: List[str]=['chair'], train: bool=True, split: float=.8):
        self.root = Path(path)
        self.paths = []
        self.synset_idxs = []
        self.synsets = self.__convert_categories(categories)
        self.labels = [synset_to_label[s] for s in self.synsets]

        # loops through desired classes
        for i in range(len(self.synsets)):
            syn = self.synsets[i]
            class_target = self.root / syn
            if not class_target.exists():
                raise ValueError(
                    'Class {0} ({1}) was not found at location {2}.'.format(
                    syn, self.labels[i], str(class_target))
                )

            # find all objects in the class
            models = sorted(class_target.glob('*'))
            stop = int(len(models) * split)
            if train:
                models = models[:stop]
            else:
                models = models[stop:]
            self.paths += models
            self.synset_idxs += [i] * len(models)

        self.names = [p.name for p in self.paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict:
        # return 3D shape (mesh) and reference
        # label = self.labels[self.synset_idxs[idx]]
        obj_location = self.paths[idx] / 'model_normalized.obj'
        mesh = trimesh.load(str(obj_location))

        return {
            'mesh': mesh, 'reference': None
        }

    def __convert_categories(self, categories):
        assert categories is not None, 'List of categories empty'
        synsets = [label_to_synset[c] for c in categories if c in label_to_synset.keys()]
        if len(synsets) < len(categories):
            warnings.warn('Selected unavailable categories - skipped')
        return synsets